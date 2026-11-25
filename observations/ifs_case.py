"""A forecast case from ECMWF open data: the IFS control (with its analyses before the run) or the IFS-ENS members.

    python observations/ifs_case.py control --run 2026-11-25T00 --start 2026-11-01 --elias DIR --archive DIR --out c.npz
    python observations/ifs_case.py members --run 2026-11-25T00 --members 1-10 --elias DIR --out m01.npz

Data come from ECMWF's open-data copy on Google Cloud only (storage.googleapis.com/ecmwf-open-data; CC BY 4.0): each
message is read by its byte range from the cycle's index. `control`: the step-0 analyses of every 00 and 12 UTC cycle
from --start to the run, then the run's own forecast to 360 h, all at 12 h, through the ELIAS 2.0 networks as
era5_wcb.py does for ERA5; adds the 500 hPa height and sea-level pressure on the networks' 1 degree grid and the zonal
means of u, T and height on all 14 levels. `members`: the same fields (no networks) for each perturbed member at
12-hourly steps, with the 300 hPa PV and its advection by the irrotational wind (the outflow's push on the tropopause).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import era5_wcb as W

BASE = "https://storage.googleapis.com/ecmwf-open-data"
UA = {"User-Agent": "qg-channel-ifs-case/0.1"}
LEVELS = (10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)


def url(run: pd.Timestamp, step: int, stream: str, kind: str) -> str:
    return f"{BASE}/{run:%Y%m%d}/{run:%H}z/ifs/0p25/{stream}/{run:%Y%m%d%H}0000-{step}h-{stream}-{kind}"


def get(u: str, rng: tuple[int, int] | None = None, tries: int = 5) -> bytes:
    for attempt in range(1, tries + 1):
        try:
            h = dict(UA)
            if rng:
                h["Range"] = f"bytes={rng[0]}-{rng[0] + rng[1] - 1}"
            with urllib.request.urlopen(urllib.request.Request(u, headers=h), timeout=120) as r:
                return r.read()
        except Exception as e:  # transient network errors: retry, then give up on this cycle
            if attempt == tries:
                raise
            print(f"    retry {attempt}: {type(e).__name__} {u[-60:]}", flush=True)
            time.sleep(10 * attempt)
    raise RuntimeError(u)


def index(run, step, stream, kind) -> list[dict]:
    return [json.loads(x) for x in get(url(run, step, stream, kind) + ".index").decode().splitlines() if x.strip()]


def decode(buf: bytes) -> np.ndarray:
    """One GRIB message -> (721, 1440) from 90N and from 0E."""
    import eccodes

    h = eccodes.codes_new_from_message(buf)
    try:
        ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        lat0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        x = eccodes.codes_get_values(h).reshape(nj, ni)
    finally:
        eccodes.codes_release(h)
    if (ni, nj) != (1440, 721) or lat0 != 90.0:
        raise SystemExit(f"unexpected grid {ni}x{nj} from {lat0}")
    return np.roll(x, round(lon0 / 0.25), axis=1)  # lon0 = -180 -> start at 0E


def want(rows: list[dict], need: dict[str, tuple], number: str | None) -> list[dict]:
    out = []
    for r in rows:
        if number is not None and r.get("number") != number:
            continue
        p = r["param"]
        if p not in need:
            continue
        pl = r.get("levtype") == "pl" and int(r["levelist"]) in need[p]
        if pl or (r.get("levtype") == "sfc" and need[p] == ()):
            out.append(r)
    return out


def read(run, step, stream, kind, need, number=None, pool=None) -> dict:
    """{param: (levels, array (levels, 721, 1440))}; surface fields have levels [0]."""
    rows = want(index(run, step, stream, kind), need, number)
    u = url(run, step, stream, kind) + ".grib2"
    bufs = list(pool.map(lambda r: get(u, (r["_offset"], r["_length"])), rows))
    got: dict[str, dict] = {}
    for r, b in zip(rows, bufs, strict=True):
        got.setdefault(r["param"], {})[int(r.get("levelist", 0))] = decode(b)
    out = {}
    for p, levs in need.items():
        have = got.get(p, {})
        ls = sorted(have) if levs == () else sorted(levs)
        missing = [lev for lev in ls if lev not in have] if levs else ([] if have else [0])
        if missing:
            raise SystemExit(f"{run} +{step} h: {p} missing {missing}")
        out[p] = (ls, np.stack([have[lev] for lev in ls]))
    return out


def extras(raw: dict, E) -> dict:
    """Per time: Z500 (m) and sea-level pressure (hPa) on the networks' 1 degree grid, and zonal means (level, 721)."""
    rows = E.ROWS.size
    gh = raw["gh"]
    z500 = W.box_mean(gh[1][gh[0].index(500)].astype(np.float64)[: 4 * (rows - 1) + 3], rows, E.LON.size)
    msl = W.box_mean(raw["msl"][1][0].astype(np.float64)[: 4 * (rows - 1) + 3], rows, E.LON.size) / 100.0
    return {
        "z500": z500[1:-1].astype(np.float32),
        "msl": msl[1:-1].astype(np.float32),
        "zm_u": raw["u"][1].mean(-1).astype(np.float32),
        "zm_t": raw["t"][1].mean(-1).astype(np.float32),
        "zm_gh": gh[1].mean(-1).astype(np.float32),
    }


def control(a, E) -> dict:
    run = pd.Timestamp(a.run)
    start = pd.Timestamp(a.start)
    times = pd.date_range(start, run + pd.Timedelta(hours=a.last_step), freq="12h")
    need = {k: tuple(sorted(set(v) | W.EXTRA.get(k, set()))) for k, v in E.LEVELS.items()}
    need.update(u=LEVELS, t=LEVELS, gh=LEVELS, msl=())
    pool = ThreadPoolExecutor(16)

    def fetch(t):
        t0 = time.time()
        cyc, step = (t, 0) if t <= run else (run, int((t - run) / pd.Timedelta(hours=1)))
        raw = read(cyc, step, "oper", "fc", need, pool=pool)
        print(
            f"  {t:%Y-%m-%d %H}Z ({'analysis' if step == 0 else f'+{step} h'}) in {time.time() - t0:.0f} s", flush=True
        )
        return raw

    lat = np.linspace(90.0, -90.0, 721)
    out = W.identify(times, fetch, E, Path(a.archive), lat, workers=2, extra=lambda raw: extras(raw, E))
    out.update(run=np.datetime64(run, "s"), levels=np.array(LEVELS), zlat=lat)
    return out


def members(a, E) -> dict:
    """Per member and 12-hourly step: Z500, MSLP, precipitation since the start (mm), zonal means, and the 300 hPa PV
    and its advection by the irrotational wind."""
    run = pd.Timestamp(a.run)
    lo, hi = (int(x) for x in a.members.split("-"))
    nums = list(range(lo, hi + 1))
    steps = list(range(0, a.last_step + 1, 12))
    need = {"u": LEVELS, "v": tuple(sorted({300, *W.PV_LEVS})), "t": LEVELS, "gh": LEVELS, "msl": (), "tp": ()}
    lat = np.linspace(90.0, -90.0, 721)
    rows = E.ROWS.size
    north = 4 * (rows - 1) + 3
    metre = E.R_EARTH * np.pi / 180.0
    cos = np.cos(np.deg2rad(E.LAT))[:, None]
    pool = ThreadPoolExecutor(16)
    keys = ("z500", "msl", "zm_u", "zm_t", "zm_gh", "pv300", "adv300", "tp")
    res = {k: [] for k in keys}
    for n in nums:
        per = {k: [] for k in keys}
        for step in steps:
            t0 = time.time()
            nd = dict(need) if step else {k: v for k, v in need.items() if k != "tp"}
            raw = read(run, step, "enfo", "pf", nd, number=str(n), pool=pool)
            x = extras(raw, E)
            fp = {
                k: {
                    lev: W.box_mean(raw[k][1][raw[k][0].index(lev)].astype(np.float64)[:north], rows, E.LON.size)
                    for lev in W.PV_LEVS
                }
                for k in ("u", "v", "t")
            }
            u3, v3 = (raw[k][1][raw[k][0].index(300)].astype(np.float64) for k in ("u", "v"))
            wind = E.irrotational_wind(W.divergence(u3, v3, lat)[:-1])
            pv = W.potential_vorticity(fp, E)
            adv = -86400.0 * (wind[0] * W.ddx(pv, metre, cos) + wind[1] * np.gradient(pv, axis=-2) / metre)
            tp = raw["tp"][1][0] * 1000.0 if step else np.zeros((721, 1440))
            x.update(
                pv300=pv.astype(np.float16),
                adv300=adv.astype(np.float16),
                tp=W.box_mean(tp[:north], rows, E.LON.size)[1:-1].astype(np.float32),
            )
            for k in keys:
                per[k].append(x[k])
            print(f"  member {n} +{step} h in {time.time() - t0:.0f} s", flush=True)
        for k in keys:
            res[k].append(np.stack(per[k]))
    return dict(
        run=np.datetime64(run, "s"),
        numbers=np.array(nums),
        steps=np.array(steps),
        levels=np.array(LEVELS),
        zlat=lat,
        lat=E.LAT,
        lon=E.LON,
        **{k: np.stack(v) for k, v in res.items()},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("control", "members"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--start")
    ap.add_argument("--members", default="1-50")
    ap.add_argument("--last-step", type=int, default=360)
    ap.add_argument("--elias", required=True)
    ap.add_argument("--archive")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.elias)
    import elias as E

    out = control(a, E) if a.mode == "control" else members(a, E)
    np.savez_compressed(a.out, **out)
    print("wrote", a.out, {k: getattr(v, "shape", None) for k, v in out.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
