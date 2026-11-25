"""A forecast case from ECMWF open data: the IFS control (with its analyses before the run) or the IFS-ENS members;
--model aifs-single / aifs-ens takes the forecast from the AIFS single run or the AIFS-ENS control and members instead.

    python observations/ifs_case.py control --run 2026-11-25T00 --start 2026-11-01 --elias DIR --archive DIR --out c.npz
    python observations/ifs_case.py members --run 2026-11-25T00 --members 1-10 --elias DIR --out m01.npz
    python observations/ifs_case.py hindcast --runs 2026-10-01T00,2026-10-03T00 --members 1-25 --elias DIR --out h.npz
    python observations/ifs_case.py analyses --start 2026-10-01 --end 2026-11-25 --elias DIR --out a.npz

Data come from ECMWF's open-data copy on Google Cloud only (storage.googleapis.com/ecmwf-open-data; CC BY 4.0): each
message is read by its byte range from the cycle's index. `control`: the step-0 analyses of every 00 and 12 UTC cycle
from --start to the run, then the run's own forecast to 360 h, all at 12 h, through the ELIAS 2.0 networks as
era5_wcb.py does for ERA5; adds the 500 hPa height and sea-level pressure on the networks' 1 degree grid and the zonal
means of u, T and height on all 14 levels. `members`: the same fields (no networks) for each perturbed member at
12-hourly steps, with the 300 hPa PV and its advection by the irrotational wind (the outflow's push on the tropopause).
`hindcast` and `analyses`: earlier IFS-ENS cycles and the verifying analyses, daily, on coarse grids (coarse()).
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
# model -> (directory, control stream and file kind, members' file kind); the AIFS-ENS files carry geopotential z
MODELS = {
    "ifs": ("ifs", ("oper", "fc"), "ef"),
    "aifs-single": ("aifs-single", ("oper", "fc"), None),
    "aifs-ens": ("aifs-ens", ("enfo", "cf"), "pf"),
}
G = 9.80665


def url(run: pd.Timestamp, step: int, stream: str, kind: str, model: str = "ifs") -> str:
    d = MODELS[model][0]
    return f"{BASE}/{run:%Y%m%d}/{run:%H}z/{d}/0p25/{stream}/{run:%Y%m%d%H}0000-{step}h-{stream}-{kind}"


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


def index(run, step, stream, kind, model="ifs") -> list[dict]:
    return [
        json.loads(x) for x in get(url(run, step, stream, kind, model) + ".index").decode().splitlines() if x.strip()
    ]


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


def read(run, step, stream, kind, need, number=None, pool=None, model="ifs") -> dict:
    """{param: (levels, array (levels, 721, 1440))}; surface fields have levels [0]. Files without geopotential
    height (AIFS-ENS) give gh from z."""
    rows = index(run, step, stream, kind, model)
    if "gh" in need and not any(r["param"] == "gh" for r in rows):
        nz = {k: v for k, v in need.items() if k != "gh"}
        nz["z"] = tuple(sorted(set(need["gh"]) | set(need.get("z", ()))))
        out = read(run, step, stream, kind, nz, number, pool, model)
        levs, z = out["z"]
        if "z" in need:
            out["z"] = (sorted(need["z"]), z[[levs.index(lev) for lev in sorted(need["z"])]])
        else:
            del out["z"]
        gh = sorted(need["gh"])
        return {**out, "gh": (gh, z[[levs.index(lev) for lev in gh]] / G)}
    rows = want(rows, need, number)
    u = url(run, step, stream, kind, model) + ".grib2"
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
        if t <= run:
            raw = read(t, 0, "oper", "fc", need, pool=pool)
            step = 0
        else:
            step = int((t - run) / pd.Timedelta(hours=1))
            raw = read(run, step, *MODELS[a.model][1], need, pool=pool, model=a.model)
        print(
            f"  {t:%Y-%m-%d %H}Z ({'analysis' if step == 0 else f'+{step} h'}) in {time.time() - t0:.0f} s", flush=True
        )
        return raw

    lat = np.linspace(90.0, -90.0, 721)
    out = W.identify(times, fetch, E, Path(a.archive), lat, workers=2, extra=lambda raw: extras(raw, E))
    out.update(run=np.datetime64(run, "s"), levels=np.array(LEVELS), zlat=lat)
    return out


def state(raw: dict, E) -> dict:
    """One member's or analysis' fields: extras() plus the 300 hPa PV and its advection by the irrotational wind
    (PVU/day), u at 250 hPa and precipitation since the start (mm), on the networks' 1 degree grid."""
    lat = np.linspace(90.0, -90.0, 721)
    rows = E.ROWS.size
    north = 4 * (rows - 1) + 3
    metre = E.R_EARTH * np.pi / 180.0
    cos = np.cos(np.deg2rad(E.LAT))[:, None]
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
    tp = raw["tp"][1][0] * 1000.0 if "tp" in raw else np.zeros((721, 1440))
    u250 = fp["u"][250][1:-1]
    x.update(
        pv300=pv.astype(np.float32),
        adv300=adv.astype(np.float32),
        u250=u250.astype(np.float32),
        tp=W.box_mean(tp[:north], rows, E.LON.size)[1:-1].astype(np.float32),
        gh500=raw["gh"][1][raw["gh"][0].index(500)],
    )
    return x


NEED = {"u": LEVELS, "v": tuple(sorted({300, *W.PV_LEVS})), "t": LEVELS, "gh": LEVELS, "msl": (), "tp": ()}
FINE = ("z500", "msl", "zm_u", "zm_t", "zm_gh", "pv300", "adv300", "tp")


def members(a, E) -> dict:
    """Per member and 12-hourly step: the state() fields on 1 degree."""
    run = pd.Timestamp(a.run)
    lo, hi = (int(x) for x in a.members.split("-"))
    nums = list(range(lo, hi + 1))
    steps = list(range(0, a.last_step + 1, 12))
    pool = ThreadPoolExecutor(16)
    res = {k: [] for k in FINE}
    for n in nums:
        per = {k: [] for k in FINE}
        for step in steps:
            t0 = time.time()
            nd = dict(NEED) if step else {k: v for k, v in NEED.items() if k != "tp"}
            x = state(read(run, step, "enfo", MODELS[a.model][2], nd, number=str(n), pool=pool, model=a.model), E)
            for k in FINE:
                per[k].append(x[k].astype(np.float16) if k in ("pv300", "adv300") else x[k])
            print(f"  member {n} +{step} h in {time.time() - t0:.0f} s", flush=True)
        for k in FINE:
            res[k].append(np.stack(per[k]))
    return dict(
        run=np.datetime64(run, "s"),
        numbers=np.array(nums),
        steps=np.array(steps),
        levels=np.array(LEVELS),
        zlat=np.linspace(90.0, -90.0, 721),
        lat=E.LAT,
        lon=E.LON,
        **{k: np.stack(v) for k, v in res.items()},
    )


COARSE = ("z500", "msl", "tp", "u250", "pv300", "adv300", "zm_u", "zm_gh")


def coarse(x: dict) -> dict:
    """Small copies for verification: Z500 on 2.5 degrees (from 0.25 degree, 20-90N, from 0E), the 1 degree fields
    on 5 degrees (22-87N, from 178W), zonal means every 2 degrees (20-90N)."""
    g = x["gh500"][1:281].reshape(28, 10, 144, 10).mean((1, 3))
    out = {"z500": g.astype(np.float32)}
    for k in ("msl", "tp", "u250", "pv300", "adv300"):
        f = x[k][26:96]  # E.LAT 20..89
        out[k] = (
            f.reshape(14, 5, 72, 5).mean((1, 3)).astype(np.float16 if k in ("pv300", "adv300", "tp") else np.float32)
        )
    out["zm_u"] = x["zm_u"][:, 0:281:8].astype(np.float32)
    out["zm_gh"] = x["zm_gh"][:, 0:281:8].astype(np.float32)
    return out


def coarse_coords() -> dict:
    return dict(
        lat25=88.625 - 2.5 * np.arange(28),
        lon25=1.125 + 2.5 * np.arange(144),
        lat5=22.0 + 5.0 * np.arange(14),
        lon5=-178.0 + 5.0 * np.arange(72),
        zlat2=90.0 - 2.0 * np.arange(36),
        levels=np.array(LEVELS),
    )


def hindcast(a, E) -> dict:
    """Earlier IFS-ENS cycles (--runs, comma separated) for members --members, daily steps to --last-step: coarse()."""
    runs = [pd.Timestamp(r) for r in a.runs.split(",")]
    lo, hi = (int(x) for x in a.members.split("-"))
    nums = list(range(lo, hi + 1))
    steps = list(range(0, a.last_step + 1, 24))
    pool = ThreadPoolExecutor(16)
    res = {k: [] for k in COARSE}
    for run in runs:
        per_run = {k: [] for k in COARSE}
        for n in nums:
            per = {k: [] for k in COARSE}
            t0 = time.time()
            for step in steps:
                nd = dict(NEED) if step else {k: v for k, v in NEED.items() if k != "tp"}
                x = coarse(state(read(run, step, "enfo", "ef", nd, number=str(n), pool=pool), E))
                for k in COARSE:
                    per[k].append(x[k])
            print(f"  {run:%Y-%m-%d %H}Z member {n}: {len(steps)} steps in {time.time() - t0:.0f} s", flush=True)
            for k in COARSE:
                per_run[k].append(np.stack(per[k]))
        for k in COARSE:
            res[k].append(np.stack(per_run[k]))
    return dict(
        runs=np.array([np.datetime64(r, "s") for r in runs]),
        numbers=np.array(nums),
        steps=np.array(steps),
        **coarse_coords(),
        **{k: np.stack(v) for k, v in res.items()},
    )


def analyses(a, E) -> dict:
    """The step-0 analyses at 00 UTC from --start to --end: coarse()."""
    days = pd.date_range(a.start, a.end, freq="1D")
    pool = ThreadPoolExecutor(16)
    need = {k: v for k, v in NEED.items() if k != "tp"}
    res = {k: [] for k in COARSE}
    for d in days:
        x = coarse(state(read(d, 0, "oper", "fc", need, pool=pool), E))
        for k in COARSE:
            res[k].append(x[k])
        print(f"  analysis {d:%Y-%m-%d}", flush=True)
    return dict(days=days.values.astype("datetime64[s]"), **coarse_coords(), **{k: np.stack(v) for k, v in res.items()})


# region mode: the Canadian Arctic, Baffin Bay and Greenland on 2 degrees (8 x 8 means of 0.25 degree, 85-45N,
# 130W-10E), every level and the surface state, ensemble mean and spread, plus every member's means over BOXES
REGION_PL = ("t", "q", "u", "v", "gh")
REGION_SFC = ("2t", "skt", "msl", "tcc", "lcc", "sd", "sithick", "tcwv", "sp")
REGION_ACC = ("tp", "str", "ssr", "ttr")  # accumulated: members only (zero in the analyses)
BOXES = {"canarctic": ((60.0, 72.0), (265.0, 295.0)), "baffin": ((65.0, 78.0), (285.0, 310.0))}
R0, R1, C0, NC = 20, 180, 920, 560  # rows 85N..45N, columns 230E..370E (wraps)


def region_grid(x: np.ndarray) -> np.ndarray:
    """(..., 721, 1440) from 90N and 0E -> (..., 20, 70) 2 degree means over 85-45N, 230-370E."""
    cols = (C0 + np.arange(NC)) % 1440
    y = x[..., R0:R1, :][..., cols]
    return y.reshape(*y.shape[:-2], 20, 8, 70, 8).mean((-3, -1))


def region_boxes(g: np.ndarray) -> dict:
    lat = 84.125 - 2.0 * np.arange(20)
    lon = 230.875 + 2.0 * np.arange(70)
    out = {}
    for k, (la, lo) in BOXES.items():
        m = ((lat >= la[0]) & (lat <= la[1]))[:, None] & ((lon >= lo[0]) & (lon <= lo[1]))[None, :]
        w = np.cos(np.deg2rad(lat))[:, None] * m
        out[k] = (g * w).sum((-2, -1)) / w.sum()
    return out


def region_state(raw: dict, acc: bool) -> dict:
    """Fields missing from a cycle's files (open data added some, e.g. low cloud, in October 2026) are NaN."""
    out = {}
    for k in REGION_PL:
        out[k] = region_grid(raw[k][1].astype(np.float64))
    for k in REGION_SFC + (REGION_ACC if acc else ()):
        out[k] = region_grid(raw[k][1][0].astype(np.float64)) if k in raw else np.full((20, 70), np.nan)
    return out


def region_read(run, step, stream, kind, need, number=None, pool=None) -> dict:
    """read() of the surface fields this cycle's files carry (pressure-level fields are required)."""
    have = {r["param"] for r in index(run, step, stream, kind)}
    return read(run, step, stream, kind, {k: v for k, v in need.items() if v or k in have}, number, pool)


def region(a, E) -> dict:
    """Members of --runs, daily steps to --last-step: ensemble mean and standard deviation on the region grid, and
    every member's BOXES means."""
    runs = [pd.Timestamp(r) for r in a.runs.split(",")]
    lo, hi = (int(x) for x in a.members.split("-"))
    nums = list(range(lo, hi + 1))
    steps = list(range(0, a.last_step + 1, 24))
    pool = ThreadPoolExecutor(16)
    keys = REGION_PL + REGION_SFC + REGION_ACC
    res: dict[str, list] = {}
    for run in runs:
        s1 = {k: np.zeros((len(steps), 14, 20, 70) if k in REGION_PL else (len(steps), 20, 70)) for k in keys}
        s2 = {k: np.zeros_like(v) for k, v in s1.items()}
        bx = {
            f"{b}_{k}": np.zeros((len(nums), len(steps)) + ((14,) if k in REGION_PL else ()))
            for b in BOXES
            for k in keys
        }
        for i, n in enumerate(nums):
            t0 = time.time()
            for j, step in enumerate(steps):
                need = dict.fromkeys(REGION_PL, LEVELS) | dict.fromkeys(REGION_SFC, ())
                if step:
                    need |= dict.fromkeys(REGION_ACC, ())
                x = region_state(region_read(run, step, "enfo", "ef", need, number=str(n), pool=pool), acc=step > 0)
                for k in keys:
                    v = x.get(k, np.zeros((20, 70)))  # accumulations at step 0
                    s1[k][j] += v
                    s2[k][j] += v * v
                    for b, m in region_boxes(v).items():
                        bx[f"{b}_{k}"][i, j] = m
            print(f"  {run:%Y-%m-%d %H}Z member {n}: {len(steps)} steps in {time.time() - t0:.0f} s", flush=True)
        n_ = len(nums)
        for k in keys:
            mean = s1[k] / n_
            res.setdefault(f"mean_{k}", []).append(mean.astype(np.float32))
            res.setdefault(f"sd_{k}", []).append(np.sqrt(np.maximum(s2[k] / n_ - mean**2, 0)).astype(np.float32))
        for k, v in bx.items():
            res.setdefault(f"box_{k}", []).append(v.astype(np.float32))
    return dict(
        runs=np.array([np.datetime64(r, "s") for r in runs]),
        numbers=np.array(nums),
        steps=np.array(steps),
        levels=np.array(LEVELS),
        rlat=84.125 - 2.0 * np.arange(20),
        rlon=230.875 + 2.0 * np.arange(70),
        **{k: np.stack(v) for k, v in res.items()},
    )


def region_analyses(a, E) -> dict:
    """The 00 UTC analyses from --start to --end on the region grid, with the BOXES means."""
    days = pd.date_range(a.start, a.end, freq="1D")
    pool = ThreadPoolExecutor(16)
    need = dict.fromkeys(REGION_PL, LEVELS) | dict.fromkeys(REGION_SFC, ())
    res: dict[str, list] = {}
    for d in days:
        x = region_state(region_read(d, 0, "oper", "fc", need, pool=pool), acc=False)
        for k, v in x.items():
            res.setdefault(k, []).append(v.astype(np.float32))
            for b, m in region_boxes(v).items():
                res.setdefault(f"box_{b}_{k}", []).append(np.float32(m) if np.ndim(m) == 0 else m.astype(np.float32))
        print(f"  analysis {d:%Y-%m-%d}", flush=True)
    return dict(
        days=days.values.astype("datetime64[s]"),
        levels=np.array(LEVELS),
        rlat=84.125 - 2.0 * np.arange(20),
        rlon=230.875 + 2.0 * np.arange(70),
        **{k: np.stack(v) for k, v in res.items()},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("control", "members", "hindcast", "analyses", "region", "region-analyses"))
    ap.add_argument("--run", default="")
    ap.add_argument("--model", choices=tuple(MODELS), default="ifs")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--runs", default="")
    ap.add_argument("--members", default="1-50")
    ap.add_argument("--last-step", type=int, default=360)
    ap.add_argument("--elias", required=True)
    ap.add_argument("--archive")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.elias)
    import elias as E

    out = {
        "control": control,
        "members": members,
        "hindcast": hindcast,
        "analyses": analyses,
        "region": region,
        "region-analyses": region_analyses,
    }[a.mode](a, E)
    np.savez_compressed(a.out, **out)
    print("wrote", a.out, {k: getattr(v, "shape", None) for k, v in out.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
