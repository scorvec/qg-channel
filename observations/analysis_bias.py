"""How the operational IFS and GFS analyses differ from ERA5, field by field, and ERA5 truth for verifying forecasts.

    python observations/analysis_bias.py days --start 2026-09-01 --end 2026-09-04 --bias-end 2026-10-15 --out d.npz
    python observations/analysis_bias.py combine --windows 2026-09-01:2026-09-30,2026-09-16:2026-10-15 --out bias.npz d*.npz

`days`: for each day up to --bias-end, the IFS analysis (step 0, ECMWF open data on Google Cloud) and the GFS analysis
(f000, NOAA's AWS S3 bucket) minus ERA5 (the ARCO copy on Google Cloud, ERA5T for recent months) at 00 and 18 UTC, for
the variables an ERA5-trained forecast model starts from: temperature, geopotential, u, v, vertical velocity and specific
humidity on 13 levels (50-1000 hPa), 2 m temperature, sea-level pressure and the 10 m and 100 m winds. Kept on 1 degree
(4 x 4 means of the 0.25 degree grid, 89.625N-89.625S, from 0.125E). Every day also gives ERA5 itself at 00 UTC on that
grid (500 hPa height, 850 hPa temperature, sea-level pressure, 2 m temperature, 250 hPa wind) for verification.
`combine`: mean differences over each window, per source and hour, and the ERA5 verification series.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ifs_case as I

ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
G = 9.80665
# name: (IFS param, GFS idx variable, ERA5 variable, on pressure levels, GFS level text for surface fields)
FIELDS = {
    "t": ("t", "TMP", "temperature", True, ""),
    "z": ("gh", "HGT", "geopotential", True, ""),
    "u": ("u", "UGRD", "u_component_of_wind", True, ""),
    "v": ("v", "VGRD", "v_component_of_wind", True, ""),
    "w": ("w", "VVEL", "vertical_velocity", True, ""),
    "q": ("q", "SPFH", "specific_humidity", True, ""),
    "t2m": ("2t", "TMP", "2m_temperature", False, "2 m above ground"),
    "msl": ("msl", "PRMSL", "mean_sea_level_pressure", False, "mean sea level"),
    "u10": ("10u", "UGRD", "10m_u_component_of_wind", False, "10 m above ground"),
    "v10": ("10v", "VGRD", "10m_v_component_of_wind", False, "10 m above ground"),
    "u100": ("100u", "UGRD", "100m_u_component_of_wind", False, "100 m above ground"),
    "v100": ("100v", "VGRD", "100m_v_component_of_wind", False, "100 m above ground"),
}
HOURS = (0, 18)


def coarse(x: np.ndarray) -> np.ndarray:
    """(..., 721, 1440) from 90N and 0E -> (..., 180, 360) 4 x 4 means (the -90 row dropped)."""
    s = x.shape[:-2]
    return x[..., :720, :].reshape(*s, 180, 4, 360, 4).mean((-3, -1))


def ifs(t: pd.Timestamp, pool) -> dict:
    need = {p: (LEVELS if pl else ()) for p, _, _, pl, _ in FIELDS.values()}
    raw = I.read(t, 0, "oper", "fc", need, pool=pool)
    out = {}
    for k, (p, _, _, pl, _) in FIELDS.items():
        x = raw[p][1] if pl else raw[p][1][0]
        out[k] = x * G if k == "z" else x
    return out


def gfs(t: pd.Timestamp, pool) -> dict:
    base = f"{GFS}/gfs.{t:%Y%m%d}/{t:%H}/atmos/gfs.t{t:%H}z.pgrb2.0p25.f000"
    rows = [x.split(":") for x in I.get(base + ".idx").decode().splitlines() if x.strip()]
    start = {f"{r[3]}:{r[4]}": int(r[1]) for r in rows}
    nxt = {int(a[1]): int(b[1]) for a, b in itertools.pairwise(rows)}
    keys = []
    for _, gv, _, pl, lev in FIELDS.values():
        keys += [f"{gv}:{L} mb" for L in LEVELS] if pl else [f"{gv}:{lev}"]
    miss = [k for k in keys if k not in start]
    if miss:
        raise SystemExit(f"GFS {t} f000 missing {miss}")
    bufs = dict(zip(keys, pool.map(lambda k: I.get(base, (start[k], nxt[start[k]] - start[k])), keys), strict=True))
    out = {}
    for k, (_, gv, _, pl, lev) in FIELDS.items():
        x = np.stack([I.decode(bufs[f"{gv}:{L} mb"]) for L in LEVELS]) if pl else I.decode(bufs[f"{gv}:{lev}"])
        out[k] = x * G if k == "z" else x
    return out


def era5(ds, t: pd.Timestamp, names: list[str]) -> dict:
    out = {}
    for k in names:
        _, _, ev, pl, _ = FIELDS[k]
        da = ds[ev].sel(time=t)
        out[k] = (da.sel(level=list(LEVELS)) if pl else da).values.astype(np.float64)
    return out


def days(a) -> dict:
    import xarray as xr

    ds = xr.open_zarr(ERA5, chunks=None, storage_options={"token": "anon"}, consolidated=True)
    pool = ThreadPoolExecutor(16)
    bias_end = pd.Timestamp(a.bias_end)
    out: dict[str, list] = {}
    dates, vdates = [], []
    for d in pd.date_range(a.start, a.end, freq="1D"):
        if d <= bias_end:
            for h in HOURS:
                t = d + pd.Timedelta(hours=h)
                t0 = time.time()
                e = era5(ds, t, list(FIELDS))
                for src, fn in (("ifs", ifs), ("gfs", gfs)):
                    f = fn(t, pool)
                    for k in FIELDS:
                        out.setdefault(f"{src}_{h:02d}_{k}", []).append(coarse(f[k] - e[k]).astype(np.float32))
                print(f"  {t:%Y-%m-%d %H}Z differences in {time.time() - t0:.0f} s", flush=True)
            dates.append(d)
        e = era5(ds, d, ["z", "t", "msl", "t2m", "u", "v"])
        lv = list(LEVELS)
        v = {
            "z500": e["z"][lv.index(500)] / G,
            "t850": e["t"][lv.index(850)],
            "msl": e["msl"] / 100.0,
            "t2m": e["t2m"],
            "u250": e["u"][lv.index(250)],
            "v250": e["v"][lv.index(250)],
        }
        for k, x in v.items():
            out.setdefault(f"era5_{k}", []).append(coarse(x).astype(np.float32))
        vdates.append(d)
        print(f"  {d:%Y-%m-%d} ERA5 verification fields", flush=True)
    res = {k: np.stack(x) for k, x in out.items()}
    res["bias_days"] = np.array(dates, dtype="datetime64[s]")
    res["verify_days"] = np.array(vdates, dtype="datetime64[s]")
    return res


def combine(a) -> dict:
    parts = [dict(np.load(f)) for f in a.files]
    bd = np.concatenate([p["bias_days"] for p in parts])
    vd = np.concatenate([p["verify_days"] for p in parts])
    out = {"lat": 89.625 - np.arange(180) * 1.0, "lon": 0.125 + np.arange(360) * 1.0, "levels": np.array(LEVELS)}
    o = np.argsort(vd)
    out["verify_days"] = vd[o]
    for k in [k for k in parts[0] if k.startswith("era5_")]:
        out[k] = np.concatenate([p[k] for p in parts])[o]
    windows = [w.split(":") for w in a.windows.split(",")]
    out["windows"] = np.array([[np.datetime64(x, "s") for x in w] for w in windows])
    keys = sorted({k for p in parts for k in p if k.startswith(("ifs_", "gfs_"))})
    for k in keys:
        x = np.concatenate([p[k] for p in parts if k in p])  # days after --bias-end carry verification fields only
        means = []
        for w0, w1 in windows:
            m = (bd >= np.datetime64(w0)) & (bd <= np.datetime64(w1))
            if not m.any():
                raise SystemExit(f"window {w0}..{w1}: no days")
            means.append(x[m].mean(0))
        out[k] = np.stack(means).astype(np.float32)
        print(k, f"window means; global mean abs {np.abs(out[k]).mean():.3g}", flush=True)
    out["n_days"] = np.array([((bd >= np.datetime64(w0)) & (bd <= np.datetime64(w1))).sum() for w0, w1 in windows])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("days", "combine"))
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--bias-end", default="1900-01-01")
    ap.add_argument("--windows", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("files", nargs="*")
    a = ap.parse_intermixed_args()
    out = days(a) if a.mode == "days" else combine(a)
    np.savez_compressed(a.out, **out)
    print("wrote", a.out, {k: v.shape for k, v in out.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
