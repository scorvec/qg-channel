"""Warm conveyor belts around one sudden stratospheric warming, from ERA5 with the ELIAS 2.0 networks.

    python observations/era5_wcb.py --date 2009-01-24 --elias DIR --archive DIR --out wcb_20090124.npz
ERA5 comes from the public analysis-ready copy on Google Cloud (ARCO ERA5, 0.25 deg, hourly), read anonymously at 00 and
12 UTC from event day -11 to +46. Each field is reduced to the networks' 1 degree grid by the mean over the 1 degree box
centred on each point. The 300 hPa divergence for the outflow's irrotational wind is taken from the winds on the whole
0.25 degree grid. Identification follows the published networks (Quinting and Grams 2022, Geosci. Model Dev. 15,
715-730; models, decision thresholds and trajectory climatology from doi:10.5281/zenodo.5154980): `elias.py` (--elias,
the directory holding it) applies them, and --archive is the unzipped archive. The ascent is found at every time; the
inflow uses the ascent 24 h later and the outflow the ascent 24 h earlier, so the output runs from day -10 to +45.

Writes: times (00/12 UTC), foot (stage, time, packed LAT x LON bits; stages inflow, ascent, outflow), clim (stage, day,
LAT x LON, float16: the archive's trajectory climatology for each valid day), lat, lon; pv300 (time, LAT x LON, PVU) and
adv300 (PVU/day: the 300 hPa PV advection by the irrotational wind, where an outflow pushes tropospheric air across the
tropopause; Archambault et al. 2013, Mon. Wea. Rev. 141, 2325-2346).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
NAMES = {
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "t": "temperature",
    "z": "geopotential",
}
A_E = 6371200.0
FIRST, LAST = -11, 46  # event days read; the stages are complete from FIRST + 1 to LAST - 1
PV_LEVS = (400, 300, 250)  # Ertel PV at the middle level; its vertical derivatives across the three
EXTRA = {"u": {300, *PV_LEVS}, "v": {300, *PV_LEVS}, "t": set(PV_LEVS)}  # levels read beyond the networks' own


def box_mean(v: np.ndarray, rows: int, nlon: int) -> np.ndarray:
    """A 0.25 degree field (rows from 90N, columns from 0E) -> 1 degree points from 90N southward and from 180W, each the
    mean over the 1 degree box centred on it (weights 1/2, 1, 1, 1, 1/2 each way; at 90N the half box south of the
    pole). Returns (rows, nlon) ordered from the south, as the networks' grid."""
    v = np.roll(v, -720, axis=-1)  # start at 180W
    v = np.concatenate([v[:, -2:], v, v[:, :2]], axis=1)
    x = 0.5 * v[:, 0::4][:, :nlon] + v[:, 1::4][:, :nlon] + v[:, 2::4][:, :nlon] + v[:, 3::4][:, :nlon]
    x = x + 0.5 * v[:, 4::4][:, :nlon]
    out = np.empty((rows, nlon), np.float32)
    out[-1] = (x[0] + x[1] + 0.5 * x[2]) / 10.0
    k = 4 * np.arange(1, rows)
    out[-2::-1] = (0.5 * x[k - 2] + x[k - 1] + x[k] + x[k + 1] + 0.5 * x[k + 2]) / 16.0
    return out


def divergence(u: np.ndarray, v: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Horizontal divergence (1/s) on the 0.25 degree grid (rows from 90N, columns from 0E), centred differences; the
    pole row takes the mean of the row next to it."""
    phi = np.deg2rad(lat)[:, None]
    cos = np.cos(phi)
    dlam = np.deg2rad(0.25)
    with np.errstate(divide="ignore", invalid="ignore"):
        du = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dlam)
        dv = np.gradient(v * cos, phi[:, 0], axis=0)
        d = (du + dv) / (A_E * cos)
    d[0] = d[1].mean()
    d[-1] = d[-2].mean()
    return d


def ddx(a: np.ndarray, metre: float, cos: np.ndarray) -> np.ndarray:
    """d/dx on the 1 degree grid, periodic in longitude."""
    return (np.roll(a, -1, axis=-1) - np.roll(a, 1, axis=-1)) / (2.0 * metre * cos)


def potential_vorticity(f: dict, E) -> np.ndarray:
    """Ertel PV (PVU) at 300 hPa on E.LAT x E.LON from u, v, t on E.ROWS at PV_LEVS:
    -g [(f + zeta) dtheta/dp - dv/dp dtheta/dx + du/dp dtheta/dy], vertical derivatives centred in ln p, theta's through
    ln theta (the operator of the site's warm-conveyor-belt forecast)."""
    metre = E.R_EARTH * np.pi / 180.0
    cos = np.cos(np.deg2rad(E.LAT))[:, None]
    at = PV_LEVS[1]
    lnp = np.log(100.0 * np.array(PV_LEVS))
    inner = (..., slice(1, -1), slice(None))

    def ddp(a):
        return np.gradient(a, lnp, axis=0)[1] / (100.0 * at)

    def ddy(a):
        return (a[..., 2:, :] - a[..., :-2, :]) / (2.0 * metre)

    u, v = (np.stack([f[k][lev] for lev in PV_LEVS]) for k in ("u", "v"))
    th = np.stack([f["t"][lev] * (1000.0 / lev) ** E.KAPPA for lev in PV_LEVS])
    cos_r = np.cos(np.deg2rad(E.ROWS))[:, None]
    vort = 2.0 * E.OMEGA * np.sin(np.deg2rad(E.LAT))[:, None] + ddx(v[1][inner], metre, cos) - ddy(u[1] * cos_r) / cos
    th_p = (th[1] * ddp(np.log(th)))[inner]
    dthx = ddx(th[1][inner], metre, cos)
    return -1.0e6 * E.G * (vort * th_p - ddp(v)[inner] * dthx + ddp(u)[inner] * ddy(th[1]))


def cal_day(d: pd.Timestamp) -> pd.Timestamp:
    """The archive has no climatology file for 29 February: use the 28th's."""
    return d - pd.Timedelta(days=1) if (d.month, d.day) == (2, 29) else d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--elias", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    sys.path.insert(0, a.elias)
    import elias as E
    import xarray as xr

    d0 = pd.Timestamp(a.date)
    times = pd.date_range(d0 + pd.Timedelta(days=FIRST), d0 + pd.Timedelta(days=LAST, hours=12), freq="12h")
    ds = xr.open_zarr(STORE, chunks=None, storage_options={"token": "anon"}, consolidated=True)
    lat = ds.latitude.values
    if lat[0] != 90.0 or lat[1] != 89.75 or ds.longitude.values[0] != 0.0:
        raise SystemExit("unexpected grid")
    north = 4 * (E.ROWS.size - 1) + 3  # 90N to 7.5S
    root = Path(a.archive)

    def fetch(t):
        """Download only (threads): {name: (levels, 721, 1440)} with 300 hPa added for the winds."""
        for attempt in range(1, 5):
            try:
                t0 = time.time()
                raw = {}
                for k, levels in E.LEVELS.items():
                    need = sorted(set(levels) | EXTRA.get(k, set()))
                    raw[k] = (need, ds[NAMES[k]].sel(time=t, level=need).values)
                print(f"  {t:%Y-%m-%d %H}Z read in {time.time() - t0:.0f} s", flush=True)
                return raw
            except Exception as e:  # a transient read error must not end the event
                print(f"  {t:%Y-%m-%d %H}Z attempt {attempt}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                time.sleep(15 * attempt)
        raise SystemExit(f"{t}: no data")

    def reduce(raw):
        """In the main thread (pyshtools is not thread-safe): the 1 degree fields, the outflow wind speed, the 300 hPa
        PV and the rate (PVU/day) at which the irrotational wind advects it (the outflow's push on the tropopause)."""
        f, fp = {}, {}
        for k, (need, x) in raw.items():
            for i, lev in enumerate(need):
                if lev in E.LEVELS[k] or lev in PV_LEVS:
                    b = box_mean(x[i, :north].astype(np.float64), E.ROWS.size, E.LON.size).astype(np.float64)
                    if lev in E.LEVELS[k]:
                        f.setdefault(k, {})[lev] = np.maximum(b, 0.0) if k == "q" else b
                    if lev in PV_LEVS and k in "uvt":
                        fp.setdefault(k, {})[lev] = b
        u, v = (raw[k][1][raw[k][0].index(300)].astype(np.float64) for k in ("u", "v"))
        wind = E.irrotational_wind(divergence(u, v, lat)[:-1])  # Driscoll-Healy: no south-pole row
        pv = potential_vorticity(fp, E)
        metre = E.R_EARTH * np.pi / 180.0
        cos = np.cos(np.deg2rad(E.LAT))[:, None]
        adv = -86400.0 * (wind[0] * ddx(pv, metre, cos) + wind[1] * np.gradient(pv, axis=-2) / metre)
        return f, np.hypot(wind[0], wind[1]).astype(np.float32), pv.astype(np.float32), adv.astype(np.float32)

    got = []
    with ThreadPoolExecutor(a.workers) as pool:
        for i in range(0, len(times), a.workers):  # bounded: a batch of raw fields in memory at a time
            got.extend(reduce(raw) for raw in pool.map(fetch, times[i : i + a.workers]))
    fields = {k: {lev: np.stack([g[0][k][lev] for g in got]) for lev in E.LEVELS[k]} for k in E.LEVELS}
    speed = np.stack([g[1] for g in got])
    pv300 = np.stack([g[2] for g in got])
    adv300 = np.stack([g[3] for g in got])
    del got
    nets = {s: E.Net(root / "models" / f"cnn_{E.STAGES[s]}.h5") for s in E.STAGES}
    days = [t.normalize() for t in times]
    clim_asc = np.stack([E.climatology(root, "ascent", cal_day(d)) for d in days])
    ascent = nets["ascent"](E.ascent_image(fields, clim_asc))
    lag = E.LAG_H // 12
    keep = np.arange(lag, len(times) - lag)
    sub = {k: {lev: arr[keep] for lev, arr in v.items()} for k, v in fields.items()}
    prob = {
        "ascent": ascent[keep],
        "inflow": nets["inflow"](E.inflow_image(sub, ascent[keep + lag])),
        "outflow": nets["outflow"](E.outflow_image(sub, speed[keep], ascent[keep - lag])),
    }
    stages = ("inflow", "ascent", "outflow")
    kt = times[keep]
    foot = np.stack([np.stack([prob[s][i] > E.thresholds(root, s, kt[i]) for i in range(len(kt))]) for s in stages])
    udays = pd.DatetimeIndex(sorted({t.normalize() for t in kt}))
    clim = np.stack([np.stack([E.climatology(root, s, cal_day(d)) for d in udays]) for s in stages]).astype(np.float16)
    np.savez_compressed(
        a.out,
        times=kt.values.astype("datetime64[s]"),
        days=udays.values.astype("datetime64[D]"),
        foot=np.packbits(foot.reshape(len(stages), len(kt), -1), axis=-1),
        clim=clim,
        lat=E.LAT,
        lon=E.LON,
        stages=np.array(stages),
        pv300=pv300[keep].astype(np.float16),
        adv300=adv300[keep].astype(np.float16),
    )
    print(
        f"wrote {a.out}: {len(kt)} times; mean shares",
        {s: round(float(foot[i].mean()), 4) for i, s in enumerate(stages)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
