"""Impose an observed stratosphere: daily zonal-mean wind anomalies from a reanalysis, nudged above ~12.5 km.

    python scripts/run_event.py ANOMALY.nc NAME [--members 50] [--dt 600] [--out runs]
ANOMALY.nc holds one variable u (time, lev, lat): the zonal-mean zonal-wind anomaly in m/s, pressure levels in hPa, northern
latitudes, one value per day for the days to run (e.g. 10 days before an SSW's central date to 79 days after). It is
mapped onto the channel with z = 7 km x ln(1000 / p) and y = (latitude - 9) x 111.2 km. Pair the result with `ref` run at
the same --dt.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import xarray as xr

import qgchannel as qg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("anomaly", type=Path)
    ap.add_argument("name")
    ap.add_argument("--members", type=int, default=50)
    ap.add_argument("--dt", type=float, default=600.0)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    a = ap.parse_args()
    c = replace(qg.Config(), dt=a.dt)
    m = qg.build(c)
    da = xr.open_dataarray(a.anomaly).transpose("time", "lev", "lat")
    du = qg.to_channel(m, da.values, da.lev.values, da.lat.values)
    clim = np.load(a.out / "control.npz")["u"].mean(0)
    restarts = np.load(a.out / "restarts.npy")[: a.members]
    res = qg.imposed(c, restarts, clim, du, days=du.shape[0])
    np.savez(a.out / f"event_{a.name}.npz", imposed=du.astype(np.float32), **res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
