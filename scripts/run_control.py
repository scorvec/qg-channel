"""Spin up the control climate and save daily zonal means and restart states.

    python scripts/run_control.py [--days 2200] [--mountains greenland,rockies] [--out runs]
Writes <out>/control.npz and <out>/restarts.npy; keep one folder per mountain setup (runs with mountains must use the
control and restarts made with the same mountains).
A 2200-day run takes about 30 min on one laptop core at the default resolution (64 x 32 x 30).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import qgchannel as qg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=2200)
    ap.add_argument("--mountains", default="", help="comma-separated: greenland, rockies (default: flat)")
    ap.add_argument("--mountain-scale", type=float, default=qg.Config.mountain_scale)
    ap.add_argument("--nx", type=int, default=qg.Config.nx, help="grid points around the circle (ny = nx / 2)")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    a = ap.parse_args()
    mountains = tuple(x for x in a.mountains.split(",") if x)
    a.out.mkdir(parents=True, exist_ok=True)
    dt = 600.0 if a.nx > qg.Config.nx else qg.Config.dt  # finer grids need the shorter step
    c = qg.Config(mountains=mountains, mountain_scale=a.mountain_scale, nx=a.nx, ny=a.nx // 2, dt=dt)
    daily, restarts = qg.control(c, days=a.days)
    np.savez(a.out / "control.npz", **daily)
    np.save(a.out / "restarts.npy", restarts)
    m = qg.build(c)
    share, efold = qg.annular_mode(daily["u"], m)
    u = daily["u"].mean(0)
    print(f"jet at 10 km: {u[int(np.argmin(abs(m.z - 1e4)))].max():.1f} m/s; surface westerlies {u[0].max():.1f} m/s")
    print(f"eddy energy at 5 km: {daily['eke'].mean(0)[int(np.argmin(abs(m.z - 5e3)))].max():.0f} m2/s2")
    print(f"annular mode: {100 * share:.0f} % of the variance, e-folding {efold:.0f} days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
