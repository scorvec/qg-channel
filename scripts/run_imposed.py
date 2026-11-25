"""Impose an idealized polar stratospheric anomaly from every restart, paired with a reference run.

    python scripts/run_imposed.py SETUP [--members 50] [--days 90] [--amplitude -20] [--dt 900] [--storms-off] [--out runs]
SETUP is one of ref, low (12-20 km only), deep (12 km up), shallow (above 22 km). Run `ref` once with the same --dt and
--members; the response of a setup is its run minus ref, member by member (scripts/analyze.py).
Writes runs/<SETUP>[_tag].npz with daily u, u'v' (member, day, z, y) and h0, anticyclonic and cyclonic folds (member,
day, y).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import qgchannel as qg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("setup", choices=sorted(qg.PROFILES))
    ap.add_argument("--members", type=int, default=50)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--amplitude", type=float, default=-20.0, help="polar wind anomaly aloft (m/s)")
    ap.add_argument("--dt", type=float, default=900.0)
    ap.add_argument("--storms-off", action="store_true", help="zero every eddy after each step")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    a = ap.parse_args()
    c = replace(qg.Config(), dt=a.dt)
    m = qg.build(c)
    clim = np.load(a.out / "control.npz")["u"].mean(0)
    restarts = np.load(a.out / "restarts.npy")[: a.members]
    res = qg.imposed(c, restarts, clim, qg.polar_anomaly(m, a.setup, a.amplitude), days=a.days, eddies=not a.storms_off)
    tag = "".join(
        [
            f"_dt{int(a.dt)}" if a.dt != 900 else "",
            f"_a{int(-a.amplitude)}" if a.amplitude != -20 else "",
            "_off" if a.storms_off else "",
        ]
    )
    np.savez(a.out / f"{a.setup}{tag}.npz", **res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
