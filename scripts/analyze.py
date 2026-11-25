"""Responses to an imposed stratosphere: each run minus its reference, member by member, by time window.

    python scripts/analyze.py low deep shallow [--ref ref] [--windows 0-14,15-44,45-89] [--out runs]
Measures (for the response = experiment - reference):
  u_sfc_pole    surface zonal wind on the polar side, y 6-8 Mm (m/s)
  u_5km_pole    the same at 5 km
  jet_shift     change of the latitude of the 5-km jet maximum (km; negative = toward the equator)
  uv_poleflank  eddy momentum flux u'v' at the fold layer on the jet's poleward flank, y 5-6.5 Mm (m2/s2)
  anti / cyc    share of the jet band (y 3-7 Mm) with overturned PV at the fold layer, by tilt (percentage points)
  *_agree       share of members whose change has the sign of the mean
Members that blew up are left out of the windows they affect.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def measures(r: dict) -> dict:
    z, y = r["z"], r["y"]
    i5 = int(np.argmin(np.abs(z - 5000)))
    it = int(r["fold_layer"])
    pole, flank, band = (y >= 6e6) & (y <= 8e6), (y >= 5e6) & (y <= 6.5e6), (y >= 3e6) & (y <= 7e6)
    u5 = r["u"][:, :, i5]
    return {
        "u_sfc_pole": r["u"][:, :, 0][:, :, pole].mean(-1),
        "u_5km_pole": u5[:, :, pole].mean(-1),
        "jet_shift": y[np.argmax(np.nan_to_num(u5, nan=-1e9), axis=-1)] / 1e3,
        "uv_poleflank": r["uv"][:, :, it][:, :, flank].mean(-1),
        "anti": 100 * r["anti"][:, :, band].mean(-1),
        "cyc": 100 * r["cyc"][:, :, band].mean(-1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default="ref")
    ap.add_argument("--windows", default="0-14,15-44,45-89")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    a = ap.parse_args()
    ref = dict(np.load(a.out / f"{a.ref}.npz"))
    windows = [tuple(int(v) for v in w.split("-")) for w in a.windows.split(",")]
    rows = []
    for name in a.runs:
        ex = dict(np.load(a.out / f"{name}.npz"))
        n = min(len(ex["u"]), len(ref["u"]))
        me = measures({k: v[:n] if v.ndim > 1 else v for k, v in ex.items()})
        mr = measures({k: v[:n] if v.ndim > 1 else v for k, v in ref.items()})
        for w0, w1 in windows:
            row = {"run": name, "days": f"{w0 + 1}-{w1 + 1}"}
            for k in me:
                bad = ~np.isfinite(ex["u"][:n, w0 : w1 + 1]).all(axis=(1, 2, 3))
                d = (me[k] - mr[k])[~bad, w0 : w1 + 1].mean(1)
                row["members"] = int((~bad).sum())
                row[k] = float(d.mean())
                row[k + "_agree"] = float((np.sign(d) == np.sign(d.mean())).mean())
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(a.out / "response.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
