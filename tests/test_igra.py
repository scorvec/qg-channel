"""The IGRA reader on a synthetic sounding in the IGRA v2 fixed-column format."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "observations"))
import igra_arctic as I


def line(p_pa: int, gph: int, t_c: float, wdir: int = 270, wspd: float = 20.0) -> str:
    """One data record: columns 10-15 pressure (Pa), 17-21 height (m), 23-27 temperature (0.1 C), 41-45 direction,
    47-51 speed (0.1 m/s)."""
    s = [" "] * 51
    s[0:2] = "21"
    for lo, hi, v in (
        (9, 15, p_pa),
        (16, 21, gph),
        (22, 27, round(t_c * 10)),
        (40, 45, wdir),
        (46, 51, round(wspd * 10)),
    ):
        s[lo:hi] = f"{v:>{hi - lo}d}"
    return "".join(s)


def sounding() -> list[str]:
    # a 6.5 K/km troposphere to a tropopause at 9 km (~300 hPa), isothermal-to-warming stratosphere above
    levels = []
    for zk in range(25):
        z = zk * 1000
        t = 0.0 - 6.5 * min(z, 9000) / 1000 + (1.5 * (z - 9000) / 1000 if z > 9000 else 0)
        p = int(100000 * np.exp(-z / 7400))
        levels.append(line(p, z, t))
    for P in (30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000):
        z = int(-7400 * np.log(P / 100000))
        t = 0.0 - 6.5 * min(z, 9000) / 1000 + (1.5 * (z - 9000) / 1000 if z > 9000 else 0)
        levels.append(line(P, z, t))
    head = f"#XXM00099999 2021 01 20 00 9999 {len(levels):>4d}"
    return [head, *levels]


def test_parse_and_tropopause():
    out = I.parse(sounding())
    assert len(out) == 1
    tm, lev = out[0]
    row = I.sounding_row(tm, lev)
    assert abs(row["z_trop"] - 9000) <= 1000
    assert row["n2_til"] > 2e-4  # a stable layer above the tropopause
    assert np.isclose(row["u100"], 20.0, atol=1e-6)  # westerly at 270 degrees
    assert np.isfinite(row["t200"])
