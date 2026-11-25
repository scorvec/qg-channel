"""Arctic radiosondes (IGRA v2): the lower stratosphere and the tropopause, sounding by sounding, for checking the model's
mechanism against observations that no reanalysis model has touched.

    python observations/igra_arctic.py [--min-lat 60] [--first-year 1958] [--out igra]
Reads the NOAA/NCEI station list, keeps stations at or north of --min-lat whose records span at least 1980-2020, downloads
each station's period-of-record file once (one at a time) and writes, for every 00 and 12 UTC sounding from October to
April:
  T, u, v at 300, 250, 200, 150, 100, 70, 50 and 30 hPa (K, m/s; NaN where missing)
  the WMO lapse-rate tropopause: pressure (hPa), height (m) and temperature (K)
  the tropopause inversion layer: mean and maximum N^2 (1/s^2) in the 2 km above the tropopause, and the temperature
  change across those 2 km
Heights come from the reported geopotential heights, filled between them hypsometrically. Output: <out>/igra_arctic.parquet
and <out>/stations.csv.
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive"
LIST = f"{BASE}/doc/igra2-station-list.txt"
DATA = f"{BASE}/access/data-por/{{id}}-data.txt.zip"
HEADERS = {"User-Agent": "qg-channel-igra-fetch/0.1"}
LEVELS = (300, 250, 200, 150, 100, 70, 50, 30)
G, RD = 9.80665, 287.05
MONTHS = {10, 11, 12, 1, 2, 3, 4}


def stations(min_lat: float, first: int, last: int) -> pd.DataFrame:
    txt = requests.get(LIST, headers=HEADERS, timeout=60).text
    rows = []
    for ln in txt.splitlines():
        try:  # a few entries have no coordinates
            rows.append(
                (ln[0:11], float(ln[12:20]), float(ln[21:30]), ln[41:71].strip(), int(ln[72:76]), int(ln[77:81]))
            )
        except ValueError:
            continue
    st = pd.DataFrame(rows, columns=["id", "lat", "lon", "name", "first", "last"])
    return st[(st.lat >= min_lat) & (st["first"] <= first) & (st["last"] >= last)].reset_index(drop=True)


def num(s: str, missing: tuple[int, ...] = (-9999, -8888)) -> float:
    try:
        v = int(s)
    except ValueError:
        return np.nan
    return np.nan if v in missing else float(v)


def parse(lines: list[str]) -> list[tuple[pd.Timestamp, np.ndarray]]:
    """Soundings as (time, array of levels: p (Pa), gph (m), T (K), wdir (deg), wspd (m/s)), Oct-Apr, 00/12 UTC."""
    out = []
    i = 0
    while i < len(lines):
        h = lines[i]
        n = int(h[32:36])
        year, month, day, hour = int(h[13:17]), int(h[18:20]), int(h[21:23]), int(h[24:26])
        block = lines[i + 1 : i + 1 + n]
        i += 1 + n
        if month not in MONTHS or hour not in (0, 12):
            continue
        lev = np.array(
            [
                (num(r[9:15]), num(r[16:21]), num(r[22:27]) / 10 + 273.15, num(r[40:45]), num(r[46:51]) / 10)
                for r in block
            ]
        )
        lev = lev[np.isfinite(lev[:, 0]) & (lev[:, 0] > 0)]
        if len(lev) < 5:
            continue
        lev = lev[np.argsort(-lev[:, 0])]  # surface first
        out.append((pd.Timestamp(year, month, day, hour), lev))
    return out


def heights(p: np.ndarray, gph: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Height at every level: reported geopotential heights, filled hypsometrically from the nearest level below that has
    one (temperature interpolated in ln p where missing)."""
    lnp = np.log(p)
    ok = np.isfinite(t)
    if ok.sum() < 3:
        return np.full_like(p, np.nan)
    tt = np.interp(-lnp, -lnp[ok], t[ok])
    z = gph.copy()
    for k in range(1, len(p)):
        if not np.isfinite(z[k]) and np.isfinite(z[k - 1]):
            z[k] = z[k - 1] + RD / G * 0.5 * (tt[k] + tt[k - 1]) * (lnp[k - 1] - lnp[k])
    return z


def tropopause(p: np.ndarray, z: np.ndarray, t: np.ndarray) -> int | None:
    """Index of the WMO lapse-rate tropopause: the lowest level above 500 hPa where the lapse rate drops to 2 K/km or less
    and the mean lapse rate to every level within the next 2 km stays at or below 2 K/km."""
    ok = np.isfinite(z) & np.isfinite(t)
    p, z, t = p[ok], z[ok], t[ok]
    idx = np.nonzero(ok)[0]
    for k in range(len(p) - 1):
        if p[k] > 50000 or p[k] < 7500:
            continue
        if (t[k] - t[k + 1]) / max(z[k + 1] - z[k], 1.0) * 1e3 > 2.0:
            continue
        above = (z > z[k]) & (z <= z[k] + 2000)
        if above.sum() == 0:
            continue
        if np.all((t[k] - t[above]) / (z[above] - z[k]) * 1e3 <= 2.0):
            return int(idx[k])
    return None


def sounding_row(time_: pd.Timestamp, lev: np.ndarray) -> dict:
    p, gph, t, wdir, wspd = lev.T
    row: dict = {"time": time_}
    u = -wspd * np.sin(np.deg2rad(wdir))
    v = -wspd * np.cos(np.deg2rad(wdir))
    for P in LEVELS:
        k = np.nonzero(np.isclose(p, P * 100.0))[0]
        k = k[0] if k.size else None
        row[f"t{P}"] = t[k] if k is not None else np.nan
        row[f"u{P}"] = u[k] if k is not None else np.nan
        row[f"v{P}"] = v[k] if k is not None else np.nan
    z = heights(p, gph, t)
    k = tropopause(p, z, t)
    row |= {
        "p_trop": np.nan,
        "z_trop": np.nan,
        "t_trop": np.nan,
        "n2_til": np.nan,
        "n2_til_max": np.nan,
        "dt_til": np.nan,
    }
    if k is None:
        return row
    row |= {"p_trop": p[k] / 100.0, "z_trop": z[k], "t_trop": t[k]}
    sel = np.isfinite(z) & np.isfinite(t) & (z >= z[k]) & (z <= z[k] + 2000)
    if sel.sum() >= 3:
        zz, tt = z[sel], t[sel]
        th = tt * (100000.0 / p[sel]) ** 0.2857
        n2 = G * np.diff(np.log(th)) / np.maximum(np.diff(zz), 1.0)
        row["n2_til"] = float(G * (np.log(th[-1]) - np.log(th[0])) / max(zz[-1] - zz[0], 1.0))
        row["n2_til_max"] = float(n2.max())
        row["dt_til"] = float(tt[-1] - tt[0])
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-lat", type=float, default=60.0)
    ap.add_argument("--first-year", type=int, default=1980)
    ap.add_argument("--last-year", type=int, default=2020)
    ap.add_argument("--out", type=Path, default=Path("igra"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    st = stations(a.min_lat, a.first_year, a.last_year)
    st.to_csv(a.out / "stations.csv", index=False)
    print(f"{len(st)} stations at or north of {a.min_lat}N spanning {a.first_year}-{a.last_year}", flush=True)
    frames = []
    for s in st.itertuples():
        t0 = time.perf_counter()
        r = requests.get(DATA.format(id=s.id), headers=HEADERS, timeout=600)
        if r.status_code != 200:
            print(f"{s.id} {s.name}: HTTP {r.status_code}, skipped", flush=True)
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            lines = zf.read(zf.namelist()[0]).decode("ascii", "replace").splitlines()
        rows = [sounding_row(tm, lev) for tm, lev in parse(lines)]
        df = pd.DataFrame(rows)
        df.insert(0, "station", s.id)
        frames.append(df.astype({c: "float32" for c in df.columns if c not in ("station", "time")}))
        print(
            f"{s.id} {s.name}: {len(df)} soundings, {df.p_trop.notna().mean():.0%} with a tropopause "
            f"({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )
    pd.concat(frames, ignore_index=True).to_parquet(a.out / "igra_arctic.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
