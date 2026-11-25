"""What Arctic radiosondes saw around each sudden stratospheric warming: the polar-cap anomaly of the tropopause and of the
lower-stratosphere temperature, from the soundings alone.

    python observations/igra_events.py [--igra igra] [--dates observations/ssw_dates.csv] [--min-lat 65]
Reads igra_arctic.parquet and stations.csv (observations/igra_arctic.py) and the event list (central dates of the
MERRA-2 SSW catalogue, CP07 at 60N 10 hPa, with the type, the depth and the AO averaged over days 15-60). For every station
north of --min-lat: daily means of the 00 and 12 UTC soundings, anomalies about the station's 1981-2020 calendar mean
(366 slots, Feb 29 its own slot, smoothed over 31 days); then the mean over the stations reporting that day (at least 10).
Writes, next to the input:
  igra_event_daily.csv    every event, days -30..+75, the cap-mean anomaly of each quantity and the number of stations
  igra_event_windows.csv  the same averaged over days -30..-1, 0..14, 15..44 and 45..74
Quantities: tropopause height/pressure/temperature, the tropopause-inversion strength (mean and maximum N^2 in the 2 km
above the tropopause, and the warming across them), and temperature at 300, 200, 150, 100, 70 and 50 hPa.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

QUANT = ("z_trop", "p_trop", "t_trop", "n2_til", "n2_til_max", "dt_til", "t300", "t200", "t150", "t100", "t70", "t50")
WINDOWS = {"d-30..-1": (-30, -1), "d0..14": (0, 14), "d15..44": (15, 44), "d45..74": (45, 74)}


def slot366(t: pd.DatetimeIndex) -> np.ndarray:
    doy = t.dayofyear.values - 1
    return np.where(~t.is_leap_year & (doy >= 59), doy + 1, doy)


def station_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(day=df.time.dt.floor("D")).groupby("day")[list(QUANT)].mean()
    d.index = pd.DatetimeIndex(d.index)
    s = slot366(d.index)
    base = d[(d.index.year >= 1981) & (d.index.year <= 2020)]
    clim = base.groupby(slot366(base.index)).mean().reindex(range(366))
    clim = (
        pd.concat([clim.iloc[-15:], clim, clim.iloc[:15]]).rolling(31, center=True, min_periods=10).mean().iloc[15:-15]
    )
    clim.index = range(366)
    return d - clim.loc[s].set_axis(d.index)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--igra", type=Path, default=Path("igra"))
    ap.add_argument("--dates", type=Path, default=Path("observations/ssw_dates.csv"))
    ap.add_argument("--min-lat", type=float, default=65.0)
    a = ap.parse_args()
    st = pd.read_csv(a.igra / "stations.csv")
    keep = set(st.id[st.lat >= a.min_lat])
    df = pd.read_parquet(a.igra / "igra_arctic.parquet")
    df = df[df.station.isin(keep)]
    anoms = {sid: station_anomalies(g) for sid, g in df.groupby("station")}
    print(f"{len(anoms)} stations north of {a.min_lat}N")
    stacked = pd.concat(anoms, names=["station", "day"])
    cap = stacked.groupby(level="day").mean()
    count = stacked["t100"].notna().groupby(level="day").sum()
    cap = cap.where(count >= 10)
    ev = pd.read_csv(a.dates, parse_dates=["date"])
    rows, wins = [], []
    for e in ev.itertuples():
        lag = pd.RangeIndex(-30, 76)
        days = e.date + pd.to_timedelta(lag, "D")
        block = cap.reindex(days).set_axis(lag)
        block["stations"] = count.reindex(days).values
        rows.append(block.assign(date=e.date.date(), lag=lag))
        w = {"date": e.date.date(), "type": e.type, "depth": e.depth, "ao_d15_60": e.ao_d15_60}
        for name, (lo, hi) in WINDOWS.items():
            sub = block.loc[lo:hi]
            for q in QUANT:
                w[f"{q}_{name}"] = sub[q].mean() if sub[q].notna().sum() >= (hi - lo + 1) // 2 else np.nan
        wins.append(w)
    pd.concat(rows).to_csv(a.igra / "igra_event_daily.csv", index=False, float_format="%.5g")
    out = pd.DataFrame(wins)
    out.to_csv(a.igra / "igra_event_windows.csv", index=False, float_format="%.5g")
    cols = ["date", "depth", "ao_d15_60", "n2_til_d0..14", "n2_til_d15..44", "z_trop_d15..44", "t150_d15..44"]
    print(out[cols].round(6).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
