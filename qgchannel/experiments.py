"""Experiments: a control climate with restart states, and paired runs with an imposed stratosphere.

The imposed-stratosphere design: from each restart, two runs nudge the stratosphere's zonal mean (above ~12.5 km, at a
1-day rate) toward (a) the control climate and (b) the control climate plus an anomaly. The troposphere and every eddy stay
free in both. The response is (b) minus (a), member by member, so the nudging itself cancels and any difference comes
from the imposed stratosphere alone.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .diagnostics import folds, zonal_stats
from .model import DAY, Channel, Config, build

NUDGE_BASE = 12.5e3  # m: the nudging switches on over ~1 km around this height
NUDGE_TAU = DAY
FOLD_LAYER = 9.0e3  # m: wave breaking is counted on the layer nearest this height
DAILY = ("u", "uv", "h0", "anti", "cyc")
MAP_LEVELS = (0.5e3, 5.5e3)  # m: the lowest layer (surface pressure) and ~500 hPa


def control(
    c: Config | None = None,
    days: int = 2200,
    spin: int = 200,
    every: int = 20,
    seed: int = 1,
    log: Callable[[str], None] = print,
) -> tuple[dict, np.ndarray]:
    """Spin up from rest plus noise and run on. Returns daily zonal means after the spin-up {u, uv, vb, eke} (t, nz, ny),
    the time-mean streamfunction near the ground and at ~5.5 km (psi_mean (2, ny, nx): the stationary waves), and the model
    state every `every` days (restart points)."""
    m = build(c)
    rng = np.random.default_rng(seed)
    m.q = m.invert_q(m.psi_eq) + m.fwd(1e-7 * rng.standard_normal((m.c.nz, m.c.ny, m.c.nx))) * m.mask[None]
    out = {k: [] for k in ("u", "uv", "vb", "eke")}
    snaps = []
    lev = [int(np.argmin(np.abs(m.z - zz))) for zz in MAP_LEVELS]
    psum = np.zeros((len(lev), m.c.ny, m.c.nx))
    t0 = time.perf_counter()
    for d in range(days):
        m.run_days(1)
        if not np.isfinite(m.q).all():
            raise FloatingPointError(f"the control run blew up on day {d}")
        if d < spin:
            continue
        if (d - spin) % every == 0:
            snaps.append(m.q.astype(np.complex64))
        s = zonal_stats(m)
        for k in out:
            out[k].append(s[k])
        psum += m.inv(m.invert_psi(m.q))[lev]
        if d % 200 == 0:
            log(f"control day {d} ({time.perf_counter() - t0:.0f} s)")
    psi_mean = (psum / (days - spin)).astype(np.float32)
    return {k: np.array(v, np.float32) for k, v in out.items()} | {
        "z": m.z,
        "y": m.y,
        "h": m.h,
        "psi_mean": psi_mean,
    }, np.array(snaps)


def ramp(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((z - lo) / (hi - lo), 0, 1)


PROFILES = {
    "ref": np.zeros_like,
    "low": lambda z: ramp(z, 12e3, 16e3) * (1 - ramp(z, 20e3, 24e3)),  # the lowermost stratosphere only
    "deep": lambda z: ramp(z, 12e3, 20e3),  # from the tropopause up, weaker at 12-20 km
    "shallow": lambda z: ramp(z, 22e3, 28e3),  # the upper stratosphere only
}


def polar_anomaly(
    m: Channel, profile: str, amplitude: float = -20.0, y0: float = 6.8e6, width: float = 1.4e6
) -> np.ndarray:
    """An idealized polar wind anomaly (nz, ny) in m/s: amplitude x vertical profile x a Gaussian centred on y0."""
    return amplitude * PROFILES[profile](m.z)[:, None] * np.exp(-(((m.y - y0) / width) ** 2))[None, :]


def target(m: Channel, ubar: np.ndarray) -> np.ndarray:
    """Spectral streamfunction whose zonal-mean wind is ubar (nz, ny) less the background U and the channel mean
    (psi = 0 on both walls keeps the channel-mean wind fixed)."""
    du = ubar - m.U[:, None]
    du = du - du.mean(1, keepdims=True)
    dy = m.y[1] - m.y[0]
    psi = -np.cumsum(du, axis=1) * dy
    psi -= psi[:, -1:] * (m.y / m.c.Ly)[None, :]
    return m.fwd(np.repeat(psi[:, :, None], m.c.nx, axis=2))


def imposed(
    c: Config,
    restarts: np.ndarray,
    clim: np.ndarray,
    anomaly: np.ndarray,
    days: int = 90,
    eddies: bool = True,
    maps: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """Runs from each restart with the stratosphere nudged toward clim + anomaly, in m/s: clim (nz, ny), anomaly either
        fixed (nz, ny) or one per day (days, nz, ny). Returns daily {u, uv (nz, ny); h0, anti, cyc (ny,)} stacked as
        (member, day, ...), NaN from the day a member blows up. maps=True adds the daily streamfunction near the ground and at
    ~5.5 km, psi_maps (member, day, 2, ny, nx)."""
    m = build(c)
    rate = 0.5 * (1 + np.tanh((m.z - NUDGE_BASE) / 1000.0)) / NUDGE_TAU
    iz = int(np.argmin(np.abs(m.z - FOLD_LAYER)))
    anomaly = np.asarray(anomaly)
    targets = [target(m, clim + a) for a in (anomaly if anomaly.ndim == 3 else anomaly[None])]
    keys = (*DAILY, "psi_maps") if maps else DAILY
    lev = [int(np.argmin(np.abs(m.z - zz))) for zz in MAP_LEVELS]
    out = {k: [] for k in keys}
    t0 = time.perf_counter()
    for k, q0 in enumerate(restarts):
        m.restart(q0)
        m.eddies = eddies
        rec = {kk: [] for kk in keys}
        for d in range(days):
            m.nudge = (targets[min(d, len(targets) - 1)], rate)
            m.run_days(1)
            if not np.isfinite(m.q).all():
                for kk in keys:
                    rec[kk] += [np.full_like(rec[kk][0], np.nan)] * (days - d)
                break
            s = zonal_stats(m)
            an, cy = folds(m, iz)
            for kk, v in (("u", s["u"]), ("uv", s["uv"]), ("h0", s["h0"]), ("anti", an), ("cyc", cy)):
                rec[kk].append(v)
            if maps:
                rec["psi_maps"].append(m.inv(m.invert_psi(m.q))[lev])
        for kk in keys:
            out[kk].append(np.array(rec[kk], np.float32))
        log(f"member {k} done ({time.perf_counter() - t0:.0f} s)")
    return {k: np.array(v) for k, v in out.items()} | {"z": m.z, "y": m.y, "fold_layer": iz, "h": m.h}


def to_channel(
    m: Channel, u_anom: np.ndarray, lev_hpa: np.ndarray, lat: np.ndarray, H: float = 7.0e3, lat0: float = 9.0
) -> np.ndarray:
    """Map observed zonal-mean wind anomalies (time, lev, lat) onto the channel (time, nz, ny): z = H ln(1000 / p),
    y = (lat - lat0) x 111.2 km, tapered to zero over the 1000 km south of the southernmost latitude given."""
    zl = H * np.log(1000.0 / np.asarray(lev_hpa, float))
    yl = (np.asarray(lat, float) - lat0) * 111.2e3
    oz, oy = np.argsort(zl), np.argsort(yl)
    zl, yl = zl[oz], yl[oy]
    taper = np.clip((m.y - (yl[0] - 1.0e6)) / 1.0e6, 0, 1)
    out = np.empty((u_anom.shape[0], m.c.nz, m.c.ny))
    for t in range(u_anom.shape[0]):
        v = np.asarray(u_anom[t])[oz][:, oy]
        vz = np.array([np.interp(m.z, zl, v[:, j]) for j in range(v.shape[1])]).T
        out[t] = np.array([np.interp(m.y, yl, vz[i], left=vz[i, 0]) for i in range(m.c.nz)]) * taper
    return out
