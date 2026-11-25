"""Diagnostics: zonal means, eddy fluxes, PV folds (wave breaking) and the channel's annular mode."""

from __future__ import annotations

import numpy as np

from .model import Channel


def zonal_stats(m: Channel) -> dict:
    """Zonal means (nz, ny) of the current state: u, eddy momentum flux u'v', eddy heat flux v'b' (b = f0 psi_z), eddy
    kinetic energy, and the bottom-layer height f0 psi / g."""
    ph = m.invert_psi(m.q)
    psi = m.inv(ph)
    u = -m.inv_dy(ph) + m.U[:, None, None]
    v = m.inv_dx(ph)
    ue = u - u.mean(2, keepdims=True)
    b = m.c.f0 * np.gradient(psi, m.z, axis=0)
    be = b - b.mean(2, keepdims=True)
    return {
        "u": u.mean(2),
        "uv": (ue * v).mean(2),
        "vb": (v * be).mean(2),
        "eke": 0.5 * (ue**2 + v**2).mean(2),
        "h0": m.c.f0 * psi[0].mean(1) / 9.80665,
    }


def folds(m: Channel, iz: int) -> tuple[np.ndarray, np.ndarray]:
    """Share of each latitude row where the PV of layer iz is overturned (total meridional PV gradient reversed by at
    least the background gradient), split by tilt: anticyclonic (q_x > 0, SW-NE, LC1) and cyclonic (q_x < 0, NW-SE,
    LC2). Returns two arrays (ny,)."""
    q = m.q[iz : iz + 1]
    qx = m.inv_dx(q)[0]
    qy = m.inv_dy(q)[0] + m.Qy[iz]
    rev = qy < -abs(m.Qy[iz])
    return (rev & (qx > 0)).mean(1), (rev & (qx < 0)).mean(1)


def total_pv(m: Channel, iz: int) -> np.ndarray:
    """Total PV of layer iz on the grid (ny, nx): the departure plus the background Q_y y."""
    return m.inv(m.q[iz : iz + 1])[0] + m.Qy[iz] * m.y[:, None]


def annular_mode(u: np.ndarray, m: Channel, top: float = 11.0e3) -> tuple[float, float]:
    """Leading pattern of the density-weighted tropospheric zonal-mean wind (daily series u: (t, nz, ny)): its share of
    the variance and its e-folding time in days."""
    trop = (m.z > 500.0) & (m.z < top)
    w = m.rho[trop] / m.rho[trop].sum()
    a = np.einsum("tzy,z->ty", u[:, trop], w)
    a = a - a.mean(0)
    _, s, vt = np.linalg.svd(a, full_matrices=False)
    pc = a @ vt[0]
    pc = pc - pc.mean()
    ac = np.array([np.dot(pc[: pc.size - k], pc[k:]) / np.dot(pc, pc) for k in range(80)])
    below = np.nonzero(ac < np.exp(-1))[0]
    return float(s[0] ** 2 / (s**2).sum()), float(below[0]) if below.size else 80.0
