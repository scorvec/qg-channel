"""Checks of the numerics against exact results."""

from __future__ import annotations

import numpy as np
import pytest

import qgchannel as qg


@pytest.fixture(scope="module")
def m() -> qg.Channel:
    return qg.build(qg.Config(nx=32, ny=16, nz=12))


def test_inversion_round_trip(m):
    rng = np.random.default_rng(0)
    q = m.fwd(rng.standard_normal((m.c.nz, m.c.ny, m.c.nx)))
    q[:, :, 0] = q[:, :, 0].real
    assert np.allclose(m.invert_q(m.invert_psi(q)), q, atol=1e-10 * np.abs(q).max())


def test_y_derivative(m):
    l = 3 * np.pi / m.c.Ly
    g = np.sin(l * m.y)[None, :, None] * np.cos(2 * np.pi * m.x / m.c.Lx)[None, None, :] * np.ones((m.c.nz, 1, 1))
    exact = l * np.cos(l * m.y)[None, :, None] * np.cos(2 * np.pi * m.x / m.c.Lx)[None, None, :]
    assert np.allclose(m.inv_dy(m.fwd(g)), exact, atol=1e-10)


def test_stretching_conserves_and_kills_constants(m):
    assert np.allclose(m.S @ np.ones(m.c.nz), 0, atol=1e-25)  # no flux through ground or lid
    # mass-weighted column integral of the stretching term vanishes
    assert abs(m.rho @ (m.S @ np.sin(np.arange(m.c.nz)))) < 1e-12 * np.abs(m.S).max()


def test_free_rossby_wave_phase_speed():
    """A single (k, l, vertical mode) Rossby wave on a resting atmosphere is an exact nonlinear solution; it must move
    west at c = -beta / (K^2 - lambda_n)."""
    c = qg.Config(nx=32, ny=16, nz=12, r_sfc=0.0, sponge_rate=0.0, wall_rate=0.0, dt=900.0)
    m = qg.Channel(c, np.zeros(c.nz), np.zeros((c.nz, c.ny)), np.zeros(c.nz))
    ik, il, n = 2, 1, 1
    k, l = m.kx[ik], m.ly[il - 1]
    ph = np.zeros_like(m.q)
    ph[:, il - 1, ik] = 1e6 * m.V[:, n]
    m.q = m.invert_q(ph)
    days = 2.0
    m.run_days(days)
    a = m.invert_psi(m.q)[:, il - 1, ik] @ m.Vi[n]
    cp = -c.beta / (k**2 + l**2 - m.lam[n])  # negative: westward
    # psi ~ cos(k (x - cp t)), so the Fourier coefficient of exp(ikx) turns by exp(-i k cp t)
    turned = np.angle(a / 1e6)
    assert np.isclose(np.angle(np.exp(1j * (turned + k * cp * days * qg.DAY))), 0, atol=1e-6)


def test_imposed_run_is_finite_and_paired():
    c = qg.Config(nx=32, ny=16, nz=12)
    m = qg.build(c)
    rng = np.random.default_rng(2)
    q0 = (m.invert_q(m.psi_eq) + m.fwd(1e-6 * rng.standard_normal((c.nz, c.ny, c.nx))) * m.mask[None])[None]
    clim = np.repeat(m.U[:, None], c.ny, axis=1)
    ref = qg.imposed(c, q0, clim, np.zeros_like(clim), days=3, log=lambda _: None)
    exp = qg.imposed(c, q0, clim, qg.polar_anomaly(m, "low"), days=3, log=lambda _: None)
    assert np.isfinite(exp["u"]).all()
    assert not np.allclose(exp["u"], ref["u"])


def test_mountain_forces_anticyclone_on_windward_side():
    """Flow over a mountain: with q = 0 and westerlies at the ground, the lowest layer's PV falls on the windward (west)
    slope and rises on the lee slope; the mountain changes the tendency of the lowest layer only."""
    flat = qg.build(qg.Config(mountains=()))
    assert flat.qtopo_x is None
    m = qg.build(qg.Config(mountains=("greenland",)))
    assert m.U[0] > 0
    t = m.inv(m.tendency(np.zeros_like(m.q)))[0]
    j = int(np.argmax(m.h.max(1)))
    i = int(np.argmax(m.h[j]))
    assert t[j, i - 2] < 0 < t[j, i + 2]
    d = m.tendency(np.zeros_like(m.q)) - flat.tendency(np.zeros_like(flat.q))
    assert np.allclose(d[1:], 0) and np.abs(d[0]).max() > 0
