"""Multi-level quasi-geostrophic beta-plane channel from the ground to 60 km: an idealized atmosphere with its own storms.

Prognostic variable: the potential vorticity of the departure from a fixed background flow U(z),

    q = lap(psi) + (f0^2 / rho) d/dz (rho / N^2 dpsi/dz),

stepped with

    dq/dt + (U + u) q_x + v (Q_y + q_y) = -S_alpha (psi - psi_eq) - r(z) lap(psi) - sponge - walls - hyperdiffusion.

z = H ln(p0 / p) and rho = exp(-z / H). The stretching operator S is built on layers with zero flux through the ground and
the lid, so the background PV gradient Q_y = beta - S U carries the ground and lid temperature gradients as PV sheets in
the end layers (Bretherton): the baroclinic instability that makes the storms is in Q_y. x is periodic (Fourier); y has
walls at 0 and Ly with psi = 0 (sine series), so inverting q for psi is a division in (vertical eigenmode, l, k).

Forcing and damping: Newtonian cooling of the temperature (b = f0 psi_z) toward the zonal-mean equilibrium psi_eq, at a
rate alpha(z); Rayleigh friction on the vorticity in the lowest 2 km; a sponge under the lid; damping zones along both
walls; an exact scale-selective hyperdiffusion. Time stepping: third-order Adams-Bashforth, 2/3-rule dealiasing.

Optionally the zonal mean (k = 0) can be nudged toward a target state at a rate that varies with height
(`Channel.nudge`): that imposes a stratosphere while leaving the troposphere and every eddy free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.fft as sf

DAY = 86400.0


@dataclass
class Config:
    nx: int = 64
    ny: int = 32
    nz: int = 30
    Lx: float = 2 * np.pi * 6.371e6 * np.cos(np.deg2rad(50.0))  # one latitude circle at 50N
    Ly: float = 9.0e6  # about 10N to the pole
    ztop: float = 60.0e3
    H: float = 7.0e3
    f0: float = 1.12e-4
    beta: float = 1.47e-11
    dt: float = 900.0  # s; 600 s is needed when a strong polar anomaly is imposed for months
    hyper_days: float = 0.3  # e-folding time of the smallest resolved scale
    zt: float = 11.0e3  # tropopause height
    n2_trop: float = 1.2e-4
    n2_strat: float = 4.5e-4
    r_sfc: float = 1.0 / DAY  # friction at the ground, falling linearly to zero at z_fric
    z_fric: float = 2.0e3
    sponge_z: float = 50.0e3
    sponge_rate: float = 0.5 / DAY
    u_jet: float = 30.0  # equilibrium jet at the tropopause (sets the heating contrast)
    wall_width: float = 7.0e5  # damping zones along both walls keep momentum from piling up there
    wall_rate: float = 1.0 / DAY


class Channel:
    def __init__(self, c: Config, U: np.ndarray, psi_eq_grid: np.ndarray, alpha: np.ndarray):
        """U (nz,): background flow; psi_eq_grid (nz, ny): zonal-mean equilibrium streamfunction; alpha (nz,): cooling
        rate (1/s)."""
        self.c = c
        nz, ny, nx = c.nz, c.ny, c.nx
        self.dz = c.ztop / nz
        self.z = (np.arange(nz) + 0.5) * self.dz
        self.zh = np.arange(1, nz) * self.dz  # interior half levels
        self.y = np.arange(1, ny + 1) * c.Ly / (ny + 1)
        self.x = np.arange(nx) * c.Lx / nx
        self.rho = np.exp(-self.z / c.H)
        self.rhoh = np.exp(-self.zh / c.H)
        self.n2h = self.n2_profile(self.zh)
        self.kx = 2 * np.pi * np.arange(nx // 2 + 1) / c.Lx
        self.ly = np.pi * np.arange(1, ny + 1) / c.Ly
        self.K2 = self.kx[None, :] ** 2 + self.ly[:, None] ** 2  # (ny, nk)
        # vertical operators: stretching S, its eigenmodes for the inversion, and the cooling operator S_alpha
        self.S = self.stretch(np.ones(nz - 1))
        lam, V = np.linalg.eig(self.S)
        o = np.argsort(-lam.real)
        self.lam, self.V = lam.real[o], V.real[:, o]
        self.Vi = np.linalg.inv(self.V)
        self.den = -self.K2[None, :, :] + self.lam[:, None, None]
        self.Sa = self.stretch(np.interp(self.zh, self.z, alpha))
        # background state
        self.U = U
        self.Qy = c.beta - self.S @ U
        self.psi_eq = self.fwd(np.repeat(psi_eq_grid[:, :, None], nx, axis=2))
        self.q_eq = self.invert_q(self.psi_eq)
        # damping
        self.r = c.r_sfc * np.clip(1 - self.z / c.z_fric, 0, 1)
        self.sponge = c.sponge_rate * np.clip((self.z - c.sponge_z) / (c.ztop - c.sponge_z), 0, 1) ** 2
        yw = np.exp(-((self.y / c.wall_width) ** 2)) + np.exp(-(((c.Ly - self.y) / c.wall_width) ** 2))
        self.wall = (c.wall_rate * yw)[None, :, None]
        self.hyp = np.exp(-((self.K2 / self.K2.max()) ** 4) * c.dt / (c.hyper_days * DAY))
        self.mask = (np.arange(nx // 2 + 1)[None, :] <= nx // 3) & (np.arange(1, ny + 1)[:, None] <= 2 * ny // 3)
        # state
        self.q = np.zeros((nz, ny, nx // 2 + 1), complex)
        self.hist: list[np.ndarray] = []
        self.t = 0.0
        self.nudge: tuple[np.ndarray, np.ndarray] | None = None  # (target psi, spectral; rate (nz,) in 1/s)
        self.eddies = True  # False zeroes every eddy after each step (the storms-off experiment)

    def n2_profile(self, z: np.ndarray) -> np.ndarray:
        c = self.c
        w = 0.5 * (1 + np.tanh((z - c.zt) / 1500.0))
        return c.n2_trop + (c.n2_strat - c.n2_trop) * w

    def stretch(self, coef_h: np.ndarray) -> np.ndarray:
        """(f0^2 / rho) d/dz (rho coef / N^2 d/dz) on the layers, with zero flux through the ground and the lid."""
        c, nz, dz = self.c, self.c.nz, self.dz
        a = self.rhoh * coef_h / self.n2h / dz**2
        S = np.zeros((nz, nz))
        i = np.arange(nz - 1)
        S[i, i + 1] += a
        S[i, i] -= a
        S[i + 1, i] += a
        S[i + 1, i + 1] -= a
        return (c.f0**2 / self.rho)[:, None] * S

    # transforms ------------------------------------------------------------------------------------------------------
    def fwd(self, g: np.ndarray) -> np.ndarray:
        """Grid (nz, ny, nx) -> spectral (nz, ny, nx // 2 + 1): sine series in y, Fourier in x."""
        return sf.rfft(sf.dst(g, type=1, axis=1, norm="ortho"), axis=2)

    def inv(self, h: np.ndarray) -> np.ndarray:
        return sf.idst(sf.irfft(h, n=self.c.nx, axis=2), type=1, axis=1, norm="ortho")

    def inv_dy(self, h: np.ndarray) -> np.ndarray:
        """d/dy on the grid: the y-derivative of a sine series is a cosine series."""
        g = sf.irfft(h * self.ly[None, :, None], n=self.c.nx, axis=2)
        a = np.zeros((g.shape[0], g.shape[1] + 2, g.shape[2]))
        a[:, 1:-1] = g
        return 0.5 * np.sqrt(2.0 / (self.c.ny + 1)) * sf.dct(a, type=1, axis=1)[:, 1:-1]

    def inv_dx(self, h: np.ndarray) -> np.ndarray:
        return self.inv(1j * self.kx[None, None, :] * h)

    # balance ---------------------------------------------------------------------------------------------------------
    def invert_psi(self, qh: np.ndarray) -> np.ndarray:
        qm = np.einsum("ij,jmk->imk", self.Vi, qh)
        return np.einsum("ij,jmk->imk", self.V, qm / self.den)

    def invert_q(self, ph: np.ndarray) -> np.ndarray:
        return -self.K2[None] * ph + np.einsum("ij,jmk->imk", self.S, ph)

    # dynamics --------------------------------------------------------------------------------------------------------
    def tendency(self, qh: np.ndarray) -> np.ndarray:
        ph = self.invert_psi(qh)
        u = -self.inv_dy(ph) + self.U[:, None, None]
        v = self.inv_dx(ph)
        adv = self.fwd(u * self.inv_dx(qh) + v * (self.inv_dy(qh) + self.Qy[:, None, None]))
        F = -np.einsum("ij,jmk->imk", self.Sa, ph - self.psi_eq)
        F += self.r[:, None, None] * self.K2[None] * ph
        F -= self.sponge[:, None, None] * (qh - self.q_eq)
        if self.c.wall_rate > 0:
            F -= self.fwd(self.wall * self.inv(qh - self.q_eq))
        if self.nudge is not None:
            target, rate = self.nudge
            zm = np.zeros_like(qh)
            zm[:, :, 0] = self.invert_q(target)[:, :, 0] - qh[:, :, 0]
            F += rate[:, None, None] * zm
        return (-adv + F) * self.mask[None]

    def step(self) -> None:
        dt = self.c.dt
        self.hist.insert(0, self.tendency(self.q))
        self.hist = self.hist[:3]
        h = self.hist
        if len(h) == 1:
            dq = dt * h[0]
        elif len(h) == 2:
            dq = dt * (1.5 * h[0] - 0.5 * h[1])
        else:
            dq = dt * (23 * h[0] - 16 * h[1] + 5 * h[2]) / 12
        self.q = (self.q + dq) * self.hyp[None]
        if not self.eddies:
            self.q[:, :, 1:] = 0
        self.t += dt

    def restart(self, q: np.ndarray) -> None:
        """Start from a saved state (the time-stepping history is cleared)."""
        self.q = np.asarray(q).astype(complex)
        self.hist = []

    def run_days(self, days: float) -> None:
        for _ in range(round(days * DAY / self.c.dt)):
            self.step()

    # diagnostics -----------------------------------------------------------------------------------------------------
    def fields(self) -> dict:
        ph = self.invert_psi(self.q)
        return {"psi": self.inv(ph), "u": -self.inv_dy(ph) + self.U[:, None, None], "v": self.inv_dx(ph)}


def default_setup(c: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Background U(z), equilibrium psi_eq (nz, ny) and cooling rate alpha(z) for a northern winter.

    The equilibrium jet is U_jet(z) sin^2(pi y / Ly): half of it is the background U(z), the y-varying half is psi_eq
    (zero channel mean). The jet rises to u_jet at the tropopause, weakens in the lower stratosphere and strengthens into
    a winter vortex aloft. Cooling: 40 days in the troposphere, 45 days just above the tropopause falling to 8 days at the
    lid."""
    z = (np.arange(c.nz) + 0.5) * c.ztop / c.nz
    y = np.arange(1, c.ny + 1) * c.Ly / (c.ny + 1)
    ujet = np.where(
        z < c.zt,
        c.u_jet * z / c.zt,
        c.u_jet - 20.0 * np.clip((z - c.zt) / 9e3, 0, 1) + 40.0 * np.clip((z - 20e3) / 30e3, 0, 1),
    )
    A = 0.5 * ujet
    psi_eq = A[:, None] * (c.Ly / (2 * np.pi)) * np.sin(2 * np.pi * y / c.Ly)[None, :]
    alpha = np.where(
        z < c.zt,
        1 / (40 * DAY),
        np.interp(z, [11e3, 20e3, 30e3, 45e3, 60e3], [1 / 45, 1 / 22, 1 / 16, 1 / 11, 1 / 8]) / DAY,
    )
    return A.copy(), psi_eq, alpha


def build(c: Config | None = None) -> Channel:
    """A channel with the default winter setup."""
    c = c or Config()
    return Channel(c, *default_setup(c))
