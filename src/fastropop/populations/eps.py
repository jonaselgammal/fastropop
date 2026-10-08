r"""Extended Press-Schechter population of merging black holes (Ellis et al. 2024, Eqs. 3-4).

    dR_BH / (dln m1 dln m2) = p_BH  \int dln M1 dln M2  dR_h / (dln M1 dln M2)
                                     x  p(ln m1 | M1, z)  p(ln m2 | M2, z)

* R_h: the halo merger rate. Press-Schechter mass function times the Lacey & Cole (1993)
  merger kernel, symmetrised in the two halo masses, built from the cosmology's linear
  theory (sigma(M), delta_c(z)). The transcription follows the KBFI implementation
  (``crosschecks/code/ville/eps_rate.py`` in the LISA_MBHBs repository).
* M*(M_h, z): the stellar-to-halo mass relation of Girelli et al. (2020), their Eq. 6 with
  the redshift evolution of their Table 3, reference case.
* p(log10 m | M_h, z): a lognormal in black-hole mass with mean
  a + b log10(M* / 1e11 Msun) + gamma log10(1 + z) and width sigma [dex]. The fit to
  inactive galaxies of Reines & Volonteri (2015) is a = 8.95, b = 1.4, sigma = 0.48,
  gamma = 0. Every halo hosts a black hole; p_BH absorbs the occupation fraction and the
  merger efficiency.

Free parameters: (log10_pBH, a, b, sigma, gamma). Native coordinates: (log10 m1, log10 m2, z)
with m2 <= m1. Integrated over the whole (m1, m2) square, dR_BH counts every merger once
(the convention of eps_rate.py, so p_BH means what it means there). The rate on the physical
triangle m2 <= m1 is therefore twice the symmetric kernel.

The halo tables depend only on the cosmology and are built once, at ``z_nodes``. A
likelihood call computes the black-hole kernel at those nodes and contracts it with the
halo table. Other redshifts are reached by interpolating log rate linearly in z.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..cosmology import PLANCK18
from ..grid import M1M2
from .base import PopulationModel

_LN10 = float(np.log(10.0))

# Girelli et al. (2020) Table 3, reference case: log M_A = B + mu z, A = C (1+z)^nu,
# gamma = D (1+z)^eta, beta = F z + E
GIRELLI2020 = dict(B=11.79, mu=0.20, C=0.046, nu=-0.38, D=0.709, eta=-0.18, F=0.043, E=0.96)

LABELS = {"log10_pBH": r"$\log_{10} p_{\rm BH}$", "a": r"$a$", "b": r"$b$", "sigma": r"$\sigma$",
          "gamma": r"$\gamma$"}
RV15_INACTIVE = dict(log10_pBH=0.0, a=8.95, b=1.4, sigma=0.48, gamma=0.0)


def log10_mstar(log10_Mh, z, shmr=GIRELLI2020):
    """log10 M*(M_h, z) [Msun], Girelli et al. (2020) Eq. 6:
    M*/M_h = 2 A [(M_h/M_A)^-beta + (M_h/M_A)^gamma]^-1."""
    s = shmr
    x = 10.0 ** (log10_Mh - (s["B"] + s["mu"] * z))
    A = s["C"] * (1.0 + z) ** s["nu"]
    gam = s["D"] * (1.0 + z) ** s["eta"]
    beta = s["F"] * z + s["E"]
    return log10_Mh + np.log10(2.0 * A) - np.log10(x ** (-beta) + x**gam)


def _loglerp(lo, hi, t):
    """Linear interpolation of log rate between two nodes; zero if either node is zero."""
    ok = (lo > 0) & (hi > 0)
    safe = lambda v: jnp.where(ok, v, 1.0)
    return jnp.where(ok, jnp.exp((1 - t) * jnp.log(safe(lo)) + t * jnp.log(safe(hi))), 0.0)


class EPS(PopulationModel):
    """Extended Press-Schechter black-hole mergers (see the module docstring).

    Parameters
    ----------
    cosmology : fastropop.cosmology.Cosmology
        Its linear theory sets the halo merger rate.
    z_nodes : array, optional
        Redshifts at which the halo tables are built (default 0.05 ... 25 in 100 steps).
    log10_Mh : tuple, optional
        (min, max, step) of the halo-mass quadrature in log10 Msun (default 7 ... 16.5, 0.1).
        With sigma >= 0.2 dex the trapezoid rule on the lognormal is accurate to < 1e-4.
    shmr : dict, optional
        Stellar-to-halo parameters (default :data:`GIRELLI2020`).
    """

    native_coords = M1M2
    param_names = ("log10_pBH", "a", "b", "sigma", "gamma")
    defaults = dict(RV15_INACTIVE)
    labels = dict(LABELS)

    def __init__(self, cosmology=PLANCK18, z_nodes=None, log10_Mh=(7.0, 16.5, 0.1), shmr=None, support=None):
        z_nodes = np.linspace(0.05, 25.0, 100) if z_nodes is None else np.asarray(z_nodes, dtype=float)
        sup = {"z": (float(z_nodes[0]), float(z_nodes[-1]))}
        sup.update(support or {})
        super().__init__(cosmology, sup)
        self.shmr = dict(GIRELLI2020 if shmr is None else shmr)
        self.z_nodes = z_nodes
        lo, hi, step = log10_Mh
        self.log10_Mh = np.arange(lo, hi + step / 2, step)
        w = np.full(self.log10_Mh.size, step)
        w[0] = w[-1] = step / 2
        R = self.halo_merger_rate(self.log10_Mh, z_nodes)                      # (z, h, h)
        # rate per dex^2 of BH mass = ln10^2 sum_h1 h2 w1 w2 R p1 p2 (p per dex; see module docstring)
        self._A = jnp.asarray(_LN10**2 * R * w[None, :, None] * w[None, None, :])
        self._x = jnp.asarray(np.stack([log10_mstar(self.log10_Mh, z, self.shmr) for z in z_nodes]))   # (z, h)
        self._zn = jnp.asarray(z_nodes)

    # ------------------------------------------------------------------ halo merger rate (fixed)
    def press_schechter(self, log10_Mh, z):
        """dn/dln M_h [Mpc^-3], Press & Schechter (1974)."""
        c = self.cosmology
        M = 10.0 ** np.asarray(log10_Mh, dtype=float)
        s = np.asarray(c.sigma_M(M))
        dsdM = s * np.asarray(c.dlnsigma_dlnM(M)) / M
        dc = float(c.delta_c(z))
        return np.sqrt(2.0 / np.pi) * c.rho_m0 * dc / s**2 * np.abs(dsdM) * np.exp(-dc**2 / (2 * s**2))

    def halo_merger_rate(self, log10_Mh, z_nodes):
        """dR_h / (dln M1 dln M2) [Mpc^-3 yr^-1] at each redshift: Press-Schechter mass
        function x Lacey & Cole (1993) kernel, symmetrised with (min, max) of the two masses."""
        c = self.cosmology
        M = 10.0 ** np.asarray(log10_Mh, dtype=float)
        A, B = np.meshgrid(M, M, indexing="ij")
        lo, hi = np.minimum(A, B), np.maximum(A, B)
        s_lo, s_hi, s_sum = (np.asarray(c.sigma_M(x)) for x in (lo, hi, A + B))
        ds = lambda x: np.asarray(c.sigma_M(x)) * np.asarray(c.dlnsigma_dlnM(x)) / x       # d sigma / d M
        ratio_ds = ds(A + B) / ds(hi)
        out = np.empty((len(z_nodes), M.size, M.size))
        for k, z in enumerate(z_nodes):
            dc = float(c.delta_c(z))
            ddc = float(c.delta_c(z + 1e-4) - c.delta_c(z - 1e-4)) / 2e-4                 # d delta_c / dz
            dtdz = -float(c.dt_dz(z))                                                       # d(age)/dz [yr]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                Q = (-hi / dtdz / c.rho_m0 * (ddc / dc) * ratio_ds * (s_hi / s_sum) ** 2
                     * (1.0 - (s_sum / s_lo) ** 2) ** -1.5
                     * np.exp(-dc**2 / 2.0 * (1.0 / s_sum**2 - 1.0 / s_lo**2 - 1.0 / s_hi**2)))
                n = self.press_schechter(log10_Mh, z)
                R = n[:, None] * n[None, :] * Q
            out[k] = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    # ------------------------------------------------------------------ black holes (per call)
    def _kernel(self, p, log10_m):
        """p(log10 m | M_h, z) per dex at the halo-table nodes: shape (z, m, h)."""
        mean = p["a"] + p["b"] * (self._x - 11.0) + p["gamma"] * jnp.log10(1.0 + self._zn)[:, None]
        u = (jnp.asarray(log10_m)[None, :, None] - mean[:, None, :]) / p["sigma"]
        return jnp.exp(-0.5 * u * u) / (p["sigma"] * jnp.sqrt(2.0 * jnp.pi))

    def rate_at_nodes(self, params, log10_m1, log10_m2):
        """Comoving rate on the triangle m2 <= m1 [Mpc^-3 yr^-1 per dex^2] for the tensor
        product of ``log10_m1`` and ``log10_m2`` at every ``z_nodes``: shape (z, m1, m2)."""
        K1, K2 = self._kernel(params, log10_m1), self._kernel(params, log10_m2)
        L = jnp.einsum("zmh,zhk,znk->zmn", K1, self._A, K2)
        tri = jnp.asarray(log10_m2)[None, :] <= jnp.asarray(log10_m1)[:, None]
        return 2.0 * 10.0 ** params["log10_pBH"] * jnp.where(tri[None], L, 0.0)

    def comoving_rate(self, params, log10_m1, log10_m2, z, chunk=512):
        """At arbitrary points: the kernel contracted point by point at the two bracketing
        redshift nodes, then log-linear in z. Evaluated in chunks of ``chunk`` points, so
        memory stays at ``chunk`` copies of one halo table."""
        import jax

        l1, l2, z = (jnp.asarray(v, dtype=float) for v in jnp.broadcast_arrays(log10_m1, log10_m2, z))
        shape = l1.shape
        n = l1.size
        pad = (-n) % chunk
        flat = [jnp.pad(v.ravel(), (0, pad), constant_values=v.ravel()[0]) for v in (l1, l2, z)]
        zn, p = self._zn, params

        def one_chunk(args):
            a1, a2, zz = args
            i = jnp.clip(jnp.searchsorted(zn, zz) - 1, 0, zn.size - 2)
            t = jnp.clip((zz - zn[i]) / (zn[i + 1] - zn[i]), 0.0, 1.0)

            def at(j):
                mean = p["a"] + p["b"] * (self._x[j] - 11.0) + p["gamma"] * jnp.log10(1.0 + zn[j])[:, None]
                k1 = jnp.exp(-0.5 * ((a1[:, None] - mean) / p["sigma"]) ** 2)
                k2 = jnp.exp(-0.5 * ((a2[:, None] - mean) / p["sigma"]) ** 2)
                return jnp.einsum("ph,phk,pk->p", k1, self._A[j], k2) / (2.0 * jnp.pi * p["sigma"] ** 2)

            return jnp.where(a2 <= a1, _loglerp(at(i), at(i + 1), t), 0.0)

        r = jax.lax.map(one_chunk, tuple(v.reshape(-1, chunk) for v in flat)).ravel()[:n]
        return (2.0 * 10.0 ** p["log10_pBH"] * r).reshape(shape)

    def intensity(self, params, grid):
        """On (log10 m1, log10 m2, z) grids: the kernel contraction on the mass axes at the halo
        nodes, then log-linear in z onto the grid's redshifts. Other grids take the
        point-wise path of the base class."""
        if grid.coords != M1M2:
            return super().intensity(params, grid)
        lm1, lm2, zg = (np.asarray(a) for a in grid.axes)
        L = self.rate_at_nodes(params, lm1, lm2)                                # (zn, m1, m2)
        zn = np.asarray(self.z_nodes)
        i = np.clip(np.searchsorted(zn, zg) - 1, 0, zn.size - 2)
        t = np.clip((zg - zn[i]) / (zn[i + 1] - zn[i]), 0.0, 1.0)
        Lz = _loglerp(L[i], L[i + 1], jnp.asarray(t)[:, None, None])
        z = jnp.asarray(zg)
        lam = jnp.moveaxis(Lz, 0, -1) * (self.cosmology.dVc_dz(z) / (1.0 + z))[None, None, :]
        lam = lam * self._inside(grid.coords, *grid.mesh())          # the whole support box, as rate_density does
        return jnp.where(grid.valid, lam, 0.0)


__all__ = ["EPS", "GIRELLI2020", "RV15_INACTIVE", "log10_mstar"]
