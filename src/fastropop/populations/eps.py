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
halo table. Other redshifts are reached by interpolating log rate linearly in z. On a grid
the mass points are the same at every redshift, so the contraction runs once per halo
node for the grid's mass points: a tensor product on (log10 m1, log10 m2) grids, the
(m1, m2) pairs of the mesh on other grids. Arbitrary points take the point-wise path.

Halo-mass quadrature: cells of width ``step`` in log10 M_h. The halo merger rate (smooth in
M_h) is taken at cell centres. The black-hole lognormal is integrated *exactly* over
each cell, with its mean linear between the cell edges (a difference of error functions).
Accuracy therefore does not depend on how the kernel's width, projected onto halo mass
(sigma / (b dlog M*/dlog M_h), down to ~0.04 dex), compares with the cell size; a pointwise
trapezoid on the same cells would. What remains is the midpoint rule for the halo merger
rate, which is steep at high mass and high z. Against a converged 0.01-dex reference, at
rho_th = 100 with ~100 events, the exact log-likelihood varies across the posterior
region by 0.036 (std; max 0.065) at the default 0.1-dex cells, 0.009 at 0.05 and 0.15 at 0.2.
Per-cell rates in rare corners can be off by more; they carry negligible rate.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..cosmology import PLANCK18
from .host_kernel import HostKernelModel, _cell_kernel, _loglerp     # noqa: F401  (re-exported)

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


class EPS(HostKernelModel):
    """Extended Press-Schechter black-hole mergers (see the module docstring).

    Parameters
    ----------
    cosmology : fastropop.cosmology.Cosmology
        Its linear theory sets the halo merger rate.
    z_nodes : array, optional
        Redshifts at which the halo tables are built (default 0.05 ... 25 in 100 steps).
    log10_Mh : tuple, optional
        (min, max, step) of the halo-mass cells in log10 Msun (default 6 ... 17, 0.1). Broad
        kernels (large sigma, small b) reach halos below 1e7 Msun, hence the low edge.
    shmr : dict, optional
        Stellar-to-halo parameters (default :data:`GIRELLI2020`).
    """

    param_names = ("log10_pBH", "a", "b", "sigma", "gamma")
    defaults = dict(RV15_INACTIVE)
    labels = dict(LABELS)

    def __init__(self, cosmology=PLANCK18, z_nodes=None, log10_Mh=(6.0, 17.0, 0.1), shmr=None, support=None):
        z_nodes = np.linspace(0.05, 25.0, 100) if z_nodes is None else np.asarray(z_nodes, dtype=float)
        sup = {"z": (float(z_nodes[0]), float(z_nodes[-1]))}
        sup.update(support or {})
        super().__init__(cosmology, sup)
        self.shmr = dict(GIRELLI2020 if shmr is None else shmr)
        self.z_nodes = z_nodes
        lo, hi, step = log10_Mh
        edges = np.arange(lo, hi + step / 2, step)
        self.log10_Mh = 0.5 * (edges[1:] + edges[:-1])                          # cell centres
        self._dh = float(step)
        R = self.halo_merger_rate(self.log10_Mh, z_nodes)                      # (z, h, h) at centres
        # rate per dex^2 of BH mass = ln10^2 sum_cells R(c1, c2) Kbar1 Kbar2, with Kbar the
        # cell-integrated lognormal (per dex of m; see _cell_kernel and the module docstring)
        self._A = jnp.asarray(_LN10**2 * R)
        self._xe = jnp.asarray(np.stack([log10_mstar(edges, z, self.shmr) for z in z_nodes]))   # (z, h+1)
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
    def _means(self, p, xe, z):
        """Mean log10 m at the halo-cell edges."""
        return p["a"] + p["b"] * (xe - 11.0) + p["gamma"] * jnp.log10(1.0 + z)[..., None]

    def _sigma(self, p):
        return p["sigma"]

    def _table(self, p):
        return self._A

    def _scale(self, p):
        return 10.0 ** p["log10_pBH"]


__all__ = ["EPS", "GIRELLI2020", "RV15_INACTIVE", "log10_mstar"]
