r"""Massive-black-hole coalescences from the holodeck semi-analytic model ("classic Phenom" of
the NANOGrav 15-yr analysis, Agazie et al. 2023; holodeck v1.6, ``sams`` and ``host_relations``).

Galaxies merge at a rate set by the stellar mass function and the pair fraction over the merger
time. Each galaxy hosts a black hole drawn from a log-normal M_BH-M_bulge relation, and the black
holes coalesce a fixed time after their galaxies merge:

* stellar mass function (Schechter), per dex of M*:
      Phi(M*, z) = ln10 Phi_0(z) (M*/M_psi(z))^(1+alpha(z)) exp(-M*/M_psi(z)),
      log10 Phi_0 = psi_0 + psi_z z,  log10 M_psi = m_psi0 + m_psi_z z,  alpha = alpha_0 + alpha_z z;
* pair fraction per unit stellar mass ratio q* = M*2/M*1 <= 1, of the primary mass:
      P(M*1, q*, z) = f_0 (M*1/M_ref)^a_P (1+z)^b_P q*^g_P   (at most f_max),
  normalised so that its integral over 0.25 <= q* <= 1 is ``gpf_frac_norm_allq``;
* galaxy merger time:  T(M*1, q*, z) = T_0 (M*1/M_ref,T)^a_T (1+z)^b_T q*^g_T;
* black-hole mass: log10 M_BH ~ N(mu + beta_M log10(f_b M* / 1e11 Msun), eps_mu);
* coalescence: a pair seen at redshift z_i (cosmic time t_i) has its black holes coalesce at
      t_c = t_i + T(M*1, q*, z_i) + tau_f,
  and pairs that would coalesce after today never do.

The comoving coalescence rate per dex of both stellar masses, at the coalescence time, is

      n(M*1, M*2; t_c) = Phi(M*1, z_i) P(M*1, q*, z_i) / T(M*1, q*, z_i) * q* ln10 * dt_i/dt_c,

with dt_c/dt_i = 1 + dT/dt_i. This is holodeck's ``rate_chirps`` (static binary density at
z_i, moved to the coalescence redshift) written per unit coalescence redshift. The black-hole
rate follows by convolving n with the log-normal for each galaxy (HostKernelModel).

Differences from holodeck: the scatter enters as an exact convolution over stellar mass
instead of holodeck's numerical redistribution between mass bins, and the merger time is that
of the galaxy pair rather than of the mean stellar masses of a black-hole bin. Both agree when
the scatter is zero.

Free parameters (those of the NANOGrav Phenom library that change coalescence rates):
psi_0 (``gsmf_phi0_log10``), m_psi0 (``gsmf_mchar0_log10``), mu (``mmb_mamp_log10``),
eps_mu (``mmb_scatter_dex``), tau_f (``hard_time``, Gyr). The inner hardening index only
shapes the evolution through the PTA band and does not enter. Everything else is fixed at the
library's values (:data:`NG15_PHENOM_FIXED`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..cosmology import PLANCK18, Z_MAX
from .host_kernel import HostKernelModel

_LN10 = float(np.log(10.0))

LABELS = {"gsmf_phi0_log10": r"$\psi_0$", "gsmf_mchar0_log10": r"$m_{\psi,0}$", "mmb_mamp_log10": r"$\mu$",
          "mmb_scatter_dex": r"$\epsilon_\mu$", "hard_time": r"$\tau_f$ [Gyr]"}
# holodeck v1.6 librarian/param_spaces_classic.py, _PS_Classic_Phenom.DEFAULTS
HOLODECK_DEFAULTS = dict(gsmf_phi0_log10=-2.77, gsmf_mchar0_log10=11.24, mmb_mamp_log10=8.69,
                         mmb_scatter_dex=0.3, hard_time=3.0)
NG15_PHENOM_FIXED = dict(
    gsmf_phiz=-0.6, gsmf_mcharz=0.11, gsmf_alpha0=-1.21, gsmf_alphaz=-0.03,
    gpf_frac_norm_allq=0.025, gpf_qlo=0.25, gpf_malpha=0.0, gpf_qgamma=0.0, gpf_zbeta=1.0, gpf_max_frac=1.0,
    gpf_mref_log10=11.0,
    gmt_norm=0.5, gmt_malpha=0.0, gmt_qgamma=-1.0, gmt_zbeta=-0.5, gmt_mref0_log10=11.0,   # gmt_norm in Gyr
    mmb_plaw=1.10, mmb_mref_log10=11.0, bulge_frac=0.615,
)


class Holodeck(HostKernelModel):
    """holodeck's classic Phenom semi-analytic model as a population (module docstring).

    Parameters
    ----------
    cosmology : fastropop.cosmology.Cosmology
    z_nodes : array, optional
        Coalescence redshifts at which the galaxy-pair table is built (default 0.05 ... 25, 100).
    log10_Mstar : tuple, optional
        (min, max, step) of the stellar-mass cells in log10 Msun (default 4 ... 13, 0.1).
    fixed : dict, optional
        Overrides of :data:`NG15_PHENOM_FIXED`.
    """

    param_names = ("gsmf_phi0_log10", "gsmf_mchar0_log10", "mmb_mamp_log10", "mmb_scatter_dex", "hard_time")
    defaults = dict(HOLODECK_DEFAULTS)
    labels = dict(LABELS)

    def __init__(self, cosmology=PLANCK18, z_nodes=None, log10_Mstar=(4.0, 13.0, 0.1), fixed=None, support=None,
                 n_bisect=48):
        z_nodes = np.linspace(0.05, 25.0, 100) if z_nodes is None else np.asarray(z_nodes, dtype=float)
        sup = {"z": (float(z_nodes[0]), float(z_nodes[-1]))}
        sup.update(support or {})
        super().__init__(cosmology, sup)
        self.fixed = {**NG15_PHENOM_FIXED, **(fixed or {})}
        if self.fixed["gmt_zbeta"] > 0:
            raise ValueError("the coalescence-time solver assumes the merger time does not grow with z "
                             "(gmt_zbeta <= 0)")
        self.z_nodes = z_nodes
        lo, hi, step = log10_Mstar
        edges = np.arange(lo, hi + step / 2, step)
        self.log10_Mstar = 0.5 * (edges[1:] + edges[:-1])                     # cell centres
        self._dh = float(step)
        self._xe = jnp.asarray(np.broadcast_to(edges, (z_nodes.size, edges.size)))
        self._zn = jnp.asarray(z_nodes)
        self._n_bisect = int(n_bisect)
        f = self.fixed
        nh = self.log10_Mstar.size
        # stellar mass ratio of cell pairs (i >= j): q* = 10^-(i-j) step
        self._k = np.subtract.outer(np.arange(nh), np.arange(nh))             # i - j
        self._qk = 10.0 ** (-step * np.arange(nh))
        pw = f["gpf_qgamma"] + 1.0
        self._gpf_norm = f["gpf_frac_norm_allq"] / ((1.0 - f["gpf_qlo"] ** pw) / pw)
        self._gmt_mref = 10.0 ** f["gmt_mref0_log10"] * 0.4 / cosmology.h

    # ------------------------------------------------------------------ galaxies
    def gsmf(self, p, log10_Mstar, z):
        """Phi = dn/dlog10 M* [Mpc^-3]."""
        f = self.fixed
        phi0 = 10.0 ** (p["gsmf_phi0_log10"] + f["gsmf_phiz"] * z)
        x = 10.0 ** (log10_Mstar - (p["gsmf_mchar0_log10"] + f["gsmf_mcharz"] * z))
        alpha = f["gsmf_alpha0"] + f["gsmf_alphaz"] * z
        return _LN10 * phi0 * x ** (1.0 + alpha) * jnp.exp(-x)

    def pair_fraction(self, log10_Mstar1, q, z):
        """Pair fraction per unit q* [dimensionless]."""
        f = self.fixed
        P = (self._gpf_norm * (10.0 ** (log10_Mstar1 - f["gpf_mref_log10"])) ** f["gpf_malpha"]
             * (1.0 + z) ** f["gpf_zbeta"] * q ** f["gpf_qgamma"])
        return jnp.minimum(P, f["gpf_max_frac"])

    def merger_time(self, log10_Mstar1, q, z):
        """Galaxy merger time [yr]."""
        f = self.fixed
        return (f["gmt_norm"] * 1e9 * (10.0 ** log10_Mstar1 / self._gmt_mref) ** f["gmt_malpha"]
                * (1.0 + z) ** f["gmt_zbeta"] * q ** f["gmt_qgamma"])

    def pair_redshift(self, p, log10_Mstar1, q, z_c):
        """Redshift z_i of the galaxy pair whose black holes coalesce at z_c, and whether it
        exists (coalescence after z = Z_MAX formation). Bisection in log(1+z): the delay
        T(z_i) + tau_f grows towards low z_i, so age(z_i) + delay is monotonic."""
        c = self.cosmology
        t_c = c.age(z_c)
        tau = p["hard_time"] * 1e9
        g = lambda z: c.age(z) + self.merger_time(log10_Mstar1, q, z) + tau - t_c       # decreasing in z
        lo = jnp.log1p(jnp.broadcast_to(z_c, jnp.broadcast_shapes(jnp.shape(z_c), jnp.shape(q))))
        hi = jnp.full_like(lo, np.log1p(Z_MAX))
        ok = g(jnp.expm1(hi)) <= 0.0

        def step(_, b):
            lo_, hi_ = b
            mid = 0.5 * (lo_ + hi_)
            pos = g(jnp.expm1(mid)) > 0.0                   # still too late: the pair formed earlier
            return jnp.where(pos, mid, lo_), jnp.where(pos, hi_, mid)

        lo, hi = jax.lax.fori_loop(0, self._n_bisect, step, (lo, hi))
        return jnp.expm1(0.5 * (lo + hi)), ok

    def _pair_rate_by_ratio(self, p):
        """Coalescence rate n per dex of both stellar masses, for every primary cell i and
        mass-ratio index k = i - j, at the redshift nodes: shape (z, i, k) [Mpc^-3 yr^-1]."""
        c, f = self.cosmology, self.fixed
        x1 = jnp.asarray(self.log10_Mstar)[None, :, None]                        # (1, i, 1)
        q = jnp.asarray(self._qk)[None, None, :]                                   # (1, 1, k)
        zc = self._zn[:, None, None]                                               # (z, 1, 1)
        if f["gmt_malpha"] == 0.0:                    # delay independent of mass: solve on (z, k) only
            zi, ok = self.pair_redshift(p, 11.0, q[:, 0, :], zc[:, :, 0])           # (z, k)
            zi, ok = zi[:, None, :], ok[:, None, :]
        else:
            zi, ok = self.pair_redshift(p, x1, q, zc)
        T = self.merger_time(x1, q, zi)
        dTdt = -f["gmt_zbeta"] * T * c.H(zi)                                       # dT/dt_i (T ~ (1+z)^b)
        n = (self.gsmf(p, x1, zi) * self.pair_fraction(x1, q, zi) / T * q * _LN10 / (1.0 + dTdt))
        return jnp.where(ok, n, 0.0)

    # ------------------------------------------------------------------ HostKernelModel hooks
    def _means(self, p, xe, z):
        f = self.fixed
        return p["mmb_mamp_log10"] + f["mmb_plaw"] * (xe + np.log10(f["bulge_frac"]) - f["mmb_mref_log10"])

    def _sigma(self, p):
        return p["mmb_scatter_dex"]

    def _table(self, p):
        """Symmetric extension of n over the stellar-mass square, half on each side, so that
        every merger is counted once (HostKernelModel convention)."""
        nk = self._pair_rate_by_ratio(p)                                           # (z, i, k)
        k = jnp.asarray(np.abs(self._k))
        hi = jnp.asarray(np.maximum.outer(np.arange(self._k.shape[0]), np.arange(self._k.shape[0])))
        return 0.5 * nk[:, hi, k]


__all__ = ["Holodeck", "HOLODECK_DEFAULTS", "NG15_PHENOM_FIXED"]
