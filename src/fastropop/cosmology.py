r"""Cosmology as an explicit object.

Every model and observable in :mod:`fastropop` takes a :class:`Cosmology`; nothing reads
cosmological parameters from module globals. Two instances are provided:

* :data:`PLANCK18` -- the default for new code (Planck 2018 VI, A&A 641, A6: H0 = 67.4,
  Omega_m = 0.315, sigma_8 = 0.811, n_s = 0.965, Omega_b h^2 = 0.0224), with radiation
  fixed by matter-radiation equality at z = 3402;
* :data:`CONCORDANCE` -- h = 0.7, Omega_m = 0.3, no radiation: the cosmology of fastropop
  0.1, kept so its results stay reproducible.

Units: distances in Mpc, times in yr, masses in Msun, wavenumbers in Mpc^-1 (no h).

The universe is flat, Omega_Lambda = 1 - Omega_m - Omega_r. Background quantities are
tabulated once per instance on 0 <= z <= 60 (dz = 0.005; four-point Gauss-Legendre per
interval) and read off by cubic Hermite interpolation with their exact derivatives
(dD_C/dz = D_H/E, d age/dz = -dt/dz), accurate to ~1e-9 at any z, so they are cheap
inside ``jit``; beyond z = 60 they are not defined. (Linear interpolation of the same
table, as lisa_mbhbs used, is ~1e-3 low below z = 0.005, where the first interval's
mean slope is not the slope at z = 0.) Linear theory --
transfer function, sigma(M), growth -- is built on first use, since only halo-based
population models need it.

Instances are immutable and hash by identity, so they can be passed to ``jit`` as static
arguments or closed over.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .constants import C_KMS, MPC_M, RHO_CRIT_H2, YR_S

jax.config.update("jax_enable_x64", True)

Z_MAX = 60.0
_DZ = 0.005
DELTA_C = 1.686                     # critical linear overdensity for collapse (EdS value)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(4)


def _cumtrapz(y, dx):
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)])


def _cumulative_integral(f, x):
    """int_{x[0]}^{x[i]} f, four-point Gauss-Legendre on every interval (exact to round-off
    for the smooth background integrands here)."""
    mid, half = 0.5 * (x[1:] + x[:-1]), 0.5 * (x[1:] - x[:-1])
    parts = sum(w * f(mid + xi * half) for xi, w in zip(_GL_X, _GL_W)) * half
    return np.concatenate([[0.0], np.cumsum(parts)])


def _hermite(x, x0, dx, y, dy, xp=jnp):
    """Cubic Hermite interpolation on the uniform grid x0 + i dx, with exact derivatives dy."""
    u = (x - x0) / dx
    i = xp.clip(xp.floor(u), 0, y.shape[0] - 2).astype(int)
    t = u - i
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * y[i] + (t3 - 2 * t2 + t) * dx * dy[i]
            + (-2 * t3 + 3 * t2) * y[i + 1] + (t3 - t2) * dx * dy[i + 1])


def _tophat(x):
    """Fourier transform of a real-space top hat, 3 j1(x)/x."""
    return 3.0 * (np.sin(x) - x * np.cos(x)) / x**3


def _dtophat_dx(x):
    return 3.0 * ((x**2 - 3.0) * np.sin(x) + 3.0 * x * np.cos(x)) / x**4


@dataclass(frozen=True, eq=False)
class Cosmology:
    r"""A flat FLRW cosmology with matter, radiation and a cosmological constant.

    Parameters
    ----------
    h : float
        H0 / (100 km s^-1 Mpc^-1).
    Omega_m : float
        Matter density today (cold dark matter + baryons).
    Omega_r : float
        Radiation density today. 0 gives a matter + Lambda model.
    Omega_b : float
        Baryon density today; only the transfer function uses it.
    n_s : float
        Scalar spectral index.
    sigma8 : float or None
        Normalisation of the linear power spectrum, sigma(R = 8/h Mpc) at z = 0. ``None``
        normalises to COBE instead (Bunn & White 1997 delta_H for flat models), as the
        EPS implementation in ``crosschecks/code/ville/eps_rate.py`` does.
    T_cmb : float
        CMB temperature [K]; enters the transfer function.
    name : str
        Label for printing.
    """

    h: float
    Omega_m: float
    Omega_r: float = 0.0
    Omega_b: float = 0.0224 / 0.674**2
    n_s: float = 0.965
    sigma8: Optional[float] = 0.811
    T_cmb: float = 2.7255
    name: str = "custom"

    def __post_init__(self):
        if not (0.0 < self.Omega_m < 1.0 and 0.0 <= self.Omega_r < 1.0 - self.Omega_m):
            raise ValueError(f"need 0 < Omega_m < 1 and 0 <= Omega_r < 1 - Omega_m, got {self}")
        if not 0.0 <= self.Omega_b <= self.Omega_m:
            raise ValueError(f"need 0 <= Omega_b <= Omega_m, got {self}")
        z = np.linspace(0.0, Z_MAX, int(round(Z_MAX / _DZ)) + 1)
        dcdz = self.hubble_distance / self._E_np(z)                # dD_C/dz [Mpc]
        dtdz = 1.0 / ((1.0 + z) * self._H_np(z))                   # |dt/dz| [yr]
        dc = _cumulative_integral(lambda x: self.hubble_distance / self._E_np(x), z)
        cum = _cumulative_integral(lambda x: 1.0 / ((1.0 + x) * self._H_np(x)), z)
        age = cum[-1] - cum + self._age_beyond(Z_MAX)
        object.__setattr__(self, "_dz", z[1] - z[0])
        for k, v in (("_z", z), ("_dc", dc), ("_dcdz", dcdz), ("_age", age), ("_dagedz", -dtdz)):
            object.__setattr__(self, k + "_np", v)
            object.__setattr__(self, k, jnp.asarray(v))

    def __repr__(self):
        s8 = "COBE" if self.sigma8 is None else f"{self.sigma8:g}"
        return (f"Cosmology({self.name!r}: h={self.h:g}, Omega_m={self.Omega_m:g}, "
                f"Omega_r={self.Omega_r:.4g}, Omega_b={self.Omega_b:.4g}, n_s={self.n_s:g}, sigma8={s8})")

    # ------------------------------------------------------------------ background
    @property
    def Omega_L(self):
        return 1.0 - self.Omega_m - self.Omega_r

    @property
    def H0(self):
        """Hubble constant [1/yr]."""
        return 100.0 * self.h / (MPC_M / 1e3) * YR_S

    @property
    def hubble_distance(self):
        """c / H0 [Mpc]."""
        return C_KMS / (100.0 * self.h)

    def E(self, z):
        """H(z) / H0."""
        z = jnp.asarray(z, dtype=float)
        return jnp.sqrt(self.Omega_m * (1 + z) ** 3 + self.Omega_r * (1 + z) ** 4 + self.Omega_L)

    def H(self, z):
        """Hubble rate [1/yr]."""
        return self.H0 * self.E(z)

    def comoving_distance(self, z):
        """Line-of-sight comoving distance [Mpc]."""
        return _hermite(jnp.asarray(z, dtype=float), 0.0, self._dz, self._dc, self._dcdz)

    def luminosity_distance(self, z):
        """Luminosity distance [Mpc]."""
        z = jnp.asarray(z, dtype=float)
        return (1.0 + z) * self.comoving_distance(z)

    def dVc_dz(self, z):
        """Comoving volume per unit redshift, whole sky [Mpc^3]."""
        z = jnp.asarray(z, dtype=float)
        return 4.0 * jnp.pi * self.comoving_distance(z) ** 2 * self.hubble_distance / self.E(z)

    def dt_dz(self, z):
        """|dt/dz| of cosmic time [yr]."""
        z = jnp.asarray(z, dtype=float)
        return 1.0 / ((1.0 + z) * self.H(z))

    def age(self, z):
        """Age of the universe at redshift ``z`` [yr]."""
        return _hermite(jnp.asarray(z, dtype=float), 0.0, self._dz, self._age, self._dagedz)

    def lookback_time(self, z):
        """Cosmic time elapsed between redshift ``z`` and today [yr]."""
        return self._age[0] - self.age(z)

    def z_at_age(self, t):
        """Redshift at which the universe has age ``t`` [yr]: inverse linear interpolation,
        then two Newton steps on :meth:`age` with its exact derivative."""
        t = jnp.asarray(t, dtype=float)
        z = jnp.interp(t, self._age[::-1], self._z[::-1])
        for _ in range(2):
            z = jnp.clip(z + (self.age(z) - t) / self.dt_dz(z), 0.0, Z_MAX)
        return z

    # NumPy twins for SciPy integrands, which call with scalars thousands of times
    def E_np(self, z):
        return self._E_np(np.asarray(z, dtype=float))

    def comoving_distance_np(self, z):
        return _hermite(np.asarray(z, dtype=float), 0.0, self._dz, self._dc_np, self._dcdz_np, xp=np)

    def dVc_dz_np(self, z):
        return 4.0 * np.pi * self.comoving_distance_np(z) ** 2 * self.hubble_distance / self.E_np(z)

    def dt_dz_np(self, z):
        return 1.0 / ((1.0 + z) * self._H_np(z))

    def _E_np(self, z):
        return np.sqrt(self.Omega_m * (1 + z) ** 3 + self.Omega_r * (1 + z) ** 4 + self.Omega_L)

    def _H_np(self, z):
        return self.H0 * self._E_np(z)

    def _age_beyond(self, z):
        """Cosmic time elapsed before redshift ``z`` [yr]: int_z^inf dz' / ((1+z') H(z'))."""
        from scipy.integrate import quad
        return quad(lambda x: 1.0 / ((1.0 + x) * self._H_np(x)), z, np.inf,
                    epsabs=0.0, epsrel=1e-12, limit=500)[0]

    # ------------------------------------------------------------------ linear theory
    @property
    def rho_m0(self):
        """Mean comoving matter density [Msun Mpc^-3]."""
        return self.Omega_m * RHO_CRIT_H2 * self.h**2

    def mass_to_radius(self, M):
        """Comoving top-hat radius enclosing mass ``M`` [Msun] at the mean density [Mpc]."""
        return (3.0 * jnp.asarray(M, dtype=float) / (4.0 * jnp.pi * self.rho_m0)) ** (1.0 / 3.0)

    def transfer(self, k):
        r"""Eisenstein & Hu (1998) transfer function with baryon oscillations, ``k`` in Mpc^-1.

        Their fitting formulae, with the fitted sound horizon. The same transcription
        as ``crosschecks/code/ville/eps_rate.py``, except the exponent of b2 in beta_c:
        -0.0266 as in the paper, where that code has -0.026.
        """
        k = np.asarray(k, dtype=float)
        h, Om, Ob = self.h, self.Omega_m, self.Omega_b
        Oc, om, ob, th = Om - Ob, Om * h**2, Ob * h**2, self.T_cmb / 2.7
        keq = 7.46e-2 * om * th**-2
        ksilk = 1.6 * ob**0.52 * om**0.73 * (1 + (10.4 * om) ** -0.95)
        s = 44.5 * np.log(9.83 / om) / np.sqrt(1 + 10 * ob**0.75)
        zeq = 2.5e4 * om * th**-4
        b1d = 0.313 * om**-0.419 * (1 + 0.607 * om**0.674)
        b2d = 0.238 * om**0.223
        zd = 1291 * om**0.251 / (1 + 0.659 * om**0.828) * (1 + b1d * ob**b2d)
        Rd = 31.5 * ob * th**-4 * (zd / 1e3) ** -1
        a1 = (46.9 * om) ** 0.670 * (1 + (32.1 * om) ** -0.532)
        a2 = (12.0 * om) ** 0.424 * (1 + (45.0 * om) ** -0.582)
        alpha_c = a1 ** (-Ob / Om) * a2 ** (-((Ob / Om) ** 3))
        bb1 = 0.944 / (1 + (458 * om) ** -0.708)
        bb2 = (0.395 * om) ** -0.0266
        beta_c = 1.0 / (1 + bb1 * ((Oc / Om) ** bb2 - 1))
        q = k / (13.41 * keq)

        def T0(al, be):
            L = np.log(np.e + 1.8 * be * q)
            return L / (L + (14.2 / al + 386.0 / (1 + 69.9 * q**1.08)) * q**2)

        f = 1.0 / (1 + (k * s / 5.4) ** 4)
        Tc = f * T0(1.0, beta_c) + (1 - f) * T0(alpha_c, beta_c)
        y = (1 + zeq) / (1 + zd)
        G = y * (-6 * np.sqrt(1 + y) + (2 + 3 * y) * np.log((np.sqrt(1 + y) + 1) / (np.sqrt(1 + y) - 1)))
        alpha_b = 2.07 * keq * s * (1 + Rd) ** -0.75 * G
        beta_b = 0.5 + Ob / Om + (3 - 2 * Ob / Om) * np.sqrt(1 + (17.2 * om) ** 2)
        beta_node = 8.41 * om**0.435
        s_tilde = s / (1 + (beta_node / (k * s)) ** 3) ** (1.0 / 3.0)
        Tb = (T0(1.0, 1.0) / (1 + (k * s / 5.2) ** 2)
              + alpha_b / (1 + (beta_b / (k * s)) ** 3) * np.exp(-((k / ksilk) ** 1.4))) * np.sinc(k * s_tilde / np.pi)
        return Ob / Om * Tb + Oc / Om * Tc

    def _delta2_shape(self, k):
        """Dimensionless power spectrum up to its normalisation: (k D_H)^(3+n_s) T(k)^2."""
        return (k * self.hubble_distance) ** (3.0 + self.n_s) * self.transfer(k) ** 2

    @cached_property
    def _k(self):
        return np.geomspace(1e-6, 1e5, 8193)                         # Mpc^-1

    def _sigma2_unnormalised(self, R):
        R = np.atleast_1d(np.asarray(R, dtype=float))
        lnk = np.log(self._k)
        d2 = self._delta2_shape(self._k)
        return np.array([np.trapezoid(d2 * _tophat(self._k * r) ** 2, lnk) for r in R])

    @cached_property
    def _amplitude(self):
        """Multiplies _delta2_shape to give Delta^2(k) at z = 0."""
        if self.sigma8 is None:                                     # COBE (Bunn & White 1997)
            n1, Om = self.n_s - 1.0, self.Omega_m
            dH = 1.94e-5 * Om ** (-0.785 - 0.05 * np.log(Om)) * np.exp(-0.95 * n1 - 0.169 * n1**2)
            return dH**2
        return self.sigma8**2 / float(self._sigma2_unnormalised(8.0 / self.h)[0])

    def delta2(self, k):
        """Dimensionless linear power spectrum Delta^2(k) = k^3 P(k) / 2 pi^2 at z = 0."""
        return self._amplitude * self._delta2_shape(np.asarray(k, dtype=float))

    def sigma_R(self, R):
        """rms linear overdensity in a top hat of comoving radius ``R`` [Mpc] at z = 0 (NumPy)."""
        return np.sqrt(self._amplitude * self._sigma2_unnormalised(R))

    @property
    def sigma8_value(self):
        """sigma(8/h Mpc): the input sigma8, or the one implied by the COBE normalisation."""
        return float(self.sigma_R(8.0 / self.h)[0])

    @cached_property
    def _sigma_table(self):
        """ln M, ln sigma(M) and d ln sigma / d ln M on 1 <= M <= 1e18 Msun (step 0.01 dex)."""
        lM = np.arange(0.0, 18.0 + 1e-9, 0.01) * np.log(10.0)
        R = (3.0 * np.exp(lM) / (4.0 * np.pi * self.rho_m0)) ** (1.0 / 3.0)
        lnk = np.log(self._k)
        d2 = self._amplitude * self._delta2_shape(self._k)
        s2, ds2 = np.empty(R.size), np.empty(R.size)
        for i, r in enumerate(R):
            x = self._k * r
            W = _tophat(x)
            s2[i] = np.trapezoid(d2 * W**2, lnk)
            ds2[i] = np.trapezoid(d2 * 2 * W * _dtophat_dx(x) * x, lnk)       # d sigma^2 / d ln R
        dlns_dlnM = ds2 / (2.0 * s2) / 3.0
        return lM, 0.5 * np.log(s2), dlns_dlnM

    def sigma_M(self, M):
        """rms linear overdensity on mass scale ``M`` [Msun] at z = 0."""
        lM, lns, _ = self._sigma_table
        return jnp.exp(jnp.interp(jnp.log(jnp.asarray(M, dtype=float)), jnp.asarray(lM), jnp.asarray(lns)))

    def dlnsigma_dlnM(self, M):
        """d ln sigma / d ln M at mass ``M`` [Msun]."""
        lM, _, d = self._sigma_table
        return jnp.interp(jnp.log(jnp.asarray(M, dtype=float)), jnp.asarray(lM), jnp.asarray(d))

    @cached_property
    def _growth_table(self):
        r"""Linear growth factor from the Heath (1977) integral,
        D(a) \propto E(a) \int_0^a da' / (a' E(a'))^3, normalised to D(z=0) = 1.

        Exact for pressureless matter + Lambda; the radiation term in E is kept for
        consistency with the background, and its effect on the growth integral is below
        Omega_r (1+z) / Omega_m (1% at z = 30 for Planck 2018).
        """
        a = np.geomspace(1e-8, 1.0, 40001)
        aE = np.sqrt(self.Omega_m / a + self.Omega_r / a**2 + self.Omega_L * a**2)
        I = _cumtrapz(a * aE**-3, np.log(a[1]) - np.log(a[0]))     # integrand in d ln a
        D = aE / a * I
        z = 1.0 / a - 1.0
        keep = z <= Z_MAX + 1.0
        return z[keep][::-1], (D / D[-1])[keep][::-1]

    def growth(self, z):
        """Linear growth factor D(z), D(0) = 1."""
        zz, D = self._growth_table
        return jnp.interp(jnp.asarray(z, dtype=float), jnp.asarray(zz), jnp.asarray(D))

    def delta_c(self, z):
        """Critical linear overdensity for collapse at redshift ``z``, extrapolated to today."""
        return DELTA_C / self.growth(z)


PLANCK18 = Cosmology(h=0.674, Omega_m=0.315, Omega_r=0.315 / 3403.0, Omega_b=0.0224 / 0.674**2,
                     n_s=0.965, sigma8=0.811, T_cmb=2.7255, name="Planck 2018")
CONCORDANCE = Cosmology(h=0.7, Omega_m=0.3, Omega_r=0.0, name="concordance (fastropop 0.1)")

__all__ = ["CONCORDANCE", "Cosmology", "DELTA_C", "PLANCK18", "Z_MAX"]
