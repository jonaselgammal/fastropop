r"""Phenomenological merger-rate densities: the separable ansatz and its variants.

    R(Mc, q, z) = n0 f_M(Mc) f_z(z) p(q)      [Mpc^-3 yr^-1 per dex Mc, per unit q, per unit z]

The baseline is the form of Sesana, Vecchio & Colacino (2008), with a power-law
mass-ratio distribution:

* mass ``"cutoff"``: f_M = (Mc / 1e7 Msun)^(-alpha_M) exp(-Mc / Mc_*);
* mass ``"bimodal"``: the cutoff law plus a lognormal light component in log10 Mc. Both
  are normalised, so ``w_light`` is the fraction of mergers in the light component;
* redshift ``"powerlaw"``: f_z = (1+z)^beta_z exp(-z / z0);
* redshift ``"delay"``: the Madau & Dickinson (2014) star-formation rate convolved with
  a power-law delay-time distribution p(t_d) ~ t_d^(-alpha_d) above t_min, with a
  smooth lower cutoff. (beta_z, z0) are replaced by (alpha_d, log10_tmin);
* p(q) = q^beta_q / N_q on [q_min, 1], and zero outside.

The defaults are the fiducial point of the LISA study: calibrated by weighted maximum
likelihood to the Q3nod_K16 catalogue (preset ``"fiducial"``).
"""

from __future__ import annotations

import jax.numpy as jnp

from ..cosmology import PLANCK18
from ..grid import MCQ
from .base import PopulationModel

FIDUCIAL = dict(log10_n0=-11.0855, alpha_M=-0.5161, log10_Mstar=5.3717, beta_z=3.5693, z0=3.7706,
                beta_q=-0.3603)
_EXTRA_DEFAULTS = dict(w_light=0.5, log10_Mlight=3.0, sigma_light=0.4, alpha_d=1.0, log10_tmin=-1.0)
LABELS = {
    "log10_n0": r"$\log_{10} n_0$", "alpha_M": r"$\alpha_M$", "log10_Mstar": r"$\log_{10}\mathcal{M}_*$",
    "beta_z": r"$\beta_z$", "z0": r"$z_0$", "beta_q": r"$\beta_q$",
    "w_light": r"$w_{\rm light}$", "log10_Mlight": r"$\log_{10}\mathcal{M}_{\rm L}$",
    "sigma_light": r"$\sigma_{\rm L}$", "alpha_d": r"$\alpha_d$",
    "log10_tmin": r"$\log_{10}(t_{\min}/\mathrm{Gyr})$",
}

_MASS_NORM_GRID = jnp.linspace(2.0, 10.5, 400)        # log10 Mc, normalises the bimodal components
_TD_GRID = jnp.geomspace(1e-3, 13.0, 120) * 1e9       # delay times [yr], 1 Myr to 13 Gyr
_CUT_W = 0.15                                          # width of the t_min cutoff in ln t_d


def q_norm(beta_q, q_min=0.01):
    """Normalisation of p(q) ~ q^beta_q on [q_min, 1], with the beta_q -> -1 limit."""
    b = beta_q + 1.0
    safe = jnp.where(jnp.abs(b) < 1e-8, 1.0, b)
    return jnp.where(jnp.abs(b) < 1e-8, -jnp.log(q_min), (1.0 - q_min**safe) / safe)


def madau_dickinson(z):
    """Cosmic star-formation rate density [Msun yr^-1 Mpc^-3], Madau & Dickinson (2014)."""
    return 0.015 * (1.0 + z) ** 2.7 / (1.0 + ((1.0 + z) / 2.9) ** 5.6)


class Phenomenological(PopulationModel):
    """Separable phenomenological family (see the module docstring).

    Parameters
    ----------
    cosmology : fastropop.cosmology.Cosmology
    mass : {"cutoff", "bimodal"}
    redshift : {"powerlaw", "delay"}
    q_min : float
        Lower edge of the mass-ratio distribution.
    support : dict, optional
        See :class:`~fastropop.populations.base.PopulationModel`.
    """

    native_coords = MCQ

    def __init__(self, cosmology=PLANCK18, mass="cutoff", redshift="powerlaw", q_min=0.01, support=None):
        super().__init__(cosmology, support)
        if mass not in ("cutoff", "bimodal"):
            raise ValueError(f"mass must be 'cutoff' or 'bimodal', got {mass!r}")
        if redshift not in ("powerlaw", "delay"):
            raise ValueError(f"redshift must be 'powerlaw' or 'delay', got {redshift!r}")
        self.mass, self.redshift, self.q_min = mass, redshift, q_min
        z_names = ("beta_z", "z0") if redshift == "powerlaw" else ("alpha_d", "log10_tmin")
        extra = ("w_light", "log10_Mlight", "sigma_light") if mass == "bimodal" else ()
        self.param_names = ("log10_n0", "alpha_M", "log10_Mstar") + z_names + ("beta_q",) + extra
        d = {**FIDUCIAL, **_EXTRA_DEFAULTS}
        if redshift == "delay":
            d["log10_n0"] = -11.0
        self.defaults = {k: d[k] for k in self.param_names}
        self.labels = {k: LABELS[k] for k in self.param_names}

    def comoving_rate(self, params, log10_Mc, q, z):
        p = params
        Mc = 10.0 ** jnp.asarray(log10_Mc, dtype=float)
        q = jnp.asarray(q, dtype=float)
        z = jnp.asarray(z, dtype=float)
        if self.mass == "cutoff":
            f_m = self._heavy(log10_Mc, p)
        else:
            g = _MASS_NORM_GRID
            light = lambda lm: jnp.exp(-0.5 * ((lm - p["log10_Mlight"]) / p["sigma_light"]) ** 2)
            f_m = ((1.0 - p["w_light"]) * self._heavy(log10_Mc, p) / jnp.trapezoid(self._heavy(g, p), g)
                   + p["w_light"] * light(log10_Mc) / jnp.trapezoid(light(g), g))
        if self.redshift == "powerlaw":
            f_z = (1.0 + z) ** p["beta_z"] * jnp.exp(-z / p["z0"])
        else:
            f_z = self._delayed_formation(z, p["alpha_d"], p["log10_tmin"])
        in_q = (q >= self.q_min) & (q <= 1.0)
        p_q = jnp.where(in_q, q ** p["beta_q"] / q_norm(p["beta_q"], self.q_min), 0.0)
        return 10.0 ** p["log10_n0"] * f_m * f_z * p_q

    @staticmethod
    def _heavy(log10_Mc, p):
        Mc = 10.0 ** jnp.asarray(log10_Mc, dtype=float)
        return (Mc / 1e7) ** (-p["alpha_M"]) * jnp.exp(-Mc / 10.0 ** p["log10_Mstar"])

    def _delayed_formation(self, z, alpha_d, log10_tmin):
        """Formation rate convolved with the delay-time distribution (un-normalised; n0 sets
        the scale). No formation before z = 60, the edge of the cosmology tables."""
        c = self.cosmology
        td = _TD_GRID
        tmin = 10.0**log10_tmin * 1e9
        # a hard cutoff makes d(lambda)/d(log10 tmin) vanish on a fixed grid between nodes;
        # a smooth one, narrow compared with the grid range but wide compared with its
        # spacing (~0.08 in ln t_d), keeps the derivative
        wt = td ** (-alpha_d) / (1.0 + jnp.exp(-(jnp.log(td) - jnp.log(tmin)) / _CUT_W))
        wt = wt / jnp.trapezoid(wt, td)
        t_f = c.age(z)[..., None] - td
        t60 = c.age(jnp.array(60.0))
        z_f = c.z_at_age(jnp.clip(t_f, t60, None))
        psi = jnp.where(t_f > t60, madau_dickinson(z_f), 0.0)
        return jnp.trapezoid(psi * wt, td, axis=-1)


__all__ = ["FIDUCIAL", "LABELS", "Phenomenological", "madau_dickinson", "q_norm"]
