"""Cosmology-specific helper functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from scipy.special import hyp2f1

from .constants import H0s, OmegaDM, Omegak, OmegaLambda, cMKS

jax.config.update("jax_enable_x64", True)


@jit
def EE(z):
    r"""
    Compute the dimensionless Hubble parameter \(E(z)\).

    The cosmology used by `fastropop` is a fixed flat
    \(\Lambda\)CDM background with
    \(\Omega_{\rm m} = 0.3\), \(\Omega_\Lambda = 0.7\),
    \(\Omega_k = 0\), and \(h = 0.7\). The corresponding expansion
    function is

    \[
    E(z) = \sqrt{\Omega_{\rm m}(1+z)^3 + \Omega_k (1+z)^2 + \Omega_\Lambda}.
    \]

    Parameters
    ----------
    z : float or jax.Array
        Redshift.

    Returns
    -------
    float or jax.Array
        Dimensionless Hubble parameter \(E(z)\).
    """
    return jnp.sqrt(OmegaDM * (1.0 + z) ** 3.0 + Omegak * (1.0 + z) ** 2.0 + OmegaLambda)


@jit
def dtodz(z):
    r"""
    Compute \(dt_r/dz\), the derivative of cosmic time with redshift.

    This helper is used throughout the semi-analytic model when converting
    merger-rate densities written in redshift into time-based quantities:

    \[
    \frac{dt_r}{dz} = \frac{1}{H_0 (1+z) E(z)}.
    \]

    Parameters
    ----------
    z : float or jax.Array
        Redshift.

    Returns
    -------
    float or jax.Array
        Derivative \(dt_r/dz\) in seconds.
    """
    return 1.0 / (H0s * (1.0 + z) * EE(z))


def Dca(z):
    r"""
    Compute the comoving distance with the analytic hypergeometric expression.

    This function is mainly used once at import time to tabulate the comoving
    distance on a dense grid. Runtime code generally calls `Dc_interp`
    instead of evaluating the hypergeometric expression repeatedly.
    """
    term1 = hyp2f1(1.0 / 3.0, 1.0 / 2.0, 4.0 / 3.0, -(OmegaDM / OmegaLambda))
    term2 = hyp2f1(
        1.0 / 3.0,
        1.0 / 2.0,
        4.0 / 3.0,
        -((1.0 + z) ** 3.0 * OmegaDM / OmegaLambda),
    )
    return (cMKS / H0s) * (1.0 / jnp.sqrt(OmegaLambda)) * (-term1 + (1.0 + z) * term2)


_Z_VALUES = np.arange(0.0, 10.001, 0.0001)
_DC_VALUES = np.array([Dca(z) for z in _Z_VALUES])
_Z_VALUES_JAX = jnp.array(_Z_VALUES)
_DC_VALUES_JAX = jnp.array(_DC_VALUES)


@jit
def Dc_interp(z):
    r"""
    Interpolate the comoving distance on a precomputed redshift grid.

    Parameters
    ----------
    z : float or jax.Array
        Redshift.

    Returns
    -------
    float or jax.Array
        Line-of-sight comoving distance in metres.
    """
    return jnp.interp(z, _Z_VALUES_JAX, _DC_VALUES_JAX)


def Dc_interp_numpy(z):
    """NumPy interpolation helper for scalar SciPy integration paths."""
    return np.interp(z, _Z_VALUES, _DC_VALUES)


@jit
def dVcdz(z):
    r"""
    Compute the comoving volume element \(dV_c/dz\).

    The quantity returned is

    \[
    \frac{dV_c}{dz} =
    \frac{4 \pi c}{H_0} \frac{D_c(z)^2}{E(z)},
    \]

    with \(D_c(z)\) the comoving distance.

    Parameters
    ----------
    z : float or jax.Array
        Redshift.

    Returns
    -------
    float or jax.Array
        Comoving volume element in SI units.
    """
    return ((4.0 * jnp.pi * cMKS) / H0s) * Dc_interp(z) ** 2 / EE(z)


@jit
def DL(z):
    r"""
    Compute the luminosity distance.

    Parameters
    ----------
    z : float or jax.Array
        Redshift.

    Returns
    -------
    float or jax.Array
        Luminosity distance \(D_L(z) = (1+z) D_c(z)\) in metres.
    """
    return (1.0 + z) * Dc_interp(z)


__all__ = [
    "DL",
    "Dc_interp",
    "Dc_interp_numpy",
    "Dca",
    "EE",
    "H0s",
    "OmegaDM",
    "OmegaLambda",
    "Omegak",
    "dVcdz",
    "dtodz",
]
