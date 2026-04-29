"""Cosmology-specific helper functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from scipy.special import hyp2f1

from .constants import cMKS, s

jax.config.update("jax_enable_x64", True)

hH0 = 0.7
OmegaDM = 0.3
OmegaLambda = 0.7
Omegak = 0.0
H0s = hH0 * 3.24078e-18 / s


@jit
def EE(z):
    """Compute the dimensionless Hubble parameter E(z)."""
    return jnp.sqrt(OmegaDM * (1.0 + z) ** 3.0 + Omegak * (1.0 + z) ** 2.0 + OmegaLambda)


@jit
def dtodz(z):
    """Compute dt/dz, the derivative of cosmic time with respect to redshift."""
    return 1.0 / (H0s * (1.0 + z) * EE(z))


def Dca(z):
    """Compute the comoving distance analytically using hypergeometric functions."""
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
    """Interpolate the comoving distance on a precomputed grid."""
    return jnp.interp(z, _Z_VALUES_JAX, _DC_VALUES_JAX)


@jit
def dVcdz(z):
    """Compute the comoving volume element dVc/dz."""
    return ((4.0 * jnp.pi * cMKS) / H0s) * Dc_interp(z) ** 2 / EE(z)


@jit
def DL(z):
    """Compute the luminosity distance."""
    return (1.0 + z) * Dc_interp(z)


__all__ = [
    "DL",
    "Dc_interp",
    "Dca",
    "EE",
    "H0s",
    "OmegaDM",
    "OmegaLambda",
    "Omegak",
    "dVcdz",
    "dtodz",
    "hH0",
]
