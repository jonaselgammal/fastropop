"""Unit and representation conversion helpers."""

import jax.numpy as jnp
from jax import jit

from .constants import H100


@jit
def hc_to_omega(hc, f):
    """Convert characteristic strain h_c(f) to Omega_GW h^2."""
    return (2 * jnp.pi**2 / (3 * H100**2)) * f**2 * hc**2


@jit
def omega_to_hc(omega_gwh2, f):
    """Convert Omega_GW h^2 to characteristic strain h_c(f)."""
    return jnp.sqrt(omega_gwh2 * 3 * H100**2 / (2 * jnp.pi**2 * f**2))


__all__ = ["hc_to_omega", "omega_to_hc"]
