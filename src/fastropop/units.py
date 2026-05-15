"""Unit and representation conversion helpers."""

import jax.numpy as jnp
from jax import jit

from .constants import H100


@jit
def hc_to_omega(hc, f):
    r"""
    Convert characteristic strain into \(\Omega_{\rm GW} h^2\).

    The conversion assumes the standard observer-frame relation

    \[
    \Omega_{\rm GW}(f)\,h^2 =
    \frac{2\pi^2}{3 H_{100}^2}\, f^2 h_c(f)^2,
    \]

    where \(H_{100} = 100\,{\rm km\,s^{-1}\,Mpc^{-1}}\) is encoded in
    `fastropop.constants.H100`.

    Parameters
    ----------
    hc : float or jax.Array
        Characteristic strain \(h_c(f)\).
    f : float or jax.Array
        Observer-frame gravitational-wave frequency in Hz.

    Returns
    -------
    float or jax.Array
        Energy-density spectrum \(\Omega_{\rm GW}(f) h^2\).
    """
    return (2 * jnp.pi**2 / (3 * H100**2)) * f**2 * hc**2


@jit
def omega_to_hc(omega_gwh2, f):
    r"""
    Convert \(\Omega_{\rm GW} h^2\) into characteristic strain.

    This is the inverse of `hc_to_omega`:

    \[
    h_c(f) =
    \sqrt{\Omega_{\rm GW}(f) h^2
    \frac{3 H_{100}^2}{2\pi^2 f^2}}.
    \]

    Parameters
    ----------
    omega_gwh2 : float or jax.Array
        Energy-density spectrum \(\Omega_{\rm GW}(f) h^2\).
    f : float or jax.Array
        Observer-frame gravitational-wave frequency in Hz.

    Returns
    -------
    float or jax.Array
        Characteristic strain \(h_c(f)\).
    """
    return jnp.sqrt(omega_gwh2 * 3 * H100**2 / (2 * jnp.pi**2 * f**2))


__all__ = ["hc_to_omega", "omega_to_hc"]
