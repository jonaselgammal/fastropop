"""HEALPix backend selection and backend-local accumulation helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

try:
    import jax_healpy as _jhp
except ImportError:  # pragma: no cover
    _jhp = None

try:
    import healpy as _hp
except ImportError:  # pragma: no cover
    _hp = None


HAS_JAX_HEALPY = _jhp is not None
HAS_HEALPY = _hp is not None

if HAS_JAX_HEALPY:
    BACKEND_NAME = "jax-healpy"
elif HAS_HEALPY:
    BACKEND_NAME = "healpy"
else:
    BACKEND_NAME = None


def require_backend() -> str:
    """Return the selected backend name or raise if no backend is installed."""
    if BACKEND_NAME is None:
        raise ImportError(
            "No HEALPix backend is installed. Install `jax-healpy` or `healpy` to use skymaps."
        )
    return BACKEND_NAME


def nside2npix(nside):
    """Return the number of pixels for a given NSIDE."""
    require_backend()
    if HAS_JAX_HEALPY:
        return int(_jhp.nside2npix(nside))
    return int(_hp.nside2npix(nside))


def init_skymaps(npix, n_freq):
    """Initialize skymap arrays for the active backend."""
    require_backend()
    if HAS_JAX_HEALPY:
        return (
            jnp.zeros((npix, n_freq), dtype=jnp.complex128),
            jnp.zeros((npix, n_freq), dtype=jnp.complex128),
        )
    return (
        np.zeros((npix, n_freq), dtype=complex),
        np.zeros((npix, n_freq), dtype=complex),
    )


def accumulate_skymap_batch(
    skymaps_plus,
    skymaps_cross,
    nside,
    theta,
    phi,
    freq_indices,
    hbar_plus_rot,
    hbar_cross_rot,
):
    """Accumulate one batch of sources into HEALPix maps."""
    require_backend()
    if HAS_JAX_HEALPY:
        pix_indices = _jhp.ang2pix(nside, theta, phi)
        skymaps_plus = skymaps_plus.at[pix_indices, freq_indices].add(hbar_plus_rot)
        skymaps_cross = skymaps_cross.at[pix_indices, freq_indices].add(hbar_cross_rot)
        return skymaps_plus, skymaps_cross

    pix_indices = _hp.ang2pix(nside, np.asarray(theta), np.asarray(phi))
    freq_indices_np = np.asarray(freq_indices)
    flat_indices = pix_indices * skymaps_plus.shape[1] + freq_indices_np
    np.add.at(skymaps_plus.ravel(), flat_indices, np.asarray(hbar_plus_rot))
    np.add.at(skymaps_cross.ravel(), flat_indices, np.asarray(hbar_cross_rot))
    return skymaps_plus, skymaps_cross


__all__ = [
    "BACKEND_NAME",
    "HAS_HEALPY",
    "HAS_JAX_HEALPY",
    "accumulate_skymap_batch",
    "init_skymaps",
    "nside2npix",
    "require_backend",
]
