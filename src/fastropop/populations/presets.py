r"""Named populations: a model, its parameters, and where they come from.

    from fastropop.populations import get_preset
    pre = get_preset("caliskan25_light")
    model, params = pre.model(), pre.params

Every entry must say where its numbers came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..cosmology import PLANCK18
from .phenomenological import FIDUCIAL, Phenomenological


@dataclass(frozen=True)
class Preset:
    """A named population. ``model(cosmology)`` builds it; ``params`` are its parameters."""

    name: str
    build: Callable = field(repr=False)
    params: dict = field(default_factory=dict)
    reference: str = ""
    notes: str = ""

    def model(self, cosmology=PLANCK18):
        return self.build(cosmology)


_CALISKAN = ("Caliskan, Anil Kumar, Kamionkowski & Cheng, arXiv:2506.18965, Tab. I: their Eq. (3) is "
             "this ansatz with the same 1e7 Msun pivot.")
_CALISKAN_NOTE = ("They assume equal masses; here p(q) is the fiducial power law, normalised, so the "
                  "rate is unchanged. The support (250 Msun < Mc < 1e9 Msun, z < 20) is the domain over "
                  "which their n0 gives their stated 200/yr; checked to 1% (tests/test_populations.py).")
_C25_SUPPORT = {"log10_Mc": (float(np.log10(250.0)), 9.0), "z": (0.0, 20.0)}


def _c25(name, n0, alpha_M, Mstar, beta_z, z0, label):
    return Preset(
        name=name,
        build=lambda cosmo: Phenomenological(cosmo, support=_C25_SUPPORT),
        params=dict(log10_n0=float(np.log10(n0)), alpha_M=alpha_M, log10_Mstar=float(np.log10(Mstar)),
                    beta_z=beta_z, z0=z0, beta_q=FIDUCIAL["beta_q"]),
        reference=_CALISKAN, notes=f"Model {label}. " + _CALISKAN_NOTE)


PRESETS = {
    "fiducial": Preset(
        name="fiducial", build=lambda cosmo: Phenomenological(cosmo), params=dict(FIDUCIAL),
        reference="The phenomenological family calibrated by weighted maximum likelihood to the Q3nod_K16 "
                  "merger catalogue of Barausse et al. 2023 (LISA_MBHBs, scripts/calibrate_fiducial.py).",
        notes="Reproduces the catalogue's (log10 Mc, q, z) quartiles to 0.1-0.2 dex."),
    "caliskan25_light": _c25("caliskan25_light", 7.14e-18, 0.8, 5e6, 7.2, 1.5, "1 (Light Seed)"),
    "caliskan25_heavy": _c25("caliskan25_heavy", 2.01e-11, -0.85, 7e4, 4.8, 3.3, "2 (Heavy Seed)"),
    "caliskan25_ultralight": _c25("caliskan25_ultralight", 2.97e-21, 1.6, 5e5, 7.2, 1.5, "3 (Ultra-Light Seed)"),
}


def get_preset(name):
    """Look up a preset, listing the alternatives if the name is unknown."""
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}") from None


def list_presets():
    return sorted(PRESETS)


__all__ = ["PRESETS", "Preset", "get_preset", "list_presets"]
