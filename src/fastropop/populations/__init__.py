"""Population models: one contract (:class:`PopulationModel`), several constructions."""

from .base import PopulationModel
from .eps import EPS, GIRELLI2020, RV15_INACTIVE
from .holodeck import HOLODECK_DEFAULTS, NG15_PHENOM_FIXED, Holodeck
from .host_kernel import HostKernelModel
from .phenomenological import FIDUCIAL, Phenomenological, madau_dickinson, q_norm
from .presets import PRESETS, Preset, get_preset, list_presets
from .tabulated import Tabulated, bin_edges, hat_average

__all__ = ["EPS", "FIDUCIAL", "GIRELLI2020", "HOLODECK_DEFAULTS", "Holodeck", "HostKernelModel", "NG15_PHENOM_FIXED",
           "RV15_INACTIVE", "PRESETS", "Phenomenological", "PopulationModel", "Preset", "Tabulated",
           "bin_edges", "get_preset", "hat_average", "list_presets", "madau_dickinson", "q_norm"]
