"""Public package interface for fastropop."""

from ._version import __version__

_SEMI_ANALYTIC_EXPORTS = {
    "SemiAnalyticPopulation",
    "binning",
    "compute_h",
    "draw_parameters",
}

_UNIT_EXPORTS = {
    "hc_to_omega",
    "omega_to_hc",
}

_PLOT_EXPORTS = {
    "plot_binned_spectrum",
    "plot_realizations",
    "plot_sample_distributions",
    "plot_skymap",
}

_CONSTANT_EXPORTS = {
    "GMKS",
    "H100",
    "Mmax",
    "Mmin",
    "MsunMKS",
    "TNG15",
    "cMKS",
    "default_Mstar",
    "default_alphaM",
    "default_betaz",
    "default_n0",
    "default_z0",
    "fmaxNG15",
    "fminNG15",
    "kg",
    "pc",
    "pcinMKS",
    "s",
    "yr",
    "yrinMKS",
    "zmax",
    "zmin",
}

__all__ = [
    "__version__",
    *_SEMI_ANALYTIC_EXPORTS,
    *_UNIT_EXPORTS,
    *_PLOT_EXPORTS,
    *_CONSTANT_EXPORTS,
]


def __getattr__(name: str):
    """Lazily import the semi-analytic API to avoid eager heavy dependencies."""
    if name in _SEMI_ANALYTIC_EXPORTS:
        from . import semi_analytic

        return getattr(semi_analytic, name)
    if name in _UNIT_EXPORTS:
        from . import units

        return getattr(units, name)
    if name in _PLOT_EXPORTS:
        from . import plots

        return getattr(plots, name)
    if name in _CONSTANT_EXPORTS:
        from . import constants

        return getattr(constants, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
