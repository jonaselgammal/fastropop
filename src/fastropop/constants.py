"""Shared constants used across the package."""

# =============================================================================
# Base Unit Conventions
# =============================================================================
# The code uses SI-like bookkeeping with explicit scale factors so expressions
# can remain dimensionally readable without introducing a full units library.
kg = 1.0
m = 1.0
s = 1.0
pc = 1.0
yr = 1.0

# =============================================================================
# Physical Constants
# =============================================================================
# Fundamental and astrophysical scale constants in MKS-like units.
MsunMKS = 1.9891e30 * kg
GMKS = 6.67384e-11 * m**3 / (kg * s**2)
cMKS = 2.99792e8 * m / s

# =============================================================================
# Unit Conversion Factors
# =============================================================================
# Conversions from the symbolic unit factors above to their MKS values.
yrinMKS = 3.15576e7 * s / yr
pcinMKS = 3.08568e16 * m / pc

# =============================================================================
# Reference Cosmology Scales
# =============================================================================
# Standard normalization used in GW energy-density conversions.
H100 = 100.0 * 1000.0 / 3.085677581e22

# =============================================================================
# Default Integration Bounds
# =============================================================================
# Fiducial domain for mass, redshift, and PTA frequency calculations.
Mmin = 1e6 * MsunMKS
Mmax = 1e11 * MsunMKS
zmin = 0.0
zmax = 5.0

TNG15 = 16 * 365.25 * 24 * 3600
fminNG15 = (1 - 0.5) / TNG15 / s
fmaxNG15 = 1e-7 / s

# =============================================================================
# Default Population Parameters
# =============================================================================
# Baseline semi-analytic population parameters used when callers do not provide
# explicit values.
default_n0 = 1e-3 / ((1e6 * pc * pcinMKS) ** 3 * (1e9 * yr * yrinMKS))
default_alphaM = 0.0
default_Mstar = 1.8e8 * MsunMKS
default_betaz = 2.0
default_z0 = 1.8

__all__ = [
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
    "m",
    "pc",
    "pcinMKS",
    "s",
    "yr",
    "yrinMKS",
    "zmax",
    "zmin",
]
