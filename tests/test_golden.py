"""Golden test: the restructured package must reproduce fastropop 0.1 (tests/golden/golden_0_1.npz).

The numbers were recorded once from the pre-restructuring code (lisa-band at dbfa02e) by
tests/golden/make_golden.py; do not regenerate them. Tolerances were fixed before the
restructuring and say why each is not 1e-12:

* COSMO -- the cosmology moves from module constants to a ``Cosmology`` object with exact
  constants. c goes from 2.99792e8 to 2.99792458e8 m/s (1.5e-6), and 100 km/s/Mpc from
  3.24078e-18 to 1e5/3.0856775814913673e22 s^-1 (2.2e-7). The comoving distance changes
  from the closed hypergeometric form to a tabulated trapezoid integral.
* QUAD  -- hc2 and the expected count are adaptive nquad integrals at epsrel 1e-6 and 1e-4.
  They are reproducible only to that accuracy once the integrand moves at 1e-6.
"""
from pathlib import Path

import numpy as np
import pytest

from fastropop.constants import MsunMKS

G = np.load(Path(__file__).parent / "golden" / "golden_0_1.npz")
EXACT, COSMO, QUAD_HC2, QUAD_N = 1e-12, 5e-6, 1e-4, 5e-4
CONCORDANCE_PARAMS = {"n0": 10**-90.4153, "alphaM": -1.3800, "Mstar": 10**8.8272 * MsunMKS,
                      "betaz": -0.1711, "z0": 4.70}


def _cosmo():
    """The 0.1 cosmology through whichever interface the current version provides."""
    from fastropop import cosmology as c
    if hasattr(c, "CONCORDANCE"):                      # restructured: an object, astrophysical units
        from fastropop.constants import MPC_M, YR_S
        k = c.CONCORDANCE
        return dict(EE=k.E, Dc=lambda z: k.comoving_distance(z) * MPC_M,
                    DL=lambda z: k.luminosity_distance(z) * MPC_M,
                    dVcdz=lambda z: k.dVc_dz(z) * MPC_M**3, dtodz=lambda z: k.dt_dz(z) * YR_S)
    return dict(EE=c.EE, Dc=c.Dc_interp, DL=c.DL, dVcdz=c.dVcdz, dtodz=c.dtodz)


@pytest.mark.parametrize("name,rtol", [("EE", EXACT), ("Dc", COSMO), ("DL", COSMO),
                                       ("dVcdz", COSMO), ("dtodz", COSMO)])
def test_background(name, rtol):
    got = np.asarray(_cosmo()[name](G["z"]))
    np.testing.assert_allclose(got, G[name], rtol=rtol, atol=0.0)


def test_strain_and_frequency_evolution():
    from fastropop.semi_analytic import dlnfdtr, h, h_average
    np.testing.assert_allclose(np.asarray(dlnfdtr(G["M"], G["f"], G["zs"])), G["dlnfdtr"], rtol=EXACT)
    np.testing.assert_allclose(np.asarray(h(G["M"], G["f"], G["zs"])), G["h"], rtol=COSMO)
    np.testing.assert_allclose(np.asarray(h_average(G["M"], G["f"], G["zs"])), G["h_average"], rtol=COSMO)


@pytest.fixture(scope="module")
def pop():
    from fastropop.semi_analytic import SemiAnalyticPopulation
    return SemiAnalyticPopulation(population_params=CONCORDANCE_PARAMS)


def test_rate_densities(pop):
    np.testing.assert_allclose(np.asarray(pop.d2ndzdM(G["zs"], G["M"])), G["d2ndzdM"], rtol=COSMO)
    np.testing.assert_allclose(np.asarray(pop.d3ndzdMdlnf(G["M"], G["f"], G["zs"])), G["d3ndzdMdlnf"], rtol=COSMO)


def test_integrals(pop):
    np.testing.assert_allclose([pop.hc2(f) for f in G["hc2_f"]], G["hc2"], rtol=QUAD_HC2)
    np.testing.assert_allclose(pop.compute_Nbinaries(), G["Nbinaries"], rtol=QUAD_N)
