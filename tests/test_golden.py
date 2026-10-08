"""Golden test: the restructured package must reproduce fastropop 0.1 (tests/golden/golden_0_1.npz).

The numbers were recorded once from the pre-restructuring code (lisa-band at dbfa02e) by
tests/golden/make_golden.py; do not regenerate them.

The cosmology moved from module constants to a ``Cosmology`` object with exact constants.
c went from 2.99792e8 to 2.99792458e8 m/s, and 100 km/s/Mpc from 3.24078e-18 to
1e5/3.0856775814913673e22 s^-1. That changes the Hubble distance by a known factor,
D_H_RATIO = 1 + 1.747e-6, and 1/H0 by H0_RATIO = 1 + 2.2e-7. Every background quantity
scales with a known power of these. The test divides that out and then demands agreement
to the accuracy of the new tables (~1e-9).

(The tolerance was first set at a flat 5e-6 relative. That was wrong for dV_c/dz, which
goes as D_H^3: 5.24e-6 from the constants alone. Dividing out the correction is both
stricter and correct.)

QUAD: h_c^2 and the expected count are adaptive nquad integrals at epsrel 1e-6 and 1e-4.
They are reproducible only to that accuracy once the integrand moves at 1e-6.
"""
from pathlib import Path

import numpy as np
import pytest

from fastropop.constants import C_MS, MPC_M, YR_S, MsunMKS
from fastropop.cosmology import CONCORDANCE

G = np.load(Path(__file__).parent / "golden" / "golden_0_1.npz")
H0_RATIO = (0.7 * 3.24078e-18) / (0.7 * 1e5 / MPC_M)          # old H0 / new H0 = new (1/H0) / old
D_H_RATIO = (C_MS / 2.99792e8) * H0_RATIO                       # new D_H / old D_H
EXACT, TABLE, QUAD_HC2, QUAD_N = 1e-12, 1e-8, 1e-4, 5e-4
PARAMS = {"n0": 10**-90.4153, "alphaM": -1.3800, "Mstar": 10**8.8272 * MsunMKS,
          "betaz": -0.1711, "z0": 4.70}

# quantity: (value in 0.1 units, power of D_H_RATIO, power of H0_RATIO)
BACKGROUND = {
    "EE": (lambda z: CONCORDANCE.E(z), 0, 0),
    "Dc": (lambda z: CONCORDANCE.comoving_distance(z) * MPC_M, 1, 0),
    "DL": (lambda z: CONCORDANCE.luminosity_distance(z) * MPC_M, 1, 0),
    "dVcdz": (lambda z: CONCORDANCE.dVc_dz(z) * MPC_M**3, 3, 0),
    "dtodz": (lambda z: CONCORDANCE.dt_dz(z) * YR_S, 0, 1),
}


@pytest.mark.parametrize("name", list(BACKGROUND))
def test_background(name):
    fn, p_dh, p_h0 = BACKGROUND[name]
    got = np.asarray(fn(G["z"])) / D_H_RATIO**p_dh / H0_RATIO**p_h0
    np.testing.assert_allclose(got, G[name], rtol=TABLE if p_dh or p_h0 else EXACT, atol=0.0)


def test_strain_and_frequency_evolution():
    from fastropop.semi_analytic import dlnfdtr, h, h_average
    np.testing.assert_allclose(np.asarray(dlnfdtr(G["M"], G["f"], G["zs"])), G["dlnfdtr"], rtol=EXACT)
    np.testing.assert_allclose(np.asarray(h(G["M"], G["f"], G["zs"])) * D_H_RATIO, G["h"], rtol=TABLE)
    np.testing.assert_allclose(np.asarray(h_average(G["M"], G["f"], G["zs"])) * D_H_RATIO,
                               G["h_average"], rtol=TABLE)


@pytest.fixture(scope="module")
def pop():
    from fastropop.semi_analytic import SemiAnalyticPopulation
    return SemiAnalyticPopulation(population_params=PARAMS)


def test_rate_densities(pop):
    # d2n/dz dM carries dt/dz; d3n/(dz dM dlnf) = d2n * (dt/dz)^-1 * dVc/dz carries D_H^3
    np.testing.assert_allclose(np.asarray(pop.d2ndzdM(G["zs"], G["M"])) / H0_RATIO, G["d2ndzdM"], rtol=TABLE)
    np.testing.assert_allclose(np.asarray(pop.d3ndzdMdlnf(G["M"], G["f"], G["zs"])) / D_H_RATIO**3,
                               G["d3ndzdMdlnf"], rtol=TABLE)


def test_integrals(pop):
    np.testing.assert_allclose([pop.hc2(f) for f in G["hc2_f"]], G["hc2"], rtol=QUAD_HC2)
    np.testing.assert_allclose(pop.compute_Nbinaries(), G["Nbinaries"], rtol=QUAD_N)
