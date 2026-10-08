"""The Cosmology object: background against independent integrals, linear theory against its definitions."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import hyp2f1

from fastropop.cosmology import CONCORDANCE, PLANCK18, Cosmology

Z = np.array([1e-5, 1e-4, 1e-3, 0.0037, 0.05, 0.123, 1.0, 3.3333, 10.0, 24.9, 59.9])


def test_comoving_distance_matches_closed_form_without_radiation():
    """Matter + Lambda has a hypergeometric closed form (fastropop 0.1 used it)."""
    c = CONCORDANCE
    f = lambda z: hyp2f1(1 / 3, 1 / 2, 4 / 3, -(1 + z) ** 3 * c.Omega_m / c.Omega_L)
    exact = c.hubble_distance / np.sqrt(c.Omega_L) * ((1 + Z) * f(Z) - f(0.0))
    np.testing.assert_allclose(np.asarray(c.comoving_distance(Z)), exact, rtol=1e-8)


@pytest.mark.parametrize("cosmo", [PLANCK18, CONCORDANCE])
def test_background_matches_quadrature(cosmo):
    dc = [quad(lambda x: cosmo.hubble_distance / cosmo.E_np(x), 0, z, epsabs=0, epsrel=1e-13)[0] for z in Z]
    age = [quad(lambda x: 1 / ((1 + x) * cosmo.H0 * cosmo.E_np(x)), z, np.inf, epsabs=0, epsrel=1e-12,
                limit=500)[0] for z in Z]
    np.testing.assert_allclose(np.asarray(cosmo.comoving_distance(Z)), dc, rtol=1e-8)
    np.testing.assert_allclose(np.asarray(cosmo.age(Z)), age, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(cosmo.luminosity_distance(Z)), (1 + Z) * np.asarray(dc), rtol=1e-8)
    np.testing.assert_allclose(np.asarray(cosmo.dVc_dz(Z)),
                               4 * np.pi * np.asarray(dc) ** 2 * cosmo.hubble_distance / cosmo.E_np(Z), rtol=2e-8)


def test_inverse_and_derivatives():
    c = PLANCK18
    np.testing.assert_allclose(np.asarray(c.z_at_age(c.age(Z))), Z, rtol=1e-9, atol=1e-12)
    # the interpolants are differentiable and carry the exact derivatives
    g = jax.vmap(jax.grad(lambda z: c.comoving_distance(z)))(jnp.asarray(Z))
    np.testing.assert_allclose(np.asarray(g), c.hubble_distance / c.E_np(Z), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(c.lookback_time(Z)), float(c.age(0.0)) - np.asarray(c.age(Z)), rtol=1e-12)


def test_numpy_twins_match():
    c = PLANCK18
    np.testing.assert_allclose(c.comoving_distance_np(Z), np.asarray(c.comoving_distance(Z)), rtol=1e-14)
    np.testing.assert_allclose(c.dVc_dz_np(Z), np.asarray(c.dVc_dz(Z)), rtol=1e-14)
    np.testing.assert_allclose(c.dt_dz_np(Z), np.asarray(c.dt_dz(Z)), rtol=1e-14)


def test_usable_as_static_jit_argument():
    f = jax.jit(lambda z, cosmo: cosmo.luminosity_distance(z), static_argnums=1)
    assert float(f(1.0, PLANCK18)) != float(f(1.0, CONCORDANCE))


def test_planck18_is_planck18():
    c = PLANCK18
    assert (c.h, c.Omega_m, c.n_s, c.sigma8) == (0.674, 0.315, 0.965, 0.811)
    assert abs(c.Omega_b * c.h**2 - 0.0224) < 1e-12
    # age today: Planck 2018 quotes 13.797 +- 0.023 Gyr
    assert abs(float(c.age(0.0)) / 1e9 - 13.797) < 0.023
    # radiation: photons at T_cmb plus N_eff = 3.046 massless neutrinos
    omega_r = 2.469e-5 * (c.T_cmb / 2.7255) ** 4 * (1 + 0.2271 * 3.046) / c.h**2
    assert abs(c.Omega_r / omega_r - 1) < 0.01


def test_validation():
    with pytest.raises(ValueError):
        Cosmology(h=0.7, Omega_m=1.2)
    with pytest.raises(ValueError):
        Cosmology(h=0.7, Omega_m=0.3, Omega_b=0.4)


def test_linear_theory():
    c = PLANCK18
    assert abs(c.sigma8_value / c.sigma8 - 1) < 1e-6                     # normalisation round trip
    assert abs(float(c.transfer(np.array([1e-6]))[0]) - 1) < 1e-4       # T -> 1 on large scales
    M = np.geomspace(1e6, 1e15, 40)
    s = np.asarray(c.sigma_M(M))
    assert np.all(np.diff(s) < 0)
    dfd = np.gradient(np.log(s), np.log(M))                              # derivative table vs the table itself
    np.testing.assert_allclose(np.asarray(c.dlnsigma_dlnM(M))[1:-1], dfd[1:-1], rtol=2e-2)
    # growth: 1 today, and in matter domination D ~ 1/(1+z)
    assert float(c.growth(0.0)) == pytest.approx(1.0, abs=1e-12)
    zz = np.array([20.0, 40.0])
    ratio = np.asarray(c.growth(zz)) * (1 + zz)
    assert abs(ratio[1] / ratio[0] - 1) < 0.02                           # radiation still bends it slightly


def test_cobe_normalisation_matches_reference_implementation():
    """sigma8 = None normalises to COBE (Bunn & White 1997), as the KBFI EPS code does.
    With its parameters that code implies sigma_8 = 0.763 (computed 2026-10-08)."""
    v = Cosmology(h=0.674, Omega_m=0.315, Omega_r=0.0, Omega_b=0.05, n_s=0.965, sigma8=None, T_cmb=2.728)
    assert abs(v.sigma8_value - 0.763) < 1e-3
