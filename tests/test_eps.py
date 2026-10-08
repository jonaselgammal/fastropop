"""The extended Press-Schechter population: its pieces against their definitions, and its two paths
against each other. (Against the KBFI implementation it agrees to 2.6e-4 at z = 0.5-10 when given the
same inputs; see crosschecks/code/eps in the LISA_MBHBs repository.)"""
import numpy as np
import jax.numpy as jnp
import pytest
from scipy.special import erfc

from fastropop.cosmology import PLANCK18
from fastropop.grid import Grid
from fastropop.populations import EPS, GIRELLI2020
from fastropop.populations.eps import log10_mstar


@pytest.fixture(scope="module")
def model():
    return EPS(PLANCK18)


def test_press_schechter_mass_fraction(model):
    """The mass fraction in halos between M1 and M2 is erfc(nu2/sqrt2) - erfc(nu1/sqrt2), nu = delta_c/sigma."""
    lM = np.arange(6.0, 16.0001, 0.01)
    for z in (0.0, 3.0, 10.0):
        n = model.press_schechter(lM, z)
        frac = np.trapezoid(10.0**lM * n, lM * np.log(10.0)) / PLANCK18.rho_m0
        nu = float(PLANCK18.delta_c(z)) / np.asarray(PLANCK18.sigma_M(10.0 ** lM[[0, -1]]))
        assert frac == pytest.approx(erfc(nu[0] / np.sqrt(2)) - erfc(nu[1] / np.sqrt(2)), rel=2e-3), z


def test_halo_merger_rate_symmetric_and_positive(model):
    R = model.halo_merger_rate(model.log10_Mh[::10], np.array([0.5, 6.0]))
    np.testing.assert_allclose(R, np.swapaxes(R, 1, 2), rtol=1e-12)
    assert np.all(R >= 0) and np.all(np.isfinite(R))


def test_girelli_peak():
    """At M_h = M_A the stellar mass fraction is A (Girelli et al. 2020 Eq. 6)."""
    for z in (0.0, 2.0):
        lMA = GIRELLI2020["B"] + GIRELLI2020["mu"] * z
        A = GIRELLI2020["C"] * (1 + z) ** GIRELLI2020["nu"]
        assert float(log10_mstar(lMA, z)) == pytest.approx(lMA + np.log10(A), abs=1e-12)


def test_black_hole_kernel_is_normalised(model):
    """Each halo cell's kernel is p(log10 m | M_h) integrated over the cell, so over log10 m it
    integrates to the cell width -- wherever its mean lies inside the range. Girelli's fit
    extrapolated to tiny halos at high z gives mean black-hole masses far below 1 Msun, which no
    mass grid reaches."""
    p = model.params()
    lm = jnp.linspace(-5.0, 20.0, 20001)
    I = np.asarray(jnp.trapezoid(model._kernel(p, lm), lm, axis=1))
    mu = p["a"] + p["b"] * (np.asarray(model._xe) - 11.0)
    inside = (mu[:, :-1] > 0.0) & (mu[:, 1:] < 15.0)
    assert inside.mean() > 0.5
    np.testing.assert_allclose(I[inside], model._dh, rtol=1e-9)


def test_cell_kernel_matches_brute_force():
    """The cell-integrated kernel equals the pointwise one integrated on a fine sub-grid."""
    from fastropop.populations.eps import _cell_kernel
    lm = jnp.linspace(2.0, 10.0, 41)[:, None]
    for mu_lo, mu_hi, sig in ((5.0, 5.3, 0.2), (5.0, 5.02, 0.48), (4.0, 7.0, 0.3)):
        x = jnp.linspace(0.0, 1.0, 400001)          # fine enough for the far tails, where u ~ 20
        mu = mu_lo + (mu_hi - mu_lo) * x
        brute = jnp.trapezoid(jnp.exp(-0.5 * ((lm - mu) / sig) ** 2) / (sig * jnp.sqrt(2 * jnp.pi)), x, axis=1) * 0.1
        np.testing.assert_allclose(np.asarray(_cell_kernel(lm[:, 0], mu_lo, mu_hi, sig, 0.1)), np.asarray(brute),
                                   rtol=1e-6, atol=1e-30)


def test_fast_path_matches_point_path_and_scales_with_pBH(model):
    g = Grid.m1m2(n_m=41, n_z=31, log10_m=(2.0, 10.0), z=(0.1, 22.0))      # z off the halo nodes
    p = model.params()
    lam = np.asarray(model.intensity(p, g))
    x1, x2, z = (np.broadcast_to(np.asarray(a), g.shape) for a in g.mesh())
    pts = [jnp.asarray(a.ravel()) for a in (x1, x2, z)]
    lam_pts = np.asarray(model.rate_density(p, g.coords, *pts)).reshape(g.shape)
    sel = np.asarray(g.valid) & (lam_pts > 1e-250)
    np.testing.assert_allclose(lam[sel], lam_pts[sel], rtol=1e-11)
    lam2 = np.asarray(model.intensity(model.params(log10_pBH=-1.0), g))
    np.testing.assert_allclose(lam2[sel], 0.1 * lam[sel], rtol=1e-12)


def test_pair_path_matches_point_path(model):
    """On a (log10 Mc, q, z) grid the intensity contracts the kernel at the mesh's (m1, m2) pairs."""
    g = Grid.mcq(n_mc=23, n_q=9, n_z=31, log10_mc=(2.3, 9.5), z=(0.1, 22.0))     # z off the halo nodes
    p = model.params()
    lam = np.asarray(model.intensity(p, g))
    lmc, q, z = (np.broadcast_to(np.asarray(a), g.shape) for a in g.mesh())
    lam_pts = np.asarray(model.intensity_at(p, *(jnp.asarray(a.ravel()) for a in (lmc, q, z)))).reshape(g.shape)
    sel = lam_pts > 1e-250
    assert sel.mean() > 0.9
    np.testing.assert_allclose(lam[sel], lam_pts[sel], rtol=1e-11)


def test_rate_is_the_same_in_both_coordinate_systems(model):
    p = model.params()
    gm = Grid.m1m2(n_m=161, n_z=60, log10_m=(3.0, 9.0), z=(0.1, 15.0))
    gq = Grid.mcq(n_mc=121, n_q=80, n_z=60, log10_mc=(2.4, 9.0), z=(0.1, 15.0))
    # the same physical region on both grids: both masses in [1e3, 1e9] and q >= 0.01 (EPS
    # has mass ratios down to ~1e-6, which the (Mc, q) grid does not reach)
    model_cut = EPS(PLANCK18, support={"log10_m1": (3.0, 9.0), "log10_m2": (3.0, 9.0), "q": (0.01, 1.0)})
    rm = float(model_cut.total_rate(p, gm))
    rq = float(model_cut.total_rate(p, gq))
    assert rq == pytest.approx(rm, rel=2e-2)
