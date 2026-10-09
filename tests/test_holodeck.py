"""The holodeck population: its pieces against their definitions, and its paths against each other.
(Against holodeck v1.6 itself, at zero scatter: total coalescence rate to 0.5%, its redshift
distribution to < 1%; see crosschecks/code/holodeck in the LISA_MBHBs repository.)"""
import numpy as np
import jax.numpy as jnp
import pytest

from fastropop.cosmology import PLANCK18
from fastropop.grid import Grid
from fastropop.populations import Holodeck


@pytest.fixture(scope="module")
def model():
    return Holodeck(PLANCK18)


def test_table_is_symmetric_and_finite(model):
    A = np.asarray(model._table(model.params()))
    assert A.shape == (model.z_nodes.size, model.log10_Mstar.size, model.log10_Mstar.size)
    assert np.all(np.isfinite(A)) and np.all(A >= 0)
    np.testing.assert_allclose(A, np.swapaxes(A, 1, 2), rtol=0, atol=0)


def test_pair_redshift_solves_the_delay_equation(model):
    p = model.params(hard_time=1.5)
    q = jnp.array([1.0, 0.3, 0.1])[:, None]
    zc = jnp.array([0.1, 0.5, 1.0, 2.0])[None, :]
    zi, ok = model.pair_redshift(p, 11.0, q, zc)
    c = model.cosmology
    resid = c.age(zi) + model.merger_time(11.0, q, zi) + 1.5e9 - c.age(zc)
    assert np.all(np.asarray(ok)[:, :3])
    assert np.max(np.abs(np.asarray(resid)[np.asarray(ok)])) < 1e3            # yr, against delays of Gyr
    assert np.all(np.asarray(zi) >= np.asarray(zc))


def test_no_coalescence_without_time():
    """A delay longer than the age of the universe leaves nothing."""
    m = Holodeck(PLANCK18, z_nodes=np.linspace(0.05, 3.0, 10))
    assert float(np.max(np.asarray(m._table(m.params(hard_time=20.0))))) == 0.0


def test_rate_scales_with_the_mass_function_normalisation(model):
    g = Grid.m1m2(n_m=41, n_z=21, log10_m=(3.0, 9.0), z=(0.1, 5.0))
    r0 = float(model.total_rate(model.params(), g))
    r1 = float(model.total_rate(model.params(gsmf_phi0_log10=model.defaults["gsmf_phi0_log10"] + 0.3), g))
    assert r1 / r0 == pytest.approx(10 ** 0.3, rel=1e-10)


def test_fast_path_matches_point_path(model):
    g = Grid.m1m2(n_m=31, n_z=17, log10_m=(3.0, 9.0), z=(0.1, 6.0))      # z off the nodes
    p = model.params()
    lam = np.asarray(model.intensity(p, g))
    x1, x2, z = (np.broadcast_to(np.asarray(a), g.shape) for a in g.mesh())
    pts = [jnp.asarray(a.ravel()) for a in (x1, x2, z)]
    lam_pts = np.asarray(model.rate_density(p, g.coords, *pts)).reshape(g.shape)
    sel = np.asarray(g.valid) & (lam_pts > 1e-250)
    np.testing.assert_allclose(lam[sel], lam_pts[sel], rtol=1e-11)


def test_scatter_conserves_the_rate():
    """The log-normal only redistributes black-hole mass: on a grid wide enough to hold the
    broadened population, the total rate does not depend on the scatter."""
    m = Holodeck(PLANCK18, z_nodes=np.linspace(0.05, 3.0, 30))
    g = Grid.m1m2(n_m=241, n_z=30, log10_m=(-1.0, 11.0), z=(0.05, 3.0))
    r = [float(m.total_rate(m.params(mmb_scatter_dex=s), g)) for s in (0.05, 0.3, 0.6)]
    assert r[1] == pytest.approx(r[0], rel=5e-3) and r[2] == pytest.approx(r[0], rel=5e-3)


def test_native_interpolation_matches_pair_path():
    """(log10 Mc, q, z) grids from a 0.05-dex (m1, m2) tensor grid: rate-weighted error below 0.5%
    (EPS ~1e-5; holodeck ~0.2%, the geometric-mean bias of log interpolation at small scatter)."""
    from fastropop.populations import EPS
    g = Grid.mcq(n_mc=40, n_q=12, n_z=20, log10_mc=(2.5, 9.5), z=(0.1, 10.0))
    W = np.asarray(g.W)
    for exact, fast, p in ((EPS(PLANCK18), EPS(PLANCK18).use_native_interpolation(0.05), {}),
                           (Holodeck(PLANCK18), Holodeck(PLANCK18).use_native_interpolation(0.05),
                            dict(gsmf_phi0_log10=-2.2, hard_time=1.0))):
        pp = exact.params(**p)
        le, lf = np.asarray(exact.intensity(pp, g)), np.asarray(fast.intensity(pp, g))
        assert np.sum(np.abs(lf - le) * W) / np.sum(le * W) < 5e-3


def test_interpolated_model_is_reusable_across_traces():
    """The cached interpolation stencil must hold constants, not tracers of the first jit."""
    import jax
    m = Holodeck(PLANCK18).use_native_interpolation(0.05)
    g = Grid.mcq(n_mc=20, n_q=6, n_z=10, log10_mc=(3.0, 9.0), z=(0.1, 5.0))
    th = jnp.asarray(m.pack(m.params()))
    a = jax.jit(lambda t: m.intensity(m.unpack(t), g).sum())(th)
    b = jax.jit(lambda t: 2.0 * m.intensity(m.unpack(t), g).sum())(th)      # a second, different trace
    assert float(b) == pytest.approx(2.0 * float(a), rel=1e-12)
