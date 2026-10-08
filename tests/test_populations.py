"""Grids, the population contract, and the models built on it."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fastropop.cosmology import CONCORDANCE, PLANCK18
from fastropop.grid import M1M2, MCQ, Grid, jacobian, to_m1m2, to_mcq, trapezoid_weights
from fastropop.populations import FIDUCIAL, Phenomenological, Tabulated, get_preset, list_presets, q_norm

rng = np.random.default_rng(1)
PTS = rng.uniform([2.5, 0.01, 0.06], [9.5, 1.0, 24.0], size=(50, 3))


# --------------------------------------------------------------------- grids and coordinates
def test_grid_weights_and_shape():
    g = Grid.mcq(n_mc=31, n_q=12, n_z=41)
    assert g.shape == (31, 12, 41)
    for a, w in zip(g.axes, g.weights):
        assert float(jnp.sum(w)) == pytest.approx(float(a[-1] - a[0]), rel=1e-12)
    assert float(jnp.sum(g.W)) == pytest.approx(7.0 * 0.99 * 24.95, rel=1e-12)
    np.testing.assert_allclose(np.asarray(g.axes[1]), np.geomspace(0.01, 1.0, 12))


def test_coordinate_round_trip_and_jacobian():
    lmc, q = jnp.asarray(PTS[:, 0]), jnp.asarray(PTS[:, 1])
    l1, l2 = to_m1m2(lmc, q)
    back = to_mcq(l1, l2)
    np.testing.assert_allclose(np.asarray(back[0]), PTS[:, 0], rtol=1e-13)
    np.testing.assert_allclose(np.asarray(back[1]), PTS[:, 1], rtol=1e-12)
    # |d(l1, l2)/d(lmc, q)| by automatic differentiation
    J = jax.vmap(jax.jacfwd(lambda x: jnp.stack(to_m1m2(x[0], x[1]))))(jnp.stack([lmc, q], axis=1))
    det = np.abs(np.linalg.det(np.asarray(J)))
    np.testing.assert_allclose(np.asarray(jacobian(M1M2, MCQ, lmc, q)), det, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(jacobian(MCQ, M1M2, l1, l2)) * det, 1.0, rtol=1e-12)


def test_rate_is_the_same_in_both_coordinate_systems():
    """A lognormal mass function, negligible at both grids' mass edges, so the two grids
    cover the same population. The (m1, m2) grid integrates over the triangle m2 <= m1."""
    m = Phenomenological(PLANCK18, mass="bimodal")
    p = m.params(w_light=1.0, log10_Mlight=5.5, sigma_light=0.4)
    gq = Grid.mcq(n_mc=300, n_q=200, n_z=120, log10_mc=(2.5, 8.5), z=(0.05, 20.0))
    gm = Grid.m1m2(n_m=600, n_z=120, log10_m=(2.5, 10.5), z=(0.05, 20.0))
    rq, rm = float(m.total_rate(p, gq)), float(m.total_rate(p, gm))
    assert rq == pytest.approx(rm, rel=2e-3)


# --------------------------------------------------------------------- phenomenological
def test_phenomenological_matches_its_formula():
    m = Phenomenological(PLANCK18)
    p = m.params()
    lm, q, z = (jnp.asarray(PTS[:, i]) for i in range(3))
    Mc = 10.0**lm
    expected = (10.0 ** p["log10_n0"] * (Mc / 1e7) ** (-p["alpha_M"]) * jnp.exp(-Mc / 10.0 ** p["log10_Mstar"])
                * (1 + z) ** p["beta_z"] * jnp.exp(-z / p["z0"]) * q ** p["beta_q"] / q_norm(p["beta_q"])
                * PLANCK18.dVc_dz(z) / (1 + z))
    np.testing.assert_allclose(np.asarray(m.intensity_at(p, lm, q, z)), np.asarray(expected), rtol=1e-13)


def test_mass_ratio_and_support_bounds():
    m = Phenomenological(PLANCK18, q_min=0.05, support={"z": (0.0, 5.0)})
    p = m.params()
    r = np.asarray(m.intensity_at(p, jnp.array([6.0, 6.0, 6.0]), jnp.array([0.04, 0.5, 0.5]),
                                  jnp.array([1.0, 1.0, 6.0])))
    assert r[0] == 0.0 and r[1] > 0.0 and r[2] == 0.0


def test_q_norm_limit():
    assert float(q_norm(-1.0)) == pytest.approx(-np.log(0.01))
    assert float(q_norm(-1.0 + 1e-6)) == pytest.approx(-np.log(0.01), rel=1e-5)


def test_bimodal_light_fraction():
    m = Phenomenological(PLANCK18, mass="bimodal")
    lm = jnp.linspace(2.0, 10.5, 4001)
    q, z = jnp.full_like(lm, 0.5), jnp.full_like(lm, 1.0)
    for w in (0.0, 0.3, 1.0):
        p = m.params(w_light=w)
        f = m.comoving_rate(p, lm, q, z)
        light = m.comoving_rate(m.params(w_light=1.0), lm, q, z)
        total = float(jnp.trapezoid(f, lm))
        assert float(jnp.trapezoid(w * light, lm)) == pytest.approx(w * float(jnp.trapezoid(light, lm)))
        assert total == pytest.approx(float(jnp.trapezoid(m.comoving_rate(m.params(w_light=0.0), lm, q, z), lm)),
                                      rel=1e-2)  # normalised components: total independent of w


def test_delay_model_has_no_formation_before_z60():
    m = Phenomenological(PLANCK18, redshift="delay")
    p = m.params()
    r = np.asarray(m._delayed_formation(jnp.array([0.5, 5.0, 25.0, 59.0]), p["alpha_d"], p["log10_tmin"]))
    assert np.all(np.isfinite(r)) and r[0] > r[2] > 0 and r[3] == 0.0


def test_params_pack_unpack():
    m = Phenomenological(PLANCK18, mass="bimodal", redshift="delay")
    assert m.param_names == ("log10_n0", "alpha_M", "log10_Mstar", "alpha_d", "log10_tmin", "beta_q",
                             "w_light", "log10_Mlight", "sigma_light")
    p = m.params(alpha_M=0.1)
    assert m.unpack(m.pack(p))["alpha_M"] == pytest.approx(0.1)
    with pytest.raises(KeyError):
        m.params(beta_z=1.0)


def test_intensity_is_jittable_and_differentiable():
    m = Phenomenological(PLANCK18)
    g = Grid.mcq(n_mc=20, n_q=6, n_z=20)
    f = jax.jit(lambda theta: m.total_rate(m.unpack(theta), g))
    theta = m.pack(m.params())
    grad = jax.grad(f)(theta)
    assert float(grad[0]) == pytest.approx(np.log(10.0) * float(f(theta)), rel=1e-10)   # d/d log10 n0


def test_cosmology_enters_through_the_volume_element():
    p = FIDUCIAL
    r_p = Phenomenological(PLANCK18).intensity_at(p, 6.0, 0.5, 2.0)
    r_c = Phenomenological(CONCORDANCE).intensity_at(p, 6.0, 0.5, 2.0)
    expected = (PLANCK18.dVc_dz(2.0) / CONCORDANCE.dVc_dz(2.0))
    assert float(r_p / r_c) == pytest.approx(float(expected), rel=1e-12)


# --------------------------------------------------------------------- tabulated
def test_tabulated_rows_conserve_rate():
    n = 5000
    rows = (rng.uniform(3.2, 7.8, n), rng.uniform(0.0, 1.0, n), rng.uniform(0.1, 9.9, n), rng.uniform(0, 1e-3, n))
    rows[1][:50] = 0.005          # below the grid's q range: placed in the first q bin
    rows[1][50:60] = 1.0          # q = 1 sits past geomspace's last node: placed in the last bin
    t = Tabulated.from_rows(*rows)
    g = Grid.mcq(n_mc=50, n_q=12, n_z=60, log10_mc=(3.0, 8.0), z=(0.05, 10.0))
    assert float(np.sum(np.asarray(g.W) * t.intensity(None, g))) == pytest.approx(rows[3].sum(), rel=1e-12)
    assert t.total_rate() == pytest.approx(rows[3].sum())


def test_tabulated_table_conserves_its_integral():
    a = (np.linspace(3, 9, 25), np.linspace(0.01, 1, 9), np.linspace(0.05, 15, 30))
    L, Q, Z = np.meshgrid(*a, indexing="ij")
    lam = np.exp(-(L - 6) ** 2) * Q ** -0.3 * Z ** 2 * np.exp(-Z / 2)
    t = Tabulated.from_table(*a, lam)
    native = float(np.sum(np.prod(np.meshgrid(*(np.asarray(trapezoid_weights(x)) for x in a), indexing="ij"),
                                  axis=0) * lam))
    g = Grid.mcq(n_mc=40, n_q=24, n_z=50, log10_mc=(2.5, 9.5), z=(0.05, 20.0))
    assert float(np.sum(np.asarray(g.W) * t.intensity(None, g))) == pytest.approx(native, rel=1e-6)
    assert t.total_rate() == pytest.approx(native, rel=1e-12)
    with pytest.raises(NotImplementedError):
        t.intensity_at(None, 6.0, 0.5, 1.0)


# --------------------------------------------------------------------- presets
def test_presets():
    assert set(list_presets()) >= {"fiducial", "caliskan25_light", "caliskan25_heavy", "caliskan25_ultralight"}
    assert get_preset("fiducial").params == FIDUCIAL
    g = Grid.mcq(n_mc=400, n_q=48, n_z=400, log10_mc=(2.3, 9.2), z=(0.0, 20.5))
    for name in ("caliskan25_light", "caliskan25_heavy", "caliskan25_ultralight"):
        pre = get_preset(name)
        assert float(pre.model().total_rate(pre.params, g)) == pytest.approx(200.0, rel=1e-2), name
    with pytest.raises(KeyError):
        get_preset("nope")
