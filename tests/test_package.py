import jax
import jax.numpy as jnp
import pytest
from importlib.util import find_spec

import fastropop
from fastropop import cosmology
from fastropop import healpix_backend
from fastropop import plots
from fastropop import units
from fastropop.semi_analytic import (
    SemiAnalyticPopulation,
    binning,
    compute_h,
    dlnfdtr,
    h,
    draw_parameters,
    h_average,
)

HAS_HEALPY = find_spec("healpy") is not None
HAS_JAX_HEALPY = find_spec("jax_healpy") is not None
HAS_ANY_HEALPIX = HAS_HEALPY or HAS_JAX_HEALPY


def test_package_version() -> None:
    assert fastropop.__version__ == "0.1.0"


def test_public_api_lists_semi_analytic_entry_point() -> None:
    assert "SemiAnalyticPopulation" in fastropop.__all__
    assert "seed_both_rngs" not in fastropop.__all__
    assert "jax_to_numpy_random" not in fastropop.__all__


def test_cosmology_module_keeps_constants_separate() -> None:
    assert "GMKS" not in cosmology.__all__
    assert "kg" not in cosmology.__all__


def test_units_round_trip_conversion() -> None:
    hc = 1e-15
    f = 1e-8
    omega = float(units.hc_to_omega(hc, f))
    recovered_hc = float(units.omega_to_hc(omega, f))
    assert recovered_hc == hc


def test_plots_module_exports_helpers() -> None:
    assert "plot_binned_spectrum" in plots.__all__
    assert "plot_sample_distributions" in plots.__all__
    assert "plot_realizations" in plots.__all__


def test_healpix_backend_selection() -> None:
    if HAS_JAX_HEALPY:
        assert healpix_backend.BACKEND_NAME == "jax-healpy"
    elif HAS_HEALPY:
        assert healpix_backend.BACKEND_NAME == "healpy"
    else:
        assert healpix_backend.BACKEND_NAME is None


def test_draw_parameters_accepts_none_and_explicit_key() -> None:
    params_default = draw_parameters()
    params_keyed = draw_parameters(key=0)
    assert params_default.keys() == params_keyed.keys()
    assert params_keyed["n0"] > 0
    assert params_keyed["Mstar"] > 0


def test_compute_h_returns_jax_array() -> None:
    hvals = compute_h(jnp.array([1.0, 2.0]), jnp.array([0.1, 0.2]), 1e-8)
    assert isinstance(hvals, jax.Array)
    assert hvals.shape == (2,)


def test_sample_dist_returns_jax_arrays_with_finite_values() -> None:
    pop = SemiAnalyticPopulation(
        sampling_grids={
            "Mgrid": jnp.geomspace(fastropop.Mmin / fastropop.kg, fastropop.Mmax / fastropop.kg, 32),
            "zgrid": jnp.linspace(fastropop.zmin, fastropop.zmax, 24),
            "fgrid": jnp.geomspace(fastropop.fminNG15 * fastropop.s, fastropop.fmaxNG15 * fastropop.s, 32),
        }
    )
    distM, distz, distlog10f = pop.sample_dist(Nbinaries=8, key=0)
    assert isinstance(distM, jax.Array)
    assert isinstance(distz, jax.Array)
    assert isinstance(distlog10f, jax.Array)
    assert distM.shape == (8,)
    assert distz.shape == (8,)
    assert distlog10f.shape == (8,)
    assert jnp.all(jnp.isfinite(distM))
    assert jnp.all(jnp.isfinite(distz))
    assert jnp.all(jnp.isfinite(distlog10f))


def test_binning_returns_jax_array() -> None:
    distM = jnp.array([1.0, 1.2, 1.4])
    distz = jnp.array([0.1, 0.2, 0.3])
    distlog10f = jnp.array([-8.8, -8.7, -8.6])
    spec = binning(distM, distz, distlog10f, do_plot=False)
    assert isinstance(spec, jax.Array)
    assert spec.ndim == 2
    assert spec.shape[1] == 2


def test_binning_uses_sky_averaged_strain_normalization() -> None:
    distM = jnp.array([1.0e8])
    distz = jnp.array([0.5])
    f_obs = 2.0 * fastropop.fminNG15 * fastropop.s
    distlog10f = jnp.array([jnp.log10(f_obs)])

    spec = binning(distM, distz, distlog10f, do_plot=False)

    expected = f_obs * h_average(distM[0] * fastropop.kg, f_obs / fastropop.s, distz[0]) ** 2
    expected /= 2.0 * fastropop.fminNG15 * fastropop.s
    wrong = f_obs * h(distM[0] * fastropop.kg, f_obs / fastropop.s, distz[0]) ** 2
    wrong /= 2.0 * fastropop.fminNG15 * fastropop.s

    nonzero_bins = jnp.nonzero(spec[:, 1] > 0, size=1, fill_value=-1)[0]
    assert int(nonzero_bins[0]) == 0
    assert jnp.allclose(spec[0, 1], expected)
    assert jnp.isclose(spec[0, 1] / wrong, expected / wrong)
    assert not jnp.isclose(spec[0, 1] / wrong, 1.0)
    assert jnp.all(spec[1:, 1] == 0)


def test_notebook_reference_formulas_match_package() -> None:
    params = {
        "n0": 10**-90.4153,
        "alphaM": -1.3800,
        "Mstar": 10 ** 8.8272 * fastropop.MsunMKS,
        "betaz": -0.1711,
        "z0": 4.70,
    }
    pop = SemiAnalyticPopulation(population_params=params)

    z = 1.0
    M = 1e8 * fastropop.MsunMKS
    f = 1e-8

    expected_d2 = (
        (1.0 / (M * jnp.log(10)))
        * params["n0"]
        * (M / (1e7 * fastropop.MsunMKS)) ** (-params["alphaM"])
        * jnp.exp(-M / params["Mstar"])
        * (1 + z) ** params["betaz"]
        * jnp.exp(-z / params["z0"])
        * cosmology.dtodz(z)
    )
    expected_h = (
        (8 * jnp.pi ** (2 / 3) / jnp.sqrt(10))
        * (fastropop.GMKS * M) ** (5 / 3)
        / (fastropop.cMKS**4 * cosmology.Dc_interp(z))
        * (f * (1 + z)) ** (2 / 3)
    )
    expected_d3 = expected_d2 * dlnfdtr(M, f, z) ** (-1) * cosmology.dtodz(z) ** (-1) * cosmology.dVcdz(z)

    assert jnp.allclose(pop.d2ndzdM(z, M), expected_d2)
    assert jnp.allclose(h_average(M, f, z), expected_h)
    assert jnp.allclose(pop.d3ndzdMdlnf(M, f, z), expected_d3)


@pytest.mark.skipif(not HAS_ANY_HEALPIX, reason="no HEALPix backend installed")
def test_generate_skymaps_is_deterministic_with_fixed_key_and_valid_with_none() -> None:
    pop = SemiAnalyticPopulation(
        sampling_grids={
            "Mgrid": jnp.geomspace(fastropop.Mmin / fastropop.kg, fastropop.Mmax / fastropop.kg, 32),
            "zgrid": jnp.linspace(fastropop.zmin, fastropop.zmax, 24),
            "fgrid": jnp.geomspace(fastropop.fminNG15 * fastropop.s, fastropop.fmaxNG15 * fastropop.s, 32),
        },
        PTA_params={"Nfreqs": 4},
    )
    skymaps_1 = pop.generate_skymaps(Nbinaries=6, Nside=1, batch_size=3, key=0)
    skymaps_2 = pop.generate_skymaps(Nbinaries=6, Nside=1, batch_size=3, key=0)
    for arr_1, arr_2 in zip(skymaps_1, skymaps_2):
        assert isinstance(jnp.asarray(arr_1), jax.Array)
        assert arr_1.shape == arr_2.shape
        assert jnp.allclose(jnp.asarray(arr_1), jnp.asarray(arr_2))

    random_skymaps = pop.generate_skymaps(Nbinaries=4, Nside=1, batch_size=2, key=None)
    assert random_skymaps[0].shape[0] == 2
    assert random_skymaps[1].shape == random_skymaps[2].shape


@pytest.mark.skipif(not HAS_ANY_HEALPIX, reason="no HEALPix backend installed")
def test_generate_skymaps_small_batch_pipeline_shapes() -> None:
    pop = SemiAnalyticPopulation(
        sampling_grids={
            "Mgrid": jnp.geomspace(fastropop.Mmin / fastropop.kg, fastropop.Mmax / fastropop.kg, 16),
            "zgrid": jnp.linspace(fastropop.zmin, fastropop.zmax, 16),
            "fgrid": jnp.geomspace(fastropop.fminNG15 * fastropop.s, fastropop.fmaxNG15 * fastropop.s, 16),
        },
        PTA_params={"Nfreqs": 3},
    )
    skymaps_tot, skymaps_plus, skymaps_cross = pop.generate_skymaps(
        Nbinaries=3,
        Nside=1,
        batch_size=2,
        key=1,
    )
    assert skymaps_tot.shape[0] == 2
    assert skymaps_plus.shape == skymaps_cross.shape
    assert skymaps_tot.shape[1:] == skymaps_plus.shape
