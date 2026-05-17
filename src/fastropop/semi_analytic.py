"""Semi-analytic SMBHB population modeling built around one concrete model."""

import secrets

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from scipy.integrate import nquad, quad
from scipy.interpolate import CubicSpline

from .constants import (
    GMKS,
    TNG15,
    Mmax,
    Mmin,
    MsunMKS,
    cMKS,
    default_alphaM,
    default_betaz,
    default_Mstar,
    default_n0,
    default_z0,
    fmaxNG15,
    fminNG15,
    kg,
    pc,
    pcinMKS,
    s,
    yr,
    yrinMKS,
    zmax,
    zmin,
)
from .cosmology import (
    DL,
    EE,
    Dc_interp,
    Dc_interp_numpy,
    H0s,
    OmegaDM,
    Omegak,
    OmegaLambda,
    dtodz,
    dVcdz,
)
from .healpix_backend import accumulate_skymap_batch, init_skymaps, nside2npix, require_backend
from .units import hc_to_omega, omega_to_hc

jax.config.update("jax_enable_x64", True)


def _coerce_key(key):
    """Normalize an optional seed or JAX key into a usable PRNG key."""
    if key is None:
        return jax.random.PRNGKey(secrets.randbits(32))
    if isinstance(key, int):
        return jax.random.PRNGKey(key)
    return key


@jit
def dlnfdtr(M, f, z):
    r"""
    Compute the GW-driven frequency evolution \(d\ln f_r / dt_r\).

    The semi-analytic model assumes binaries evolve only through quadrupolar
    gravitational-wave emission. In that limit,

    \[
    \frac{d \ln f_r}{dt_r} =
    \frac{96}{5}\pi^{8/3}
    \frac{(G \mathcal{M})^{5/3}}{c^5}
    f_r^{8/3},
    \]

    where \(\mathcal{M}\) is the chirp mass and
    \(f_r = (1+z) f\) is the rest-frame GW frequency.

    Parameters
    ----------
    M : float or jax.Array
        Chirp mass in kilograms.
    f : float or jax.Array
        Observer-frame GW frequency in Hz.
    z : float or jax.Array
        Redshift.

    Returns
    -------
    float or jax.Array
        Frequency-evolution term in \({\rm s^{-1}}\).
    """
    return (
        (96 / 5)
        * jnp.pi ** (8 / 3)
        * (M * GMKS) ** (5 / 3)
        * (f * (1 + z)) ** (8 / 3)
        * cMKS ** (-5)
    )


@jit
def h(M, f, z):
    r"""
    Compute the source strain amplitude without angular averaging.

    The normalization used throughout the package is

    \[
    h(\mathcal{M}, f, z) =
    \frac{4\pi^{2/3}(G\mathcal{M})^{5/3}}{c^4 D_c(z)}
    \left[f(1+z)\right]^{2/3},
    \]

    with \(D_c(z)\) the comoving distance. This quantity is appropriate
    for source-level calculations before averaging over sky position and binary
    orientation.
    """
    return (
        (4 * jnp.pi ** (2 / 3))
        * (GMKS * M) ** (5 / 3)
        / (cMKS**4 * Dc_interp(z))
        * (f * (1 + z)) ** (2 / 3)
    )


@jit
def h_average(M, f, z):
    r"""
    Compute the sky-and-orientation averaged strain amplitude.

    `fastropop` uses the stochastic-background convention

    \[
    \bar h(\mathcal{M}, f, z) = \sqrt{\frac{2}{5}}\, h(\mathcal{M}, f, z),
    \]

    which is the amplitude entering the Monte Carlo binning of unresolved
    populations.
    """
    return (
        (8 * jnp.pi ** (2 / 3) / jnp.sqrt(10))
        * (GMKS * M) ** (5 / 3)
        / (cMKS**4 * Dc_interp(z))
        * (f * (1 + z)) ** (2 / 3)
    )


def draw_parameters(param_ranges=None, key=None):
    """
    Draw one random semi-analytic population parameter set.

    This helper samples each model parameter independently from a broad range
    motivated by the original semi-analytic model literature and related
    posterior explorations. It is mainly intended for toy scans, demonstrations,
    and stochastic parameter studies rather than for calibrated inference.

    Parameters
    ----------
    param_ranges : dict, optional
        Mapping from parameter name to ``[min, max]`` interval. The parameters
        are ``n0``, ``alphaM``, ``Mstar``, ``betaz``, and ``z0``.
        ``n0`` and ``Mstar`` are sampled uniformly in log-space; the others are
        sampled uniformly in linear space.
    key : jax.random.PRNGKey or int, optional
        Random key or integer seed. If ``None``, a fresh nondeterministic key
        is generated internally.

    Returns
    -------
    dict
        Dictionary containing one sampled parameter set with scalar Python
        values ready to pass into `SemiAnalyticPopulation`.
    """
    if param_ranges is None:
        param_ranges = {
            "n0": [
                10 ** (-7.82) / ((1e6 * pc * pcinMKS) ** 3 * (1e9 * yr * yrinMKS)),
                10 ** (0.96) / ((1e6 * pc * pcinMKS) ** 3 * (1e9 * yr * yrinMKS)),
            ],
            "alphaM": [-2.75, 2.71],
            "Mstar": [10 ** (6.14) * MsunMKS, 10 ** (8.84) * MsunMKS],
            "betaz": [-1.57, 6.51],
            "z0": [0.44, 4.77],
        }

    key = _coerce_key(key)
    key_n0, key_mstar, key_alpha, key_beta, key_z0 = jax.random.split(key, 5)

    n0_min, n0_max = param_ranges["n0"]
    Mstar_min, Mstar_max = param_ranges["Mstar"]
    alphaM_min, alphaM_max = param_ranges["alphaM"]
    betaz_min, betaz_max = param_ranges["betaz"]
    z0_min, z0_max = param_ranges["z0"]

    log10_n0 = jax.random.uniform(
        key_n0,
        minval=jnp.log10(n0_min),
        maxval=jnp.log10(n0_max),
    )
    log10_Mstar = jax.random.uniform(
        key_mstar,
        minval=jnp.log10(Mstar_min),
        maxval=jnp.log10(Mstar_max),
    )

    n0 = 10**log10_n0
    Mstar = 10**log10_Mstar

    alphaM = jax.random.uniform(key_alpha, minval=alphaM_min, maxval=alphaM_max)
    betaz = jax.random.uniform(key_beta, minval=betaz_min, maxval=betaz_max)
    z0 = jax.random.uniform(key_z0, minval=z0_min, maxval=z0_max)

    return {
        "n0": float(n0),
        "alphaM": float(alphaM),
        "Mstar": float(Mstar),
        "betaz": float(betaz),
        "z0": float(z0),
    }


@jit
def compute_h_jitted(distM, distz, f_obs):
    """
    JIT-compiled version to compute strain values.

    Parameters
    ----------
    distM : jax array
        Mass values (not in kg units)
    distz : jax array
        Redshift values
    f_obs : float or jax array
        Observed frequency in Hz

    Returns
    -------
    h_vals : jax array
        Computed strain values
    """
    return h(distM * kg, f_obs, distz)


def compute_h(distM, distz, f_obs):
    """
    Compute source strain amplitudes for sampled masses and redshifts.

    Parameters
    ----------
    distM : array-like
        Sampled chirp masses in the sampling convention of the package, i.e.
        masses expressed in units of kilograms divided by `fastropop.kg`.
    distz : array-like
        Sampled redshifts.
    f_obs : float or array-like
        Observer-frame GW frequency in Hz.

    Returns
    -------
    jax.Array
        Source strain amplitudes evaluated with `h`.
    """
    return compute_h_jitted(jnp.asarray(distM), jnp.asarray(distz), f_obs)


@jit
def binning_jitted(distM, distz, distlog10f, bin_edges):
    """
    JIT-compiled binning function.

    Parameters
    ----------
    distM : jax array
        Sampled masses (not in kg units)
    distz : jax array
        Sampled redshifts
    distlog10f : jax array
        Sampled log10 frequencies
    bin_edges : jax array
        Bin edges

    Returns
    -------
    binned_sum : jax array
        Sum of f*h^2 in each bin
    """
    f_vals = 10**distlog10f
    h_vals = h_average(distM * kg, f_vals / s, distz)
    h2_vals = f_vals * h_vals**2

    # Use JAX's digitize for binning
    bin_indices = jnp.digitize(f_vals, bin_edges) - 1
    nbins = len(bin_edges) - 1

    # Sum values in each bin using segment_sum
    binned_sum = jnp.zeros(nbins)
    binned_sum = binned_sum.at[bin_indices].add(h2_vals)

    return binned_sum


def binning(distM, distz, distlog10f, freqs=None, hc2_values=None, do_plot=True):
    r"""
    Bin one Monte Carlo realization into PTA-style frequency bins.

    For each sampled source, the code computes the averaged strain
    \(\bar h\), forms \(f \bar h^2\), assigns the source to a PTA
    frequency bin, and sums the contributions in that bin. The returned
    spectrum therefore represents the discretized realization of the stochastic
    background before taking the final square root for plotting as
    \(h_c(f)\).

    Parameters
    ----------
    distM : array-like
        Sampled chirp masses in package sampling units (``kg``-scaled).
    distz : array-like
        Sampled redshifts.
    distlog10f : array-like
        Sampled \(\log_{10} f\) values, with \(f\) in Hz.
    freqs : array-like, optional
        Smooth reference frequency grid for the optional diagnostic plot.
    hc2_values : array-like, optional
        Smooth reference characteristic-strain curve for the optional plot.
    do_plot : bool, optional
        If ``True``, call `fastropop.plot_binned_spectrum`.

    Returns
    -------
    jax.Array
        Array of shape ``(n_bins, 2)`` with columns
        ``[log10(f_bin_center), summed_bin_value]``.
    """
    nbins = 28

    bin_edges_linear = jnp.array(
        [[(2 * i + 1) * (fminNG15/2) * s, (2 * i + 3) * fminNG15/2 * s] for i in range(nbins)]
    )
    bin_edges_log10 = jnp.log10(bin_edges_linear)
    bin_edges = jnp.concatenate((jnp.array([bin_edges_linear[0, 0]]), bin_edges_linear[:, 1]))
    bin_centers_log10 = jnp.log10((10 ** bin_edges_log10[:, 0] + 10 ** bin_edges_log10[:, 1]) / 2)

    binned_sum = jnp.asarray(
        binning_jitted(
            jnp.asarray(distM),
            jnp.asarray(distz),
            jnp.asarray(distlog10f),
            bin_edges,
        )
    )

    Spec_y = binned_sum / (fminNG15 * s)
    Spec_x = bin_centers_log10
    Spec = jnp.column_stack((Spec_x, Spec_y))

    if do_plot:
        from .plots import plot_binned_spectrum

        plot_binned_spectrum(Spec, freqs=freqs, hc2_values=hc2_values)

    return Spec


# =============================================================================
# Main Population Class
# =============================================================================


class SemiAnalyticPopulation:
    r"""
    Semi-analytic population model for supermassive black hole binaries.

    This is the main high-level interface of :mod:`fastropop`. It wraps the
    phenomenological merger-rate density, the cosmology helpers, the
    gravitational-wave background integrals, and the Monte Carlo sampling tools
    used to realize discrete SMBHB populations.

    The underlying population ansatz is

    \[
    \frac{d^2 n}{dz\,d\mathcal{M}}
    \propto
    \frac{1}{\mathcal{M}\ln 10}
    \left(\frac{\mathcal{M}}{10^7 M_\odot}\right)^{-\alpha_M}
    e^{-\mathcal{M}/\mathcal{M}_*}
    (1+z)^{\beta_z} e^{-z/z_0}
    \frac{dt_r}{dz},
    \]

    with free parameters ``n0``, ``alphaM``, ``Mstar``, ``betaz``, and ``z0``.
    Public methods then derive quantities such as \(h_c^2(f)\), the expected
    number of binaries, Monte Carlo source realizations, and skymaps.
    """

    def __init__(
        self, population_params=None, integration_limits=None, sampling_grids=None, PTA_params=None
    ):
        """
        Initialize one semi-analytic population model.

        Parameters
        ----------
        population_params : dict, optional
            Population-law parameters. Supported keys are ``n0``, ``alphaM``,
            ``Mstar``, ``betaz``, and ``z0``.
        integration_limits : dict, optional
            Integration bounds with optional keys ``Mbounds``, ``zbounds``,
            and ``fbounds``.
        sampling_grids : dict, optional
            Sampling grids with optional keys ``Mgrid``, ``zgrid``, and
            ``fgrid``. These control the inverse-CDF sampling resolution.
        PTA_params : dict, optional
            PTA frequency-grid configuration with optional keys ``Tobs``,
            ``fmin``, ``fmax``, and ``Nfreqs``.
        """
        population_params = {} if population_params is None else population_params
        integration_limits = {} if integration_limits is None else integration_limits
        sampling_grids = {} if sampling_grids is None else sampling_grids
        PTA_params = {} if PTA_params is None else PTA_params

        # Set population parameters
        self.n0 = population_params.get("n0", default_n0)
        self.alphaM = population_params.get("alphaM", default_alphaM)
        self.Mstar = population_params.get("Mstar", default_Mstar)
        self.betaz = population_params.get("betaz", default_betaz)
        self.z0 = population_params.get("z0", default_z0)

        # Set integration limits
        self.Mbounds = integration_limits.get("Mbounds", [Mmin / kg, Mmax / kg])
        self.zbounds = integration_limits.get("zbounds", [zmin, zmax])
        self.fbounds = integration_limits.get("fbounds", [fminNG15 * s, fmaxNG15 * s])

        # Set sampling grids
        self.Mgrid = sampling_grids.get("Mgrid", np.geomspace(Mmin / kg, Mmax / kg, 3000))
        self.zgrid = sampling_grids.get("zgrid", np.linspace(zmin, zmax, 1500))
        self.fgrid = sampling_grids.get("fgrid", np.geomspace(fminNG15 * s, fmaxNG15 * s, 2000))

        # Check that the bounds and the grids cover the same ranges, otherwise warn the user
        if (self.Mbounds[0] != self.Mgrid[0]) or (self.Mbounds[1] != self.Mgrid[-1]):
            print("Warning: Mass bounds and mass grid do not cover the same range.")
        if (self.zbounds[0] != self.zgrid[0]) or (self.zbounds[1] != self.zgrid[-1]):
            print("Warning: Redshift bounds and redshift grid do not cover the same range.")
        if (self.fbounds[0] != self.fgrid[0]) or (self.fbounds[1] != self.fgrid[-1]):
            print("Warning: Frequency bounds and frequency grid do not cover the same range.")

        # Set PTA parameters
        self.Tobs = PTA_params.get("Tobs", TNG15)  # Observation time in seconds
        self.fmin = PTA_params.get("fmin", fminNG15)  # Minimum frequency in Hz
        self.fmax = PTA_params.get("fmax", fmaxNG15)  # Maximum frequency in Hz
        self.Nfreqs = PTA_params.get("Nfreqs", 14)  # Number of frequency bins
        self.PTA_frequencies = np.geomspace(self.fmin, self.fmax, self.Nfreqs)

        # Initialize mean number of binaries
        self.Nbinaries_mean = None  # To be set after integration

    def d2ndzdM(self, z, M):
        r"""
        Compute the differential merger-rate density \(d^2n/(dz\,d\mathcal{M})\).

        This is the core semi-analytic population law used throughout the
        package:

        \[
        \frac{d^2 n}{dz\,d\mathcal{M}}
        =
        \frac{n_0}{\mathcal{M}\ln 10}
        \left(\frac{\mathcal{M}}{10^7 M_\odot}\right)^{-\alpha_M}
        e^{-\mathcal{M}/\mathcal{M}_*}
        (1+z)^{\beta_z}
        e^{-z/z_0}
        \frac{dt_r}{dz}.
        \]

        Parameters
        ----------
        z : float or array-like
            Redshift.
        M : float or array-like
            Chirp mass in kilograms.

        Returns
        -------
        float or jax.Array
            Differential number density in SI-based units consistent with the
            rest of the package.
        """
        return (
            (1.0 / (M * jnp.log(10)))
            * self.n0
            * ((M / (1e7 * MsunMKS)) ** (-self.alphaM) * jnp.exp(-M / self.Mstar))
            * (((1 + z) ** self.betaz) * jnp.exp(-z / self.z0))
            * dtodz(z)
        )

    def dndlog10M(self, M, zmin=0, zmax=5):
        r"""
        Integrate the population over redshift to obtain \(dn/d\log_{10}\mathcal{M}\).

        Parameters
        ----------
        M : float
            Chirp mass in kilograms.
        zmin : float, optional
            Lower redshift integration bound.
        zmax : float, optional
            Upper redshift integration bound.

        Returns
        -------
        float
            Mass function in \({\rm Mpc^{-3}}\).
        """

        def integrand(z):
            return M * np.log(10) * self._d2ndzdM_numpy(z, M)

        result, _ = quad(integrand, zmin, zmax)
        return (1e6 * pc * pcinMKS) ** 3 * result

    def d3ndzdMdlnf(self, M, f, z):
        r"""
        Compute \(d^3n/(dz\,d\mathcal{M}\,d\ln f)\).

        The extra frequency dimension is obtained by converting the population
        into time spent per logarithmic frequency interval:

        \[
        \frac{d^3 n}{dz\,d\mathcal{M}\,d\ln f}
        =
        \frac{d^2 n}{dz\,d\mathcal{M}}
        \left(\frac{d\ln f_r}{dt_r}\right)^{-1}
        \left(\frac{dt_r}{dz}\right)^{-1}
        \frac{dV_c}{dz}.
        \]
        """
        return self.d2ndzdM(z, M) * dlnfdtr(M, f, z) ** (-1) * (dtodz(z)) ** (-1) * dVcdz(z)

    def _integrand(self, M, z, f):
        """Compute the integrand for the characteristic strain spectrum."""
        prefactor = (4 * GMKS ** (5 / 3)) / (3 * jnp.pi ** (1 / 3) * cMKS**2)
        return (
            prefactor * f ** (-4 / 3) * (1 + z) ** (-1 / 3) * M ** (5 / 3) * self.d2ndzdM(z, M * kg)
        )

    def _integrand_log(self, x, z, f):
        """Compute the integrand in log10(M) space."""
        M = 10**x
        return self._integrand(M, z, f) * M * jnp.log(10)

    def _dtodz_numpy(self, z):
        """NumPy scalar helper for SciPy integration paths."""
        Ez = np.sqrt(OmegaDM * (1.0 + z) ** 3.0 + Omegak * (1.0 + z) ** 2.0 + OmegaLambda)
        return 1.0 / (H0s * (1.0 + z) * Ez)

    def _dVcdz_numpy(self, z):
        """NumPy scalar helper for SciPy integration paths."""
        Ez = np.sqrt(OmegaDM * (1.0 + z) ** 3.0 + Omegak * (1.0 + z) ** 2.0 + OmegaLambda)
        return ((4.0 * np.pi * cMKS) / H0s) * Dc_interp_numpy(z) ** 2 / Ez

    def _d2ndzdM_numpy(self, z, M):
        """NumPy scalar helper for SciPy integration paths."""
        return (
            (1.0 / (M * np.log(10)))
            * self.n0
            * ((M / (1e7 * MsunMKS)) ** (-self.alphaM) * np.exp(-M / self.Mstar))
            * (((1 + z) ** self.betaz) * np.exp(-z / self.z0))
            * self._dtodz_numpy(z)
        )

    def _dlnfdtr_numpy(self, M, f, z):
        """NumPy scalar helper for SciPy integration paths."""
        return (
            (96 / 5)
            * np.pi ** (8 / 3)
            * (M * GMKS) ** (5 / 3)
            * (f * (1 + z)) ** (8 / 3)
            * cMKS ** (-5)
        )

    def _d3ndzdMdlnf_numpy(self, M, f, z):
        """NumPy scalar helper for SciPy integration paths."""
        return (
            self._d2ndzdM_numpy(z, M)
            * self._dlnfdtr_numpy(M, f, z) ** (-1)
            * self._dtodz_numpy(z) ** (-1)
            * self._dVcdz_numpy(z)
        )

    def _integrand_numpy(self, M, z, f):
        """NumPy scalar helper for characteristic-strain integration."""
        prefactor = (4 * GMKS ** (5 / 3)) / (3 * np.pi ** (1 / 3) * cMKS**2)
        return (
            prefactor
            * f ** (-4 / 3)
            * (1 + z) ** (-1 / 3)
            * M ** (5 / 3)
            * self._d2ndzdM_numpy(z, M * kg)
        )

    def _integrand_log_numpy(self, x, z, f):
        """NumPy scalar helper in log10(M) space."""
        M = 10**x
        return self._integrand_numpy(M, z, f) * M * np.log(10)

    def hc2(self, ff):
        r"""
        Compute the ensemble characteristic strain squared \(h_c^2(f)\).

        This method evaluates the smooth background integral implied by the
        semi-analytic population model. In the current implementation the
        integration is performed with SciPy using NumPy scalar helpers for
        efficiency.

        Parameters
        ----------
        ff : float
            Observer-frame frequency in Hz.

        Returns
        -------
        float
            Ensemble characteristic strain squared at frequency ``ff``.
        """
        x_min = np.log10(Mmin)
        x_max = np.log10(Mmax)

        def integrand_nquad(z, x):
            return self._integrand_log_numpy(x, z, ff)

        result, _ = nquad(
            integrand_nquad, [[zmin, zmax], [x_min, x_max]], opts={"epsabs": 1e-8, "epsrel": 1e-6}
        )
        return result

    def compute_Nbinaries(self, Mbounds=None, zbounds=None, fbounds=None, verbose=False):
        r"""
        Integrate the expected number of binaries in the chosen domain.

        The integral is carried out over chirp mass, redshift, and logarithmic
        frequency:

        \[
        \langle N \rangle =
        \int d\mathcal{M}\,dz\,d\log_{10}f\;
        \ln(10)\,\mathcal{M}_{\rm unit}\,
        \frac{d^3 n}{dz\,d\mathcal{M}\,d\ln f}.
        \]

        Parameters
        ----------
        Mbounds : sequence of float, optional
            Mass bounds ``[M_min, M_max]`` in the package sampling convention.
        zbounds : sequence of float, optional
            Redshift bounds ``[z_min, z_max]``.
        fbounds : sequence of float, optional
            Frequency bounds ``[f_min, f_max]`` in SI-scaled package units.
        verbose : bool, optional
            If ``True``, print the resulting mean binary count.

        Returns
        -------
        float
            Expected number of binaries in the specified domain.
        """
        Mbounds = self.Mbounds if Mbounds is None else Mbounds
        zbounds = self.zbounds if zbounds is None else zbounds
        fbounds = self.fbounds if fbounds is None else fbounds

        def integrand(log10f, z, M):
            f = 10**log10f / s
            return np.log(10) * kg * self._d3ndzdMdlnf_numpy(M * kg, f, z)

        result, _ = nquad(
            integrand,
            [np.log10(fbounds), zbounds, Mbounds],
            opts={"epsabs": 1e-6, "epsrel": 1e-4},
        )
        self.Nbinaries_mean = result
        if verbose:
            print(f"[compute_Nbinaries] Total number of binaries: {result:.3e}")
        return result

    def generate_poisson_realization(
        self,
        Nrealizations=1,
        Nbinaries_mean=None,
        key=None,
        verbose=False,
    ):
        """
        Draw Poisson realizations of the binary count.

        Parameters
        ----------
        Nrealizations : int, optional
            Number of Poisson draws to generate.
        Nbinaries_mean : int or float, optional
            Mean count used as the Poisson expectation value. If omitted,
            ``self.Nbinaries_mean`` is used.
        key : jax.random.PRNGKey or int, optional
            Random key or seed for reproducibility. If ``None``, a fresh
            nondeterministic key is created internally.
        verbose : bool, optional
            If ``True``, print the realized counts.

        Returns
        -------
        jax.Array
            Integer-valued array of shape ``(Nrealizations,)``.
        """
        if Nbinaries_mean is None:
            if self.Nbinaries_mean is None:
                raise ValueError(
                    "[generate_poisson_realization] Nbinaries_mean is not set. "
                    "Please compute it first."
                )
            Nbinaries_mean = self.Nbinaries_mean

        realized_Nbinaries = jax.random.poisson(
            _coerce_key(key),
            lam=Nbinaries_mean,
            shape=(Nrealizations,),
        )
        if verbose:
            print(
                f"[generate_poisson_realization] Realized number of binaries: {realized_Nbinaries}"
            )
        return realized_Nbinaries

    def _prepare_sampling_distributions(self, Mgrid=None, zgrid=None, fgrid=None):
        """
        Prepare CDFs for sampling (called once, then cached).

        Returns
        -------
        tuple
            (Mgrid, cdfM, zgrid, cdfz, fgrid_log10, cdflogf)
        """
        ########################
        # 1. MASS DISTRIBUTION #
        #########################
        Mgrid = self.Mgrid if Mgrid is None else Mgrid

        pdfM = kg * self.d3ndzdMdlnf(Mgrid * kg, 10**-8.75 / s, 1)
        splineM = CubicSpline(Mgrid, pdfM)
        normM = splineM.integrate(Mgrid[0], Mgrid[-1])
        pdfM /= normM

        cdfM = np.cumsum(pdfM * np.gradient(Mgrid))
        cdfM /= cdfM[-1]

        ############################
        # 2. REDSHIFT DISTRIBUTION #
        ############################
        zgrid = self.zgrid if zgrid is None else zgrid

        pdfz = 1e80 * kg * self.d3ndzdMdlnf(1e9 * MsunMKS, 10**-8.75 / s, zgrid)

        splineZ = CubicSpline(zgrid, pdfz)
        normZ = splineZ.integrate(zgrid[0], zgrid[-1])
        pdfz /= normZ

        cdfz = np.cumsum(pdfz)
        cdfz /= cdfz[-1]

        #############################
        # 3. FREQUENCY DISTRIBUTION #
        #############################
        fgrid = self.fgrid if fgrid is None else fgrid

        pdflogf = (np.log(10)) * kg * self.d3ndzdMdlnf(1e9 * MsunMKS, fgrid / s, 1)

        splineF = CubicSpline(np.log10(fgrid), pdflogf)
        normF = splineF.integrate(np.log10(fgrid[0]), np.log10(fgrid[-1]))
        pdflogf /= normF

        cdflogf = np.cumsum(pdflogf)
        cdflogf /= cdflogf[-1]

        return (
            jnp.asarray(Mgrid),
            jnp.asarray(cdfM),
            jnp.asarray(zgrid),
            jnp.asarray(cdfz),
            jnp.asarray(np.log10(fgrid)),
            jnp.asarray(cdflogf),
        )

    def _sample_batch_from_distributions(self, current_batch_size, key_sample, distributions):
        """Sample one batch of masses, redshifts, and frequencies from cached CDFs."""
        (
            Mgrid_jax,
            cdfM_jax,
            zgrid_jax,
            cdfz_jax,
            fgrid_log10_jax,
            cdflogf_jax,
        ) = distributions
        key_M, key_z, key_f = jax.random.split(key_sample, 3)
        uM = jax.random.uniform(key_M, shape=(current_batch_size,))
        uz = jax.random.uniform(key_z, shape=(current_batch_size,))
        ulogf = jax.random.uniform(key_f, shape=(current_batch_size,))
        distM = jnp.interp(uM, cdfM_jax, Mgrid_jax)
        distz = jnp.interp(uz, cdfz_jax, zgrid_jax)
        distlog10f = jnp.interp(ulogf, cdflogf_jax, fgrid_log10_jax)
        return distM, distz, distlog10f

    def _sample_sky_orientations(self, current_batch_size, key_sky):
        """Sample isotropic sky positions and source orientations."""
        key_phi, key_theta, key_iota, key_phi0, key_psi = jax.random.split(key_sky, 5)
        phi = jax.random.uniform(
            key_phi,
            shape=(current_batch_size,),
            minval=0,
            maxval=2 * jnp.pi,
        )
        cos_theta = jax.random.uniform(key_theta, shape=(current_batch_size,), minval=-1, maxval=1)
        theta = jnp.arccos(cos_theta)
        iota = jnp.arccos(
            jax.random.uniform(key_iota, shape=(current_batch_size,), minval=-1, maxval=1)
        )
        phi0 = jax.random.uniform(
            key_phi0,
            shape=(current_batch_size,),
            minval=0,
            maxval=2 * jnp.pi,
        )
        psi = jax.random.uniform(key_psi, shape=(current_batch_size,), minval=0, maxval=jnp.pi)
        return theta, phi, iota, phi0, psi

    def _compute_rotated_polarizations(self, h_vals, iota, phi0, psi):
        """Compute plus/cross polarizations and rotate them into the Earth frame."""
        hbar_plus = h_vals * (1 + jnp.cos(iota) ** 2) / 2 * jnp.exp(1j * phi0)
        hbar_cross = 1j * h_vals * jnp.cos(iota) * jnp.exp(1j * phi0)
        hbar_plus_rot = hbar_plus * jnp.cos(2 * psi) - hbar_cross * jnp.sin(2 * psi)
        hbar_cross_rot = hbar_plus * jnp.sin(2 * psi) + hbar_cross * jnp.cos(2 * psi)
        return hbar_plus_rot, hbar_cross_rot

    def _assign_frequency_bins(self, f_vals, PTA_frequencies_jax):
        """Assign sampled frequencies to the nearest PTA frequency bin."""
        return jnp.argmin(jnp.abs(PTA_frequencies_jax[None, :] - f_vals[:, None]), axis=1)

    def _accumulate_skymap_batch(
        self,
        skymaps_plus,
        skymaps_cross,
        theta,
        phi,
        freq_indices,
        hbar_plus_rot,
        hbar_cross_rot,
        Nside,
        n_freq,
    ):
        """Accumulate one batch into the active HEALPix backend."""
        del n_freq
        return accumulate_skymap_batch(
            skymaps_plus,
            skymaps_cross,
            Nside,
            theta,
            phi,
            freq_indices,
            hbar_plus_rot,
            hbar_cross_rot,
        )

    def sample_dist(
        self,
        Nbinaries=None,
        Mgrid=None,
        zgrid=None,
        fgrid=None,
        do_plot=False,
        key=None,
    ):
        r"""
        Sample masses, redshifts, and frequencies from the semi-analytic population.

        The sampler builds one-dimensional inverse-CDF approximations for the
        mass, redshift, and logarithmic-frequency distributions and then draws
        ``Nbinaries`` independent sources.

        Parameters
        ----------
        Nbinaries : int, optional
            Number of binaries to sample. If omitted, the method uses
            ``int(self.Nbinaries_mean)``.
        Mgrid : array-like, optional
            Custom mass grid used to build the inverse-CDF sampler.
        zgrid : array-like, optional
            Custom redshift grid used to build the inverse-CDF sampler.
        fgrid : array-like, optional
            Custom frequency grid used to build the inverse-CDF sampler.
        do_plot : bool, optional
            If ``True``, produce diagnostic sampling plots.
        key : jax.random.PRNGKey or int, optional
            JAX random key or integer seed for reproducibility. If ``None``, a
            fresh nondeterministic key is created internally.

        Returns
        -------
        tuple of jax.Array
            ``(distM, distz, distlog10f)`` with shapes ``(Nbinaries,)``.
            ``distM`` is returned in the package sampling mass convention,
            ``distz`` in redshift, and ``distlog10f`` in \(\log_{10} f\).
        """
        if Nbinaries is None:
            if self.Nbinaries_mean is None:
                raise ValueError(
                    "[sample_dist] Nbinaries_mean is not set. Please compute it first."
                )
            Nbinaries = int(self.Nbinaries_mean)

        key = _coerce_key(key)

        distributions = self._prepare_sampling_distributions(Mgrid, zgrid, fgrid)
        distM, distz, distlog10f = self._sample_batch_from_distributions(
            Nbinaries,
            key,
            distributions,
        )

        if do_plot:
            Mgrid_jax, _, zgrid_jax, _, fgrid_log10_jax, _ = distributions
            # Recompute PDFs for plotting
            Mgrid_plot = np.asarray(Mgrid_jax)
            zgrid_plot = np.asarray(zgrid_jax)
            fgrid_log10_plot = np.asarray(fgrid_log10_jax)

            pdfM_plot = kg * self.d3ndzdMdlnf(Mgrid_plot * kg, 10**-8.75 / s, 1)
            pdfM_plot /= np.trapezoid(pdfM_plot, Mgrid_plot)

            pdfz_plot = 1e80 * kg * self.d3ndzdMdlnf(1e9 * MsunMKS, 10**-8.75 / s, zgrid_plot)
            pdfz_plot /= np.trapezoid(pdfz_plot, zgrid_plot)

            pdflogf_plot = (
                (np.log(10))
                * kg
                * self.d3ndzdMdlnf(
                    1e9 * MsunMKS,
                    10**fgrid_log10_plot / s,
                    1,
                )
            )
            pdflogf_plot /= np.trapezoid(pdflogf_plot, fgrid_log10_plot)

            from .plots import plot_sample_distributions

            plot_sample_distributions(
                distM,
                distz,
                distlog10f,
                Mgrid_plot,
                pdfM_plot,
                zgrid_plot,
                pdfz_plot,
                fgrid_log10_plot,
                pdflogf_plot,
            )

        return distM, distz, distlog10f

    def generate_skymaps(
        self,
        Nbinaries=None,
        PTA_frequencies=None,
        Mgrid=None,
        zgrid=None,
        fgrid=None,
        Nside=16,
        batch_size=int(1e7),
        verbose=False,
        key=None,
    ):
        """
        Generate HEALPix skymaps from a sampled realization.

        Sources are sampled in batches, assigned isotropic sky positions and
        binary orientations, converted into plus and cross polarizations, and
        accumulated into HEALPix maps frequency bin by frequency bin.

        Parameters
        ----------
        Nbinaries : int, optional
            Total number of binaries to realize. If omitted, the method uses
            ``int(self.Nbinaries_mean)``.
        PTA_frequencies : array-like, optional
            Observer-frame PTA frequency-bin centers in Hz.
        Mgrid : array-like, optional
            Custom mass grid used to build the inverse-CDF sampler.
        zgrid : array-like, optional
            Custom redshift grid used to build the inverse-CDF sampler.
        fgrid : array-like, optional
            Custom frequency grid used to build the inverse-CDF sampler.
        Nside : int, optional
            HEALPix resolution parameter.
        batch_size : int, optional
            Number of sources processed per batch.
        verbose : bool, optional
            If ``True``, print batch progress information.
        key : jax.random.PRNGKey or int, optional
            JAX random key or integer seed for reproducibility. If ``None``, a
            fresh nondeterministic key is created internally.

        Returns
        -------
        tuple of jax.Array
            ``(skymaps_tot, skymaps_plus, skymaps_cross)`` where
            ``skymaps_tot`` has shape ``(2, npix, n_frequencies)`` and the
            polarization-resolved maps have shape ``(npix, n_frequencies)``.
        """
        if Nbinaries is None:
            if self.Nbinaries_mean is None:
                raise ValueError(
                    "[generate_skymaps] Nbinaries_mean is not set. Please compute it first."
                )
            Nbinaries = int(self.Nbinaries_mean)
        key = _coerce_key(key)
        require_backend()

        PTA_frequencies = self.PTA_frequencies if PTA_frequencies is None else PTA_frequencies
        PTA_frequencies_jax = jnp.array(PTA_frequencies)
        npix = nside2npix(Nside)
        n_freq = len(PTA_frequencies)

        # Initialize empty skymaps
        skymaps_plus, skymaps_cross = init_skymaps(npix, n_freq)

        # Calculate number of batches
        num_batches = int(np.ceil(Nbinaries / batch_size))

        if verbose:
            print(
                "[generate_skymaps] Generating skymaps with "
                f"{Nbinaries} sources in {num_batches} batches"
            )
            print(f"[generate_skymaps] Nside={Nside}, npix={npix}, n_frequencies={n_freq}")

        # Prepare sampling distributions once (for efficiency)
        distributions = self._prepare_sampling_distributions(Mgrid, zgrid, fgrid)

        # Process in batches
        sources_processed = 0
        for batch_idx in range(num_batches):
            # Determine batch size
            current_batch_size = min(batch_size, Nbinaries - sources_processed)

            if verbose:
                print(
                    f"[generate_skymaps] Batch {batch_idx + 1}/{num_batches}: "
                    f"Processing {current_batch_size} sources..."
                )

            key, key_sample, key_sky = jax.random.split(key, 3)

            distM, distz, distlog10f = self._sample_batch_from_distributions(
                current_batch_size,
                key_sample,
                distributions,
            )

            # 2. Compute frequencies and amplitudes using JIT
            f_vals = 10**distlog10f
            h_vals = compute_h_jitted(distM, distz, f_vals / s)

            # 3. Generate random sky positions and orientations
            theta, phi, iota, phi0, psi = self._sample_sky_orientations(current_batch_size, key_sky)

            # 4. Compute polarization amplitudes (JIT-compiled operations)
            hbar_plus_rot, hbar_cross_rot = self._compute_rotated_polarizations(
                h_vals,
                iota,
                phi0,
                psi,
            )

            # 5. Assign to frequency bins and add to skymaps
            freq_indices = self._assign_frequency_bins(f_vals, PTA_frequencies_jax)
            skymaps_plus, skymaps_cross = self._accumulate_skymap_batch(
                skymaps_plus,
                skymaps_cross,
                theta,
                phi,
                freq_indices,
                hbar_plus_rot,
                hbar_cross_rot,
                Nside,
                n_freq,
            )

            sources_processed += current_batch_size

            if verbose:
                print(
                    f"[generate_skymaps] Total sources processed: {sources_processed}/{Nbinaries}"
                )

        # Combine into final format
        skymaps_plus = jnp.asarray(skymaps_plus)
        skymaps_cross = jnp.asarray(skymaps_cross)
        skymaps_tot = jnp.stack((skymaps_plus, skymaps_cross), axis=0)

        if verbose:
            print("[generate_skymaps] Skymap generation complete!")

        return skymaps_tot, skymaps_plus, skymaps_cross

    def compute_many_realizations(
        self,
        Nbinaries_mean,
        nrealizations=100,
        freqs=None,
        hc2_values=None,
        do_plot=True,
        key=None,
    ):
        """
        Generate many Monte Carlo realizations of the binned GW spectrum.

        Each realization first draws a Poisson-distributed binary count with
        mean ``Nbinaries_mean``, then samples a discrete population and bins it
        into PTA-style frequency bins.

        Parameters
        ----------
        Nbinaries_mean : int or float
            Poisson mean used for the source-count draw in each realization.
        nrealizations : int, optional
            Number of independent realizations to generate.
        freqs : array-like, optional
            Optional smooth comparison frequency grid in Hz.
        hc2_values : array-like, optional
            Optional smooth comparison strain curve passed to the plotting
            helper.
        do_plot : bool, optional
            If ``True``, visualize the ensemble of realizations together with
            the median and central interval.
        key : jax.random.PRNGKey or int, optional
            Random key or seed for reproducibility. If ``None``, a fresh
            nondeterministic key is created internally.

        Returns
        -------
        tuple
            ``(tabreal, log10f, median, q_low, q_high)`` where ``tabreal`` has
            shape ``(nrealizations, n_bins, 2)`` and stores the transformed
            spectra used for plotting.
        """
        key = _coerce_key(key)
        tabreal = []

        for _ in range(nrealizations):
            key, key_poisson, key_sample = jax.random.split(key, 3)
            Nbinaries_sample = int(jax.random.poisson(key_poisson, lam=Nbinaries_mean))
            distM, distz, distlog10f = self.sample_dist(
                Nbinaries_sample,
                do_plot=False,
                key=key_sample,
            )
            Spec = binning(distM, distz, distlog10f, do_plot=False)
            Spec_transformed = jnp.column_stack((Spec[:, 0], jnp.log10(jnp.sqrt(Spec[:, 1]))))
            tabreal.append(Spec_transformed)

        tabreal = jnp.stack(tabreal)

        log10f = tabreal[0, :, 0]
        yvals = tabreal[:, :, 1]

        median = jnp.mean(yvals, axis=0)
        q_low = jnp.quantile(yvals, (1 - 0.68) / 2, axis=0)
        q_high = jnp.quantile(yvals, (1 + 0.68) / 2, axis=0)

        if do_plot:
            from .plots import plot_realizations

            plot_realizations(
                log10f,
                yvals,
                median,
                q_low,
                q_high,
                freqs=freqs,
                hc2_values=hc2_values,
            )

        return tabreal, log10f, median, q_low, q_high


__all__ = [
    "DL",
    "Dc_interp",
    "EE",
    "GMKS",
    "Mmax",
    "Mmin",
    "MsunMKS",
    "SemiAnalyticPopulation",
    "TNG15",
    "binning",
    "cMKS",
    "compute_h",
    "default_Mstar",
    "default_alphaM",
    "default_betaz",
    "default_n0",
    "default_z0",
    "draw_parameters",
    "fmaxNG15",
    "fminNG15",
    "h",
    "h_average",
    "hc_to_omega",
    "kg",
    "omega_to_hc",
    "pc",
    "pcinMKS",
    "s",
    "yr",
    "yrinMKS",
    "zmax",
    "zmin",
]
