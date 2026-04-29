"""Semi-analytic SMBHB population modeling built around one concrete model."""

import secrets

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from scipy.integrate import quad, nquad
from scipy.interpolate import CubicSpline

from .constants import (
    GMKS,
    Mmax,
    Mmin,
    MsunMKS,
    TNG15,
    cMKS,
    default_Mstar,
    default_alphaM,
    default_betaz,
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
    Dc_interp,
    EE,
    dVcdz,
    dtodz,
)
from .healpix_backend import accumulate_skymap_batch, init_skymaps, nside2npix, require_backend
from .units import hc_to_omega, omega_to_hc

jax.config.update("jax_enable_x64", True)


def _coerce_key(key):
    """Normalize an optional seed or JAX key into a PRNG key."""
    if key is None:
        return jax.random.PRNGKey(secrets.randbits(32))
    if isinstance(key, int):
        return jax.random.PRNGKey(key)
    return key


@jit
def dlnfdtr(M, f, z):
    """Compute the frequency evolution term d(ln f)/dt_r."""
    return (96 / 5) * jnp.pi ** (8 / 3) * (M * GMKS) ** (5 / 3) * (f * (1 + z)) ** (8 / 3) * cMKS**(-5)


@jit
def h(M, f, z):
    """Compute the gravitational wave strain amplitude."""
    return (
        (4 * jnp.pi**(2/3))
        * (GMKS * M)**(5/3)
        / (cMKS**4 * Dc_interp(z))
        * (f * (1+z))**(2/3)
    )


@jit
def h_average(M, f, z):
    """Compute the sky and polarization averaged strain amplitude."""
    return (
        (8 * jnp.pi**(2/3) / jnp.sqrt(10))
        * (GMKS * M)**(5/3)
        / (cMKS**4 * Dc_interp(z))
        * (f * (1+z))**(2/3)
    )

def draw_parameters(param_ranges=None, key=None):
    """Draw one random parameter set from posterior ranges using JAX RNG."""
    if param_ranges is None:
        param_ranges = {
            "n0": [10**(-7.82) / ((1e6 * pc * pcinMKS)**3 * (1e9 * yr * yrinMKS)),
                   10**(0.96) / ((1e6 * pc * pcinMKS)**3 * (1e9 * yr * yrinMKS))],
            "alphaM": [-2.75, 2.71],
            "Mstar": [10**(6.14) * MsunMKS, 10**(8.84) * MsunMKS],
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
    """Compute strain values for sampled masses, redshifts, and frequencies."""
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
    h_vals = h(distM * kg, f_vals / s, distz)
    h2_vals = f_vals * h_vals**2

    # Use JAX's digitize for binning
    bin_indices = jnp.digitize(f_vals, bin_edges) - 1
    nbins = len(bin_edges) - 1

    # Sum values in each bin using segment_sum
    binned_sum = jnp.zeros(nbins)
    binned_sum = binned_sum.at[bin_indices].add(h2_vals)

    return binned_sum


def binning(distM, distz, distlog10f, freqs=None, hc2_values=None, do_plot=True):
    """
    Bin the sampled sources into frequency bins and compute spectrum.

    This is a standalone function that doesn't depend on population parameters.

    Parameters
    ----------
    distM : array
        Sampled masses (not in kg units)
    distz : array
        Sampled redshifts
    distlog10f : array
        Sampled log10 frequencies
    freqs : array, optional
        Reference frequencies for plotting
    hc2_values : array, optional
        Reference h_c values for plotting
    do_plot : bool, optional
        Whether to generate diagnostic plot

    Returns
    -------
    Spec : jax.Array
        Array of shape (nbins, 2) containing [log10(f), f*h^2]
    """
    nbins = 14

    bin_edges_linear = jnp.array(
        [[(2 * i + 1) * fminNG15 * s, (2 * i + 3) * fminNG15 * s] for i in range(nbins)]
    )
    bin_edges_log10 = jnp.log10(bin_edges_linear)
    bin_edges = jnp.concatenate((jnp.array([bin_edges_linear[0, 0]]), bin_edges_linear[:, 1]))
    bin_centers_log10 = jnp.log10((10**bin_edges_log10[:, 0] + 10**bin_edges_log10[:, 1]) / 2)

    binned_sum = jnp.asarray(
        binning_jitted(
            jnp.asarray(distM),
            jnp.asarray(distz),
            jnp.asarray(distlog10f),
            bin_edges,
        )
    )

    Spec_y = binned_sum / (2 * fminNG15 * s)
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
    """
    Semi-analytical population model for supermassive black hole binaries.

    This class encapsulates all population-dependent calculations and provides
    methods for sampling sources, computing spectra, and generating skymaps.

    Parameters
    ----------
    n0 : float, optional
        Normalization parameter for the population distribution
    alphaM : float, optional
        Mass power-law index
    Mstar : float, optional
        Characteristic mass scale in kg
    betaz : float, optional
        Redshift power-law index
    z0 : float, optional
        Characteristic redshift scale
    """

    def __init__(self, population_params=None,
                 integration_limits=None,
                 sampling_grids=None,
                 PTA_params=None):
        """Initialize the population model with specified parameters."""
        population_params = {} if population_params is None else population_params
        integration_limits = {} if integration_limits is None else integration_limits
        sampling_grids = {} if sampling_grids is None else sampling_grids
        PTA_params = {} if PTA_params is None else PTA_params

        # Set population parameters
        self.n0 = population_params.get('n0', default_n0)
        self.alphaM = population_params.get('alphaM', default_alphaM)
        self.Mstar = population_params.get('Mstar', default_Mstar)
        self.betaz = population_params.get('betaz', default_betaz)
        self.z0 = population_params.get('z0', default_z0)

        # Set integration limits
        self.Mbounds = integration_limits.get('Mbounds', [Mmin/kg, Mmax/kg])
        self.zbounds = integration_limits.get('zbounds', [zmin, zmax])
        self.fbounds = integration_limits.get('fbounds', [fminNG15*s, fmaxNG15*s])

        # Set sampling grids
        self.Mgrid = sampling_grids.get('Mgrid', 
                                        np.geomspace(Mmin/kg, Mmax/kg, 3000))
        self.zgrid = sampling_grids.get('zgrid', 
                                        np.linspace(zmin, zmax, 1500))
        self.fgrid = sampling_grids.get('fgrid', 
                                        np.geomspace(fminNG15*s, fmaxNG15*s, 2000))
        
        # Check that the bounds and the grids cover the same ranges, otherwise warn the user
        if (self.Mbounds[0] != self.Mgrid[0]) or (self.Mbounds[1] != self.Mgrid[-1]):
            print("Warning: Mass bounds and mass grid do not cover the same range.")
        if (self.zbounds[0] != self.zgrid[0]) or (self.zbounds[1] != self.zgrid[-1]):
            print("Warning: Redshift bounds and redshift grid do not cover the same range.")
        if (self.fbounds[0] != self.fgrid[0]) or (self.fbounds[1] != self.fgrid[-1]):
            print("Warning: Frequency bounds and frequency grid do not cover the same range.")

        # Set PTA parameters
        self.Tobs = PTA_params.get('Tobs', TNG15)  # Observation time in seconds
        self.fmin = PTA_params.get('fmin', fminNG15)  # Minimum frequency in Hz
        self.fmax = PTA_params.get('fmax', fmaxNG15)  # Maximum frequency in Hz
        self.Nfreqs = PTA_params.get('Nfreqs', 14)  # Number of frequency bins
        self.PTA_frequencies = np.geomspace(self.fmin, self.fmax, self.Nfreqs)
        
        # Initialize mean number of binaries
        self.Nbinaries_mean = None  # To be set after integration

    def d2ndzdM(self, z, M):
        """
        Compute the differential number density d²n/(dz dM).

        Parameters
        ----------
        z : float or array
            Redshift
        M : float or array
            Mass in kg

        Returns
        -------
        float or array
            Differential number density
        """
        return (
            (1.0 / (M * jnp.log(10)))
            * self.n0
            * ((M / (1e7 * MsunMKS)) ** (-self.alphaM) * jnp.exp(-M / self.Mstar))
            * (((1 + z) ** self.betaz) * jnp.exp(-z / self.z0))
            * dtodz(z)
        )

    def dndlog10M(self, M, zmin=0, zmax=5):
        """
        Compute dn/dlog10(M) by integrating over redshift.

        Parameters
        ----------
        M : float
            Mass in kg
        zmin, zmax : float, optional
            Redshift integration bounds

        Returns
        -------
        float
            Number density in Mpc^-3
        """
        integrand = lambda z: M * jnp.log(10) * self.d2ndzdM(z, M)
        result, _ = quad(integrand, zmin, zmax)
        return (1e6 * pc * pcinMKS)**3 * result

    def d3ndzdMdlnf(self, M, f, z):
        """
        Compute the 3D differential number density d³n/(dz dM d ln f).

        Parameters
        ----------
        M : float or array
            Mass in kg
        f : float or array
            Frequency in Hz
        z : float or array
            Redshift

        Returns
        -------
        float or array
            3D differential number density
        """
        return (
            self.d2ndzdM(z, M)
            * dlnfdtr(M, f, z)**(-1)
            * (dtodz(z))**(-1)
            * dVcdz(z)
        )

    def _integrand(self, M, z, f):
        """Compute the integrand for the characteristic strain spectrum."""
        prefactor = (4 * GMKS ** (5 / 3)) / (3 * jnp.pi ** (1 / 3) * cMKS**2)
        return prefactor * f**(-4/3) * (1 + z)**(-1/3) * M**(5/3) * self.d2ndzdM(z, M * kg)

    def _integrand_log(self, x, z, f):
        """Compute the integrand in log10(M) space."""
        M = 10**x
        return self._integrand(M, z, f) * M * jnp.log(10)

    def hc2(self, ff):
        """
        Compute the characteristic strain squared h_c^2(f).

        Parameters
        ----------
        ff : float
            Frequency in Hz

        Returns
        -------
        float
            Characteristic strain squared
        """
        x_min = np.log10(Mmin)
        x_max = np.log10(Mmax)

        def integrand_nquad(z, x):
            return self._integrand_log(x, z, ff)

        result, _ = nquad(
            integrand_nquad,
            [[zmin, zmax], [x_min, x_max]],
            opts={'epsabs':1e-8, 'epsrel':1e-6}
        )
        return result

    def compute_Nbinaries(self, Mbounds=None, zbounds=None, fbounds=None,
                          verbose=False):
        """
        Integrate to find the total number of binaries.

        Parameters
        ----------
        Mass_bounds : list of 2 floats, optional
            [M_min, M_max] in kg
        redshift_bounds : list of 2 floats, optional
            [z_min, z_max]
        frequency_bounds : list of 2 floats, optional
            [f_min, f_max] in Hz

        Returns
        -------
        float
            Total number of binaries
        """
        Mbounds = self.Mbounds if Mbounds is None else Mbounds
        zbounds = self.zbounds if zbounds is None else zbounds
        fbounds = self.fbounds if fbounds is None else fbounds

        def integrand(log10f, z, M):
            f = 10**log10f / s
            return jnp.log(10) * kg * self.d3ndzdMdlnf(M * kg, f, z)

        result, _ = nquad(
            integrand,
            [np.log10(fbounds), zbounds, Mbounds],
            opts={"epsabs": 1e-6, "epsrel": 1e-4}
        )
        self.Nbinaries_mean = result
        if verbose:
            print(f"[compute_Nbinaries] Total number of binaries: {result:.3e}")
        return result
    
    def generate_poisson_realization(self, Nrealizations=1, Nbinaries_mean=None, key=None, verbose=False):
        """
        Generate a Poisson realization of the number of binaries.

        Parameters
        ----------
        Nrealizations : int, optional
            Number of realizations to generate
        Nbinaries_mean : int, optional
            Mean (expectation) value of the number of binaries. If None, uses self.Nbinaries_mean
        key : jax.random.PRNGKey or int, optional
            Random key or seed for reproducibility. If None, a fresh
            nondeterministic key is created internally.

        Returns
        -------
        int or array of int
            Realized number of binaries
        """
        if Nbinaries_mean is None:
            if self.Nbinaries_mean is None:
                raise ValueError("[generate_poisson_realization] Nbinaries_mean is not set. Please compute it first.")
            Nbinaries_mean = self.Nbinaries_mean

        realized_Nbinaries = jax.random.poisson(
            _coerce_key(key),
            lam=Nbinaries_mean,
            shape=(Nrealizations,),
        )
        if verbose:
            print(f"[generate_poisson_realization] Realized number of binaries: {realized_Nbinaries}")
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

        pdfM = kg * self.d3ndzdMdlnf(Mgrid * kg, 10**-8.75/s, 1)
        splineM = CubicSpline(Mgrid, pdfM)
        normM = splineM.integrate(Mgrid[0], Mgrid[-1])
        pdfM /= normM

        cdfM = np.cumsum(pdfM * np.gradient(Mgrid))
        cdfM /= cdfM[-1]

        ############################
        # 2. REDSHIFT DISTRIBUTION #
        ############################
        zgrid = self.zgrid if zgrid is None else zgrid

        pdfz = 1e80 * kg * self.d3ndzdMdlnf(1e9 * MsunMKS, 10**-8.75/s, zgrid)

        splineZ = CubicSpline(zgrid, pdfz)
        normZ = splineZ.integrate(zgrid[0], zgrid[-1])
        pdfz /= normZ

        cdfz = np.cumsum(pdfz)
        cdfz /= cdfz[-1]

        #############################
        # 3. FREQUENCY DISTRIBUTION #
        #############################
        fgrid = self.fgrid if fgrid is None else fgrid

        pdflogf = (np.log(10)) * kg * self.d3ndzdMdlnf(1e9 * MsunMKS, fgrid/s, 1)

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
        Mgrid_jax, cdfM_jax, zgrid_jax, cdfz_jax, fgrid_log10_jax, cdflogf_jax = distributions
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
        phi = jax.random.uniform(key_phi, shape=(current_batch_size,), minval=0, maxval=2 * jnp.pi)
        cos_theta = jax.random.uniform(key_theta, shape=(current_batch_size,), minval=-1, maxval=1)
        theta = jnp.arccos(cos_theta)
        iota = jnp.arccos(
            jax.random.uniform(key_iota, shape=(current_batch_size,), minval=-1, maxval=1)
        )
        phi0 = jax.random.uniform(key_phi0, shape=(current_batch_size,), minval=0, maxval=2 * jnp.pi)
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

    def sample_dist(self, Nbinaries=None, Mgrid=None, zgrid=None, fgrid=None, do_plot=False, key=None):
        """
        Sample mass, redshift, and frequency distributions for a population of binaries.

        Parameters
        ----------
        Nbinaries : int, optional
            Number of binaries to sample. If None, uses self.Nbinaries_mean
        do_plot : bool, optional
            Whether to generate diagnostic plots
        Mgrid : array, optional
            Custom mass grid for sampling
        zgrid : array, optional
            Custom redshift grid for sampling
        fgrid : array, optional
            Custom frequency grid for sampling
        key : jax.random.PRNGKey or int, optional
            JAX random key or integer seed for reproducibility. If None, a
            fresh nondeterministic key is created internally.

        Returns
        -------
        distM : jax.Array
            Sampled masses (not in kg units)
        distz : jax.Array
            Sampled redshifts
        distlog10f : jax.Array
            Sampled log10 frequencies
        """
        if Nbinaries is None:
            if self.Nbinaries_mean is None:
                raise ValueError("[sample_dist] Nbinaries_mean is not set. Please compute it first.")
            Nbinaries = int(self.Nbinaries_mean)

        key = _coerce_key(key)

        distributions = self._prepare_sampling_distributions(Mgrid, zgrid, fgrid)
        distM, distz, distlog10f = self._sample_batch_from_distributions(Nbinaries, key, distributions)

        if do_plot:
            Mgrid_jax, _, zgrid_jax, _, fgrid_log10_jax, _ = distributions
            # Recompute PDFs for plotting
            Mgrid_plot = np.asarray(Mgrid_jax)
            zgrid_plot = np.asarray(zgrid_jax)
            fgrid_log10_plot = np.asarray(fgrid_log10_jax)

            pdfM_plot = kg * self.d3ndzdMdlnf(Mgrid_plot * kg, 10**-8.75/s, 1)
            pdfM_plot /= np.trapezoid(pdfM_plot, Mgrid_plot)

            pdfz_plot = 1e80 * kg * self.d3ndzdMdlnf(1e9 * MsunMKS, 10**-8.75/s, zgrid_plot)
            pdfz_plot /= np.trapezoid(pdfz_plot, zgrid_plot)

            pdflogf_plot = (np.log(10)) * kg * self.d3ndzdMdlnf(1e9 * MsunMKS, 10**fgrid_log10_plot/s, 1)
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

    def generate_skymaps(self, Nbinaries=None, PTA_frequencies=None,
                                 Mgrid=None, zgrid=None, fgrid=None,
                                 Nside=16,
                                 batch_size=int(1e7), verbose=False, key=None):
        """
        Generate HEALPix skymaps by processing sources in batches.

        This method is memory-efficient as it only keeps the skymaps in memory,
        not all the sampled sources and amplitudes. It processes sources in batches,
        computing their sky positions, orientations, and amplitudes, then adding them
        to the skymaps before discarding the batch.

        Parameters
        ----------
        Nbinaries : int, optional
            Total number of binaries to generate. Default: self.Nbinaries_mean
        PTA_frequencies : array, optional
            Array of frequency bin centers in Hz. Default: self.PTA_frequencies
        Mgrid : array, optional
            Custom mass grid for sampling
        zgrid : array, optional
            Custom redshift grid for sampling
        fgrid : array, optional
            Custom frequency grid for sampling
        Nside : int, optional
            HEALPix resolution parameter. Default: 16
        batch_size : int, optional
            Number of sources to process per batch. Default: 10000000
        verbose : bool, optional
            Whether to print progress messages. Default: False
        key : jax.random.PRNGKey or int, optional
            JAX random key or integer seed for reproducibility. If None, a
            fresh nondeterministic key is created internally.

        Returns
        -------
        skymaps_tot : numpy.ndarray
            Combined skymaps, shape (2, npix, n_frequencies)
        skymaps_plus : numpy.ndarray
            Plus polarization skymaps, shape (npix, n_frequencies)
        skymaps_cross : numpy.ndarray
            Cross polarization skymaps, shape (npix, n_frequencies)
        """
        if Nbinaries is None:
            if self.Nbinaries_mean is None:
                raise ValueError("[generate_skymaps] Nbinaries_mean is not set. Please compute it first.")
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
            print(f"[generate_skymaps] Generating skymaps with {Nbinaries} sources in {num_batches} batches")
            print(f"[generate_skymaps] Nside={Nside}, npix={npix}, n_frequencies={n_freq}")

        # Prepare sampling distributions once (for efficiency)
        distributions = self._prepare_sampling_distributions(Mgrid, zgrid, fgrid)

        # Process in batches
        sources_processed = 0
        for batch_idx in range(num_batches):
            # Determine batch size
            current_batch_size = min(batch_size, Nbinaries - sources_processed)

            if verbose:
                print(f"[generate_skymaps] Batch {batch_idx + 1}/{num_batches}: Processing {current_batch_size} sources...")

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
            hbar_plus_rot, hbar_cross_rot = self._compute_rotated_polarizations(h_vals, iota, phi0, psi)

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
                print(f"[generate_skymaps] Total sources processed: {sources_processed}/{Nbinaries}")

        # Combine into final format
        skymaps_plus = jnp.asarray(skymaps_plus)
        skymaps_cross = jnp.asarray(skymaps_cross)
        skymaps_tot = jnp.stack((skymaps_plus, skymaps_cross), axis=0)

        if verbose:
            print("[generate_skymaps] Skymap generation complete!")

        return skymaps_tot, skymaps_plus, skymaps_cross

    def compute_many_realizations(self, Nbinaries, nrealizations=100, freqs=None,
                                  hc2_values=None, do_plot=True, key=None):
        """
        Compute many Monte Carlo realizations of the GW spectrum.

        Parameters
        ----------
        Nbinaries : int
            Number of binaries per realization
        nrealizations : int, optional
            Number of realizations to perform
        freqs : array, optional
            Reference frequencies for plotting
        hc2_values : array, optional
            Reference h_c values for plotting
        do_plot : bool, optional
            If True, plot ensemble average, individual realizations, and 68% interval
        key : jax.random.PRNGKey or int, optional
            Random key or seed for reproducibility. If None, a fresh
            nondeterministic key is created internally.

        Returns
        -------
        tabreal : jax.Array
            Array of shape (nrealizations, nbins, 2): [log10(f), log10(sqrt(f h^2))]
        log10f : jax.Array
            Log10 of frequency bin centers
        median : jax.Array
            Median of realizations at each frequency
        q_low : jax.Array
            Lower 68% quantile
        q_high : jax.Array
            Upper 68% quantile
        """
        key = _coerce_key(key)
        tabreal = []

        for _ in range(nrealizations):
            key, key_poisson, key_sample = jax.random.split(key, 3)
            Nbinaries_sample = int(jax.random.poisson(key_poisson, lam=Nbinaries))
            distM, distz, distlog10f = self.sample_dist(Nbinaries_sample, do_plot=False, key=key_sample)
            Spec = binning(distM, distz, distlog10f, do_plot=False)
            Spec_transformed = jnp.column_stack((Spec[:, 0], jnp.log10(jnp.sqrt(Spec[:, 1]))))
            tabreal.append(Spec_transformed)

        tabreal = jnp.stack(tabreal)

        log10f = tabreal[0, :, 0]
        yvals = tabreal[:, :, 1]

        median = jnp.median(yvals, axis=0)
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
