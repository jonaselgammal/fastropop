# Quickstart

The core workflow is built around `SemiAnalyticPopulation`. In practice, you usually want to do three things:

- choose the semi-analytic population parameters
- compute the expected number of binaries in the PTA band
- generate and bin one or more Monte Carlo realizations

## Define a Population Model

The main model parameters are:

- \(n_0\): merger-rate normalization
- \(\alpha_M\): mass power-law slope
- \(M_\star\): exponential cutoff mass
- \(\beta_z\): redshift power-law slope
- \(z_0\): exponential cutoff redshift

Here is a minimal setup using the parameter choice from the Poisson notebook:

```python
import jax.numpy as jnp
import fastropop

population_params = {
    "n0": 10**-90.4153,
    "alphaM": -1.3800,
    "Mstar": 10**8.8272 * fastropop.MsunMKS,
    "betaz": -0.1711,
    "z0": 4.70,
}

sampling_grids = {
    "Mgrid": jnp.geomspace(fastropop.Mmin / fastropop.kg, fastropop.Mmax / fastropop.kg, 300),
    "zgrid": jnp.linspace(fastropop.zmin, fastropop.zmax, 300),
    "fgrid": jnp.geomspace(fastropop.fminNG15 * fastropop.s, fastropop.fmaxNG15 * fastropop.s, 300),
}

PTA_params = {
    "Tobs": fastropop.TNG15,
    "fmin": fastropop.fminNG15,
    "fmax": fastropop.fmaxNG15,
    "Nfreqs": 14,
}

pop = fastropop.SemiAnalyticPopulation(
    population_params=population_params,
    sampling_grids=sampling_grids,
    PTA_params=PTA_params,
)
```

## Compute the Expected Number of Binaries

The expected source count in the chosen mass, redshift, and frequency range is:

```python
Nbinaries_mean = pop.compute_Nbinaries()
```

## Draw One Realization and Bin It

To generate one Monte Carlo population, first draw a Poisson realization of the total binary count and then sample masses, redshifts, and frequencies:

```python
Nbinaries = int(pop.generate_poisson_realization(Nrealizations=1, Nbinaries_mean=Nbinaries_mean, key=0)[0])

distM, distz, distlog10f = pop.sample_dist(Nbinaries=Nbinaries, key=1)
spec = fastropop.binning(distM, distz, distlog10f, do_plot=False)
```

Here:

- `distM` contains sampled chirp masses in the mass convention used by the sampling API
- `distz` contains sampled redshifts
- `distlog10f` contains sampled \(\log_{10} f\) values
- `spec` is a two-column array containing the PTA-bin centers and the corresponding \(h_c^2\)-style binned quantity

## Generate Several Realizations

For repeated realizations, you can either loop manually or use the built-in helper:

```python
tabreal, log10f, median, q_low, q_high = pop.compute_many_realizations(
    Nbinaries_mean=Nbinaries_mean,
    nrealizations=10,
    freqs=freqs,
    hc2_values=hc2_values,
    key=2,
    do_plot=False,
)
```

This is the easiest way to estimate the realization-to-realization scatter of the spectrum.

!!! note
    `sample_dist(...)` does **not** perform a Poisson draw by itself. It samples exactly the number of binaries you pass in via `Nbinaries`.

    By contrast, `compute_many_realizations(...)` **does** Poisson-sample the binary count internally for each realization, using `Nbinaries_mean` as the mean.

## Reproducibility

For stochastic methods, pass either:

- a JAX PRNG key
- an integer seed

For example:

```python
params = fastropop.draw_parameters(key=3)
Nbinaries_realization = pop.generate_poisson_realization(Nrealizations=1, key=4)
```

If `key=None`, `fastropop` creates a fresh nondeterministic JAX key internally.

## Generate and Plot a Skymap

If you have a HEALPix backend installed, you can also generate one skymap realization and visualize a frequency slice:

```python
Nbinaries_skymap = min(int(Nbinaries_mean), 100000)

skymaps_tot, skymaps_plus, skymaps_cross = pop.generate_skymaps(
    Nbinaries=Nbinaries_skymap,
    Nside=16,
    batch_size=100000,
    key=5,
)

fastropop.plot_skymap(
    skymaps_tot,
    freq_index=0,
    polarization="total",
    log_scale=True,
    title="Total skymap, first PTA frequency bin",
)
```

## Next Step

For more complete worked examples corresponding to the workflows on this page, see:

- `examples/population-spectra.ipynb`
- `examples/skymaps.ipynb`
