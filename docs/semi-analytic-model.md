# Semi-Analytic Population Model

This page summarizes the semi-analytic SMBHB population model implemented in `fastropop` and shows how the main equations map onto the code.

The implementation is based on the semi-analytic SMBHB framework introduced in [Sesana, Vecchio, and Colacino (2008)](https://arxiv.org/abs/0804.4476).

## Overview

The package models the nanohertz gravitational-wave background as a superposition of many distant supermassive black hole binaries (SMBHBs). Each binary is characterized by:

- chirp mass \(M\)
- redshift \(z\)
- observed frequency \(f\)
- sky position and orientation angles for skymaps and polarization realizations

The workflow in the code is:

1. define a merger-rate density in mass and redshift
2. convert that into a source-count density in mass, redshift, and frequency
3. integrate to get ensemble quantities such as `h_c^2(f)` and the expected source count
4. sample a Poisson realization of binaries from that distribution
5. bin the sampled binaries into PTA-style strain spectra

## Cosmology

The cosmology helpers are collected in `src/fastropop/cosmology.py`.

The model uses

$$
E(z) = \sqrt{\Omega_M (1+z)^3 + \Omega_k (1+z)^2 + \Omega_\Lambda}
$$

and

$$
\frac{dt_r}{dz} = \frac{1}{H_0 (1+z) E(z)}.
$$

In the code these correspond to:

- `EE(z)` -> \(E(z)\)
- `dtodz(z)` -> \(dt_r / dz\)
- `Dc_interp(z)` -> comoving distance interpolation
- `DL(z)` -> luminosity distance
- `dVcdz(z)` -> \(dV_c / dz\)

## Population Density in Mass and Redshift

The core semi-analytic model is the merger-rate density

$$
\frac{d^2 n}{dz \, d\log_{10}\mathcal{M}}
=
\dot n_0
\left[
\left(\frac{\mathcal{M}}{10^7 M_\odot}\right)^{-\alpha_{\mathcal M}}
e^{-\mathcal M / \mathcal M_*}
\right]
\left[
(1+z)^{\beta_z} e^{-z/z_0}
\right]
\frac{dt_r}{dz}.
$$

The free parameters are

$$
\theta = \{\dot n_0, \alpha_{\mathcal M}, \mathcal M_*, \beta_z, z_0\}.
$$

`fastropop` works internally with the density per unit mass rather than per unit `log10(M)`, so the implemented quantity is

$$
\frac{d^2 n}{dz \, d\mathcal M}
=
\frac{1}{\mathcal M \ln 10}
\frac{d^2 n}{dz \, d\log_{10}\mathcal M}.
$$

This is implemented in `src/fastropop/semi_analytic.py`:

- `SemiAnalyticPopulation.d2ndzdM(z, M)`

The constructor parameters map directly as:

- `n0` -> \(\dot n_0\)
- `alphaM` -> \(\alpha_{\mathcal M}\)
- `Mstar` -> \(\mathcal M_*\)
- `betaz` -> \(\beta_z\)
- `z0` -> \(z_0\)

## Frequency Evolution

Assuming circular, GW-driven binaries, the rest-frame frequency evolves as

$$
\frac{d\ln f_r}{dt_r}
=
\frac{96}{5}\pi^{8/3}
\frac{(G \mathcal M)^{5/3}}{c^5}
f_r^{8/3}.
$$

Since the code uses observed frequency `f`, the rest-frame frequency is

$$
f_r = f(1+z).
$$

This is implemented as:

- `dlnfdtr(M, f, z)`

where the `(1+z)` factor is already included internally.

## Source Count Density in Mass, Redshift, and Frequency

The differential source count is built from the population density, the residence time in frequency, and the comoving volume element:

$$
\frac{d^3 N}{dz \, d\mathcal M \, d\ln f}
=
\frac{d^2 n}{dz \, d\mathcal M}
\left(\frac{d\ln f_r}{dt_r}\right)^{-1}
\left(\frac{dt_r}{dz}\right)^{-1}
\frac{dV_c}{dz}.
$$

This is implemented in:

- `SemiAnalyticPopulation.d3ndzdMdlnf(M, f, z)`

This is the key sampling density used to generate Monte Carlo realizations of binaries.

## Ensemble Characteristic Strain

For the ensemble-averaged characteristic strain, the model uses

$$
h_c^2(f)
=
\frac{4 G^{5/3}}{3 \pi^{1/3} c^2}
f^{-4/3}
\int d\mathcal M \int dz \,
(1+z)^{-1/3} \mathcal M^{5/3}
\frac{d^2 n}{dz \, d\mathcal M}.
$$

This is implemented by:

- `SemiAnalyticPopulation.hc2(ff)`
- internal helpers `_integrand`, `_integrand_log`
- NumPy/SciPy scalar integration helpers `_integrand_numpy`, `_integrand_log_numpy`

The code currently uses SciPy for this integration path on purpose. The JAX-first version was slower here because adaptive scalar quadrature and JAX callbacks are a bad combination on CPU.

## Expected Number of Binaries

The expected source count over a mass, redshift, and frequency range is

$$
\langle N \rangle
=
\int_{\ln f_{\min}}^{\ln f_{\max}} d\ln f
\int_{z_{\min}}^{z_{\max}} dz
\int_{\mathcal M_{\min}}^{\mathcal M_{\max}} d\mathcal M \,
\frac{d^3 N}{dz \, d\mathcal M \, d\ln f}.
$$

In the code this is:

- `SemiAnalyticPopulation.compute_Nbinaries(...)`

The implementation integrates in `log10(f)` instead of `ln(f)`, which introduces the expected Jacobian factor `ln(10)` in the integrand.

## Strain of Individual Binaries

For a single binary, the polarization amplitudes can be written as

$$
\tilde h_+(f) = h(f)\frac{1+\cos^2\iota}{2} e^{i\Psi(f)},
$$

$$
\tilde h_\times(f) = i h(f)\cos\iota \, e^{i\Psi(f)}.
$$

The code distinguishes between:

- `h(M, f, z)` -> the un-averaged amplitude normalization
- `h_average(M, f, z)` -> the sky-and-orientation averaged amplitude

The averaged amplitude used for stochastic background binning is

$$
\bar h(f)
=
\frac{8\pi^{2/3}}{\sqrt{10}}
\frac{(G \mathcal M (1+z))^{5/3}}{c^4 d_L}
f^{2/3}.
$$

This is implemented by:

- `h_average(M, f, z)`

The two are related by the sky-and-orientation average

$$
\bar h(f) = \sqrt{\frac{2}{5}} \, h(f).
$$

## Monte Carlo Realizations and Binning

The practical Monte Carlo recipe is:

1. compute \(\langle N \rangle\)
2. draw a Poisson realization \(N_s\)
3. sample \(N_s\) binaries from \(d^3N/(dz \, d\mathcal M \, d\ln f)\)
4. add the binary contributions within frequency bins

The binned characteristic strain estimator is

$$
h_c^2(f_i) = \frac{\sum_k \bar h_k^2 f_k}{\Delta f_i},
$$

where the sum runs over sources in the `i`-th frequency bin.

This corresponds to:

- `generate_poisson_realization(...)` -> Poisson draw of the source count
- `sample_dist(...)` -> draw masses, redshifts, and frequencies
- `binning(...)` -> compute the binned spectrum

More specifically, the code path in `src/fastropop/semi_analytic.py` is:

- `_prepare_sampling_distributions(...)` builds separable sampling CDFs
- `sample_dist(...)` draws \((M, z, \log_{10} f)\)
- `binning_jitted(...)` computes \(f \bar h^2\) and sums into PTA bins
- `binning(...)` normalizes by the bin width and returns the spectrum array

## Skymaps and Angular Variables

For spatially isotropic realizations, the manuscript samples:

- \(\phi \sim U[0, 2\pi)\)
- \(\cos\theta \sim U[-1,1]\)
- \(\cos\iota \sim U[-1,1]\)
- \(\psi \sim U[0,\pi)\)

These correspond to the random angular draws used in:

- `SemiAnalyticPopulation.generate_skymaps(...)`

The skymap code combines sampled binary parameters with random sky position, inclination, polarization angle, and phase, then projects them into HEALPix pixels.

## Summary of Equation-to-Code Correspondence

| Physics quantity | Equation | Code |
| --- | --- | --- |
| \(E(z)\) | cosmology factor | `EE` |
| \(dt_r/dz\) | cosmic time-redshift relation | `dtodz` |
| \(dV_c/dz\) | comoving volume element | `dVcdz` |
| \(d^2 n / (dz \, dM)\) | semi-analytic merger density | `SemiAnalyticPopulation.d2ndzdM` |
| \(d\ln f_r / dt_r\) | GW-driven inspiral | `dlnfdtr` |
| \(d^3 N / (dz \, dM \, d\ln f)\) | source-count density | `SemiAnalyticPopulation.d3ndzdMdlnf` |
| \(h_c^2(f)\) | ensemble strain spectrum | `SemiAnalyticPopulation.hc2` |
| \(\langle N \rangle\) | expected number of binaries | `SemiAnalyticPopulation.compute_Nbinaries` |
| \(\bar h(f)\) | averaged single-source strain | `h_average` |
| \(h_c^2(f_i)\) binned | Monte Carlo estimator | `binning` |

## Related Notebook

For a worked tutorial focused on the model itself and its ensemble quantities, see:

- `examples/notebooks/semi-analytic.ipynb`
