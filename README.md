# fastropop

`fastropop` is the package that generates your SMBHB populations at lightning speed. It is designed to be a fast, flexible, and user-friendly tool for simulating the cosmic population of supermassive black hole binaries (SMBHBs) and their gravitational wave signatures in the nanohertz regime.

The package is centered on the semi analytic population from [0804.4476](https://arxiv.org/abs/0804.4476), which is implemented in the `SemiAnalyticPopulation` class. It also includes a suite of utilities for computing characteristic-strain quantities, sampling population realizations, binning into PTA-style spectra

## What It Does

- defines a semi-analytic SMBHB population model
- computes characteristic-strain quantities and expected binary counts
- samples population realizations with JAX-backed randomness
- bins sampled realizations into PTA-style spectra
- generates HEALPix skymaps with either `jax-healpy` or `healpy`

## Installation

```bash
git clone git@github.com:jonaselgammal/fastropop.git
cd fastropop
pip install -e .
```

Notes:
- `jax` and `jaxlib` are core dependencies
- skymap generation requires a HEALPix backend
- `jax-healpy` is preferred when installed
- standard `healpy` is supported as a fallback backend

## Quick Start

```python
import fastropop

pop = fastropop.SemiAnalyticPopulation()

distM, distz, distlog10f = pop.sample_dist(Nbinaries=1000, key=0)
spec = fastropop.binning(distM, distz, distlog10f, do_plot=False)
```

For reproducible stochastic methods, pass a JAX key or integer seed:

```python
params = fastropop.draw_parameters(key=0)
Nbinaries = pop.generate_poisson_realization(1, key=1)
```

If `key=None`, the package creates a fresh nondeterministic JAX key internally.

## Notebooks

The main examples currently live in:
- `examples/notebooks/SMBHB_pop_poisson.ipynb`
- `examples/notebooks/test_sam.ipynb`

## Repository Layout

```text
src/fastropop/        Package source
tests/                Unit and regression tests
examples/             Example scripts and notebooks
.github/workflows/    CI and release automation
```
