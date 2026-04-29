# fastropop

<p align="center">
  <img src="img/fastropop-logo-transparent-light.png" alt="fastropop logo" width="880">
</p>

`fastropop` is the package that generates your SMBHB populations at lightning speed. It is designed to be a fast, flexible, and user-friendly tool for simulating the cosmic population of supermassive black hole binaries (SMBHBs) and their gravitational wave signatures in the nanohertz regime.

The package is centered on the semi analytic population from [arXiv:0804.4476](https://arxiv.org/abs/0804.4476), which is implemented in the `SemiAnalyticPopulation` class. It also includes a suite of utilities for computing characteristic-strain quantities, sampling population realizations, and binning into PTA-style spectra.

The docs are intentionally compact. The goal is to cover the main workflow clearly, not to exhaustively document every internal helper.

## What You Can Do

- define a semi-analytic population with `SemiAnalyticPopulation`
- compute expected binary counts and strain quantities
- sample Poisson realizations of the population
- bin realizations into PTA-style characteristic-strain spectra
- generate HEALPix skymaps with `jax-healpy` or `healpy`

## Start Here

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Semi-Analytic Model](semi-analytic-model.md)
- [API reference](api.md)
