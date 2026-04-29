# fastropop

`fastropop` is a standalone Python package for fast astrophysical population
generation, with a focus on JAX-based semi-analytic compact-binary models.

This repository is scaffolded to support:

- installable `src/` package layout
- automated tests with `pytest`
- linting and formatting with `ruff`
- runnable examples under `examples/`
- GitHub Actions CI for tests and package builds
- a release path to PyPI

The package is centered around one concrete semi-analytic population model for
SMBHB populations, with small support modules for shared cosmology and helper
functions.

Core modules:
- `fastropop.constants` for units, physical constants, and default parameter values
- `fastropop.cosmology` for cosmological distance/time helper functions
- `fastropop.semi_analytic` for the concrete `SemiAnalyticPopulation` implementation

## Planned scope

- JAX-first semi-analytic population generation utilities
- a concrete `SemiAnalyticPopulation` API for sampling and derived quantities
- examples and notebooks that reproduce core workflows

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

For plotting utilities and HEALPix skymap generation, install the visualization
extras:

```bash
pip install -e ".[dev,viz]"
```

## Repository layout

```text
src/fastropop/        Package source
tests/                Unit and regression tests
examples/             Lightweight runnable examples
.github/workflows/    CI and release automation
```
