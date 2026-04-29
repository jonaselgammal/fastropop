# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Local checks

```bash
ruff check .
ruff format .
pytest
python -m build
```

## Near-term refactor plan

1. Keep the public API centered on `SemiAnalyticPopulation`.
2. Split monolithic logic only where there is a real boundary:
   - cosmology helpers
   - population model and sampling
   - derived observables and plotting helpers
3. Add regression tests against the current trusted outputs.
4. Add example notebooks once the API stabilizes.
