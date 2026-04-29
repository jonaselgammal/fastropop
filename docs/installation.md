# Installation

Clone the repository and install it in editable mode:

```bash
git clone git@github.com:jonaselgammal/fastropop.git
cd fastropop
pip install -e .
```

Optional extras:

```bash
pip install -e ".[docs]"
pip install -e ".[viz]"
pip install -e ".[docs,viz]"
```

Notes:

- `jax` and `jaxlib` are core dependencies
- skymap generation requires a HEALPix backend
- `jax-healpy` is preferred when installed
- standard `healpy` is supported as a fallback
