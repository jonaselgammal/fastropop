# Cosmology

Every model and observable takes a `Cosmology` object; nothing in `fastropop` reads
cosmological parameters from module globals. Two instances are provided:

- `PLANCK18`: Planck 2018 (h = 0.674, Ω_m = 0.315, σ₈ = 0.811, n_s = 0.965), the default for new code;
- `CONCORDANCE`: h = 0.7, Ω_m = 0.3, no radiation. This was the cosmology of fastropop 0.1, kept so its results stay reproducible.

Units are Mpc, yr and Msun, with wavenumbers in Mpc⁻¹ (no h). Background quantities are accurate to
~1e-9 at any redshift up to z = 60. Linear theory (transfer function, σ(M), growth) is built on first use.

```python
from fastropop.cosmology import PLANCK18, Cosmology

PLANCK18.luminosity_distance(1.0)          # Mpc
PLANCK18.age(0.0) / 1e9                    # Gyr
PLANCK18.sigma_M(1e12)                     # rms linear overdensity at z = 0
my = Cosmology(h=0.7, Omega_m=0.3, Omega_r=0.0, sigma8=0.8, name="mine")
```

::: fastropop.cosmology.Cosmology
    options:
      heading_level: 2
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
      separate_signature: true
      members_order: source
