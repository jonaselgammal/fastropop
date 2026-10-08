# Changelog

## Unreleased (0.2.0, breaking)

Developed on branch `lisa-band`.

### Cosmology is an object

- `fastropop.cosmology.Cosmology` holds the cosmological parameters: h, Ω_m, Ω_r, Ω_b, n_s, σ₈ and T_cmb (flat).
  - **Background:** E, H, comoving and luminosity distance, dV_c/dz, dt/dz, age, lookback time and z(age), in Mpc and yr, accurate to ~1e-9 up to z = 60.
  - **Linear theory:** the Eisenstein & Hu (1998) transfer function, Δ²(k), σ(R), σ(M) and d ln σ/d ln M, the growth factor and δ_c(z). Normalised to σ₈, or to COBE with `sigma8=None`.
- Two instances are provided: `PLANCK18` (Planck 2018) and `CONCORDANCE` (h = 0.7, Ω_m = 0.3, no radiation: the 0.1 cosmology).
- **Removed:**
  - `cosmology.EE`, `dtodz`, `Dca`, `Dc_interp`, `Dc_interp_numpy`, `dVcdz` and `DL`;
  - the constants `hH0`, `OmegaDM`, `OmegaLambda`, `Omegak` and `H0s`;
  - the re-exports of `DL`, `Dc_interp` and `EE` from `semi_analytic`.
- **Added:** `cosmology=` arguments on `SemiAnalyticPopulation`, `h`, `h_average`, `compute_h` and `binning`. They default to `CONCORDANCE`, so 0.1 results are unchanged except for the constants below.
- **New constants:** exact or IAU-nominal astrophysical units (`C_MS`, `C_KMS`, `MPC_M`, `YR_S`, `GM_SUN`, `MSUN_S`, `RHO_CRIT_H2`).

### Numbers that moved

- **The Hubble distance grows by 1.747e-6.** The cosmology now uses c = 299792458 m/s (was 2.99792e8) and the IAU Mpc (100 km/s/Mpc was 3.24078e-18 s⁻¹). Distances and strains move by that factor; dV_c/dz and d³n/(dz dM d ln f) move by its cube.
- **Everything else in `tests/test_golden.py` reproduces 0.1 to 1e-8** once that factor is divided out.
