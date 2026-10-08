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

### Population models

- **`fastropop.grid`:** quadrature grids in named coordinates, (log10 Mc, q, z) or (log10 m1, log10 m2, z). The latter uses the trapezoid rule on the physical triangle m2 <= m1. The module also provides the exact coordinate conversions and Jacobian, and `component_masses`.
- **`fastropop.populations`:** one contract for every model (`PopulationModel`). A model supplies a comoving merger-rate density in its native coordinates; the shared code turns it into the observer-frame intensity λ [yr⁻¹ per unit coordinate] on any grid, applies a support box, and handles parameters as dicts (`pack`/`unpack` for samplers).
  - **`Phenomenological`:** the separable family (Sesana, Vecchio & Colacino 2008) with a power-law p(q). The mass part is a cutoff or bimodal; the redshift part is a power law, or Madau–Dickinson star formation convolved with delays.
  - **`EPS`:** black-hole mergers from extended Press–Schechter halo mergers (Ellis et al. 2024, Eqs. 3–4).
    - **Construction:** Press–Schechter × Lacey–Cole halo merger rate from the cosmology's linear theory, then the Girelli et al. (2020) stellar-to-halo relation (Table 3 reference case), then a log-normal black-hole–stellar mass relation.
    - **Free parameters:** log10 p_BH, a, b, σ, γ. Native coordinates (log10 m1, log10 m2, z), with a fast path on those grids.
    - **Validation:** given identical inputs, it reproduces the KBFI implementation to 2.6e-4 at z = 0.5–10.
  - **`Tabulated`:** catalogue rows or a gridded intensity, as rate-conserving injections.
  - **Presets** (`get_preset`): the fiducial, and the three Çalışkan et al. (2025) models with their published support.
- **Removed:** `seed_models`, whose models are now presets.

### Numbers that moved

- **The Hubble distance grows by 1.747e-6.** The cosmology now uses c = 299792458 m/s (was 2.99792e8) and the IAU Mpc (100 km/s/Mpc was 3.24078e-18 s⁻¹). Distances and strains move by that factor; dV_c/dz and d³n/(dz dM d ln f) move by its cube.
- **Everything else in `tests/test_golden.py` reproduces 0.1 to 1e-8** once that factor is divided out.
