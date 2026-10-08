# Populations and grids

A population model is defined by its physics alone: a source-frame, comoving merger-rate
density in its natural coordinates. The shared code turns that into the observer-frame rate
λ (the intensity of the Poisson process) on any grid. Every observable consumes λ.

```python
from fastropop.cosmology import PLANCK18
from fastropop.grid import Grid
from fastropop.populations import Phenomenological, get_preset

model = Phenomenological(PLANCK18)                     # separable family, Sesana et al. (2008) form
grid = Grid.mcq(n_mc=70, n_q=24, n_z=100)              # (log10 Mc, q, z), trapezoid weights
lam = model.intensity(model.params(), grid)            # yr^-1 per unit (log10 Mc, q, z)
rate = model.total_rate(model.params(), grid)          # yr^-1

pre = get_preset("caliskan25_light")                   # model + parameters + provenance
pre.model().total_rate(pre.params, grid)
```

::: fastropop.grid.Grid
    options:
      heading_level: 2
      members_order: source
      show_root_heading: true
      show_root_full_path: false

::: fastropop.populations.base.PopulationModel
    options:
      heading_level: 2
      members_order: source
      show_root_heading: true
      show_root_full_path: false

::: fastropop.populations.phenomenological.Phenomenological
    options:
      heading_level: 2
      members_order: source
      merge_init_into_class: true
      show_root_heading: true
      show_root_full_path: false

::: fastropop.populations.eps.EPS
    options:
      heading_level: 2
      members_order: source
      merge_init_into_class: true
      show_root_heading: true
      show_root_full_path: false

::: fastropop.populations.tabulated.Tabulated
    options:
      heading_level: 2
      members_order: source
      show_root_heading: true
      show_root_full_path: false

::: fastropop.populations.presets
    options:
      heading_level: 2
      members: [Preset, get_preset, list_presets]
      show_root_heading: true
      show_root_full_path: false
