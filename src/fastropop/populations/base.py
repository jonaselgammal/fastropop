r"""The contract every population model implements.

A model is defined by its physics alone: a source-frame, comoving merger-rate density
in its natural coordinates,

    comoving_rate(params, x1, x2, z)   [Mpc^-3 yr^-1 per unit (x1, x2, z)],

with (x1, x2, z) one of the systems in :mod:`fastropop.grid`. Everything else is shared
and lives here. :meth:`PopulationModel.rate_density` turns that into the
observer-frame rate in any coordinate system: it applies the Jacobian, then
dV_c/dz / (1+z), then the model's support:

    lambda(x) = comoving_rate * |d native / d x| * dV_c/dz / (1+z)   [yr^-1 per unit x].

This lambda is the intensity of the Poisson process that every observable consumes. A
model may override :meth:`intensity` with a fast path for grids in its native
coordinates; it must agree with the generic one.

Parameters are dict pytrees keyed by :attr:`param_names`; :meth:`pack` and
:meth:`unpack` convert to and from flat arrays for samplers. Scale parameters are
carried as log10, masses in Msun, times in yr.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..cosmology import PLANCK18
from ..grid import MCQ, SYSTEMS, convert, jacobian


class PopulationModel:
    """Base class. Subclasses set :attr:`native_coords`, :attr:`param_names`,
    :attr:`defaults`, and implement :meth:`comoving_rate`.

    Parameters
    ----------
    cosmology : fastropop.cosmology.Cosmology
        Background used for dV_c/dz (and by models that need cosmic time).
    support : dict, optional
        Bounds outside which the rate is zero, keyed by coordinate name, e.g.
        ``{"log10_Mc": (2.398, 9.0), "z": (0.0, 20.0)}``. Any coordinate of either
        system may be used.
    """

    native_coords: tuple = MCQ
    param_names: tuple = ()
    defaults: dict = {}
    labels: dict = {}

    def __init__(self, cosmology=PLANCK18, support=None):
        self.cosmology = cosmology
        self.support = dict(support or {})
        known = {c for s in SYSTEMS for c in s}
        unknown = set(self.support) - known
        if unknown:
            raise ValueError(f"unknown support coordinates {unknown}; use any of {sorted(known)}")

    # ---------------------------------------------------------------- physics
    def comoving_rate(self, params, x1, x2, z):
        """Source-frame comoving merger-rate density in :attr:`native_coords`
        [Mpc^-3 yr^-1 per unit native coordinate]."""
        raise NotImplementedError

    # ---------------------------------------------------------------- shared
    def rate_density(self, params, coords, x1, x2, z):
        """Observer-frame merger rate [yr^-1 per unit ``coords``] at points given in ``coords``."""
        z = jnp.asarray(z, dtype=float)
        n1, n2 = convert(coords, self.native_coords, x1, x2)
        lam = (self.comoving_rate(params, n1, n2, z) * jacobian(self.native_coords, coords, x1, x2)
               * self.cosmology.dVc_dz(z) / (1.0 + z))
        return lam * self._inside(coords, x1, x2, z)

    def intensity_at(self, params, log10_Mc, q, z):
        """Observer-frame rate per yr per unit (log10 Mc, q, z) at arbitrary points."""
        return self.rate_density(params, MCQ, log10_Mc, q, z)

    def intensity(self, params, grid):
        """Observer-frame rate per yr per unit grid coordinate, on the grid's nodes (zero on
        unphysical nodes)."""
        x1, x2, z = grid.mesh()
        lam = self.rate_density(params, grid.coords, x1, x2, z)
        return jnp.where(grid.valid, jnp.broadcast_to(lam, grid.shape), 0.0)

    def total_rate(self, params, grid):
        """Total observer-frame merger rate on ``grid`` [yr^-1]."""
        return jnp.sum(grid.W * self.intensity(params, grid))

    def _inside(self, coords, x1, x2, z):
        if not self.support:
            return 1.0
        values = {"z": z, coords[0]: x1, coords[1]: x2}
        for system in SYSTEMS:
            if system != coords:
                a, b = convert(coords, system, x1, x2)
                values.update({system[0]: a, system[1]: b})
        inside = 1.0
        for name, (lo, hi) in self.support.items():
            inside = inside * ((values[name] >= lo) & (values[name] <= hi))
        return inside

    # ---------------------------------------------------------------- parameters
    def params(self, **overrides):
        """The defaults, updated with ``overrides``."""
        unknown = set(overrides) - set(self.param_names)
        if unknown:
            raise KeyError(f"unknown parameters {unknown}; {type(self).__name__} has {self.param_names}")
        return {**self.defaults, **overrides}

    def pack(self, params):
        """dict -> flat array in :attr:`param_names` order."""
        return jnp.stack([jnp.asarray(params[n], dtype=float) for n in self.param_names])

    def unpack(self, theta):
        """Flat array -> dict."""
        return {n: theta[i] for i, n in enumerate(self.param_names)}

    def __repr__(self):
        sup = f", support={self.support}" if self.support else ""
        return f"{type(self).__name__}({self.cosmology.name}{sup})"
