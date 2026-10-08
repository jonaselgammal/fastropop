r"""Quadrature grids in named population coordinates.

A population model evaluates its rate density on a :class:`Grid`, and observables
integrate over the grid with its trapezoid weights. Two coordinate systems are supported:

* ``MCQ = ("log10_Mc", "q", "z")``: source-frame chirp mass [Msun], mass ratio
  q = m2/m1 <= 1, and redshift;
* ``M1M2 = ("log10_m1", "log10_m2", "z")``: source-frame component masses [Msun]. Only
  m2 <= m1 is physical: nodes with m2 > m1 carry no rate (:attr:`Grid.valid`), and the
  weights :attr:`Grid.W` are the trapezoid rule on that triangle, with half weight on the
  diagonal.

A density per unit (log10 Mc, q) equals one per unit (log10 m1, log10 m2) times
|d(log10 m1, log10 m2) / d(log10 Mc, q)| = 1 / (q ln 10) (:func:`jacobian`).

Every grid also gives (log10 Mc, q, z) at its nodes (:meth:`Grid.node_mcqz`), which is what
SNR and detection need, whatever the grid's own coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

MCQ = ("log10_Mc", "q", "z")
M1M2 = ("log10_m1", "log10_m2", "z")
SYSTEMS = (MCQ, M1M2)
_LN10 = float(np.log(10.0))


def trapezoid_weights(x):
    """Trapezoid quadrature weights for a 1-D grid of any spacing."""
    x = jnp.asarray(x, dtype=float)
    dx = jnp.diff(x)
    return jnp.zeros(x.size).at[:-1].add(dx / 2).at[1:].add(dx / 2)


def to_mcq(log10_m1, log10_m2):
    """(log10 m1, log10 m2) -> (log10 Mc, q), with q = m2 / m1."""
    log10_q = log10_m2 - log10_m1
    q = 10.0**log10_q
    log10_Mc = log10_m1 + 0.6 * log10_q - 0.2 * jnp.log10(1.0 + q)
    return log10_Mc, q


def to_m1m2(log10_Mc, q):
    """(log10 Mc, q) -> (log10 m1, log10 m2): m1 = Mc (1+q)^(1/5) q^(-3/5), m2 = q m1."""
    log10_m1 = log10_Mc + 0.2 * jnp.log10(1.0 + q) - 0.6 * jnp.log10(q)
    return log10_m1, log10_m1 + jnp.log10(q)


def component_masses(Mc, q):
    """(chirp mass, mass ratio) -> (m1, m2, total mass), in the units of ``Mc``."""
    m1 = Mc * (1.0 + q) ** 0.2 / q**0.6
    return m1, q * m1, m1 * (1.0 + q)


def convert(src, dst, x1, x2):
    """Express the first two coordinates of a point in another coordinate system."""
    if src == dst:
        return x1, x2
    if (src, dst) == (M1M2, MCQ):
        return to_mcq(x1, x2)
    if (src, dst) == (MCQ, M1M2):
        return to_m1m2(x1, x2)
    raise ValueError(f"unknown coordinate systems {src} -> {dst}")


def jacobian(src, dst, x1, x2):
    """Factor turning a density per unit ``src`` coordinates into one per unit ``dst``,
    at points given in ``dst`` coordinates."""
    if src == dst:
        return jnp.ones_like(jnp.asarray(x1, dtype=float))
    if (src, dst) == (M1M2, MCQ):          # x2 = q
        return 1.0 / (jnp.asarray(x2, dtype=float) * _LN10)
    if (src, dst) == (MCQ, M1M2):          # q = 10^(x2 - x1)
        return 10.0 ** (jnp.asarray(x2, dtype=float) - jnp.asarray(x1, dtype=float)) * _LN10
    raise ValueError(f"unknown coordinate systems {src} -> {dst}")


@dataclass(frozen=True, eq=False)
class Grid:
    """Tensor-product grid with trapezoid weights in one of the :data:`SYSTEMS`.

    Build it with :meth:`mcq` or :meth:`m1m2`. Instances hash by identity, so they can
    be static ``jit`` arguments.
    """

    coords: tuple
    axes: tuple

    def __post_init__(self):
        if self.coords not in SYSTEMS:
            raise ValueError(f"coords must be one of {SYSTEMS}, got {self.coords}")
        axes = tuple(jnp.asarray(a, dtype=float) for a in self.axes)
        w = tuple(trapezoid_weights(a) for a in axes)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "weights", w)
        W = w[0][:, None, None] * w[1][None, :, None] * w[2][None, None, :]
        if self.coords == M1M2:
            # trapezoid rule on the physical triangle m2 <= m1: nodes on the diagonal m2 = m1
            # carry half their cell, nodes above it none
            a, b = axes[0][:, None, None], axes[1][None, :, None]
            W = W * jnp.where(b < a, 1.0, jnp.where(b == a, 0.5, 0.0))
        object.__setattr__(self, "W", W)

    @classmethod
    def mcq(cls, n_mc=70, n_q=24, n_z=100, log10_mc=(3.0, 10.0), q=(0.01, 1.0), z=(0.05, 25.0),
             q_spacing="log"):
        """Grid in (log10 Mc, q, z): linear in log10 Mc and z, and log-spaced (default) or
        linear in q. Log-spacing resolves the q -> q_min edge, which dominates the merger
        and ringdown at fixed chirp mass."""
        if q_spacing not in ("log", "linear"):
            raise ValueError(f"q_spacing must be 'log' or 'linear', got {q_spacing!r}")
        qa = jnp.geomspace(*q, n_q) if q_spacing == "log" else jnp.linspace(*q, n_q)
        return cls(MCQ, (jnp.linspace(*log10_mc, n_mc), qa, jnp.linspace(*z, n_z)))

    @classmethod
    def m1m2(cls, n_m=81, n_z=100, log10_m=(2.0, 10.0), z=(0.05, 25.0)):
        """Grid in (log10 m1, log10 m2, z), both mass axes the same."""
        m = jnp.linspace(*log10_m, n_m)
        return cls(M1M2, (m, m, jnp.linspace(*z, n_z)))

    @property
    def shape(self):
        return tuple(a.size for a in self.axes)

    def mesh(self):
        """The three axes broadcast against each other: shapes (n,1,1), (1,n,1), (1,1,n)."""
        a, b, c = self.axes
        return a[:, None, None], b[None, :, None], c[None, None, :]

    @property
    def valid(self):
        """Nodes inside the physical region (all of them, except m2 > m1 on an M1M2 grid)."""
        if self.coords == MCQ:
            return jnp.ones(self.shape, dtype=bool)
        x1, x2, _ = self.mesh()
        return jnp.broadcast_to(x2 <= x1, self.shape)

    def node_mcqz(self):
        """(log10 Mc, q, z) at every node, broadcast to the grid's shape. Nodes outside the
        physical region come back with q > 1 and must be masked with :attr:`valid`."""
        x1, x2, z = self.mesh()
        lmc, q = convert(self.coords, MCQ, x1, x2)
        return tuple(jnp.broadcast_to(a, self.shape) for a in (lmc, q, z))


__all__ = ["Grid", "M1M2", "MCQ", "SYSTEMS", "component_masses", "convert", "jacobian", "to_m1m2", "to_mcq", "trapezoid_weights"]
