r"""Populations given as data: catalogue rows, or a tabulated intensity.

These have no parameters and serve as injections (e.g. semi-analytic merger catalogues, or a
holodeck realisation), not as models to infer. On a grid they give the rate-conserving
*bin average* of the intensity: for each cell, the rate it holds divided by its quadrature
weight. That is what drawing events or counting needs. It is not the intensity at the node.

.. warning::
   Do not pair a bin average with something evaluated at the node that varies strongly
   across the cell. A GW amplitude ~ 1/D_L^2 near z = 0 is an example: at n_z = 100 that
   pairing puts half the sub-mHz background at the first redshift node and doubles it.
   For such quantities, sum over :meth:`Tabulated.sources` instead.
"""

from __future__ import annotations

import numpy as np

from ..cosmology import PLANCK18
from ..grid import MCQ, trapezoid_weights
from .base import PopulationModel


def bin_edges(x):
    """Edges at the midpoints and at the ends, so each bin's width is its node's trapezoid weight."""
    x = np.asarray(x, dtype=float)
    return np.concatenate([[x[0]], 0.5 * (x[1:] + x[:-1]), [x[-1]]])


def _bin_average(f, e, support, n_sub=64):
    """Average over each bin [e_k, e_k+1] of f, taken as zero outside ``support``."""
    out = np.empty(len(e) - 1)
    for k in range(len(e) - 1):
        lo, hi = max(e[k], support[0]), min(e[k + 1], support[1])
        if hi <= lo:
            out[k] = 0.0
            continue
        x = np.linspace(lo, hi, n_sub)
        out[k] = np.trapezoid(f(x), x) / (e[k + 1] - e[k])
    return out


def hat_average(nodes, e):
    """Matrix A such that (A f)_k is the average over bin k of the piecewise-linear
    interpolant of node values f (zero outside the nodes)."""
    nodes = np.asarray(nodes, dtype=float)
    cols = [_bin_average(lambda x, c=c: np.interp(x, nodes, c, left=0.0, right=0.0), e, (nodes[0], nodes[-1]))
            for c in np.eye(len(nodes))]
    return np.stack(cols, axis=1)


class Tabulated(PopulationModel):
    """A population given as catalogue rows or as a gridded intensity (see the module
    docstring). Build it with :meth:`from_rows` or :meth:`from_table`."""

    native_coords = MCQ
    param_names = ()

    def __init__(self, cosmology=PLANCK18, rows=None, table=None, name="tabulated", reference=""):
        super().__init__(cosmology)
        if (rows is None) == (table is None):
            raise ValueError("give exactly one of rows or table")
        self.rows, self.table, self.name, self.reference = rows, table, name, reference
        self.defaults = {}

    @classmethod
    def from_rows(cls, log10_Mc, q, z, rate, cosmology=PLANCK18, name="catalogue", reference=""):
        """One row per merger: source-frame log10 Mc, q, z, and the observer-frame rate it
        stands for [yr^-1]."""
        rows = tuple(np.asarray(a, dtype=float) for a in (log10_Mc, q, z, rate))
        if len({a.shape for a in rows}) != 1:
            raise ValueError("log10_Mc, q, z and rate must have the same shape")
        return cls(cosmology, rows=rows, name=name, reference=reference)

    @classmethod
    def from_table(cls, log10_Mc, q, z, intensity, cosmology=PLANCK18, name="table", reference=""):
        """An observer-frame intensity [yr^-1 per unit (log10 Mc, q, z)] on the nodes
        ``log10_Mc x q x z``, read as its trilinear interpolant, zero outside."""
        axes = tuple(np.asarray(a, dtype=float) for a in (log10_Mc, q, z))
        lam = np.asarray(intensity, dtype=float)
        if lam.shape != tuple(a.size for a in axes):
            raise ValueError(f"intensity has shape {lam.shape}, axes give {tuple(a.size for a in axes)}")
        return cls(cosmology, table=(axes, lam), name=name, reference=reference)

    def comoving_rate(self, params, x1, x2, z):
        raise NotImplementedError("a tabulated population has no point-wise rate density; "
                                  "use intensity(grid) or sources()")

    def intensity(self, params, grid):
        """Rate-conserving bin average on ``grid`` (log10 Mc, q, z grids only). Rows outside
        the grid in Mc or z are dropped. Rows below the grid's q range are placed in its
        first q bin, and rows at q = 1 in its last."""
        if grid.coords != MCQ:
            raise NotImplementedError("tabulated populations bin onto (log10 Mc, q, z) grids")
        E = tuple(bin_edges(np.asarray(a)) for a in grid.axes)
        W = np.asarray(grid.W)
        if self.rows is not None:
            l, q, z, r = self.rows
            keep = (l >= E[0][0]) & (l <= E[0][-1]) & (z >= E[2][0]) & (z <= E[2][-1])
            H, _ = np.histogramdd((l[keep], np.clip(q[keep], E[1][0], E[1][-1]), z[keep]), bins=E,
                                  weights=r[keep])
            return np.where(W > 0, H / np.where(W > 0, W, 1.0), 0.0)
        (a0, a1, a2), lam = self.table
        ops = [hat_average(n, e) for n, e in zip((a0, a1, a2), E)]
        return np.einsum("ai,bj,ck,ijk->abc", *ops, lam)

    def sources(self):
        """(log10 Mc, q, z, rate [yr^-1]) per row, or per table node with the node's share of
        the rate: the representation for quantities that vary strongly within a cell."""
        if self.rows is not None:
            return self.rows
        (a0, a1, a2), lam = self.table
        w = [np.asarray(trapezoid_weights(a)) for a in (a0, a1, a2)]
        W = w[0][:, None, None] * w[1][None, :, None] * w[2][None, None, :]
        L, Q, Z = np.meshgrid(a0, a1, a2, indexing="ij")
        rate = (W * lam).ravel()
        keep = rate > 0
        return L.ravel()[keep], Q.ravel()[keep], Z.ravel()[keep], rate[keep]

    def total_rate(self, params=None, grid=None):
        """Total observer-frame rate [yr^-1], over all rows or the whole table (no grid needed)."""
        return float(np.sum(self.sources()[3]))


__all__ = ["Tabulated", "bin_edges", "hat_average"]
