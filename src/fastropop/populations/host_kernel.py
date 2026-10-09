r"""Black-hole mergers from host mergers: a host-pair merger rate convolved with a log-normal
black-hole-host relation.

    dR_BH / (dlog10 m1 dlog10 m2) = s  \int dh1 dh2  A(h1, h2, z)  p(log10 m1 | h1, z)  p(log10 m2 | h2, z)

h is the host's log mass coordinate (halo mass for :class:`~fastropop.populations.EPS`, stellar
mass for :class:`~fastropop.populations.Holodeck`). A is the host-pair merger rate per unit h1
and h2, symmetric and counting every merger once over the whole (h1, h2) square; s is an
overall scale. p is a normal distribution in log10 m whose mean is linear in h within each host
cell, with width ``sigma``.

The rate on the physical triangle m2 <= m1 is twice the symmetric expression.

Host quadrature: cells of width ``dh``. The host-pair rate is taken at cell centres, and the
black-hole log-normal is integrated exactly over each cell (a difference of error functions),
so accuracy does not depend on how narrow the kernel is compared with a cell.

A subclass sets ``self._xe`` (host-cell edges, shape (z, h+1), in the coordinate the mean is
linear in), ``self._dh``, ``self.z_nodes`` / ``self._zn``, and implements :meth:`_means`,
:meth:`_sigma`, :meth:`_table` and :meth:`_scale`. The table and the kernel are evaluated at
the redshift nodes; other redshifts are reached by interpolating log rate linearly in z.

Grids in other coordinates, e.g. (log10 Mc, q, z), are evaluated at the mesh's (m1, m2) pairs.
Exact, but each pair needs its own contraction. :meth:`use_native_interpolation` instead
contracts once on a fine (m1, m2) tensor grid and interpolates log-bilinearly onto the pairs
(3-13x fewer operations; its accuracy is set by ``step``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erfc

from ..grid import M1M2, convert, jacobian
from .base import PopulationModel


def _cell_kernel(log10_m, mu_lo, mu_hi, sigma, dh):
    """int over a host cell of p(log10 m | h) dh, the mean running linearly from mu_lo to
    mu_hi across the cell of width dh. Broadcasts log10_m against the mu arrays."""
    u_lo, u_hi = (log10_m - mu_lo) / sigma, (log10_m - mu_hi) / sigma
    a, b = jnp.maximum(u_lo, u_hi), jnp.minimum(u_lo, u_hi)
    r2 = jnp.sqrt(2.0)
    # Phi(a) - Phi(b) from the side where both tails are small, so it does not cancel
    diff = jnp.where(b > 0, 0.5 * (erfc(b / r2) - erfc(a / r2)), 0.5 * (erfc(-a / r2) - erfc(-b / r2)))
    dmu = jnp.abs(mu_hi - mu_lo)
    u_mid = (log10_m - 0.5 * (mu_lo + mu_hi)) / sigma
    flat = jnp.exp(-0.5 * u_mid * u_mid) / (sigma * jnp.sqrt(2.0 * jnp.pi))
    small = dmu < 1e-3 * sigma
    return dh * jnp.where(small, flat, diff / jnp.where(small, 1.0, dmu))


def _loglerp(lo, hi, t):
    """Linear interpolation of log rate between two nodes; zero if either node is zero."""
    ok = (lo > 0) & (hi > 0)
    safe = lambda v: jnp.where(ok, v, 1.0)
    return jnp.where(ok, jnp.exp((1 - t) * jnp.log(safe(lo)) + t * jnp.log(safe(hi))), 0.0)


class HostKernelModel(PopulationModel):
    """Base class for black-hole populations built from host mergers (module docstring)."""

    native_coords = M1M2

    # ------------------------------------------------------------------ supplied by subclasses
    def _means(self, p, xe, z):
        """Mean log10 m at host-cell edges ``xe`` (shape (..., h+1)) and redshifts ``z``."""
        raise NotImplementedError

    def _sigma(self, p):
        """Width of the black-hole log-normal [dex]."""
        raise NotImplementedError

    def _table(self, p):
        """Host-pair merger rate at the cell centres and redshift nodes, shape (z, h, h):
        symmetric, per unit h1 and h2, every merger counted once over the square
        [Mpc^-3 yr^-1]."""
        raise NotImplementedError

    def _scale(self, p):
        return 1.0

    # ------------------------------------------------------------------ evaluation paths
    _native_step = None

    def use_native_interpolation(self, step=0.05):
        """Evaluate non-(m1, m2) grids by log-bilinear interpolation from an (m1, m2) tensor
        grid of spacing ``step`` [dex] (None: exact pair path). Returns self."""
        self._native_step = None if step is None else float(step)
        self._interp_cache = {}
        return self

    def _pair_interpolation(self, l1, l2):
        """Tensor nodes covering the pairs, and the bilinear stencil of every pair (cached)."""
        key = (l1.shape, float(l1.ravel()[0]), float(l1.ravel()[-1]), float(l2.ravel()[0]), float(l2.ravel()[-1]))
        if key not in self._interp_cache:
            h = self._native_step
            lo = np.floor(min(l1.min(), l2.min()) / h) * h - h
            hi = np.ceil(max(l1.max(), l2.max()) / h) * h + h
            nodes = np.arange(lo, hi + h / 2, h)
            u, v = (l1.ravel() - lo) / h, (l2.ravel() - lo) / h
            i, j = np.floor(u).astype(int), np.floor(v).astype(int)
            self._interp_cache[key] = (jnp.asarray(nodes), i, j, jnp.asarray(u - i), jnp.asarray(v - j))
        return self._interp_cache[key]

    def _symmetric_rate(self, params, log10_m):
        """K A K^T on the tensor product of ``log10_m`` with itself, at every redshift node,
        before the m2 <= m1 restriction and the factor 2 s: shape (z, m, m)."""
        K = self._kernel(params, log10_m)
        return jnp.einsum("zmh,zhk,znk->zmn", K, self._table(params), K)

    def _rate_at_pairs_interpolated(self, params, l1, l2):
        nodes, i, j, tu, tv = self._pair_interpolation(np.asarray(l1), np.asarray(l2))
        L = self._symmetric_rate(params, nodes)
        c = [L[:, i, j], L[:, i + 1, j], L[:, i, j + 1], L[:, i + 1, j + 1]]
        w = [(1 - tu) * (1 - tv), tu * (1 - tv), (1 - tu) * tv, tu * tv]
        ok = (c[0] > 0) & (c[1] > 0) & (c[2] > 0) & (c[3] > 0)
        safe = [jnp.where(ok, x, 1.0) for x in c]
        logi = jnp.exp(sum(wk * jnp.log(xk) for wk, xk in zip(w, safe)))        # log-bilinear
        lin = sum(wk * xk for wk, xk in zip(w, c))                                 # where a corner is empty
        Lp = jnp.where(ok, logi, lin)
        return 2.0 * self._scale(params) * jnp.where(jnp.asarray(l2.ravel() <= l1.ravel()), Lp, 0.0)

    # ------------------------------------------------------------------ shared
    def _kernel(self, p, log10_m):
        """Cell-integrated p(log10 m | host cell, z) at the redshift nodes: shape (z, m, h)."""
        mu = self._means(p, self._xe, self._zn)                                # (z, h+1)
        lm = jnp.asarray(log10_m)[None, :, None]
        return _cell_kernel(lm, mu[:, None, :-1], mu[:, None, 1:], self._sigma(p), self._dh)

    def rate_at_nodes(self, params, log10_m1, log10_m2):
        """Comoving rate on the triangle m2 <= m1 [Mpc^-3 yr^-1 per dex^2] for the tensor
        product of ``log10_m1`` and ``log10_m2`` at every ``z_nodes``: shape (z, m1, m2)."""
        K1, K2 = self._kernel(params, log10_m1), self._kernel(params, log10_m2)
        L = jnp.einsum("zmh,zhk,znk->zmn", K1, self._table(params), K2)
        tri = jnp.asarray(log10_m2)[None, :] <= jnp.asarray(log10_m1)[:, None]
        return 2.0 * self._scale(params) * jnp.where(tri[None], L, 0.0)

    def rate_at_pairs(self, params, log10_m1, log10_m2):
        """Comoving rate [Mpc^-3 yr^-1 per dex^2] at the mass pairs (``log10_m1[i]``,
        ``log10_m2[i]``) at every ``z_nodes``: shape (z, pairs). Zero where m2 > m1."""
        l1, l2 = jnp.asarray(log10_m1, dtype=float), jnp.asarray(log10_m2, dtype=float)
        L = jnp.einsum("zph,zhk,zpk->zp", self._kernel(params, l1), self._table(params), self._kernel(params, l2))
        return 2.0 * self._scale(params) * jnp.where(l2 <= l1, L, 0.0)

    def comoving_rate(self, params, log10_m1, log10_m2, z, chunk=512):
        """At arbitrary points: the kernel contracted point by point at the two bracketing
        redshift nodes, then log-linear in z. Evaluated in chunks of ``chunk`` points, so
        memory stays at ``chunk`` copies of one host table."""
        import jax

        l1, l2, z = (jnp.asarray(v, dtype=float) for v in jnp.broadcast_arrays(log10_m1, log10_m2, z))
        shape = l1.shape
        n = l1.size
        pad = (-n) % chunk
        flat = [jnp.pad(v.ravel(), (0, pad), constant_values=v.ravel()[0]) for v in (l1, l2, z)]
        zn, p = self._zn, params
        A, sig = self._table(p), self._sigma(p)

        def one_chunk(args):
            a1, a2, zz = args
            i = jnp.clip(jnp.searchsorted(zn, zz) - 1, 0, zn.size - 2)
            t = jnp.clip((zz - zn[i]) / (zn[i + 1] - zn[i]), 0.0, 1.0)

            def at(j):
                mu = self._means(p, self._xe[j], zn[j])                        # (points, h+1)
                k1 = _cell_kernel(a1[:, None], mu[:, :-1], mu[:, 1:], sig, self._dh)
                k2 = _cell_kernel(a2[:, None], mu[:, :-1], mu[:, 1:], sig, self._dh)
                return jnp.einsum("ph,phk,pk->p", k1, A[j], k2)

            return jnp.where(a2 <= a1, _loglerp(at(i), at(i + 1), t), 0.0)

        r = jax.lax.map(one_chunk, tuple(v.reshape(-1, chunk) for v in flat)).ravel()[:n]
        return (2.0 * self._scale(p) * r).reshape(shape)

    def intensity(self, params, grid):
        """The kernel contracted at the redshift nodes for the grid's mass points (a tensor
        product on (log10 m1, log10 m2) grids, the mesh's (m1, m2) pairs otherwise), then
        log-linear in z onto the grid's redshifts."""
        x1, x2, zg = (np.asarray(a) for a in grid.axes)
        if grid.coords == M1M2:
            L = self.rate_at_nodes(params, x1, x2)                              # (zn, n1, n2)
            jac = 1.0
        else:
            X1, X2 = np.meshgrid(x1, x2, indexing="ij")
            with jax.ensure_compile_time_eval():          # the grid is static: concrete arrays under jit
                l1, l2 = (np.asarray(a) for a in convert(grid.coords, M1M2, X1, X2))
            if self._native_step is None:
                L = self.rate_at_pairs(params, l1.ravel(), l2.ravel())
            else:
                L = self._rate_at_pairs_interpolated(params, l1, l2)
            L = L.reshape((-1,) + X1.shape)
            jac = jacobian(M1M2, grid.coords, X1, X2)[..., None]               # per unit grid coordinates
        zn = np.asarray(self.z_nodes)
        i = np.clip(np.searchsorted(zn, zg) - 1, 0, zn.size - 2)
        t = np.clip((zg - zn[i]) / (zn[i + 1] - zn[i]), 0.0, 1.0)
        Lz = _loglerp(L[i], L[i + 1], jnp.asarray(t)[:, None, None])
        z = jnp.asarray(zg)
        lam = jnp.moveaxis(Lz, 0, -1) * jac * (self.cosmology.dVc_dz(z) / (1.0 + z))[None, None, :]
        lam = lam * self._inside(grid.coords, *grid.mesh())          # the whole support box, as rate_density does
        return jnp.where(grid.valid, lam, 0.0)


__all__ = ["HostKernelModel"]
