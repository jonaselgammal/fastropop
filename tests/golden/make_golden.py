"""Record fastropop 0.1 outputs before the cosmology/population restructuring.

Run ONCE on the pre-restructuring code (lisa-band at dbfa02e). tests/test_golden.py then
checks every later version against these numbers. Regenerating them after the
restructuring would defeat the purpose.

    python tests/golden/make_golden.py   ->  tests/golden/golden_0_1.npz
"""
import os, time
import numpy as np
import jax.numpy as jnp

import fastropop
from fastropop import cosmology as cosmo
from fastropop.constants import MsunMKS
from fastropop.semi_analytic import SemiAnalyticPopulation, dlnfdtr, h, h_average

HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.array([0.0, 1e-3, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 9.99])
M = np.array([1e8, 1e9, 1e10]) * MsunMKS            # chirp masses [kg]
F = np.array([1e-9, 1e-8])                          # Hz
ZS = np.array([0.1, 1.0, 3.0])
MM, FF, ZZ = (a.ravel() for a in np.meshgrid(M, F, ZS, indexing="ij"))
QUICKSTART = {"n0": 10**-90.4153, "alphaM": -1.3800, "Mstar": 10**8.8272 * MsunMKS,
              "betaz": -0.1711, "z0": 4.70}

out = dict(version=fastropop.__version__, z=Z, M=MM, f=FF, zs=ZZ,
           EE=np.asarray(cosmo.EE(jnp.asarray(Z))), Dc=np.asarray(cosmo.Dc_interp(jnp.asarray(Z))),
           DL=np.asarray(cosmo.DL(jnp.asarray(Z))), dVcdz=np.asarray(cosmo.dVcdz(jnp.asarray(Z))),
           dtodz=np.asarray(cosmo.dtodz(jnp.asarray(Z))),
           h=np.asarray(h(MM, FF, ZZ)), h_average=np.asarray(h_average(MM, FF, ZZ)),
           dlnfdtr=np.asarray(dlnfdtr(MM, FF, ZZ)))
pop = SemiAnalyticPopulation(population_params=QUICKSTART)
out["d2ndzdM"] = np.asarray(pop.d2ndzdM(ZZ, MM))
out["d3ndzdMdlnf"] = np.asarray(pop.d3ndzdMdlnf(MM, FF, ZZ))
t0 = time.time()
out["hc2_f"] = np.array([2e-9, 1e-8])
out["hc2"] = np.array([pop.hc2(f) for f in out["hc2_f"]])
print(f"hc2: {time.time() - t0:.1f}s")
t0 = time.time()
out["Nbinaries"] = np.array(pop.compute_Nbinaries())
print(f"Nbinaries: {time.time() - t0:.1f}s  -> {float(out['Nbinaries']):.6g}")
np.savez(os.path.join(HERE, "golden_0_1.npz"), **out)
print("saved", os.path.join(HERE, "golden_0_1.npz"))
