r"""Named population models for the semi-analytic merger-rate ansatz.

Every model here is a parameter set for the *same* phenomenological rate
density that :class:`~fastropop.semi_analytic.SemiAnalyticPopulation` already
implements,

.. math::
   \frac{d^2 n}{dz\, d\log_{10}\mathcal{M}}
   = \dot n_0
     \left(\frac{\mathcal{M}}{10^7 M_\odot}\right)^{-\alpha_M}
     e^{-\mathcal{M}/\mathcal{M}_*}
     (1+z)^{\beta_z} e^{-z/z_0}
     \frac{dt_r}{dz},

so switching between a "light seed" and a "heavy seed" scenario is a change of
five numbers, not a change of physics. The names follow the literature and
describe the *shape* a seeding channel is expected to produce; none of these
models resolves seed formation, occupation fractions or accretion. A physical
seeding chain is a separate piece of work -- see ``docs/seeds.md``.

Mass ratio
----------
The ansatz has no mass-ratio dependence. ``q_prescription`` records what the
source assumed so that comparisons are like-for-like: ``"equal"`` means every
binary is taken to have :math:`q=1`, which *maximises* the radiated energy at
fixed chirp mass, while ``"powerlaw"`` carries an index ``beta_q`` for
:math:`p(q)\propto q^{\beta_q}`.
"""

from dataclasses import dataclass, field
from typing import Optional

from .constants import MsunMKS, pc, pcinMKS, yr, yrinMKS

__all__ = ["SeedModel", "SEED_MODELS", "get_seed_model", "list_seed_models"]

_MPC_M = 1e6 * pc * pcinMKS          # one megaparsec in metres
_YR_S = yr * yrinMKS                 # one year in seconds


def _n0_si(n0_per_Mpc3_per_yr):
    """Convert a rate density from Mpc^-3 yr^-1 to the package's SI units."""
    return n0_per_Mpc3_per_yr / (_MPC_M ** 3 * _YR_S)


@dataclass(frozen=True)
class SeedModel:
    """One named parameter set for the semi-analytic rate density.

    Attributes
    ----------
    alphaM, Mstar, betaz, z0, n0
        The five population parameters. ``Mstar`` is in kilograms and ``n0`` in
        SI (m^-3 s^-1), matching :mod:`fastropop.constants`.
    q_prescription
        ``"equal"`` (q = 1) or ``"powerlaw"``; see the module docstring.
    beta_q
        Index of p(q) when ``q_prescription == "powerlaw"``.
    total_rate_per_yr
        The all-sky merger rate the source normalised to, if stated. Useful as
        a check after integrating, and ``None`` when the source did not fix one.
    norm_zmax, norm_mass_range
        The integration domain under which ``total_rate_per_yr`` is recovered.
        These are not cosmetic: the Caliskan models only reproduce their stated
        200/yr when integrated to z ~ 20, because the Heavy Seed rate peaks near
        z ~ 15 and gains a further 10% between z = 20 and z = 60.
    reference, notes
        Provenance. Every model must say where its numbers came from.
    """

    name: str
    alphaM: float
    Mstar: float
    betaz: float
    z0: float
    n0: float
    reference: str
    q_prescription: str = "equal"
    beta_q: Optional[float] = None
    total_rate_per_yr: Optional[float] = None
    norm_zmax: Optional[float] = None
    norm_mass_range: Optional[tuple] = None
    notes: str = ""

    def population_params(self):
        """A ``population_params`` dict for :class:`SemiAnalyticPopulation`."""
        return {"n0": self.n0, "alphaM": self.alphaM, "Mstar": self.Mstar,
                "betaz": self.betaz, "z0": self.z0}


_CALISKAN = ("Caliskan, Anil Kumar, Kamionkowski & Cheng, arXiv:2506.18965, Tab. I. "
             "Normalised so the total merger rate integrates to 200/yr.")
_CALISKAN_NOTE = ("Their Eq. (3) is this ansatz with the same 10^7 Msun pivot. They "
                  "assume spinless, equal-mass binaries; they note q=1 maximises "
                  "Omega_GW at fixed chirp mass. VERIFIED: integrating this parameter "
                  "set over 250 Msun < Mc < 1e9 Msun and z < 20 reproduces their stated "
                  "200/yr to better than 1% for all three models (199.1 / 198.1 / 199.0 "
                  "for light / heavy / ultra-light).")

SEED_MODELS = {
    "caliskan25_light": SeedModel(
        name="caliskan25_light", alphaM=0.8, Mstar=5e6 * MsunMKS, betaz=7.2, z0=1.5,
        n0=_n0_si(7.14e-18), reference=_CALISKAN, q_prescription="equal",
        total_rate_per_yr=200.0, norm_zmax=20.0,
        norm_mass_range=(250.0, 1e9), notes="Model 1 (Light Seed). " + _CALISKAN_NOTE),
    "caliskan25_heavy": SeedModel(
        name="caliskan25_heavy", alphaM=-0.85, Mstar=7e4 * MsunMKS, betaz=4.8, z0=3.3,
        n0=_n0_si(2.01e-11), reference=_CALISKAN, q_prescription="equal",
        total_rate_per_yr=200.0, norm_zmax=20.0,
        norm_mass_range=(250.0, 1e9), notes="Model 2 (Heavy Seed). " + _CALISKAN_NOTE),
    "caliskan25_ultralight": SeedModel(
        name="caliskan25_ultralight", alphaM=1.6, Mstar=5e5 * MsunMKS, betaz=7.2, z0=1.5,
        n0=_n0_si(2.97e-21), reference=_CALISKAN, q_prescription="equal",
        total_rate_per_yr=200.0, norm_zmax=20.0,
        norm_mass_range=(250.0, 1e9), notes="Model 3 (Ultra-Light Seed). " + _CALISKAN_NOTE),
}


def get_seed_model(name):
    """Look up a model by name, with a helpful error listing the alternatives."""
    try:
        return SEED_MODELS[name]
    except KeyError:
        raise KeyError(
            f"unknown seed model {name!r}; available: {sorted(SEED_MODELS)}") from None


def list_seed_models():
    """Every registered model name."""
    return sorted(SEED_MODELS)
