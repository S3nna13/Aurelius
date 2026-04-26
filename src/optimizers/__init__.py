"""Optimizer algorithms for Aurelius."""

from .adabelief_optimizer import AdaBelief
from .adafactor import (
    Adafactor,
    adafactor_factorized_second_moment,
    adafactor_reconstruct_second_moment,
)
from .lamb import LAMB, lamb_trust_ratio
from .lookahead import Lookahead
from .mars import Mars
from .muonclip import MuonClip
from .mup_scaling import MuPConfig, MuPScaler
from .prodigy import Prodigy
from .radam import RAdam, radam_rectification, radam_rho_inf, radam_rho_t
from .ranger_optimizer import Ranger
from .schedule_free_adamw import ScheduleFreeAdamW

__all__ = [
    "MuPConfig",
    "MuPScaler",
    "Adafactor",
    "adafactor_factorized_second_moment",
    "adafactor_reconstruct_second_moment",
    "LAMB",
    "lamb_trust_ratio",
    "Lookahead",
    "Mars",
    "MuonClip",
    "Prodigy",
    "RAdam",
    "radam_rectification",
    "radam_rho_inf",
    "radam_rho_t",
    "ScheduleFreeAdamW",
    "Ranger",
    "AdaBelief",
]

# ---------------------------------------------------------------------------
# OPTIMIZER_REGISTRY — central lookup used by training harnesses.
# Created here if it does not already exist in the surrounding namespace so
# that integration tests can import it directly from this package.
# ---------------------------------------------------------------------------
try:
    OPTIMIZER_REGISTRY  # type: ignore[name-defined]  # noqa: F821
except NameError:
    OPTIMIZER_REGISTRY: dict = {}  # type: ignore[no-redef]

OPTIMIZER_REGISTRY["schedule_free_adamw"] = ScheduleFreeAdamW  # type: ignore[name-defined]
OPTIMIZER_REGISTRY["mars"] = Mars  # type: ignore[name-defined]
OPTIMIZER_REGISTRY["muonclip"] = MuonClip  # type: ignore[name-defined]
OPTIMIZER_REGISTRY["ranger"] = Ranger  # type: ignore[name-defined]
OPTIMIZER_REGISTRY["adabelief"] = AdaBelief  # type: ignore[name-defined]
