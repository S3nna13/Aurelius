"""Optimizer algorithms for Aurelius."""

from .adafactor import (
    Adafactor,
    adafactor_factorized_second_moment,
    adafactor_reconstruct_second_moment,
)
from .lamb import LAMB, lamb_trust_ratio
from .lookahead import Lookahead
from .mars import Mars
from .prodigy import Prodigy
from .radam import RAdam, radam_rectification, radam_rho_inf, radam_rho_t
from .schedule_free_adamw import ScheduleFreeAdamW

__all__ = [
    "Adafactor",
    "adafactor_factorized_second_moment",
    "adafactor_reconstruct_second_moment",
    "LAMB",
    "lamb_trust_ratio",
    "Lookahead",
    "Mars",
    "Prodigy",
    "RAdam",
    "radam_rectification",
    "radam_rho_inf",
    "radam_rho_t",
    "ScheduleFreeAdamW",
]

# Register in OPTIMIZER_REGISTRY if the project exposes one.
try:
    OPTIMIZER_REGISTRY  # type: ignore[name-defined]  # noqa: F821
except NameError:
    pass
else:
    OPTIMIZER_REGISTRY["schedule_free_adamw"] = ScheduleFreeAdamW  # type: ignore[name-defined]  # noqa: F821
    OPTIMIZER_REGISTRY["mars"] = Mars  # type: ignore[name-defined]  # noqa: F821
