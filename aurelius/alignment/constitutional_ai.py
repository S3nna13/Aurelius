"""Package shim for Constitutional AI.

Re-exports the source-side implementation so the legacy
``aurelius.alignment`` import path remains valid.
"""

from __future__ import annotations

from src.alignment.constitutional_ai import (
    CAILoss,
    CAITrainer,
    Constitution,
    ConstitutionalPrinciple,
    ConstitutionalScorer,
)

__all__ = [
    "CAILoss",
    "CAITrainer",
    "Constitution",
    "ConstitutionalPrinciple",
    "ConstitutionalScorer",
]
