"""Package shim for Constitutional AI v3.

Keeps the legacy ``aurelius.alignment`` namespace aligned with the
source-side implementation used by the test suite.
"""

from __future__ import annotations

from src.alignment.constitutional_ai_v3 import (
    CAITrainer,
    ConstitutionalFilter,
    ConstitutionalPrinciple,
    CritiqueHead,
    RevisionScorer,
)

__all__ = [
    "CAITrainer",
    "ConstitutionalFilter",
    "ConstitutionalPrinciple",
    "CritiqueHead",
    "RevisionScorer",
]
