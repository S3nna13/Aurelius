"""ZClip — adaptive gradient clipping via EMA z-score anomaly detection.

Re-exports the canonical implementation from ``src.optimizers.adaptive_clipper``
so that the training loop and optimizers share a single source of truth.
"""

from __future__ import annotations

from src.optimizers.adaptive_clipper import ZClip

__all__ = ["ZClip"]
