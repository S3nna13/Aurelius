"""Quantization utilities for Aurelius."""

from __future__ import annotations

from .gptq_calibration import (
    CalibrationStats,
    GPTQCalibrator,
    GPTQConfig,
)
from .mixed_precision_planner import (
    LayerSensitivity,
    MixedPrecisionPlan,
    MixedPrecisionPlanner,
)
from .awq_quantizer import (
    AWQConfig,
    AWQScaleSearch,
    AWQQuantizer,
    QUANTIZATION_REGISTRY,
)
from .bnb_emulation import (
    BnBConfig,
    OutlierDetector,
    BnBQuantizer,
)
from .quantization_aware_training import (
    QATConfig,
    FakeQuantize,
    QATWrapper,
)

__all__ = [
    # GPTQ calibration
    "GPTQConfig",
    "CalibrationStats",
    "GPTQCalibrator",
    # Mixed-precision planning
    "LayerSensitivity",
    "MixedPrecisionPlan",
    "MixedPrecisionPlanner",
    # AWQ
    "AWQConfig",
    "AWQScaleSearch",
    "AWQQuantizer",
    "QUANTIZATION_REGISTRY",
    # BnB emulation
    "BnBConfig",
    "OutlierDetector",
    "BnBQuantizer",
    # QAT
    "QATConfig",
    "FakeQuantize",
    "QATWrapper",
]
