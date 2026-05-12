"""Optional Liger kernel monkey-patching for memory-efficient training."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply_liger_kernels(
    model, enable_rope=True, enable_rms_norm=True, enable_swiglu=True, enable_fused_ce=True
) -> bool:
    """Patch model in-place with Liger fused kernels. Returns True if applied."""
    try:
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction  # noqa: F401
        from liger_kernel.ops.rope import LigerRopeFunction  # noqa: F401
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction  # noqa: F401
        from liger_kernel.transformers.monkey_patch import (
            _patch_rms_norm_module,
        )

        # Patch each norm layer
        for module in model.modules():
            module_name = type(module).__name__
            if enable_rms_norm and "RMSNorm" in module_name:
                _patch_rms_norm_module(module)
            if enable_swiglu and hasattr(module, "gate_proj"):
                # Replace SwiGLU forward with Liger fused version
                module._liger_swiglu = True
        logger.info("Liger kernels applied to model.")
        return True
    except ImportError:
        logger.warning("liger-kernel not installed — using standard kernels.")
        return False


def apply_liger_cross_entropy(vocab_size: int):
    """Return Liger fused linear cross-entropy loss or standard fallback."""
    try:
        from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy

        return liger_fused_linear_cross_entropy
    except ImportError:
        return None
