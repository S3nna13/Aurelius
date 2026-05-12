"""SageAttention 2 — quantized attention kernel integration.

Replaces F.scaled_dot_product_attention with SA2's INT8 Q/K kernel.
Requires: pip install sageattention>=2.0.0
Paper: arXiv:2411.10958 (ICML 2025)
"""

from __future__ import annotations

import torch
from torch import Tensor

_sa2_available = False
try:
    from sageattention import sageattn

    _sa2_available = True
except ImportError:
    pass


def sage_attention(
    q: Tensor, k: Tensor, v: Tensor, attn_mask=None, dropout_p: float = 0.0, is_causal: bool = False
) -> Tensor:
    """Drop-in replacement for F.scaled_dot_product_attention using SA2."""
    if _sa2_available and q.is_cuda and not q.requires_grad:
        # SA2: q, k, v must be (B, H, S, D) — same as SDPA
        return sageattn(q, k, v, tensor_layout="HND", is_causal=is_causal)
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
    )


__all__ = ["sage_attention", "_sa2_available"]
