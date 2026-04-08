"""Dense-to-MoE upcycling for AureliusTransformer.

Converts a trained dense transformer's SwiGLUFFN layers into SparseMoEFFN layers
by copying the dense FFN weights into each expert copy (identical initialization).
The router is initialized with small random weights so all experts start equal.

Reference: Komatsuzaki et al. (2022) "Sparse Upcycling" arXiv:2212.09535
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import AureliusConfig
from .ffn import SwiGLUFFN
from .moe import MoEConfig, SparseMoEFFN


@dataclass
class UpcycleConfig:
    n_experts: int = 8       # how many expert copies to make per layer
    top_k: int = 2           # experts activated per token
    load_balance_alpha: float = 0.01
    upcycle_all_layers: bool = True   # if False, only upcycle layers in layer_indices
    layer_indices: list[int] | None = None  # only used when upcycle_all_layers=False


def upcycle_ffn(
    dense_ffn: SwiGLUFFN,
    moe_cfg: MoEConfig,
    model_cfg: AureliusConfig,
) -> SparseMoEFFN:
    """Convert one dense SwiGLUFFN into a SparseMoEFFN.

    Each expert is initialized as a deep copy of the dense FFN.
    The router is re-initialized with small random weights (std=0.01).

    Args:
        dense_ffn: The trained dense FFN to copy into experts.
        moe_cfg: MoE configuration (n_experts, top_k, load_balance_alpha).
        model_cfg: Model configuration (needed to construct SparseMoEFFN).

    Returns:
        SparseMoEFFN with n_experts copies of the dense FFN weights.
    """
    moe = SparseMoEFFN(model_cfg, moe_cfg)
    # Replace each expert's weights with a deep copy of the dense FFN
    for i in range(moe_cfg.n_experts):
        moe.experts[i] = copy.deepcopy(dense_ffn)
    # Router stays randomly initialized (small std=0.01 from SparseMoEFFN.__init__)
    return moe


def _make_moe_block_forward(block):
    """Return a bound forward method for a TransformerBlock whose ffn is SparseMoEFFN.

    The standard TransformerBlock.forward does:
        x = x + self.ffn(self.ffn_norm(x))
    But SparseMoEFFN returns (output, aux_loss). This wrapper unpacks the tuple
    and discards aux_loss so the block's output contract is unchanged.
    """
    def forward(
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask=None,
        past_kv=None,
    ):
        attn_out, kv = block.attn(block.attn_norm(x), freqs_cis, mask, past_kv)
        x = x + attn_out
        ffn_out, _aux_loss = block.ffn(block.ffn_norm(x))
        x = x + ffn_out
        return x, kv

    return forward


def upcycle_model(
    model: nn.Module,
    model_cfg: AureliusConfig,
    upcycle_cfg: UpcycleConfig | None = None,
) -> nn.Module:
    """Upcycle a dense AureliusTransformer in-place.

    Replaces SwiGLUFFN layers with SparseMoEFFN layers. All other
    weights (attention, norms, embeddings) are kept identical.

    Patches each upcycled TransformerBlock's forward method so that the
    SparseMoEFFN's (output, aux_loss) return is handled correctly (aux_loss
    is discarded; the block's output contract remains unchanged).

    Args:
        model: The trained dense AureliusTransformer.
        model_cfg: AureliusConfig used to build the model.
        upcycle_cfg: Upcycling configuration. Uses defaults if None.

    Returns:
        The modified model (same object, modified in-place).

    Raises:
        ValueError: If no SwiGLUFFN layers are found.
    """
    if upcycle_cfg is None:
        upcycle_cfg = UpcycleConfig()

    moe_cfg = MoEConfig(
        n_experts=upcycle_cfg.n_experts,
        top_k=upcycle_cfg.top_k,
        load_balance_alpha=upcycle_cfg.load_balance_alpha,
    )

    layers = model.layers  # nn.ModuleList of TransformerBlock

    if upcycle_cfg.upcycle_all_layers:
        target_indices = list(range(len(layers)))
    else:
        target_indices = list(upcycle_cfg.layer_indices or [])

    # Verify at least one SwiGLUFFN exists among targets
    found = False
    for i in target_indices:
        if isinstance(layers[i].ffn, SwiGLUFFN):
            found = True
            break
    if not found:
        raise ValueError(
            "No SwiGLUFFN layers found among the target layer indices. "
            "Ensure the model has dense SwiGLUFFN FFN layers before upcycling."
        )

    for i in target_indices:
        layer = layers[i]
        if isinstance(layer.ffn, SwiGLUFFN):
            layer.ffn = upcycle_ffn(layer.ffn, moe_cfg, model_cfg)
            # Patch forward to handle (output, aux_loss) tuple from SparseMoEFFN.
            # Assign as an instance attribute so it takes precedence over the class method.
            # The closure captures `layer` directly (no implicit self needed).
            layer.forward = _make_moe_block_forward(layer)

    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total, trainable, and frozen parameters.

    Returns dict with keys: total, trainable, frozen.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
