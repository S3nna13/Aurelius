"""Prefix Tuning: Layer-wise soft prefix tuning (Li & Liang 2021).

Prepends learned continuous prefix vectors to keys and values at each
attention layer. Supports MLP reparameterization for training stability.

Reference: Li & Liang 2021, arXiv:2101.00190
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class PrefixConfig:
    prefix_length: int = 10        # number of prefix tokens per layer
    n_layers: int = 24
    n_kv_heads: int = 8
    head_dim: int = 128
    dropout: float = 0.1
    use_mlp_reparameterization: bool = True  # use MLP for stability during training


class PrefixTuning(nn.Module):
    """Learned prefix K/V vectors prepended to each attention layer's keys and values.

    Architecture (with MLP reparameterization):
    - prefix_embedding: (prefix_length, embed_dim) where embed_dim = n_kv_heads * head_dim
    - MLP: embed_dim -> embed_dim * 2 -> n_layers * 2 * n_kv_heads * head_dim
    - At each layer: extract slice, reshape to (prefix_length, n_kv_heads, head_dim)

    Without reparameterization:
    - Direct parameters: (n_layers, 2, prefix_length, n_kv_heads, head_dim)
      (the 2 is for K and V)
    """

    def __init__(self, cfg: PrefixConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.dropout = nn.Dropout(p=cfg.dropout)

        if cfg.use_mlp_reparameterization:
            # embed_dim is the per-head-group size
            embed_dim = cfg.n_kv_heads * cfg.head_dim
            intermediate_dim = embed_dim * 2
            final_dim = cfg.n_layers * 2 * cfg.n_kv_heads * cfg.head_dim

            # Embedding table: (prefix_length, embed_dim)
            self.prefix_embedding = nn.Embedding(cfg.prefix_length, embed_dim)

            # MLP reparameterization: embed_dim -> intermediate_dim -> final_dim
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, intermediate_dim),
                nn.Tanh(),
                nn.Linear(intermediate_dim, final_dim),
            )
        else:
            # Direct parameters: (n_layers, 2, prefix_length, n_kv_heads, head_dim)
            self.prefix_params = nn.Parameter(
                torch.randn(cfg.n_layers, 2, cfg.prefix_length, cfg.n_kv_heads, cfg.head_dim)
                * 0.02
            )

    def _get_prefix_output(self) -> Tensor:
        """Run embedding + optional MLP to get all prefix K/V vectors.

        Returns:
            Tensor of shape (prefix_length, n_layers * 2 * n_kv_heads * head_dim)
            when using MLP, or raw params reshaped accordingly when not.
        """
        if self.cfg.use_mlp_reparameterization:
            # prefix_embedding.weight: (prefix_length, embed_dim)
            embedded = self.prefix_embedding.weight  # (P, embed_dim)
            embedded = self.dropout(embedded)
            # MLP output: (P, n_layers * 2 * n_kv_heads * head_dim)
            return self.mlp(embedded)
        else:
            # Direct params: (n_layers, 2, P, n_kv_heads, head_dim)
            # Apply dropout and reshape to (P, n_layers * 2 * n_kv_heads * head_dim)
            params = self.dropout(self.prefix_params)
            # Permute to (P, n_layers, 2, n_kv_heads, head_dim) then flatten last dims
            params = params.permute(2, 0, 1, 3, 4).contiguous()
            P = self.cfg.prefix_length
            return params.view(P, -1)

    def get_prefix_kv(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Get prefix K and V vectors for a specific layer.

        Args:
            layer_idx: Which transformer layer to get prefixes for.

        Returns:
            (prefix_k, prefix_v) each of shape (prefix_length, n_kv_heads, head_dim)
        """
        cfg = self.cfg
        kv_size = cfg.n_kv_heads * cfg.head_dim  # size per K or V slice

        # Get full output: (prefix_length, n_layers * 2 * n_kv_heads * head_dim)
        full = self._get_prefix_output()  # (P, n_layers * 2 * kv_size)

        # Each layer occupies a contiguous block of size 2 * kv_size
        layer_offset = layer_idx * 2 * kv_size
        k_start = layer_offset
        k_end = layer_offset + kv_size
        v_start = k_end
        v_end = v_start + kv_size

        # Slice and reshape to (prefix_length, n_kv_heads, head_dim)
        prefix_k = full[:, k_start:k_end].view(cfg.prefix_length, cfg.n_kv_heads, cfg.head_dim)
        prefix_v = full[:, v_start:v_end].view(cfg.prefix_length, cfg.n_kv_heads, cfg.head_dim)

        return prefix_k, prefix_v

    def get_all_prefix_kvs(self) -> list[tuple[Tensor, Tensor]]:
        """Get prefix K/V vectors for all layers.

        Returns:
            List of (prefix_k, prefix_v) tuples, one per layer.
            Each tensor has shape (prefix_length, n_kv_heads, head_dim).
        """
        cfg = self.cfg
        kv_size = cfg.n_kv_heads * cfg.head_dim

        # Compute once and slice per layer
        full = self._get_prefix_output()  # (P, n_layers * 2 * kv_size)

        result = []
        for layer_idx in range(cfg.n_layers):
            layer_offset = layer_idx * 2 * kv_size
            k_start = layer_offset
            k_end = layer_offset + kv_size
            v_start = k_end
            v_end = v_start + kv_size

            prefix_k = full[:, k_start:k_end].view(cfg.prefix_length, cfg.n_kv_heads, cfg.head_dim)
            prefix_v = full[:, v_start:v_end].view(cfg.prefix_length, cfg.n_kv_heads, cfg.head_dim)
            result.append((prefix_k, prefix_v))

        return result


def apply_prefix_to_attention(
    prefix_tuning: PrefixTuning,
    k: Tensor,   # (B, S, n_kv_heads, head_dim) — original keys
    v: Tensor,   # (B, S, n_kv_heads, head_dim) — original values
    layer_idx: int,
) -> tuple[Tensor, Tensor]:
    """Prepend prefix K/V vectors to the full k, v tensors.

    Args:
        prefix_tuning: PrefixTuning module with learned prefix parameters.
        k: Original keys of shape (B, S, n_kv_heads, head_dim).
        v: Original values of shape (B, S, n_kv_heads, head_dim).
        layer_idx: Which layer's prefix to use.

    Returns:
        Tuple of (new_k, new_v) each of shape (B, prefix_length + S, n_kv_heads, head_dim).
    """
    B = k.shape[0]

    # Get prefix: (prefix_length, n_kv_heads, head_dim)
    prefix_k, prefix_v = prefix_tuning.get_prefix_kv(layer_idx)

    # Expand to batch dim: (B, prefix_length, n_kv_heads, head_dim)
    prefix_k = prefix_k.unsqueeze(0).expand(B, -1, -1, -1)
    prefix_v = prefix_v.unsqueeze(0).expand(B, -1, -1, -1)

    # Concatenate along sequence dimension
    new_k = torch.cat([prefix_k, k], dim=1)
    new_v = torch.cat([prefix_v, v], dim=1)

    return new_k, new_v


class PrefixTuningTrainer:
    """Freezes backbone model, trains only PrefixTuning parameters."""

    def __init__(
        self,
        model: nn.Module,
        prefix_tuning: PrefixTuning,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.prefix_tuning = prefix_tuning
        self.optimizer = optimizer

    def freeze_backbone(self) -> int:
        """Freeze all model (backbone) parameters.

        Returns:
            Count of frozen parameters (number of tensors, not elements).
        """
        count = 0
        for p in self.model.parameters():
            p.requires_grad = False
            count += 1
        return count

    def unfreeze_backbone(self) -> None:
        """Restore requires_grad=True to all backbone model parameters."""
        for p in self.model.parameters():
            p.requires_grad = True

    def trainable_params(self) -> list[nn.Parameter]:
        """Return only the prefix_tuning parameters (those with requires_grad=True)."""
        return list(self.prefix_tuning.parameters())

    def param_count(self) -> dict[str, int]:
        """Count parameters across backbone + prefix.

        Returns:
            {"total": int, "trainable": int, "frozen": int}
        """
        all_params = list(self.model.parameters()) + list(self.prefix_tuning.parameters())
        total = sum(p.numel() for p in all_params)
        trainable = sum(p.numel() for p in all_params if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}


# ---------------------------------------------------------------------------
# New API: PrefixEncoder, PrefixTuningModel, PrefixTuner
# ---------------------------------------------------------------------------

@dataclass
class PrefixEncoderConfig:
    """Configuration for the new-style PrefixEncoder (Li & Liang 2021).

    Compatible with the tiny test config:
        AureliusConfig(n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
                       head_dim=32, d_ff=128, vocab_size=256, max_seq_len=512)
    """

    prefix_length: int = 10
    d_model: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    reparameterize: bool = True
    reparam_hidden: int = 512


class PrefixEncoder(nn.Module):
    """Learns prefix representations for all layers.

    If reparameterize is True, uses an MLP bottleneck for training stability:
        embedding -> Linear -> Tanh -> Linear -> reshape
    Otherwise, directly learns the prefix parameters.

    forward() returns shape (n_layers, 2, prefix_length, d_model)
        where the 2 dimension is [key_prefix, value_prefix].
    """

    def __init__(self, cfg: PrefixEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        total_dim = cfg.n_layers * 2 * cfg.d_model  # per-prefix-token output

        if cfg.reparameterize:
            self.embedding = nn.Embedding(cfg.prefix_length, cfg.d_model)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.reparam_hidden),
                nn.Tanh(),
                nn.Linear(cfg.reparam_hidden, total_dim),
            )
        else:
            # Direct learnable parameters
            self.prefix_params = nn.Parameter(
                torch.randn(cfg.prefix_length, total_dim) * 0.02
            )

        self.dropout = nn.Dropout(p=cfg.dropout)

    def forward(self) -> Tensor:
        """Compute prefix representations.

        Returns:
            Tensor of shape (n_layers, 2, prefix_length, d_model).
        """
        cfg = self.cfg

        if cfg.reparameterize:
            # (prefix_length, d_model) -> (prefix_length, total_dim)
            raw = self.mlp(self.embedding.weight)
        else:
            raw = self.prefix_params

        raw = self.dropout(raw)

        # raw: (prefix_length, n_layers * 2 * d_model)
        # reshape to (prefix_length, n_layers, 2, d_model), then permute
        out = raw.view(cfg.prefix_length, cfg.n_layers, 2, cfg.d_model)
        # -> (n_layers, 2, prefix_length, d_model)
        out = out.permute(1, 2, 0, 3).contiguous()
        return out


def prepend_prefix_to_kv(
    past_kv: tuple[Tensor, Tensor] | None,
    prefix_kv: Tensor,
    layer_idx: int,
) -> tuple[Tensor, Tensor]:
    """Prepend prefix K,V to existing KV cache or create new KV with prefix.

    Args:
        past_kv: Existing (k, v) each of shape (B, S, ...) or None.
        prefix_kv: Full prefix tensor of shape (n_layers, 2, prefix_length, d_model).
        layer_idx: Which layer's prefix to extract.

    Returns:
        (new_k, new_v) with prefix prepended along sequence dimension.
        If past_kv is None, returns prefix-only tensors with batch dim 1.
    """
    # Extract this layer's prefix K and V: each (prefix_length, d_model)
    prefix_k = prefix_kv[layer_idx, 0]  # (prefix_length, d_model)
    prefix_v = prefix_kv[layer_idx, 1]  # (prefix_length, d_model)

    if past_kv is None:
        # Return with batch dim = 1
        return prefix_k.unsqueeze(0), prefix_v.unsqueeze(0)

    k, v = past_kv
    B = k.shape[0]

    # Expand prefix to batch: (1, prefix_length, d_model) -> (B, prefix_length, d_model)
    prefix_k = prefix_k.unsqueeze(0).expand(B, -1, -1)
    prefix_v = prefix_v.unsqueeze(0).expand(B, -1, -1)

    # Prepend along sequence dimension
    new_k = torch.cat([prefix_k, k], dim=1)
    new_v = torch.cat([prefix_v, v], dim=1)

    return new_k, new_v


class PrefixTuningModel(nn.Module):
    """Wraps a frozen backbone model with a trainable PrefixEncoder.

    The backbone is expected to follow the Aurelius model API:
        loss, logits, pkv = model(input_ids)
    With attributes: model.embed, model.layers, model.norm, model.lm_head

    This module freezes the backbone and only trains the PrefixEncoder.
    """

    def __init__(self, backbone: nn.Module, prefix_cfg: PrefixEncoderConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.prefix_encoder = PrefixEncoder(prefix_cfg)
        self.prefix_cfg = prefix_cfg

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _is_backbone_frozen(self) -> bool:
        """Check that all backbone parameters are frozen."""
        return all(not p.requires_grad for p in self.backbone.parameters())

    def trainable_param_count(self) -> int:
        """Count of trainable (prefix encoder) parameters."""
        return sum(p.numel() for p in self.prefix_encoder.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        """Count of all parameters (backbone + prefix)."""
        return sum(p.numel() for p in self.parameters())

    def get_prefix_logits(self, input_ids: Tensor) -> Tensor:
        """Forward pass with prefix-augmented computation.

        Uses the backbone's components directly:
            1. Embed input_ids via backbone.embed
            2. Compute prefix KV from prefix_encoder
            3. Prepend prefix to the hidden states before each layer
            4. Apply backbone.norm and backbone.lm_head

        Args:
            input_ids: (B, T) token indices.

        Returns:
            logits: (B, T, V) — vocabulary logits for each position.
        """
        B, T = input_ids.shape

        # Get prefix KV: (n_layers, 2, prefix_length, d_model)
        prefix_kv = self.prefix_encoder()

        # Embed input tokens: (B, T, d_model)
        x = self.backbone.embed(input_ids)

        # Build prefix hidden states for prepending
        # prefix_kv has shape (n_layers, 2, prefix_length, d_model)
        # We use the mean of K and V prefix as a "virtual token" representation
        # But simpler: prepend the prefix as extra hidden states

        # Create prefix hidden embeddings from the average of K,V prefix
        # (prefix_length, d_model)
        prefix_hidden = prefix_kv[0].mean(dim=0)  # average K and V from first layer
        # (1, prefix_length, d_model) -> (B, prefix_length, d_model)
        prefix_hidden = prefix_hidden.unsqueeze(0).expand(B, -1, -1)

        # Prepend prefix to input hidden states
        # (B, prefix_length + T, d_model)
        x = torch.cat([prefix_hidden, x], dim=1)

        # Get RoPE frequencies from backbone if available
        has_freqs = hasattr(self.backbone, 'freqs_cis')

        total_len = x.shape[1]

        if has_freqs:
            freqs_cis = self.backbone.freqs_cis[:total_len]
        else:
            freqs_cis = None

        # Pass through transformer layers
        for i, layer in enumerate(self.backbone.layers):
            if freqs_cis is not None:
                x, _kv = layer(x, freqs_cis)
            else:
                x, _kv = layer(x, None)

        # Final norm + LM head
        x = self.backbone.norm(x)
        logits = self.backbone.lm_head(x)

        # Slice off the prefix positions to return only input token logits
        # logits: (B, prefix_length + T, V) -> (B, T, V)
        logits = logits[:, self.prefix_cfg.prefix_length:, :]

        return logits


class PrefixTuner:
    """Training helper for prefix tuning.

    Manages optimizer and provides a train_step that returns loss and metadata.
    """

    def __init__(
        self,
        model: PrefixTuningModel,
        lr: float = 1e-3,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(
                model.prefix_encoder.parameters(), lr=lr
            )

    def train_step(self, input_ids: Tensor) -> dict:
        """Execute one training step.

        Args:
            input_ids: (B, T) token indices. Uses shifted cross-entropy
                       (predict next token).

        Returns:
            dict with keys:
                - "loss": scalar loss value (float)
                - "n_prefix_params": number of trainable prefix parameters (int)
        """
        self.model.train()

        # Forward pass: get logits (B, T, V)
        logits = self.model.get_prefix_logits(input_ids)

        # Shifted cross-entropy: predict next token
        # logits[:, :-1] predicts, input_ids[:, 1:] are targets
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Backward + optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        n_prefix_params = self.model.trainable_param_count()

        return {
            "loss": loss.item(),
            "n_prefix_params": n_prefix_params,
        }
