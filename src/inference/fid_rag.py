"""Fusion-in-Decoder RAG — multi-passage retrieval-augmented generation.

Implements FiD-style encoding where each retrieved passage is independently
encoded through the backbone, then fused via one of three strategies
(concatenation, cross-attention, or averaging) before being prepended to the
input hidden states for decoding.

Reference: Izacard & Grave (2020), "Leveraging Passage Retrieval with
Generative Models for Open Domain Question Answering".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List


@dataclass
class FiDConfig:
    """Configuration for Fusion-in-Decoder RAG."""
    n_passages: int = 5
    max_passage_len: int = 128
    fusion_method: str = "concat"  # "concat" | "cross_attention" | "average"
    d_model: int = 64


def encode_passages(
    model: nn.Module,
    passage_ids_list: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Encode each passage independently through the backbone.

    Uses a forward hook on the last transformer block to capture hidden
    states before the final norm and LM head.

    Args:
        model: AureliusTransformer instance.
        passage_ids_list: List of (B, T) token ID tensors, one per passage.

    Returns:
        List of hidden-state tensors, each (B, T, d_model).
    """
    encoded: List[torch.Tensor] = []

    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured[0] = h.detach()

    last_layer = model.layers[-1]
    handle = last_layer.register_forward_hook(_hook)

    try:
        for pids in passage_ids_list:
            if pids.dim() == 1:
                pids = pids.unsqueeze(0)
            captured: list = [None]
            with torch.no_grad():
                _ = model(pids)  # loss, logits, pkv — plain tuple
            encoded.append(captured[0])
    finally:
        handle.remove()

    return encoded


def concat_fusion(encoded_passages: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate encoded passages along the sequence dimension.

    Args:
        encoded_passages: List of (B, T_i, d) tensors.

    Returns:
        (B, T_total, d) — all passages concatenated on dim=1.
    """
    return torch.cat(encoded_passages, dim=1)


def average_fusion(encoded_passages: List[torch.Tensor]) -> torch.Tensor:
    """Mean-pool each passage to (B, d), then stack and mean to (B, 1, d).

    Args:
        encoded_passages: List of (B, T_i, d) tensors.

    Returns:
        (B, 1, d) — averaged representation.
    """
    pooled = [p.mean(dim=1) for p in encoded_passages]  # each (B, d)
    stacked = torch.stack(pooled, dim=0)  # (N, B, d)
    return stacked.mean(dim=0).unsqueeze(1)  # (B, 1, d)


class FusionCrossAttention(nn.Module):
    """Cross-attention: query from input, key/value from passages.

    Args:
        d_model: Hidden dimension.
        n_heads: Number of attention heads (default 4).
    """

    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        kv_source: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, T_q, d) — input hidden states.
            kv_source: (B, T_kv, d) — concatenated passage hidden states.

        Returns:
            (B, T_q, d) — cross-attended output.
        """
        B, T_q, _ = query.shape
        T_kv = kv_source.shape[1]

        q = self.q_proj(query).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_source).view(B, T_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_source).view(B, T_kv, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, T_q, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(out)


class FiDModel(nn.Module):
    """Fusion-in-Decoder wrapper around an Aurelius backbone.

    Encodes passages through the backbone, fuses them, prepends the fused
    representation to the input hidden states, and computes logits.

    Args:
        backbone: AureliusTransformer instance.
        config: FiDConfig with fusion settings.
    """

    def __init__(self, backbone: nn.Module, config: FiDConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config

        if config.fusion_method == "cross_attention":
            n_heads = max(1, config.d_model // 16)
            # Ensure d_model divisible by n_heads
            while config.d_model % n_heads != 0:
                n_heads -= 1
            self.cross_attn = FusionCrossAttention(config.d_model, n_heads=n_heads)
        else:
            self.cross_attn = None

    def forward(
        self,
        input_ids: torch.Tensor,
        passage_ids_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) — input token IDs.
            passage_ids_list: List of (B, T_p) passage token ID tensors.

        Returns:
            (B, T_out, V) — logits over vocabulary.
        """
        # 1. Encode passages
        encoded = encode_passages(self.backbone, passage_ids_list)

        # 2. Get input hidden states (via hook on last layer)
        input_hidden = self._get_hidden(input_ids)

        # 3. Fuse passages
        if self.config.fusion_method == "concat":
            fused = concat_fusion(encoded)
        elif self.config.fusion_method == "average":
            fused = average_fusion(encoded)
        elif self.config.fusion_method == "cross_attention":
            kv_source = concat_fusion(encoded)
            fused = self.cross_attn(input_hidden, kv_source)
        else:
            raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")

        # 4. For cross_attention, fused replaces input; for others, prepend
        if self.config.fusion_method == "cross_attention":
            combined = torch.cat([fused, input_hidden], dim=1)
        else:
            combined = torch.cat([fused, input_hidden], dim=1)

        # 5. Project to vocab via backbone's norm + lm_head
        normed = self.backbone.norm(combined)
        logits = self.backbone.lm_head(normed)

        return logits

    def _get_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract last-layer hidden states for input_ids."""
        captured: list = [None]

        def _hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[0] = h.detach()

        handle = self.backbone.layers[-1].register_forward_hook(_hook)
        try:
            with torch.no_grad():
                _ = self.backbone(input_ids)
            return captured[0]
        finally:
            handle.remove()


class FiDGenerator:
    """Autoregressive generator using Fusion-in-Decoder.

    Args:
        fid_model: FiDModel instance.
        eos_token_id: Optional end-of-sequence token.
    """

    def __init__(
        self,
        fid_model: FiDModel,
        eos_token_id: int | None = None,
    ) -> None:
        self.fid_model = fid_model
        self.eos_token_id = eos_token_id

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        passages: List[torch.Tensor],
        max_tokens: int = 16,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate tokens conditioned on input and retrieved passages.

        Args:
            input_ids: (B, T) — prompt token IDs.
            passages: List of (B, T_p) passage token ID tensors.
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature (>0). Uses greedy if <= 0.

        Returns:
            (B, T + max_tokens) — concatenated input + generated tokens.
        """
        generated = input_ids.clone()

        for _ in range(max_tokens):
            logits = self.fid_model(generated, passages)  # (B, T_out, V)
            # Take logits at last position
            next_logits = logits[:, -1, :]  # (B, V)

            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if self.eos_token_id is not None and (next_token == self.eos_token_id).all():
                break

        return generated
