"""BitNet b1.58 — Ma et al. 2024.

Ternary weight quantization {-1, 0, +1} with straight-through estimator (STE)
and per-token 8-bit activation quantization.

Pure PyTorch — no external quantization libraries required.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# TernaryQuantizer
# ---------------------------------------------------------------------------


class TernaryQuantizer:
    """Quantize weights to {-1, 0, +1} using absmean scaling."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def quantize(self, W: Tensor) -> Tensor:
        """Return ternary weight in {-1, 0, +1} (same shape as W)."""
        scale = W.abs().mean()
        W_scaled = W / (scale + 1e-8)
        W_q = W_scaled.sign() * (W_scaled.abs() > self.threshold).float()
        return W_q

    def straight_through(self, W: Tensor) -> Tensor:
        """STE: forward sees quantized values, backward sees full gradients."""
        W_q = self.quantize(W)
        return W + (W_q - W).detach()

    def bit_width(self, W: Tensor) -> float:
        """Fraction of non-zero weights (1.0 = all non-zero, 0.0 = all zero)."""
        return (W != 0).float().mean().item()


# ---------------------------------------------------------------------------
# AbsMaxQuantizer
# ---------------------------------------------------------------------------


class AbsMaxQuantizer:
    """Per-token activation quantization (simulated n-bit integer)."""

    def __init__(self, n_bits: int = 8) -> None:
        self.n_bits = n_bits
        self._max_int = 2 ** (n_bits - 1) - 1
        self._min_int = -(2 ** (n_bits - 1))

    def quantize(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize per-token activations.

        Args:
            x: (B, T, D) float tensor

        Returns:
            (x_dequant, scale) where scale has shape (B, T, 1)
        """
        scale = x.abs().amax(dim=-1, keepdim=True) / (self._max_int)
        x_int = (x / (scale + 1e-8)).round().clamp(self._min_int, self._max_int)
        x_dequant = x_int * scale
        return x_dequant, scale

    def quantization_error(self, x: Tensor) -> float:
        """Mean absolute error between x and its dequantized approximation."""
        x_dequant, _ = self.quantize(x)
        return (x - x_dequant).abs().mean().item()


# ---------------------------------------------------------------------------
# BitLinear
# ---------------------------------------------------------------------------


class BitLinear(nn.Module):
    """Linear layer with 1.58-bit ternary weights and 8-bit activations."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # (1) Quantize activations per-token
        x_q, _ = AbsMaxQuantizer().quantize(x)
        # (2) STE-quantize weights to ternary
        W_q = TernaryQuantizer().straight_through(self.weight)
        # (3) Linear with ternary weights
        return F.linear(x_q, W_q, self.bias)


# ---------------------------------------------------------------------------
# BitNetBlock
# ---------------------------------------------------------------------------


class BitNetBlock(nn.Module):
    """Transformer block using BitLinear for all projection layers."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"  # noqa: S101
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Attention projections
        self.q_proj = BitLinear(d_model, d_model)
        self.k_proj = BitLinear(d_model, d_model)
        self.v_proj = BitLinear(d_model, d_model)
        self.out_proj = BitLinear(d_model, d_model)

        # FFN: 2-layer, 4x hidden
        self.ffn_up = BitLinear(d_model, 4 * d_model)
        self.ffn_down = BitLinear(4 * d_model, d_model)

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def _attention(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        Q = self.q_proj(x)  # (B, T, D)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split heads: (B, n_heads, T, head_dim)
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, T, T)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

    def forward(self, x: Tensor) -> Tensor:
        # Self-attention sublayer with residual
        x = x + self._attention(self.norm1(x))
        # FFN sublayer with residual
        h = self.ffn_up(self.norm2(x))
        h = F.gelu(h)
        h = self.ffn_down(h)
        x = x + h
        return x


# ---------------------------------------------------------------------------
# BitNetModel
# ---------------------------------------------------------------------------


class BitNetModel(nn.Module):
    """Full BitNet b1.58 language model."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([BitNetBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids: (B, T) long tensor

        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.embedding(input_ids)  # (B, T, D)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# BitNetAnalyzer
# ---------------------------------------------------------------------------


class BitNetAnalyzer:
    """Analyze quantization properties of a BitNet model."""

    def __init__(self) -> None:
        pass

    def _bitlinear_layers(self, model: nn.Module):
        return [m for m in model.modules() if isinstance(m, BitLinear)]

    def count_bitlinear_layers(self, model: nn.Module) -> int:
        """Count the total number of BitLinear layers in the model."""
        return len(self._bitlinear_layers(model))

    def model_sparsity(self, model: nn.Module) -> float:
        """Mean fraction of zero weights across all BitLinear layers.

        Uses TernaryQuantizer to determine what would be zeroed out.
        """
        layers = self._bitlinear_layers(model)
        if not layers:
            return 0.0
        quantizer = TernaryQuantizer()
        sparsities = []
        for layer in layers:
            W_q = quantizer.quantize(layer.weight.detach())
            sparsity = (W_q == 0).float().mean().item()
            sparsities.append(sparsity)
        return sum(sparsities) / len(sparsities)

    def effective_bits(self, model: nn.Module) -> float:
        """Effective bits per weight: log2(3) * (1 - sparsity)."""
        sparsity = self.model_sparsity(model)
        return math.log2(3) * (1.0 - sparsity)
