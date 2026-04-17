"""Modern activation function variants for LLMs.

Implements SwiGLU, GeGLU, ReGLU, Squared ReLU, and benchmarking utilities.

References:
    - Shazeer, 2020: "GLU Variants Improve Transformer"
    - So et al., 2021: "Primer: Searching for Efficient Transformers for Language Modeling"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ActivationConfig:
    """Configuration for a feed-forward network activation variant.

    Attributes:
        activation: One of "swiglu", "geglu", "reglu", "squared_relu",
                    "gelu", "silu".
        d_model:    Input / output dimension.
        d_ff:       Hidden (inner) dimension.
    """

    activation: str = "swiglu"
    d_model: int = 512
    d_ff: int = 2048


# ---------------------------------------------------------------------------
# GLU-family FFN modules
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network.

    output = W_down( SiLU(W_gate(x)) * W_up(x) )

    Uses two separate input projections (W_gate and W_up) of shape
    (d_model -> d_ff), and one output projection W_down (d_ff -> d_model).
    All projections are bias-free.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up   = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))


class GeGLUFFN(nn.Module):
    """GeGLU feed-forward network.

    output = W_down( GELU(W_gate(x)) * W_up(x) )

    Identical structure to SwiGLUFFN but uses GELU instead of SiLU.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up   = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.W_down(F.gelu(self.W_gate(x)) * self.W_up(x))


class ReGLUFFN(nn.Module):
    """ReGLU feed-forward network.

    output = W_down( ReLU(W_gate(x)) * W_up(x) )

    Identical structure to SwiGLUFFN but uses ReLU instead of SiLU.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up   = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.W_down(F.relu(self.W_gate(x)) * self.W_up(x))


# ---------------------------------------------------------------------------
# Squared-ReLU FFN
# ---------------------------------------------------------------------------

class SquaredReLUFFN(nn.Module):
    """Squared ReLU feed-forward network (from Primer).

    output = W2( relu(W1(x))^2 )

    Standard 2-layer MLP where the activation is ReLU^2(.).
    All projections are bias-free.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h = F.relu(self.W1(x))
        return self.W2(h * h)


# ---------------------------------------------------------------------------
# Standard 2-layer MLPs (GELU / SiLU)
# ---------------------------------------------------------------------------

class _StandardFFN(nn.Module):
    """Standard 2-layer MLP with a configurable pointwise activation."""

    def __init__(self, d_model: int, d_ff: int, activation: str) -> None:
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)
        self._activation = activation

    def forward(self, x: Tensor) -> Tensor:
        if self._activation == "gelu":
            return self.W2(F.gelu(self.W1(x)))
        elif self._activation == "silu":
            return self.W2(F.silu(self.W1(x)))
        raise ValueError(f"Unknown activation: {self._activation}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_GLU_VARIANTS = {"swiglu", "geglu", "reglu"}
_STANDARD_VARIANTS = {"gelu", "silu"}
_ALL_VARIANTS = _GLU_VARIANTS | {"squared_relu"} | _STANDARD_VARIANTS


class FFNFactory:
    """Factory for creating and analysing feed-forward network modules."""

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def create(self, config: ActivationConfig) -> nn.Module:
        """Instantiate the FFN described by *config*.

        Args:
            config: An ActivationConfig specifying the variant, d_model,
                    and d_ff.

        Returns:
            An nn.Module ready for use.

        Raises:
            ValueError: If config.activation is not recognised.
        """
        act = config.activation.lower()
        if act == "swiglu":
            return SwiGLUFFN(config.d_model, config.d_ff)
        elif act == "geglu":
            return GeGLUFFN(config.d_model, config.d_ff)
        elif act == "reglu":
            return ReGLUFFN(config.d_model, config.d_ff)
        elif act == "squared_relu":
            return SquaredReLUFFN(config.d_model, config.d_ff)
        elif act in _STANDARD_VARIANTS:
            return _StandardFFN(config.d_model, config.d_ff, act)
        raise ValueError(
            f"Unknown activation '{config.activation}'. "
            f"Choose from: {sorted(_ALL_VARIANTS)}"
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def parameter_count(self, module: nn.Module) -> int:
        """Return the total number of trainable parameters in *module*."""
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def flop_estimate(
        self,
        config: ActivationConfig,
        batch_size: int,
        seq_len: int,
    ) -> Dict[str, int]:
        """Estimate FLOPs for a single forward (and approximate backward) pass.

        Counting convention: one multiply-add = 2 FLOPs.

        For GLU variants and Squared ReLU we count three matmuls:
            W_gate and W_up  (each  B * T * d_model * d_ff * 2)
            W_down           (      B * T * d_ff    * d_model * 2)
        Total forward = 2 * 2 * B * T * d_model * d_ff

        For standard 2-layer MLPs we count two matmuls:
            W1: B * T * d_model * d_ff  * 2
            W2: B * T * d_ff    * d_model * 2
        Total forward = 2 * B * T * d_model * d_ff * 2

        The backward pass is approximated as 2x the forward FLOPs.

        Args:
            config:     Network configuration.
            batch_size: Number of sequences in the batch.
            seq_len:    Sequence length (tokens per sequence).

        Returns:
            Dict with keys "forward_flops" and "backward_flops".
        """
        B, T = batch_size, seq_len
        d_m, d_ff = config.d_model, config.d_ff
        act = config.activation.lower()

        if act in _GLU_VARIANTS or act == "squared_relu":
            # Two input projections + one output projection
            forward_flops = 2 * 2 * B * T * d_m * d_ff
        else:
            # Two projections (standard MLP)
            forward_flops = 2 * B * T * d_m * d_ff * 2

        return {
            "forward_flops": forward_flops,
            "backward_flops": 2 * forward_flops,
        }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class ActivationBenchmark:
    """Compare activation variants on the same input and measure sparsity.

    Args:
        configs: List of ActivationConfig objects, one per variant to
                 benchmark.
    """

    def __init__(self, configs: List[ActivationConfig]) -> None:
        self._configs = configs
        self._factory = FFNFactory()
        self._ffns: Dict[str, nn.Module] = {}
        for cfg in configs:
            key = cfg.activation
            if key in self._ffns:
                key = f"{cfg.activation}_{cfg.d_ff}"
            ffn = self._factory.create(cfg)
            ffn.train(False)   # inference mode — no eval() call
            self._ffns[key] = ffn

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_outputs(self, x: Tensor) -> Dict[str, Tensor]:
        """Run each registered FFN on *x* and return a dict of outputs.

        Args:
            x: Input tensor of shape (batch, seq, d_model).

        Returns:
            Dict mapping activation name to output tensor.
        """
        results: Dict[str, Tensor] = {}
        with torch.no_grad():
            for name, ffn in self._ffns.items():
                results[name] = ffn(x)
        return results

    # ------------------------------------------------------------------
    # Sparsity
    # ------------------------------------------------------------------

    def sparsity(self, x: Tensor, ffn: nn.Module) -> float:
        """Measure the fraction of near-zero hidden activations.

        Runs the first linear projection through its activation function
        and counts values with absolute magnitude below 1e-6.

        Args:
            x:   Input tensor of shape (batch, seq, d_model).
            ffn: An FFN module produced by FFNFactory.

        Returns:
            Float in [0, 1] proportion of activations that are
            effectively zero.
        """
        with torch.no_grad():
            if isinstance(ffn, SwiGLUFFN):
                hidden = F.silu(ffn.W_gate(x))
            elif isinstance(ffn, GeGLUFFN):
                hidden = F.gelu(ffn.W_gate(x))
            elif isinstance(ffn, ReGLUFFN):
                hidden = F.relu(ffn.W_gate(x))
            elif isinstance(ffn, SquaredReLUFFN):
                h = F.relu(ffn.W1(x))
                hidden = h * h
            elif isinstance(ffn, _StandardFFN):
                if ffn._activation == "gelu":
                    hidden = F.gelu(ffn.W1(x))
                else:
                    hidden = F.silu(ffn.W1(x))
            else:
                hidden = ffn(x)

        total = hidden.numel()
        if total == 0:
            return 0.0
        near_zero = (hidden.abs() < 1e-6).sum().item()
        return float(near_zero) / float(total)
