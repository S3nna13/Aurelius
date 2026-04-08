"""Activation function library with properties and learnable meta-activation for NAS.

Provides:
- ACTIVATION_REGISTRY: catalog of activation functions with metadata
- get_activation: factory returning nn.Module for a named activation
- Mish, Squareplus, StarReLU: custom activation nn.Module subclasses
- LearnableActivation: 2-layer MLP approximating any smooth activation
- ActivationSearchSpace: discrete + continuous NAS search space
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Activation metadata ───────────────────────────────────────────────


@dataclass
class ActivationInfo:
    name: str
    is_gated: bool          # gated = uses two projections (like SwiGLU)
    is_smooth: bool         # smooth = differentiable everywhere
    approx_monotonic: bool  # approximately monotonic
    description: str


ACTIVATION_REGISTRY: dict[str, ActivationInfo] = {
    "gelu": ActivationInfo(
        name="gelu",
        is_gated=False,
        is_smooth=True,
        approx_monotonic=True,
        description=(
            "Gaussian Error Linear Unit. Smooth approximation to ReLU using the "
            "Gaussian CDF: x * Φ(x). Default activation in most modern transformers."
        ),
    ),
    "relu": ActivationInfo(
        name="relu",
        is_gated=False,
        is_smooth=False,
        approx_monotonic=True,
        description=(
            "Rectified Linear Unit. f(x) = max(0, x). Simple, fast, non-smooth at 0."
        ),
    ),
    "silu": ActivationInfo(
        name="silu",
        is_gated=False,
        is_smooth=True,
        approx_monotonic=True,
        description=(
            "Sigmoid Linear Unit (Swish). f(x) = x * sigmoid(x). Smooth, "
            "non-monotonic with a small dip below zero."
        ),
    ),
    "mish": ActivationInfo(
        name="mish",
        is_gated=False,
        is_smooth=True,
        approx_monotonic=True,
        description=(
            "Mish activation. f(x) = x * tanh(softplus(x)). Smooth, self-regularized, "
            "non-monotonic. Often outperforms ReLU in deep networks."
        ),
    ),
    "squareplus": ActivationInfo(
        name="squareplus",
        is_gated=False,
        is_smooth=True,
        approx_monotonic=True,
        description=(
            "Squareplus. f(x) = (x + sqrt(x^2 + b)) / 2 with b=4. Smooth, always "
            "positive, monotonic approximation to ReLU without approximation overhead."
        ),
    ),
    "starrelu": ActivationInfo(
        name="starrelu",
        is_gated=False,
        is_smooth=False,
        approx_monotonic=True,
        description=(
            "StarReLU. f(x) = s * relu(x)^2 + b with learnable s and b. From "
            "'MetaFormer Baselines' (Yu et al. 2022). Squared ReLU with scale/bias."
        ),
    ),
}


# ── Custom activation modules ─────────────────────────────────────────


class Mish(nn.Module):
    """Mish activation: f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))."""

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(F.softplus(x))


class Squareplus(nn.Module):
    """Squareplus activation: f(x) = (x + sqrt(x^2 + b)) / 2.

    Smooth, always-positive, monotonic approximation to ReLU.
    Default b=4 matches the original paper.
    """

    def __init__(self, b: float = 4.0) -> None:
        super().__init__()
        self.b = b

    def forward(self, x: Tensor) -> Tensor:
        return (x + torch.sqrt(x * x + self.b)) / 2.0


class StarReLU(nn.Module):
    """StarReLU: f(x) = s * relu(x)^2 + b with learnable s and b.

    From "MetaFormer Baselines for Vision" (Yu et al. 2022).
    Initialized with s=1, b=0.
    """

    def __init__(self) -> None:
        super().__init__()
        self.s = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        return self.s * F.relu(x) ** 2 + self.b


# ── Activation factory ────────────────────────────────────────────────


def get_activation(name: str) -> nn.Module:
    """Return an nn.Module implementing the named activation.

    Supported: "gelu", "relu", "silu", "mish", "squareplus", "starrelu".

    Raises ValueError for unknown names.
    """
    name = name.lower()
    match name:
        case "gelu":
            return nn.GELU()
        case "relu":
            return nn.ReLU()
        case "silu":
            return nn.SiLU()
        case "mish":
            return Mish()
        case "squareplus":
            return Squareplus()
        case "starrelu":
            return StarReLU()
        case _:
            raise ValueError(
                f"get_activation: unknown activation '{name}'. "
                f"Supported: {sorted(ACTIVATION_REGISTRY.keys())}"
            )


# ── Learnable meta-activation ─────────────────────────────────────────


class LearnableActivation(nn.Module):
    """Approximates any smooth activation via a small 2-layer MLP.

    Architecture: Linear(1, hidden) → GELU → Linear(hidden, 1)
    Applied element-wise (flatten input → MLP → reshape to original shape).

    Can be initialized to mimic a known activation via init_from, then fine-tuned
    during NAS or end-to-end training.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        init_from: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        if init_from is not None:
            target_act = get_activation(init_from)
            self.fit_to(lambda x: target_act(x), n_steps=50, lr=0.01)

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        # Flatten to (N, 1), apply MLP, reshape back
        flat = x.reshape(-1, 1)
        out = self.net(flat)
        return out.reshape(original_shape)

    def fit_to(
        self,
        target_fn: Callable,
        n_steps: int = 100,
        lr: float = 0.01,
    ) -> float:
        """Fit this activation to match target_fn on range [-3, 3].

        Returns the final MSE loss.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        x = torch.linspace(-3.0, 3.0, 256).unsqueeze(1)  # (256, 1)

        final_loss = float("inf")
        for _ in range(n_steps):
            optimizer.zero_grad()
            pred = self.net(x).squeeze(1)
            with torch.no_grad():
                target = target_fn(x.squeeze(1))
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        return final_loss


# ── Activation search space ───────────────────────────────────────────


class ActivationSearchSpace:
    """Discrete + continuous search space for activation NAS.

    Supports random search over a fixed list of activation choices.
    """

    choices: list[str] = ["gelu", "relu", "silu", "mish", "squareplus", "starrelu"]

    def sample(self) -> str:
        """Uniformly sample one activation name."""
        return random.choice(self.choices)

    def evaluate(
        self,
        activation_name: str,
        model_fn: Callable[[str], nn.Module],
        eval_fn: Callable[[nn.Module], float],
    ) -> dict[str, float]:
        """Evaluate a single activation choice.

        Args:
            activation_name: name of the activation to evaluate.
            model_fn: fn(act_name) → model to build a model with that activation.
            eval_fn: fn(model) → float metric (higher is better).

        Returns:
            {"activation": name, "score": float}
        """
        model = model_fn(activation_name)
        score = eval_fn(model)
        return {"activation": activation_name, "score": score}

    def search(
        self,
        model_fn: Callable[[str], nn.Module],
        eval_fn: Callable[[nn.Module], float],
        n_trials: int = 6,
    ) -> str:
        """Random search over activation choices; return best activation name.

        Each choice is evaluated once (n_trials capped at len(choices) if
        n_trials >= len(choices), ensuring full coverage).
        """
        # If n_trials >= number of choices, evaluate all to ensure determinism;
        # otherwise sample randomly.
        if n_trials >= len(self.choices):
            candidates = list(self.choices)
        else:
            seen: set[str] = set()
            candidates = []
            while len(candidates) < n_trials:
                name = self.sample()
                if name not in seen:
                    seen.add(name)
                    candidates.append(name)

        best_name = candidates[0]
        best_score = float("-inf")
        for name in candidates:
            result = self.evaluate(name, model_fn, eval_fn)
            if result["score"] > best_score:
                best_score = result["score"]
                best_name = result["activation"]

        return best_name
