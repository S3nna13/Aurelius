"""LoRA (Low-Rank Adaptation) parameter-efficient fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""

    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    merge_weights: bool = False


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with low-rank adaptation (LoRA).

    The original linear weight is frozen; only lora_A and lora_B are trained.
    Forward pass: linear(x) + scale * dropout(x) @ lora_A.T @ lora_B.T
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha

        # Freeze the original linear weight (and bias if present)
        linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False

        in_features = linear.in_features
        out_features = linear.out_features

        # LoRA matrices: A ~ N(0,1), B = 0
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.normal_(self.lora_A)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    @property
    def scale(self) -> float:
        """Scaling factor alpha / r."""
        return self.alpha / self.r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute linear(x) + scale * dropout(x) @ lora_A.T @ lora_B.T."""
        base = self.linear(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + self.scale * lora_out

    def merge(self) -> nn.Linear:
        """Return a new nn.Linear with merged weights W + scale * B @ A."""
        merged_weight = self.linear.weight + self.scale * (self.lora_B @ self.lora_A)
        new_linear = nn.Linear(
            self.linear.in_features,
            self.linear.out_features,
            bias=self.linear.bias is not None,
        )
        new_linear.weight = nn.Parameter(merged_weight.detach().clone())
        if self.linear.bias is not None:
            new_linear.bias = nn.Parameter(self.linear.bias.detach().clone())
        return new_linear


def apply_lora(model: nn.Module, config: LoRAConfig) -> int:
    """Replace target nn.Linear modules with LoRALinear.

    Walks model.named_modules() and replaces any nn.Linear whose name
    contains a string from config.target_modules.

    Args:
        model: The PyTorch model to modify in-place.
        config: LoRA configuration.

    Returns:
        Number of replaced modules.
    """
    replaced = 0
    # Collect replacements first to avoid mutation during iteration
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []

    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Check if any target string appears in the module's leaf name
        leaf_name = full_name.split(".")[-1] if full_name else ""
        if any(target in full_name or target in leaf_name for target in config.target_modules):
            # Find parent module and attribute name
            parts = full_name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            replacements.append((parent, parts[-1], module))

    for parent, attr, original_linear in replacements:
        lora_linear = LoRALinear(
            original_linear,
            r=config.r,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        setattr(parent, attr, lora_linear)
        replaced += 1

    return replaced


def _is_lora_param(name: str) -> bool:
    """Return True if the parameter name corresponds to a LoRA matrix."""
    return "lora_A" in name or "lora_B" in name


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Return only LoRA trainable parameters (lora_A and lora_B).

    Args:
        model: The model containing LoRALinear modules.

    Returns:
        List of lora_A / lora_B parameters.
    """
    return [p for name, p in model.named_parameters() if _is_lora_param(name)]


def get_lora_param_count(model: nn.Module) -> dict[str, int]:
    """Return parameter element counts for trainable (LoRA), frozen, and total.

    'trainable' counts only lora_A / lora_B elements.
    'frozen' counts everything else.

    Args:
        model: The model to inspect.

    Returns:
        Dict with keys 'trainable', 'frozen', 'total' (element counts).
    """
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        n = param.numel()
        if _is_lora_param(name):
            trainable += n
        else:
            frozen += n
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


def save_lora_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a dict of only the LoRA (lora_A / lora_B) parameters.

    Args:
        model: The model with LoRALinear modules.

    Returns:
        Dict mapping parameter name to tensor for all LoRA params.
    """
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if _is_lora_param(name)
    }


def load_lora_weights(model: nn.Module, weights: dict[str, torch.Tensor]) -> None:
    """Load LoRA weights back into matching model parameters.

    Args:
        model: The model with LoRALinear modules.
        weights: Dict of parameter name -> tensor (as from save_lora_weights).
    """
    state = model.state_dict()
    state.update(weights)
    model.load_state_dict(state, strict=False)
