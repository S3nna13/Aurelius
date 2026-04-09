"""Efficient fine-tuning variants: LoRA+, IA3, and VeRA adapters."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class LoRAVariantConfig:
    """Configuration for LoRA variant adapters."""

    rank: int = 8
    alpha: float = 16.0          # scaling: scale = alpha / rank
    dropout: float = 0.0
    variant: str = "lora_plus"   # "lora_plus" | "ia3" | "vera"
    lora_plus_lr_ratio: float = 16.0  # LoRA+: B gets lr_ratio * base_lr
    vera_shared_dim: int = 256   # VeRA shared random projection dim


class LoRAPlusAdapter(nn.Module):
    """LoRA+ adapter (Hayou et al. 2024).

    Standard LoRA with asymmetric learning rates: B matrix receives a higher
    learning rate than A. At initialization B is zeroed so the adapter
    contributes nothing until training begins.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        config: LoRAVariantConfig with rank, alpha, dropout, and lr_ratio.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAVariantConfig,
    ) -> None:
        super().__init__()
        self.config = config
        rank = config.rank
        self.scale = config.alpha / rank

        # A: Gaussian init; B: zero init (adapter is identity at init)
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Return adapter output: (x @ lora_A @ lora_B) * scale."""
        return (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scale

    def get_param_groups(self, base_lr: float) -> list[dict]:
        """Return two param groups: A at base_lr, B at base_lr * lr_ratio.

        Args:
            base_lr: Learning rate for lora_A.

        Returns:
            List of two optimizer param-group dicts.
        """
        return [
            {"params": [self.lora_A], "lr": base_lr},
            {"params": [self.lora_B], "lr": base_lr * self.config.lora_plus_lr_ratio},
        ]


class IA3Adapter(nn.Module):
    """IA3 adapter (Liu et al. 2022): element-wise rescaling of activations.

    Learns a single scale vector of the same dimension as the feature axis.
    Initialized to ones so the adapter is an identity transform at init.

    Args:
        features: Feature dimension to rescale.
    """

    def __init__(self, features: int) -> None:
        super().__init__()
        self.scale_vector = nn.Parameter(torch.ones(features))

    def forward(self, x: Tensor) -> Tensor:
        """Return x * scale_vector (broadcast over batch/seq dims)."""
        return x * self.scale_vector

    def merge_into_linear(self, linear: nn.Linear) -> nn.Linear:
        """Create a new Linear whose weights are scaled by scale_vector.

        The IA3 scaling is folded into the weight matrix so inference
        requires no extra multiply.

        Args:
            linear: The base nn.Linear to merge into.

        Returns:
            New nn.Linear with merged weights.
        """
        with torch.no_grad():
            new_linear = nn.Linear(
                linear.in_features,
                linear.out_features,
                bias=linear.bias is not None,
            )
            # weight shape: (out_features, in_features)
            # scale_vector shape: (out_features,) — scale each output row
            new_linear.weight.copy_(linear.weight * self.scale_vector.unsqueeze(1))
            if linear.bias is not None:
                new_linear.bias.copy_(linear.bias * self.scale_vector)
        return new_linear


class VeRAAdapter(nn.Module):
    """VeRA adapter (Kopiczko et al. 2023).

    Uses shared frozen random projection matrices A and B; only small
    per-dimension vectors d and b are trained, drastically reducing the
    number of trainable parameters.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        config: LoRAVariantConfig with rank, alpha, and vera_shared_dim.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAVariantConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.scale = config.alpha / config.rank
        shared_dim = config.vera_shared_dim

        # Frozen shared random projections registered as buffers (not parameters)
        A_frozen = torch.randn(in_features, shared_dim)
        B_frozen = torch.randn(shared_dim, out_features)
        self.register_buffer("A_frozen", A_frozen)
        self.register_buffer("B_frozen", B_frozen)

        # Trainable per-dimension scaling vectors
        self.d = nn.Parameter(torch.ones(shared_dim))   # intermediate scaling
        self.b = nn.Parameter(torch.zeros(out_features))  # output bias

    def forward(self, x: Tensor) -> Tensor:
        """Compute VeRA output: ((x @ A_frozen) * d @ B_frozen + b) * scale."""
        hidden = (x @ self.A_frozen) * self.d   # (..., shared_dim)
        out = hidden @ self.B_frozen + self.b    # (..., out_features)
        return out * self.scale


def get_adapter_params(model: nn.Module, adapter_type: str) -> list[nn.Parameter]:
    """Return all trainable parameters belonging to adapter modules in model.

    Searches for submodules whose type name matches the adapter_type string
    (case-insensitive) and collects their parameters that require grad.

    Args:
        model: The model containing adapters.
        adapter_type: One of "lora_plus", "ia3", or "vera".

    Returns:
        List of trainable adapter Parameters.
    """
    type_map = {
        "lora_plus": LoRAPlusAdapter,
        "ia3": IA3Adapter,
        "vera": VeRAAdapter,
    }
    adapter_cls = type_map.get(adapter_type.lower())
    params: list[nn.Parameter] = []

    for module in model.modules():
        if adapter_cls is not None and isinstance(module, adapter_cls):
            for p in module.parameters():
                if p.requires_grad:
                    params.append(p)
        elif adapter_cls is None:
            # Fallback: any adapter type
            if isinstance(module, (LoRAPlusAdapter, IA3Adapter, VeRAAdapter)):
                for p in module.parameters():
                    if p.requires_grad:
                        params.append(p)

    return params


def apply_adapters_to_model(
    model: nn.Module,
    config: LoRAVariantConfig,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Attach adapters to Linear layers in model and freeze all original params.

    For each targeted nn.Linear, an adapter module is added as a
    ``{name}_adapter`` attribute on the parent module. All original model
    parameters are frozen; only adapter parameters remain trainable.

    Args:
        model: The model to modify (mutated in-place).
        config: LoRAVariantConfig controlling adapter type and hyperparams.
        target_modules: List of module names to target (e.g. ["q_proj",
            "v_proj"]). If None, all nn.Linear layers are targeted.

    Returns:
        The modified model.
    """
    # First freeze all existing parameters
    for param in model.parameters():
        param.requires_grad_(False)

    # Collect (parent, child_name, full_path) for matching Linear layers
    adapter_cls_map = {
        "lora_plus": LoRAPlusAdapter,
        "ia3": IA3Adapter,
        "vera": VeRAAdapter,
    }
    adapter_cls = adapter_cls_map.get(config.variant, LoRAPlusAdapter)

    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        # Determine if this module is a target
        leaf_name = full_name.split(".")[-1] if "." in full_name else full_name
        if target_modules is not None and leaf_name not in target_modules:
            continue

        in_f = module.in_features
        out_f = module.out_features

        # Build adapter
        if config.variant == "lora_plus":
            adapter = LoRAPlusAdapter(in_f, out_f, config)
        elif config.variant == "ia3":
            adapter = IA3Adapter(out_f)
        elif config.variant == "vera":
            adapter = VeRAAdapter(in_f, out_f, config)
        else:
            adapter = LoRAPlusAdapter(in_f, out_f, config)

        # Find parent module and attach adapter as {leaf_name}_adapter
        if "." in full_name:
            parent_name = full_name.rsplit(".", 1)[0]
            parent = model
            for part in parent_name.split("."):
                parent = getattr(parent, part)
        else:
            parent = model

        attr_name = f"{leaf_name}_adapter"
        setattr(parent, attr_name, adapter)

    return model
