"""Warm-start utilities for partially loading model weights."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Original helpers (preserved)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WarmStartReport:
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    shape_mismatch_keys: tuple[str, ...]


def warm_start_state(
    target_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], WarmStartReport]:
    """Load matching source tensors into a target state dict copy."""
    updated: dict[str, torch.Tensor] = {}
    loaded_keys: list[str] = []
    missing_keys: list[str] = []
    shape_mismatch_keys: list[str] = []

    for key, target_tensor in target_state.items():
        if key not in source_state:
            updated[key] = target_tensor.clone()
            missing_keys.append(key)
            continue
        source_tensor = source_state[key]
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            updated[key] = target_tensor.clone()
            shape_mismatch_keys.append(key)
            continue
        updated[key] = source_tensor.detach().clone()
        loaded_keys.append(key)

    return updated, WarmStartReport(
        loaded_keys=tuple(loaded_keys),
        missing_keys=tuple(missing_keys),
        shape_mismatch_keys=tuple(shape_mismatch_keys),
    )


def interpolation_warm_start(
    target_tensor: torch.Tensor,
    source_tensor: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Blend target and source tensors for gentle warm-starting."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if target_tensor.shape != source_tensor.shape:
        raise ValueError("target_tensor and source_tensor must match")
    return (1.0 - alpha) * target_tensor + alpha * source_tensor


def prefix_warm_start(
    target_tensor: torch.Tensor,
    source_tensor: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """Copy the largest matching prefix from source into target."""
    if target_tensor.dim() != source_tensor.dim():
        raise ValueError("target_tensor and source_tensor must have the same rank")
    if dim < 0 or dim >= target_tensor.dim():
        raise ValueError(f"dim must be in [0, {target_tensor.dim() - 1}], got {dim}")
    result = target_tensor.clone()
    copy_shape = list(target_tensor.shape)
    copy_shape[dim] = min(target_tensor.shape[dim], source_tensor.shape[dim])
    slices = tuple(slice(0, size) for size in copy_shape)
    result[slices] = source_tensor[slices]
    return result


# ---------------------------------------------------------------------------
# New warm-start strategies
# ---------------------------------------------------------------------------

@dataclass
class WarmStartConfig:
    """Configuration for warm-starting a model."""

    strategy: str = "interpolate"  # "interpolate" | "stack_layers" | "depth_upscale" | "scratch"
    source_checkpoint: str | None = None  # path to source weights
    interpolation_alpha: float = 0.5      # for interpolate: blend (0=random, 1=source)
    target_n_layers: int | None = None    # for stack_layers / depth_upscale


class WarmStartInitializer:
    """Manages warm-starting strategies for model initialization.

    Args:
        config: WarmStartConfig
    """

    def __init__(self, config: WarmStartConfig | None = None) -> None:
        self.config = config or WarmStartConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpolate_weights(
        self,
        target_model: nn.Module,
        source_state_dict: dict,
        alpha: float = 0.5,
    ) -> None:
        """In-place blend: target_param = alpha * source_param + (1-alpha) * target_param.

        Only blends parameters with matching names AND shapes.
        Logs (print) how many parameters were blended vs skipped.
        """
        target_state = dict(target_model.named_parameters())
        n_blended = 0
        n_skipped = 0

        with torch.no_grad():
            for name, target_param in target_state.items():
                if name not in source_state_dict:
                    n_skipped += 1
                    continue
                source_param = source_state_dict[name]
                if not isinstance(source_param, torch.Tensor):
                    n_skipped += 1
                    continue
                if source_param.shape != target_param.shape:
                    n_skipped += 1
                    continue
                target_param.copy_(
                    alpha * source_param.to(target_param.dtype) + (1.0 - alpha) * target_param
                )
                n_blended += 1

        print(f"[WarmStart] interpolate_weights: blended={n_blended}, skipped={n_skipped}")

    def stack_layers(
        self,
        model: nn.Module,
        source_state_dict: dict,
        target_n_layers: int,
    ) -> None:
        """Initialize a deeper model by stacking source layers.

        If source has n_source layers and target has target_n_layers:
        - Map source layer i → target layer j where j = i * (target_n_layers // n_source)
        - Remaining target layers get their nearest source layer (wrap with modulo)

        Only copies matching-shape parameters.
        """
        # Infer source layer count from state dict keys like "layers.{i}.*"
        source_layer_indices: set[int] = set()
        for key in source_state_dict:
            parts = key.split(".")
            if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
                source_layer_indices.add(int(parts[1]))

        n_source = len(source_layer_indices) if source_layer_indices else 1

        with torch.no_grad():
            for target_idx in range(target_n_layers):
                # Nearest source layer (modulo wrap)
                source_idx = target_idx % n_source

                # Build a remapped sub-dict for this target layer
                prefix_src = f"layers.{source_idx}."
                prefix_tgt = f"layers.{target_idx}."

                target_params = {
                    name: param
                    for name, param in model.named_parameters()
                    if name.startswith(prefix_tgt)
                }

                for tgt_name, tgt_param in target_params.items():
                    src_name = prefix_src + tgt_name[len(prefix_tgt):]
                    if src_name not in source_state_dict:
                        continue
                    src_tensor = source_state_dict[src_name]
                    if not isinstance(src_tensor, torch.Tensor):
                        continue
                    if src_tensor.shape != tgt_param.shape:
                        continue
                    tgt_param.copy_(src_tensor.to(tgt_param.dtype))

    def apply(self, model: nn.Module, source_state_dict: dict | None = None) -> None:
        """Apply the configured strategy to model (in-place)."""
        strategy = self.config.strategy

        if strategy == "scratch":
            # Leave weights as-is (cold start)
            return

        if strategy == "interpolate":
            if source_state_dict is None and self.config.source_checkpoint is not None:
                source_state_dict = torch.load(
                    self.config.source_checkpoint, map_location="cpu", weights_only=True
                )
            if source_state_dict is None:
                raise ValueError("interpolate strategy requires source_state_dict or source_checkpoint")
            self.interpolate_weights(model, source_state_dict, alpha=self.config.interpolation_alpha)

        elif strategy in ("stack_layers", "depth_upscale"):
            if source_state_dict is None and self.config.source_checkpoint is not None:
                source_state_dict = torch.load(
                    self.config.source_checkpoint, map_location="cpu", weights_only=True
                )
            if source_state_dict is None:
                raise ValueError(f"{strategy} requires source_state_dict or source_checkpoint")
            n_layers = self.config.target_n_layers
            if n_layers is None:
                raise ValueError(f"{strategy} requires target_n_layers in config")
            self.stack_layers(model, source_state_dict, target_n_layers=n_layers)

        else:
            raise ValueError(f"Unknown warm-start strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# LayerDropout
# ---------------------------------------------------------------------------

class LayerDropout(nn.Module):
    """Layer dropout: randomly skip entire transformer layers during training.

    At inference, all layers are active.

    Used for gradual depth growth: start with high dropout rate,
    anneal to 0 over training (LayerDrop, Fan et al. 2019).

    Args:
        layer: nn.Module (a transformer block)
        drop_prob: float (probability of skipping this layer during training)
    """

    def __init__(self, layer: nn.Module, drop_prob: float = 0.1) -> None:
        super().__init__()
        self.layer = layer
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """If training and random draw < drop_prob: return x unchanged. Else: return layer(x)."""
        if self.training and torch.rand(1).item() < self.drop_prob:
            return x
        return self.layer(x, **kwargs)


# ---------------------------------------------------------------------------
# DepthGrowthScheduler
# ---------------------------------------------------------------------------

class DepthGrowthScheduler:
    """Schedule for gradually reducing LayerDropout rates during training.

    All layers start with drop_prob=initial_prob.
    Linearly anneal to 0 over anneal_steps.

    Args:
        layer_drop_modules: list[LayerDropout]
        initial_prob: float (default 0.5)
        anneal_steps: int (default 10000)
    """

    def __init__(
        self,
        layer_drop_modules: list,
        initial_prob: float = 0.5,
        anneal_steps: int = 10000,
    ) -> None:
        self.layer_drop_modules = layer_drop_modules
        self.initial_prob = initial_prob
        self.anneal_steps = anneal_steps

    def step(self, current_step: int) -> float:
        """Update all LayerDropout modules' drop_prob. Returns current drop_prob."""
        if self.anneal_steps <= 0:
            prob = 0.0
        else:
            fraction = min(current_step / self.anneal_steps, 1.0)
            prob = self.initial_prob * (1.0 - fraction)

        for module in self.layer_drop_modules:
            module.drop_prob = prob

        return prob


# ---------------------------------------------------------------------------
# muggle_init
# ---------------------------------------------------------------------------

def muggle_init(model: nn.Module) -> None:
    """'Muggles' initialization (depth-scaled init from μP / Greg Yang 2022).

    Scale embedding weights by 1/sqrt(d_model).
    Scale output projection weights by 1/n_layers.
    Leave everything else default.
    This is a simple approximation — not full μP.
    """
    d_model = model.embed.weight.shape[1]
    n_layers = len(model.layers)

    with torch.no_grad():
        model.embed.weight.mul_(1.0 / math.sqrt(d_model))
        # lm_head may be tied to embed; only scale if it is a separate tensor
        if model.lm_head.weight.data_ptr() != model.embed.weight.data_ptr():
            model.lm_head.weight.mul_(1.0 / n_layers)


# ---------------------------------------------------------------------------
# count_matchable_params
# ---------------------------------------------------------------------------

def count_matchable_params(target_model: nn.Module, source_state_dict: dict) -> dict:
    """Return counts of parameter matches between target model and source state dict.

    Returns:
        {
            'n_matched': int,          # params with same name and shape
            'n_shape_mismatch': int,   # same name, different shape
            'n_source_only': int,      # only in source
            'n_target_only': int,      # only in target
        }
    """
    target_params = dict(target_model.named_parameters())
    target_keys = set(target_params.keys())

    # Filter source_state_dict to only tensor entries
    source_tensors = {
        k: v for k, v in source_state_dict.items() if isinstance(v, torch.Tensor)
    }
    source_keys = set(source_tensors.keys())

    n_matched = 0
    n_shape_mismatch = 0
    common_keys = target_keys & source_keys

    for key in common_keys:
        if target_params[key].shape == source_tensors[key].shape:
            n_matched += 1
        else:
            n_shape_mismatch += 1

    n_source_only = len(source_keys - target_keys)
    n_target_only = len(target_keys - source_keys)

    return {
        "n_matched": n_matched,
        "n_shape_mismatch": n_shape_mismatch,
        "n_source_only": n_source_only,
        "n_target_only": n_target_only,
    }
