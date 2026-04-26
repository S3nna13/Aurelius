"""Post-hoc model architecture modification utilities.

Provides tools for inserting/removing layers, swapping modules,
and progressive depth scaling of AureliusTransformer models.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch.nn as nn

from .transformer import AureliusTransformer, TransformerBlock


def _count_params(model: nn.Module) -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def insert_layer(
    model: AureliusTransformer,
    layer_idx: int,
    init_from: str = "copy",
) -> None:
    """Insert a new TransformerBlock into model.layers at layer_idx.

    Args:
        model: The AureliusTransformer to modify in place.
        layer_idx: Insert BEFORE this index. 0 = prepend, n = append.
        init_from: "copy" = deepcopy adjacent layer, "random" = fresh init.
    """
    current_layers = list(model.layers)
    n = len(current_layers)

    if layer_idx < 0 or layer_idx > n:
        raise IndexError(f"layer_idx {layer_idx} out of range [0, {n}]")

    if init_from == "copy":
        # Copy adjacent layer: use layer_idx if not appending, else last layer
        src_idx = layer_idx if layer_idx < n else n - 1
        new_layer = copy.deepcopy(current_layers[src_idx])
    elif init_from == "random":
        new_layer = TransformerBlock(model.config)
    else:
        raise ValueError(f"init_from must be 'copy' or 'random', got '{init_from}'")

    current_layers.insert(layer_idx, new_layer)
    model.layers = nn.ModuleList(current_layers)
    model.config.n_layers += 1


def remove_layer(model: AureliusTransformer, layer_idx: int) -> nn.Module:
    """Remove and return the TransformerBlock at layer_idx.

    Args:
        model: The AureliusTransformer to modify in place.
        layer_idx: Index of the layer to remove.

    Returns:
        The removed TransformerBlock.
    """
    current_layers = list(model.layers)
    n = len(current_layers)

    if layer_idx < 0 or layer_idx >= n:
        raise IndexError(f"layer_idx {layer_idx} out of range [0, {n - 1}]")

    removed = current_layers.pop(layer_idx)
    model.layers = nn.ModuleList(current_layers)
    model.config.n_layers -= 1

    return removed


def swap_ffn(model: AureliusTransformer, layer_idx: int, new_ffn: nn.Module) -> nn.Module:
    """Replace the FFN in model.layers[layer_idx] with new_ffn.

    Args:
        model: The AureliusTransformer to modify in place.
        layer_idx: Index of the layer whose FFN to replace.
        new_ffn: The new FFN module to install.

    Returns:
        The old FFN module that was replaced.
    """
    layer = model.layers[layer_idx]
    old_ffn = layer.ffn
    layer.ffn = new_ffn
    return old_ffn


def clone_model_with_depth(
    model: AureliusTransformer,
    target_n_layers: int,
    strategy: str = "uniform",
) -> AureliusTransformer:
    """Create a new model with target_n_layers by inserting/removing layers.

    Copies weights from the original model where possible.

    Args:
        model: Source AureliusTransformer.
        target_n_layers: Desired number of layers in the new model.
        strategy: "uniform" = evenly spaced source indices (linspace);
                  "repeat_middle" = duplicate middle layers to grow.

    Returns:
        A new AureliusTransformer with target_n_layers layers.
    """
    src_layers = list(model.layers)
    n_src = len(src_layers)

    if target_n_layers <= 0:
        raise ValueError(f"target_n_layers must be > 0, got {target_n_layers}")

    # Build a new config with updated depth
    new_config = copy.deepcopy(model.config)
    new_config.n_layers = target_n_layers

    # Build the list of new layers
    if strategy == "uniform":
        if target_n_layers == n_src:
            # Exact copy
            new_layer_list = [copy.deepcopy(line) for line in src_layers]
        else:
            # Use linspace to pick evenly-spaced source indices
            indices = [
                round(i * (n_src - 1) / (target_n_layers - 1)) if target_n_layers > 1 else 0
                for i in range(target_n_layers)
            ]
            new_layer_list = [copy.deepcopy(src_layers[idx]) for idx in indices]

    elif strategy == "repeat_middle":
        if target_n_layers <= n_src:
            # Shrink: keep evenly spaced
            indices = [
                round(i * (n_src - 1) / (target_n_layers - 1)) if target_n_layers > 1 else 0
                for i in range(target_n_layers)
            ]
            new_layer_list = [copy.deepcopy(src_layers[idx]) for idx in indices]
        else:
            # Grow: start with all src layers, insert copies of middle layers
            new_layer_list = [copy.deepcopy(line) for line in src_layers]
            mid = n_src // 2
            while len(new_layer_list) < target_n_layers:
                new_layer_list.insert(mid, copy.deepcopy(src_layers[mid]))
    else:
        raise ValueError(f"strategy must be 'uniform' or 'repeat_middle', got '{strategy}'")

    # Build new model and replace its layers
    new_model = AureliusTransformer(new_config)
    new_model.layers = nn.ModuleList(new_layer_list)

    # Copy embedding and norm weights from source
    new_model.embed.weight.data.copy_(model.embed.weight.data)
    new_model.norm.weight.data.copy_(model.norm.weight.data)
    if not new_config.tie_embeddings:
        new_model.lm_head.weight.data.copy_(model.lm_head.weight.data)

    return new_model


@dataclass
class ScalingResult:
    """Statistics from a depth-scaling operation."""

    original_params: int
    new_params: int
    n_layers_original: int
    n_layers_new: int
    layers_added: int
    layers_removed: int


def scale_model_depth(
    model: AureliusTransformer,
    target_n_layers: int,
    strategy: str = "uniform",
) -> ScalingResult:
    """Grow or shrink model.layers in-place to target_n_layers.

    Args:
        model: The AureliusTransformer to modify in place.
        target_n_layers: Desired number of transformer layers.
        strategy: "uniform" = evenly spaced layer selection.

    Returns:
        ScalingResult with before/after statistics.
    """
    original_params = _count_params(model)
    n_layers_original = model.config.n_layers
    src_layers = list(model.layers)
    n_src = len(src_layers)

    if target_n_layers <= 0:
        raise ValueError(f"target_n_layers must be > 0, got {target_n_layers}")

    layers_added = max(0, target_n_layers - n_src)
    layers_removed = max(0, n_src - target_n_layers)

    if target_n_layers == n_src:
        new_layer_list = src_layers
    elif strategy == "uniform":
        if target_n_layers == 1:
            indices = [0]
        else:
            indices = [
                round(i * (n_src - 1) / (target_n_layers - 1)) for i in range(target_n_layers)
            ]
        new_layer_list = [copy.deepcopy(src_layers[idx]) for idx in indices]
    else:
        raise ValueError(f"strategy must be 'uniform', got '{strategy}'")

    model.layers = nn.ModuleList(new_layer_list)
    model.config.n_layers = target_n_layers

    new_params = _count_params(model)

    return ScalingResult(
        original_params=original_params,
        new_params=new_params,
        n_layers_original=n_layers_original,
        n_layers_new=target_n_layers,
        layers_added=layers_added,
        layers_removed=layers_removed,
    )
