"""Model merging: SLERP and TIES-Merging.

Merge multiple fine-tuned model variants into a single model that
combines their capabilities.

SLERP (Spherical Linear Interpolation):
- Interpolates weight vectors on a unit hypersphere
- Preserves angular relationships better than linear interpolation
- Best for merging 2 models

TIES-Merging (Trim, Elect Sign, Disjoint Merge) - Yadav et al., 2023:
- Step 1: Trim — zero out small task vectors (|delta_W| below threshold)
- Step 2: Elect Sign — for each parameter, pick the majority sign
- Step 3: Disjoint Merge — average only parameters that agree with elected sign
- Best for merging 3+ models without sign conflicts
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def slerp(
    t: float,
    v0: torch.Tensor,
    v1: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Spherical Linear Interpolation between two flat weight vectors.

    Interpolates along the great-circle arc on the unit hypersphere.
    Falls back to linear interpolation when vectors are nearly parallel.

    Args:
        t: Interpolation factor (0 = v0, 1 = v1).
        v0: First weight vector (any shape, will be flattened internally).
        v1: Second weight vector (same shape as v0).
        eps: Small value for numerical stability.

    Returns:
        Interpolated weight vector (same shape as v0).
    """
    shape = v0.shape
    v0_flat = v0.float().flatten()
    v1_flat = v1.float().flatten()

    # Normalize to unit sphere
    v0_norm = v0_flat / (v0_flat.norm() + eps)
    v1_norm = v1_flat / (v1_flat.norm() + eps)

    # Compute angle between vectors
    dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
    theta = torch.arccos(dot.abs())  # angle in [0, pi/2]

    # Fall back to linear interpolation when nearly parallel
    if theta.abs() < 1e-6:
        return ((1 - t) * v0_flat + t * v1_flat).reshape(shape).to(v0.dtype)

    # SLERP formula: (sin((1-t)*θ) * v0 + sin(t*θ) * v1) / sin(θ)
    sin_theta = torch.sin(theta)
    result = (
        torch.sin((1 - t) * theta) / sin_theta * v0_flat
        + torch.sin(t * theta) / sin_theta * v1_flat
    )

    return result.reshape(shape).to(v0.dtype)


def slerp_merge(
    base: nn.Module,
    model_a: nn.Module,
    model_b: nn.Module,
    t: float = 0.5,
) -> nn.Module:
    """Merge two fine-tuned models using SLERP.

    Interpolates each parameter between model_a and model_b using SLERP.
    The base model is used only to establish the parameter keys — it is
    not mixed into the result. If you want to interpolate from the base,
    pass base as model_a and t accordingly.

    Args:
        base: Reference model (defines architecture/parameter names).
        model_a: First model to merge.
        model_b: Second model to merge.
        t: Interpolation factor (0 = model_a, 1 = model_b).

    Returns:
        A new model with merged parameters (base's state dict is modified in-place).
    """
    import copy

    merged = copy.deepcopy(base)

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    merged_state = {}

    for name in merged.state_dict().keys():
        if name not in state_a or name not in state_b:
            merged_state[name] = merged.state_dict()[name]
            continue

        wa = state_a[name]
        wb = state_b[name]

        # Skip complex tensors (e.g. freqs_cis rotary embeddings) and non-float
        if wa.dtype.is_floating_point and not wa.is_complex() and wa.shape == wb.shape:
            merged_state[name] = slerp(t, wa, wb)
        else:
            # Non-float or complex buffers: use model_a's value
            merged_state[name] = wa.clone()

    merged.load_state_dict(merged_state)
    return merged


def ties_merge(
    base: nn.Module,
    models: Sequence[nn.Module],
    density: float = 0.2,
    t: float = 1.0,
) -> nn.Module:
    """Merge multiple fine-tuned models using TIES-Merging.

    Algorithm:
    1. Compute task vectors: delta_i = model_i - base
    2. Trim: zero out delta_i values below density threshold
    3. Elect sign: for each parameter, pick the majority sign across models
    4. Disjoint merge: average only task vectors that agree with elected sign
    5. Apply: merged = base + t * merged_task_vector

    Args:
        base: Base pretrained model (task vectors computed relative to this).
        models: List of fine-tuned models.
        density: Top fraction of task vector parameters to keep (0.2 = keep top 20%).
        t: Scaling factor applied to the merged task vector.

    Returns:
        A new model with TIES-merged parameters.
    """
    import copy

    if not models:
        return copy.deepcopy(base)

    base_state = base.state_dict()
    model_states = [m.state_dict() for m in models]

    merged_state = {}

    for name in base_state.keys():
        base_param = base_state[name]

        # Skip non-float and complex buffers (e.g. freqs_cis)
        if not base_param.dtype.is_floating_point or base_param.is_complex():
            merged_state[name] = base_param.clone()
            continue

        # Collect task vectors: delta_i = model_i_param - base_param
        task_vectors = []
        for ms in model_states:
            if name in ms and ms[name].shape == base_param.shape:
                delta = ms[name].float() - base_param.float()
                task_vectors.append(delta)

        if not task_vectors:
            merged_state[name] = base_param.clone()
            continue

        # Step 1: Trim — keep only top `density` fraction of each task vector
        trimmed = []
        for tv in task_vectors:
            flat = tv.abs().flatten()
            if len(flat) > 1 and density < 1.0:
                threshold = torch.quantile(flat, 1.0 - density)
                mask = tv.abs() >= threshold
                trimmed.append(tv * mask.float())
            else:
                trimmed.append(tv)

        # Step 2: Elect sign — majority vote per parameter
        stacked = torch.stack(trimmed)  # (n_models, *shape)
        sign_sum = stacked.sign().sum(dim=0)  # positive sum = majority positive
        elected_sign = sign_sum.sign()
        # Ties broken by keeping positive
        elected_sign = torch.where(elected_sign == 0, torch.ones_like(elected_sign), elected_sign)

        # Step 3: Disjoint merge — average task vectors that agree with elected sign
        # Zero out parameters that disagree with elected sign
        merged_tv = torch.zeros_like(base_param, dtype=torch.float)
        count = torch.zeros_like(base_param, dtype=torch.float)

        for tv in trimmed:
            agrees = (tv.sign() == elected_sign) | (tv == 0)
            merged_tv += tv * agrees.float()
            count += agrees.float()

        # Avoid division by zero
        count = count.clamp(min=1.0)
        merged_tv = merged_tv / count

        # Step 5: Apply merged task vector
        merged_state[name] = (base_param.float() + t * merged_tv).to(base_param.dtype)

    merged = copy.deepcopy(base)
    merged.load_state_dict(merged_state)
    return merged
