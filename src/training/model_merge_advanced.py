"""Advanced model merging: DARE (random pruning) and TIES with sign consensus voting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class MergeConfig:
    """Configuration for advanced model merging.

    Attributes:
        method: Merge strategy — "dare" | "ties" | "dare_ties" | "linear".
        density: Fraction of task-vector weights to *keep* in DARE (0 < density <= 1).
        lambda_: Scaling factor applied to the merged task vector before adding to base.
        top_k_fraction: TIES trim step — keep this fraction of largest-magnitude values.
        seed: Base random seed for DARE pruning; each tensor uses a derived seed.
    """

    method: str = "dare_ties"
    density: float = 0.5
    lambda_: float = 1.0
    top_k_fraction: float = 0.2
    seed: int = 42


# ---------------------------------------------------------------------------
# Step 1 — task vectors
# ---------------------------------------------------------------------------


def compute_task_vector(
    finetuned: dict[str, Tensor],
    base: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Compute per-parameter task vectors: delta[k] = finetuned[k] - base[k].

    Only keys present in both dicts with matching shapes are included.

    Args:
        finetuned: Fine-tuned model state dict.
        base: Base (pretrained) model state dict.

    Returns:
        Dict mapping parameter name to delta tensor (float32).
    """
    task_vector: dict[str, Tensor] = {}
    for key in base:
        if key in finetuned and finetuned[key].shape == base[key].shape:
            task_vector[key] = finetuned[key].float() - base[key].float()
    return task_vector


# ---------------------------------------------------------------------------
# Step 2a — DARE random pruning
# ---------------------------------------------------------------------------


def dare_prune(
    task_vector: dict[str, Tensor],
    density: float,
    seed: int = 42,
) -> dict[str, Tensor]:
    """Randomly zero out (1 - density) fraction of each tensor in the task vector.

    Surviving values are rescaled by 1 / density so that the expected value of
    each element is preserved (unbiased estimator).

    A per-tensor reproducible torch.Generator is used: the seed for key *k* is
    ``(seed + hash(k)) & 0xFFFF_FFFF``.

    Args:
        task_vector: Dict of delta tensors.
        density: Fraction of values to *keep* (0 < density <= 1).
        seed: Base random seed.

    Returns:
        New dict with DARE-pruned tensors.
    """
    if not (0.0 < density <= 1.0):
        raise ValueError(f"density must be in (0, 1], got {density}")

    result: dict[str, Tensor] = {}
    rescale = 1.0 / density

    for key, delta in task_vector.items():
        key_seed = (seed + hash(key)) & 0xFFFF_FFFF
        gen = torch.Generator()
        gen.manual_seed(key_seed)

        # Bernoulli mask: 1 = keep, 0 = drop
        keep_mask = torch.bernoulli(
            torch.full(delta.shape, density, dtype=torch.float), generator=gen
        )
        result[key] = delta * keep_mask * rescale

    return result


# ---------------------------------------------------------------------------
# Step 2b — TIES trim
# ---------------------------------------------------------------------------


def ties_trim(
    task_vector: dict[str, Tensor],
    top_k_fraction: float,
) -> dict[str, Tensor]:
    """Keep only the top_k_fraction largest-magnitude values per tensor.

    All values below the magnitude threshold are zeroed out.

    Args:
        task_vector: Dict of delta tensors.
        top_k_fraction: Fraction of values to keep (e.g. 0.2 keeps the top 20%).

    Returns:
        New dict with trimmed tensors.
    """
    result: dict[str, Tensor] = {}
    for key, delta in task_vector.items():
        flat = delta.abs().flatten()
        if flat.numel() > 1 and top_k_fraction < 1.0:
            threshold = torch.quantile(flat, 1.0 - top_k_fraction)
            mask = (delta.abs() >= threshold).float()
            result[key] = delta * mask
        else:
            result[key] = delta.clone()
    return result


# ---------------------------------------------------------------------------
# Step 3 — TIES elect sign (majority vote)
# ---------------------------------------------------------------------------


def ties_elect_sign(
    task_vectors: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Elect the dominant sign for each parameter via majority vote.

    For each parameter position:
        sign_sum = sum of signs across all task vectors
        elected_sign = sign(sign_sum), with ties broken toward +1

    Args:
        task_vectors: List of task-vector dicts (same keys / shapes).

    Returns:
        Dict mapping each key to a tensor of elected signs in {-1, 0, 1}.
        In practice ties are broken to +1 so values are in {-1, +1}.
    """
    if not task_vectors:
        return {}

    elected: dict[str, Tensor] = {}
    for key in task_vectors[0]:
        tensors = [tv[key].float() for tv in task_vectors if key in tv]
        if not tensors:
            continue
        sign_sum = torch.stack([t.sign() for t in tensors]).sum(dim=0)
        e_sign = sign_sum.sign()
        # Break ties (sign_sum == 0) toward +1
        e_sign = torch.where(e_sign == 0, torch.ones_like(e_sign), e_sign)
        elected[key] = e_sign

    return elected


# ---------------------------------------------------------------------------
# Step 4 — TIES disjoint merge
# ---------------------------------------------------------------------------


def ties_disjoint_merge(
    task_vectors: list[dict[str, Tensor]],
    elected_signs: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Average only values that agree with the elected sign; zero out the rest.

    Zeros in a task vector are excluded from the average denominator (i.e. they
    do not contribute to either the numerator or the count).

    Args:
        task_vectors: List of (possibly trimmed / pruned) task-vector dicts.
        elected_signs: Dict of elected-sign tensors from :func:`ties_elect_sign`.

    Returns:
        Single merged task-vector dict.
    """
    if not task_vectors:
        return {}

    merged: dict[str, Tensor] = {}
    all_keys = elected_signs.keys()

    for key in all_keys:
        e_sign = elected_signs[key]
        accumulated = torch.zeros_like(e_sign, dtype=torch.float)
        count = torch.zeros_like(e_sign, dtype=torch.float)

        for tv in task_vectors:
            if key not in tv:
                continue
            delta = tv[key].float()
            # A value "agrees" if its sign matches the elected sign AND it is non-zero
            agrees = (delta.sign() == e_sign) & (delta != 0)
            accumulated = accumulated + delta * agrees.float()
            count = count + agrees.float()

        # Avoid division by zero
        count = count.clamp(min=1.0)
        merged[key] = accumulated / count

    return merged


# ---------------------------------------------------------------------------
# Step 5 — apply task vector
# ---------------------------------------------------------------------------


def apply_task_vector(
    base: dict[str, Tensor],
    task_vector: dict[str, Tensor],
    lambda_: float,
) -> dict[str, Tensor]:
    """Add a scaled task vector back to the base model weights.

    merged[k] = base[k] + lambda_ * task_vector[k]

    Keys present in base but absent from task_vector are copied unchanged.

    Args:
        base: Base model state dict.
        task_vector: Merged task-vector dict.
        lambda_: Scaling coefficient.

    Returns:
        New merged state dict (preserves original dtype of each tensor).
    """
    merged: dict[str, Tensor] = {}
    for key, base_param in base.items():
        if key in task_vector:
            delta = task_vector[key].float()
            merged[key] = (base_param.float() + lambda_ * delta).to(base_param.dtype)
        else:
            merged[key] = base_param.clone()
    return merged


# ---------------------------------------------------------------------------
# Unified merge_models entry point
# ---------------------------------------------------------------------------


def merge_models(
    base_state: dict[str, Tensor],
    finetuned_states: list[dict[str, Tensor]],
    config: MergeConfig,
) -> dict[str, Tensor]:
    """Merge multiple fine-tuned models into a single model.

    Supported pipelines (``config.method``):

    * ``"linear"``    — average all fine-tuned states with equal weights.
    * ``"dare"``      — compute task vectors, DARE-prune each, average, apply.
    * ``"ties"``      — compute task vectors, TIES-trim, elect sign,
                         disjoint merge, apply.
    * ``"dare_ties"`` — DARE-prune first, then full TIES pipeline.

    Args:
        base_state: Base model state dict.
        finetuned_states: List of fine-tuned model state dicts.
        config: :class:`MergeConfig` controlling the merge hyperparameters.

    Returns:
        Merged state dict.

    Raises:
        ValueError: If ``config.method`` is not one of the supported values.
    """
    if not finetuned_states:
        return {k: v.clone() for k, v in base_state.items()}

    method = config.method

    if method == "linear":
        return _linear_merge(base_state, finetuned_states)

    if method == "dare":
        return _dare_merge(base_state, finetuned_states, config)

    if method == "ties":
        return _ties_merge(base_state, finetuned_states, config)

    if method == "dare_ties":
        return _dare_ties_merge(base_state, finetuned_states, config)

    raise ValueError(
        f"Unknown merge method '{method}'. Supported: 'linear', 'dare', 'ties', 'dare_ties'."
    )


# ---------------------------------------------------------------------------
# Private pipeline helpers
# ---------------------------------------------------------------------------


def _linear_merge(
    base_state: dict[str, Tensor],
    finetuned_states: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    merged: dict[str, Tensor] = {}
    for key, base_param in base_state.items():
        candidates = [
            ft[key].float()
            for ft in finetuned_states
            if key in ft and ft[key].shape == base_param.shape
        ]
        if not candidates:
            merged[key] = base_param.clone()
        else:
            merged[key] = torch.stack(candidates).mean(dim=0).to(base_param.dtype)
    return merged


def _dare_merge(
    base_state: dict[str, Tensor],
    finetuned_states: list[dict[str, Tensor]],
    config: MergeConfig,
) -> dict[str, Tensor]:
    task_vectors = []
    for i, ft in enumerate(finetuned_states):
        tv = compute_task_vector(ft, base_state)
        pruned = dare_prune(tv, density=config.density, seed=config.seed + i * 100_000)
        task_vectors.append(pruned)

    # Average task vectors across models
    avg_tv: dict[str, Tensor] = {}
    for key in base_state:
        deltas = [tv[key].float() for tv in task_vectors if key in tv]
        if deltas:
            avg_tv[key] = torch.stack(deltas).mean(dim=0)

    return apply_task_vector(base_state, avg_tv, config.lambda_)


def _ties_merge(
    base_state: dict[str, Tensor],
    finetuned_states: list[dict[str, Tensor]],
    config: MergeConfig,
) -> dict[str, Tensor]:
    task_vectors = [compute_task_vector(ft, base_state) for ft in finetuned_states]
    trimmed = [ties_trim(tv, top_k_fraction=config.top_k_fraction) for tv in task_vectors]
    elected = ties_elect_sign(trimmed)
    merged_tv = ties_disjoint_merge(trimmed, elected)
    return apply_task_vector(base_state, merged_tv, config.lambda_)


def _dare_ties_merge(
    base_state: dict[str, Tensor],
    finetuned_states: list[dict[str, Tensor]],
    config: MergeConfig,
) -> dict[str, Tensor]:
    task_vectors = []
    for i, ft in enumerate(finetuned_states):
        tv = compute_task_vector(ft, base_state)
        pruned = dare_prune(tv, density=config.density, seed=config.seed + i * 100_000)
        task_vectors.append(pruned)

    trimmed = [ties_trim(tv, top_k_fraction=config.top_k_fraction) for tv in task_vectors]
    elected = ties_elect_sign(trimmed)
    merged_tv = ties_disjoint_merge(trimmed, elected)
    return apply_task_vector(base_state, merged_tv, config.lambda_)
