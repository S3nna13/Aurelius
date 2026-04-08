"""DARE and improved TIES model merging methods.

DARE (Drop And REscale) — Yu et al. 2023:
    Randomly drops p% of task vector parameters (sets to zero), then
    rescales remaining values by 1/(1-p) to preserve expected magnitude.

TIES (Trim, Elect Sign, Merge) — Yadav et al. 2023 (improved):
    1. Trim: keep only top-k% of task vector values by absolute magnitude.
    2. Elect sign: for each parameter, use sign of sum of task vectors
       (majority vote when magnitudes are uniform).
    3. Merge: average only same-sign parameters, then add to base.

DARE-TIES combines both: apply DARE first, then TIES.
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def compute_task_vector(
    finetuned_state: dict[str, torch.Tensor],
    base_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute task vector δ_i = finetuned_i - base_i for all matching keys."""
    task_vector: dict[str, torch.Tensor] = {}
    for key in base_state:
        if key in finetuned_state and finetuned_state[key].shape == base_state[key].shape:
            task_vector[key] = finetuned_state[key].float() - base_state[key].float()
    return task_vector


def dare_drop(
    task_vector: dict[str, torch.Tensor],
    drop_rate: float = 0.9,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Randomly zero out `drop_rate` fraction of each parameter's values.

    Rescales remaining values by 1 / (1 - drop_rate) to preserve expected
    magnitude. Uses per-tensor RNG seeded by (seed + hash(key)) for
    reproducibility. Returns a new task vector dict (does not modify in place).

    Args:
        task_vector: Dict mapping parameter names to delta tensors.
        drop_rate: Fraction of values to zero out (0.0 to 1.0).
        seed: Base random seed; each key gets seed + hash(key) & 0xFFFFFFFF.

    Returns:
        New task vector dict with DARE applied.
    """
    result: dict[str, torch.Tensor] = {}
    rescale = 1.0 / (1.0 - drop_rate)

    for key, delta in task_vector.items():
        # Per-tensor reproducible RNG
        key_seed = (seed + hash(key)) & 0xFFFFFFFF
        gen = torch.Generator()
        gen.manual_seed(key_seed)

        # Bernoulli mask: 1 = keep, 0 = drop
        keep_mask = torch.bernoulli(
            torch.full(delta.shape, 1.0 - drop_rate), generator=gen
        )
        result[key] = delta * keep_mask * rescale

    return result


def ties_trim(
    task_vector: dict[str, torch.Tensor],
    top_k: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Keep only top_k fraction of parameter values by absolute magnitude.

    Zero out the rest. Returns a new dict.

    Args:
        task_vector: Dict mapping parameter names to delta tensors.
        top_k: Fraction of values to keep (e.g. 0.2 keeps top 20%).

    Returns:
        New task vector dict with small values zeroed.
    """
    result: dict[str, torch.Tensor] = {}
    for key, delta in task_vector.items():
        flat = delta.abs().flatten()
        if flat.numel() > 1 and top_k < 1.0:
            threshold = torch.quantile(flat, 1.0 - top_k)
            mask = (delta.abs() >= threshold).float()
            result[key] = delta * mask
        else:
            result[key] = delta.clone()
    return result


def ties_elect_sign(
    task_vectors: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Compute the elected sign for each parameter across multiple task vectors.

    The elected sign is the sign of the element-wise sum of all task vectors.
    Zeros remain zero (ties or exactly-cancelled values).

    Args:
        task_vectors: List of task vector dicts (same keys/shapes).

    Returns:
        Dict of elected signs (+1, 0, or -1 per element).
    """
    if not task_vectors:
        return {}

    elected: dict[str, torch.Tensor] = {}
    all_keys = task_vectors[0].keys()

    for key in all_keys:
        tensors = [tv[key] for tv in task_vectors if key in tv]
        if not tensors:
            continue
        total = torch.zeros_like(tensors[0], dtype=torch.float)
        for t in tensors:
            total = total + t.float()
        elected[key] = torch.sign(total)

    return elected


def ties_merge(
    task_vectors: list[dict[str, torch.Tensor]],
    base_state: dict[str, torch.Tensor],
    top_k: float = 0.2,
    scaling_coeff: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Full TIES merge.

    1. Trim each task vector (keep top_k by magnitude).
    2. Elect sign (sum-based).
    3. For each param: average only task vectors whose sign agrees with
       the elected sign (zero values are ignored from the count).
    4. Merged state: base + scaling_coeff * merged_delta.

    Args:
        task_vectors: List of task vector dicts.
        base_state: Base model state dict.
        top_k: Fraction of values to keep per task vector (TIES trim step).
        scaling_coeff: Scalar multiplied onto the merged delta before adding.

    Returns:
        Merged state dict.
    """
    if not task_vectors:
        return {k: v.clone() for k, v in base_state.items()}

    # Step 1: Trim
    trimmed = [ties_trim(tv, top_k=top_k) for tv in task_vectors]

    # Step 2: Elect sign
    elected_sign = ties_elect_sign(trimmed)

    # Step 3 & 4: Average same-sign contributions, then add to base
    merged_state: dict[str, torch.Tensor] = {}

    for key in base_state:
        base_param = base_state[key]

        if key not in elected_sign:
            merged_state[key] = base_param.clone()
            continue

        e_sign = elected_sign[key]  # shape = base_param.shape
        merged_tv = torch.zeros_like(base_param, dtype=torch.float)
        count = torch.zeros_like(base_param, dtype=torch.float)

        for tv in trimmed:
            if key not in tv:
                continue
            delta = tv[key].float()
            # A value agrees if its sign matches the elected sign,
            # or the elected sign is 0 (sum was exactly 0).
            agrees = (delta.sign() == e_sign) & (delta != 0)
            merged_tv = merged_tv + delta * agrees.float()
            count = count + agrees.float()

        # Avoid division by zero
        count = count.clamp(min=1.0)
        merged_tv = merged_tv / count

        merged_state[key] = (base_param.float() + scaling_coeff * merged_tv).to(
            base_param.dtype
        )

    return merged_state


# ---------------------------------------------------------------------------
# Combined DARE-TIES
# ---------------------------------------------------------------------------


def dare_ties_merge(
    finetuned_states: list[dict[str, torch.Tensor]],
    base_state: dict[str, torch.Tensor],
    drop_rate: float = 0.9,
    top_k: float = 0.2,
    scaling_coeff: float = 1.0,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """DARE + TIES combined.

    1. Compute task vectors (finetuned - base).
    2. Apply DARE drop to each task vector.
    3. Apply TIES merge (trim, elect sign, average same-sign, add to base).

    Args:
        finetuned_states: List of fine-tuned model state dicts.
        base_state: Base model state dict.
        drop_rate: DARE drop fraction (default 0.9).
        top_k: TIES trim fraction to keep (default 0.2).
        scaling_coeff: Scaling applied to merged delta (default 1.0).
        seed: Base random seed for DARE; each model/key uses a derived seed.

    Returns:
        Merged state dict.
    """
    task_vectors = []
    for i, ft_state in enumerate(finetuned_states):
        tv = compute_task_vector(ft_state, base_state)
        # Give each model a different seed offset so their drops are independent
        dropped = dare_drop(tv, drop_rate=drop_rate, seed=seed + i * 100_000)
        task_vectors.append(dropped)

    return ties_merge(
        task_vectors,
        base_state,
        top_k=top_k,
        scaling_coeff=scaling_coeff,
    )


# ---------------------------------------------------------------------------
# Unified MergingPipeline
# ---------------------------------------------------------------------------


class MergingPipeline:
    """Unified pipeline for model merging.

    Supports methods: 'dare', 'ties', 'dare_ties', 'simple_average',
    'weighted_average'.

    Args:
        base_state: Base model weights (state dict).
        finetuned_states: List of fine-tuned model weights (state dicts).
    """

    SUPPORTED_METHODS = frozenset(
        {"dare", "ties", "dare_ties", "simple_average", "weighted_average"}
    )

    def __init__(
        self,
        base_state: dict[str, torch.Tensor],
        finetuned_states: list[dict[str, torch.Tensor]],
    ) -> None:
        self.base_state = base_state
        self.finetuned_states = finetuned_states

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge(
        self,
        method: str = "dare_ties",
        weights: list[float] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Dispatch to the appropriate merge method.

        Args:
            method: One of 'dare', 'ties', 'dare_ties', 'simple_average',
                'weighted_average'.
            weights: Per-model weights for 'weighted_average'. Defaults to
                uniform weights if None.
            **kwargs: Forwarded to the underlying merge function.

        Returns:
            Merged state dict.

        Raises:
            ValueError: If method is not supported or finetuned_states is empty.
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown merge method '{method}'. "
                f"Supported: {sorted(self.SUPPORTED_METHODS)}"
            )
        if not self.finetuned_states:
            return {k: v.clone() for k, v in self.base_state.items()}

        if method == "dare":
            return self._merge_dare(**kwargs)
        elif method == "ties":
            return self._merge_ties(**kwargs)
        elif method == "dare_ties":
            return dare_ties_merge(
                self.finetuned_states, self.base_state, **kwargs
            )
        elif method == "simple_average":
            return self._merge_simple_average()
        elif method == "weighted_average":
            return self._merge_weighted_average(weights=weights)
        # Should never reach here due to guard above
        raise RuntimeError("Unreachable")  # pragma: no cover

    def evaluate_merge_quality(
        self,
        merged_state: dict[str, torch.Tensor],
        reference_state: dict[str, torch.Tensor],
    ) -> dict:
        """Compute merge quality metrics.

        Args:
            merged_state: The merged model state dict.
            reference_state: Reference model state dict to compare against.

        Returns:
            Dict with keys:
                'l2_distance': float — total ||merged - reference||_2.
                'cosine_similarity': float — mean cosine similarity per param.
                'param_count': int — total number of scalar parameters compared.
        """
        l2_sq_total = 0.0
        cosine_similarities: list[float] = []
        param_count = 0

        common_keys = [
            k
            for k in merged_state
            if k in reference_state
            and merged_state[k].shape == reference_state[k].shape
        ]

        for key in common_keys:
            m = merged_state[key].float().flatten()
            r = reference_state[key].float().flatten()

            diff = m - r
            l2_sq_total += diff.dot(diff).item()
            param_count += m.numel()

            m_norm = m.norm()
            r_norm = r.norm()
            if m_norm > 0 and r_norm > 0:
                cos_sim = (m.dot(r) / (m_norm * r_norm)).item()
            else:
                cos_sim = 1.0 if m_norm == r_norm else 0.0
            cosine_similarities.append(cos_sim)

        l2_distance = float(l2_sq_total**0.5)
        mean_cosine = (
            float(sum(cosine_similarities) / len(cosine_similarities))
            if cosine_similarities
            else 1.0
        )

        return {
            "l2_distance": l2_distance,
            "cosine_similarity": mean_cosine,
            "param_count": param_count,
        }

    # ------------------------------------------------------------------
    # Private merge implementations
    # ------------------------------------------------------------------

    def _merge_dare(self, **kwargs) -> dict[str, torch.Tensor]:
        """DARE only: compute task vectors, apply DARE drop, simple average."""
        drop_rate: float = kwargs.get("drop_rate", 0.9)
        seed: int = kwargs.get("seed", 42)
        scaling_coeff: float = kwargs.get("scaling_coeff", 1.0)

        task_vectors = []
        for i, ft_state in enumerate(self.finetuned_states):
            tv = compute_task_vector(ft_state, self.base_state)
            dropped = dare_drop(tv, drop_rate=drop_rate, seed=seed + i * 100_000)
            task_vectors.append(dropped)

        # Average the dropped task vectors, then add to base
        merged_state: dict[str, torch.Tensor] = {}
        for key in self.base_state:
            base_param = self.base_state[key]
            deltas = [
                tv[key].float() for tv in task_vectors if key in tv
            ]
            if not deltas:
                merged_state[key] = base_param.clone()
                continue
            avg_delta = torch.stack(deltas).mean(dim=0)
            merged_state[key] = (
                base_param.float() + scaling_coeff * avg_delta
            ).to(base_param.dtype)

        return merged_state

    def _merge_ties(self, **kwargs) -> dict[str, torch.Tensor]:
        """TIES only (no DARE)."""
        task_vectors = [
            compute_task_vector(ft, self.base_state)
            for ft in self.finetuned_states
        ]
        return ties_merge(task_vectors, self.base_state, **kwargs)

    def _merge_simple_average(self) -> dict[str, torch.Tensor]:
        """Element-wise average of all fine-tuned states."""
        merged_state: dict[str, torch.Tensor] = {}
        for key in self.base_state:
            candidates = [
                ft[key].float()
                for ft in self.finetuned_states
                if key in ft and ft[key].shape == self.base_state[key].shape
            ]
            if not candidates:
                merged_state[key] = self.base_state[key].clone()
            else:
                avg = torch.stack(candidates).mean(dim=0)
                merged_state[key] = avg.to(self.base_state[key].dtype)
        return merged_state

    def _merge_weighted_average(
        self, weights: list[float] | None = None
    ) -> dict[str, torch.Tensor]:
        """Weighted average of fine-tuned states."""
        n = len(self.finetuned_states)
        if weights is None:
            weights = [1.0 / n] * n
        else:
            total = sum(weights)
            weights = [w / total for w in weights]  # normalise

        merged_state: dict[str, torch.Tensor] = {}
        for key in self.base_state:
            ref = self.base_state[key]
            weighted_sum = torch.zeros_like(ref, dtype=torch.float)
            weight_sum = 0.0
            for ft, w in zip(self.finetuned_states, weights):
                if key in ft and ft[key].shape == ref.shape:
                    weighted_sum = weighted_sum + w * ft[key].float()
                    weight_sum += w
            if weight_sum == 0.0:
                merged_state[key] = ref.clone()
            else:
                merged_state[key] = (weighted_sum / weight_sum).to(ref.dtype)
        return merged_state
