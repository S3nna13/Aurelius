"""Weight-space model merging for Aurelius family members.

Implements four strategies that operate on plain ``dict[str, torch.Tensor]``
state dicts without requiring a model instance:

* **LINEAR** (model soup) — weighted average of state dicts.
* **SLERP** — spherical linear interpolation between two state dicts.
* **TIES** — trim + elect sign + disjoint merge (Yadav et al. 2023).
* **DARE** — drop and rescale (Yu et al. 2024).

All routines are deterministic under ``torch.manual_seed`` and raise
:class:`MergeError` on key-set or shape mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import torch


class MergeError(Exception):
    """Raised when a merge cannot proceed (key/shape mismatch, bad args)."""


class MergeStrategy(str, Enum):
    """Available weight-space merge strategies."""

    LINEAR = "linear"
    SLERP = "slerp"
    TIES = "ties"
    DARE = "dare"


@dataclass
class MergeResult:
    """Outcome of a merge operation."""

    state_dict: Dict[str, torch.Tensor]
    strategy: MergeStrategy
    contributors: Tuple[str, ...]
    metadata: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_same_keys(states: Sequence[Dict[str, torch.Tensor]]) -> None:
    if not states:
        raise MergeError("at least one state dict is required")
    ref = set(states[0].keys())
    for i, sd in enumerate(states[1:], start=1):
        cur = set(sd.keys())
        if cur != ref:
            missing = sorted(ref - cur)
            extra = sorted(cur - ref)
            raise MergeError(
                f"key-set mismatch at state[{i}]: "
                f"missing={missing!r} extra={extra!r}"
            )


def _check_same_shapes(states: Sequence[Dict[str, torch.Tensor]]) -> None:
    ref = states[0]
    for i, sd in enumerate(states[1:], start=1):
        for k, t in sd.items():
            rt = ref[k]
            if t.shape != rt.shape:
                raise MergeError(
                    f"shape mismatch at key {k!r}: "
                    f"state[0]={tuple(rt.shape)} vs state[{i}]={tuple(t.shape)}"
                )
            if t.is_complex() or rt.is_complex():
                raise MergeError(f"complex tensors not supported at key {k!r}")


def _validate(states: Sequence[Dict[str, torch.Tensor]]) -> None:
    _check_same_keys(states)
    _check_same_shapes(states)


# ---------------------------------------------------------------------------
# LINEAR
# ---------------------------------------------------------------------------


def linear_merge(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Weighted mean of ``state_dicts``.

    If ``weights`` is ``None`` the uniform average is used. Weights are
    normalised to sum to 1.
    """
    if not state_dicts:
        raise MergeError("linear_merge requires at least one state dict")
    _validate(state_dicts)

    n = len(state_dicts)
    if weights is None:
        w = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise MergeError(
                f"weights length {len(weights)} != state_dicts length {n}"
            )
        s = float(sum(weights))
        if s == 0.0:
            raise MergeError("weights must not sum to zero")
        w = [float(x) / s for x in weights]

    out: Dict[str, torch.Tensor] = {}
    for k in state_dicts[0]:
        ref = state_dicts[0][k]
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for sd, wi in zip(state_dicts, w):
            acc = acc + sd[k].to(torch.float32) * wi
        out[k] = acc.to(ref.dtype)
    return out


# ---------------------------------------------------------------------------
# SLERP
# ---------------------------------------------------------------------------


def _slerp_tensor(a: torch.Tensor, b: torch.Tensor, t: float, eps: float = 1e-6) -> torch.Tensor:
    af = a.to(torch.float32).reshape(-1)
    bf = b.to(torch.float32).reshape(-1)
    na = torch.linalg.vector_norm(af)
    nb = torch.linalg.vector_norm(bf)
    if na < eps or nb < eps:
        return (1.0 - t) * a.to(torch.float32) + t * b.to(torch.float32)
    dot = torch.dot(af, bf) / (na * nb)
    dot = torch.clamp(dot, -1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    # near-parallel → fall back to linear
    if float(sin_omega) < eps:
        return ((1.0 - t) * a.to(torch.float32) + t * b.to(torch.float32))
    c_a = torch.sin((1.0 - t) * omega) / sin_omega
    c_b = torch.sin(t * omega) / sin_omega
    merged = c_a * a.to(torch.float32) + c_b * b.to(torch.float32)
    return merged


def slerp_merge(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    t: float,
) -> Dict[str, torch.Tensor]:
    """Per-tensor spherical linear interpolation between ``a`` and ``b``.

    ``t=0`` returns ``a``; ``t=1`` returns ``b``. When the two flattened
    tensors are nearly parallel, falls back to linear interpolation.
    """
    if not (0.0 <= t <= 1.0):
        raise MergeError(f"slerp t must lie in [0, 1], got {t}")
    _validate([a, b])

    out: Dict[str, torch.Tensor] = {}
    for k in a:
        ta = a[k]
        tb = b[k]
        if t == 0.0:
            out[k] = ta.clone()
            continue
        if t == 1.0:
            out[k] = tb.clone()
            continue
        merged = _slerp_tensor(ta, tb, t)
        out[k] = merged.reshape(ta.shape).to(ta.dtype)
    return out


# ---------------------------------------------------------------------------
# TIES
# ---------------------------------------------------------------------------


def ties_merge(
    base: Dict[str, torch.Tensor],
    task_deltas: List[Dict[str, torch.Tensor]],
    trim_ratio: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """TIES-Merging: trim + elect sign + disjoint mean.

    Parameters
    ----------
    base
        Pretrained state dict.
    task_deltas
        List of ``finetuned - base`` deltas (one per task).
    trim_ratio
        Fraction (0–1) of smallest-magnitude entries to zero **per delta**.
    """
    if not task_deltas:
        raise MergeError("ties_merge requires at least one task delta")
    if not (0.0 <= trim_ratio < 1.0):
        raise MergeError(f"trim_ratio must lie in [0, 1), got {trim_ratio}")
    _validate([base, *task_deltas])

    # 1. Trim: zero out bottom trim_ratio by |value| per delta.
    trimmed: List[Dict[str, torch.Tensor]] = []
    for d in task_deltas:
        td: Dict[str, torch.Tensor] = {}
        for k, v in d.items():
            vf = v.to(torch.float32)
            if trim_ratio <= 0.0 or vf.numel() == 0:
                td[k] = vf.clone()
                continue
            flat = vf.reshape(-1).abs()
            k_keep = max(1, int(round(flat.numel() * (1.0 - trim_ratio))))
            if k_keep >= flat.numel():
                td[k] = vf.clone()
                continue
            # threshold = (numel - k_keep)-th smallest absolute value
            threshold = torch.topk(flat, k=flat.numel() - k_keep, largest=False).values.max()
            mask = vf.abs() > threshold
            td[k] = vf * mask.to(vf.dtype)
        trimmed.append(td)

    out: Dict[str, torch.Tensor] = {}
    for k, bv in base.items():
        ref_dtype = bv.dtype
        stacks = torch.stack([td[k] for td in trimmed], dim=0)  # [T, *shape]
        # 2. Elect sign by summed-magnitude majority.
        pos = (stacks > 0).to(torch.float32) * stacks
        neg = (stacks < 0).to(torch.float32) * stacks
        pos_sum = pos.sum(dim=0)
        neg_sum = neg.sum(dim=0)  # negative number
        elected_sign = torch.where(
            pos_sum.abs() >= neg_sum.abs(),
            torch.ones_like(pos_sum),
            -torch.ones_like(pos_sum),
        )
        # 3. Disjoint mean: average only entries whose sign matches elected.
        sign_match = (torch.sign(stacks) == elected_sign.unsqueeze(0)).to(torch.float32)
        contrib = stacks * sign_match
        denom = sign_match.sum(dim=0).clamp(min=1.0)
        merged_delta = contrib.sum(dim=0) / denom
        out[k] = (bv.to(torch.float32) + merged_delta).to(ref_dtype)
    return out


# ---------------------------------------------------------------------------
# DARE
# ---------------------------------------------------------------------------


def dare_merge(
    base: Dict[str, torch.Tensor],
    task_delta: Dict[str, torch.Tensor],
    drop_rate: float = 0.5,
    scale_mode: str = "rescale",
) -> Dict[str, torch.Tensor]:
    """DARE: drop ``drop_rate`` of entries in ``task_delta``, rescale, add.

    ``scale_mode`` = ``"rescale"`` (default) multiplies survivors by
    ``1 / (1 - drop_rate)`` so the expected magnitude is preserved.
    ``"none"`` leaves them unscaled.
    """
    if not (0.0 <= drop_rate <= 1.0):
        raise MergeError(f"drop_rate must lie in [0, 1], got {drop_rate}")
    if scale_mode not in ("rescale", "none"):
        raise MergeError(f"unknown scale_mode {scale_mode!r}")
    _validate([base, task_delta])

    out: Dict[str, torch.Tensor] = {}
    for k, bv in base.items():
        d = task_delta[k].to(torch.float32)
        if drop_rate >= 1.0:
            # Everything dropped → base unchanged.
            out[k] = bv.clone()
            continue
        if drop_rate <= 0.0:
            out[k] = (bv.to(torch.float32) + d).to(bv.dtype)
            continue
        # Bernoulli keep mask.
        keep = torch.bernoulli(
            torch.full_like(d, 1.0 - drop_rate)
        )
        if scale_mode == "rescale":
            scale = 1.0 / (1.0 - drop_rate)
        else:
            scale = 1.0
        merged = bv.to(torch.float32) + d * keep * scale
        out[k] = merged.to(bv.dtype)
    return out


# ---------------------------------------------------------------------------
# Unified wrapper
# ---------------------------------------------------------------------------


class ModelMerger:
    """Strategy-agnostic wrapper around the four merge primitives."""

    def __init__(self, strategy: MergeStrategy, **params: object) -> None:
        if not isinstance(strategy, MergeStrategy):
            try:
                strategy = MergeStrategy(strategy)
            except ValueError as exc:
                raise MergeError(f"unknown strategy {strategy!r}") from exc
        self.strategy = strategy
        self.params = dict(params)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _default_names(n: int) -> Tuple[str, ...]:
        return tuple(f"model_{i}" for i in range(n))

    # -- dispatch ----------------------------------------------------------
    def merge(
        self,
        state_dicts: List[Dict[str, torch.Tensor]],
        names: Optional[Sequence[str]] = None,
    ) -> MergeResult:
        if not state_dicts:
            raise MergeError("merge() requires at least one state dict")
        contributors = (
            tuple(names) if names is not None else self._default_names(len(state_dicts))
        )
        if len(contributors) != len(state_dicts):
            raise MergeError(
                f"names length {len(contributors)} != state_dicts length {len(state_dicts)}"
            )
        meta: Dict[str, object] = {"n_inputs": len(state_dicts)}

        if self.strategy is MergeStrategy.LINEAR:
            weights = self.params.get("weights")
            sd = linear_merge(state_dicts, weights=weights)  # type: ignore[arg-type]
            meta["weights"] = weights
        elif self.strategy is MergeStrategy.SLERP:
            if len(state_dicts) != 2:
                raise MergeError("SLERP requires exactly two state dicts")
            t = float(self.params.get("t", 0.5))
            sd = slerp_merge(state_dicts[0], state_dicts[1], t)
            meta["t"] = t
        elif self.strategy is MergeStrategy.TIES:
            if len(state_dicts) < 2:
                raise MergeError("TIES requires base + at least one task delta")
            base = state_dicts[0]
            deltas = list(state_dicts[1:])
            trim_ratio = float(self.params.get("trim_ratio", 0.2))
            sd = ties_merge(base, deltas, trim_ratio=trim_ratio)
            meta["trim_ratio"] = trim_ratio
        elif self.strategy is MergeStrategy.DARE:
            if len(state_dicts) != 2:
                raise MergeError("DARE requires exactly [base, task_delta]")
            drop_rate = float(self.params.get("drop_rate", 0.5))
            scale_mode = str(self.params.get("scale_mode", "rescale"))
            sd = dare_merge(
                state_dicts[0],
                state_dicts[1],
                drop_rate=drop_rate,
                scale_mode=scale_mode,
            )
            meta["drop_rate"] = drop_rate
            meta["scale_mode"] = scale_mode
        else:  # pragma: no cover — enum exhausted
            raise MergeError(f"unsupported strategy {self.strategy!r}")

        return MergeResult(
            state_dict=sd,
            strategy=self.strategy,
            contributors=contributors,
            metadata=meta,
        )


MERGING_REGISTRY: Dict[str, object] = {
    "linear": linear_merge,
    "slerp": slerp_merge,
    "ties": ties_merge,
    "dare": dare_merge,
}


__all__ = [
    "MERGING_REGISTRY",
    "MergeError",
    "MergeResult",
    "MergeStrategy",
    "ModelMerger",
    "dare_merge",
    "linear_merge",
    "slerp_merge",
    "ties_merge",
]
