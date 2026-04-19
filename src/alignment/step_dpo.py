"""Step-DPO: Step-level Direct Preference Optimization.

Implements per-step preference optimization for chain-of-thought reasoning
from Lai et al. 2024 (arXiv:2406.18629). Instead of comparing whole responses,
compares individual reasoning STEPS at the point where they diverge given a
shared prefix.

Loss:
    L = -log sigma( beta * (
            log pi_theta(s_c | x, s_prefix) - log pi_ref(s_c | x, s_prefix)
          - log pi_theta(s_r | x, s_prefix) + log pi_ref(s_r | x, s_prefix) ) )

Numerically stable via softplus(-x) = -log(sigmoid(x)).

Reference:
    Lai et al., "Step-DPO: Step-wise Preference Optimization for Long-chain
    Reasoning of LLMs", arXiv:2406.18629, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepPreferenceExample:
    """A single step-level preference example.

    All log-probabilities are sums over tokens within the corresponding step
    (i.e. sequence log-probs conditioned on the shared prefix).

    Args:
        prefix_logprobs: Log-probability (sum) of the shared prefix under the
            policy. Not directly used by the loss but retained for downstream
            diagnostics / logging. Shape: scalar tensor.
        chosen_step_logprobs: Sum log-prob of the chosen step under the
            current policy pi_theta. Scalar tensor.
        rejected_step_logprobs: Sum log-prob of the rejected step under the
            current policy pi_theta. Scalar tensor.
        chosen_step_ref_logprobs: Sum log-prob of chosen step under reference.
        rejected_step_ref_logprobs: Sum log-prob of rejected step under ref.
    """

    prefix_logprobs: Tensor
    chosen_step_logprobs: Tensor
    rejected_step_logprobs: Tensor
    chosen_step_ref_logprobs: Tensor
    rejected_step_ref_logprobs: Tensor

    def __post_init__(self) -> None:
        for name in (
            "prefix_logprobs",
            "chosen_step_logprobs",
            "rejected_step_logprobs",
            "chosen_step_ref_logprobs",
            "rejected_step_ref_logprobs",
        ):
            val = getattr(self, name)
            if not isinstance(val, Tensor):
                raise TypeError(
                    f"StepPreferenceExample.{name} must be a torch.Tensor, "
                    f"got {type(val).__name__}"
                )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def step_dpo_loss(
    chosen_logprobs: Tensor,
    rejected_logprobs: Tensor,
    chosen_ref_logprobs: Tensor,
    rejected_ref_logprobs: Tensor,
    beta: float = 0.1,
) -> Tensor:
    """Step-DPO loss.

    L = softplus( -beta * ( (lp_c - lp_r_c) - (lp_r - lp_r_r) ) )

    where lp_* are policy log-probs and lp_r_* are reference log-probs for
    chosen and rejected steps respectively.

    Args:
        chosen_logprobs: Policy log-prob of chosen step. Shape (B,) or scalar.
        rejected_logprobs: Policy log-prob of rejected step.
        chosen_ref_logprobs: Reference log-prob of chosen step.
        rejected_ref_logprobs: Reference log-prob of rejected step.
        beta: Preference temperature; must be >= 0.

    Returns:
        Scalar tensor (mean over batch).
    """
    if beta < 0:
        raise ValueError(f"beta must be >= 0, got {beta}")

    tensors = [
        chosen_logprobs,
        rejected_logprobs,
        chosen_ref_logprobs,
        rejected_ref_logprobs,
    ]
    shape = tensors[0].shape
    for t in tensors[1:]:
        if t.shape != shape:
            raise ValueError(
                "step_dpo_loss: log-prob tensors must share shape; got "
                f"{[tuple(x.shape) for x in tensors]}"
            )

    chosen_ratio = chosen_logprobs - chosen_ref_logprobs
    rejected_ratio = rejected_logprobs - rejected_ref_logprobs
    margin = chosen_ratio - rejected_ratio
    # softplus(-x) is numerically stable for -log sigmoid(x)
    losses = F.softplus(-beta * margin)
    return losses.mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class StepDPOTrainer:
    """Trainer for Step-DPO.

    The trainer is model-agnostic: the user supplies two callables that, given
    a :class:`StepPreferenceExample` (or its prefix+step contents) return the
    per-example sum log-prob of a given step. In typical usage these closures
    wrap a transformer forward pass.

    Args:
        policy_logprob_fn: Callable mapping a StepPreferenceExample to a dict
            with keys ``chosen`` and ``rejected`` (scalar tensors) under the
            current policy. Alternatively the trainer can be used purely on
            already-computed log-probs via :func:`step_dpo_loss` directly.
        ref_logprob_fn: Same, but under the frozen reference policy.
        beta: Preference temperature.
    """

    def __init__(
        self,
        policy_logprob_fn: Optional[Callable[[StepPreferenceExample], dict]] = None,
        ref_logprob_fn: Optional[Callable[[StepPreferenceExample], dict]] = None,
        beta: float = 0.1,
    ) -> None:
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        self.policy_logprob_fn = policy_logprob_fn
        self.ref_logprob_fn = ref_logprob_fn
        self.beta = float(beta)

    # -- helpers ----------------------------------------------------------

    def _gather(self, batch: List[StepPreferenceExample]):
        """Materialise per-example log-prob tensors from the batch.

        If ``policy_logprob_fn`` is provided it is called per-example and is
        expected to return differentiable policy log-probs. Otherwise the
        pre-computed ``*_logprobs`` fields on each example are used directly
        (useful for unit tests and for pipelines that pre-cache log-probs).
        """
        if len(batch) == 0:
            raise ValueError("StepDPOTrainer: batch must be non-empty")

        chosen_lp: List[Tensor] = []
        rejected_lp: List[Tensor] = []
        chosen_ref_lp: List[Tensor] = []
        rejected_ref_lp: List[Tensor] = []

        for ex in batch:
            if self.policy_logprob_fn is not None:
                pol = self.policy_logprob_fn(ex)
                chosen_lp.append(pol["chosen"].reshape(()))
                rejected_lp.append(pol["rejected"].reshape(()))
            else:
                chosen_lp.append(ex.chosen_step_logprobs.reshape(()))
                rejected_lp.append(ex.rejected_step_logprobs.reshape(()))

            if self.ref_logprob_fn is not None:
                ref = self.ref_logprob_fn(ex)
                chosen_ref_lp.append(ref["chosen"].reshape(()).detach())
                rejected_ref_lp.append(ref["rejected"].reshape(()).detach())
            else:
                chosen_ref_lp.append(ex.chosen_step_ref_logprobs.reshape(()).detach())
                rejected_ref_lp.append(ex.rejected_step_ref_logprobs.reshape(()).detach())

        return (
            torch.stack(chosen_lp),
            torch.stack(rejected_lp),
            torch.stack(chosen_ref_lp),
            torch.stack(rejected_ref_lp),
        )

    # -- API --------------------------------------------------------------

    def compute_loss(self, batch: List[StepPreferenceExample]) -> Tensor:
        """Compute the Step-DPO loss over a batch of examples."""
        c, r, cr, rr = self._gather(batch)
        return step_dpo_loss(c, r, cr, rr, beta=self.beta)

    def compute_loss_and_metrics(self, batch: List[StepPreferenceExample]):
        c, r, cr, rr = self._gather(batch)
        loss = step_dpo_loss(c, r, cr, rr, beta=self.beta)
        with torch.no_grad():
            reward_margin = ((c - cr) - (r - rr)).mean()
        return loss, {
            "loss": float(loss.detach().item()),
            "reward_margin": float(reward_margin.item()),
        }

    def step(self, optimizer, batch: List[StepPreferenceExample]) -> dict:
        """Run one optimization step and return metrics."""
        optimizer.zero_grad()
        loss, metrics = self.compute_loss_and_metrics(batch)
        loss.backward()
        optimizer.step()
        return metrics


__all__ = [
    "StepPreferenceExample",
    "step_dpo_loss",
    "StepDPOTrainer",
]
