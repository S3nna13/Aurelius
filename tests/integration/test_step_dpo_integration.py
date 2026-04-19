"""Integration tests for Step-DPO surfaced via src.alignment."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

import src.alignment as alignment
from src.alignment import StepPreferenceExample, StepDPOTrainer, step_dpo_loss


def test_surface_exports_present():
    assert hasattr(alignment, "StepPreferenceExample")
    assert hasattr(alignment, "StepDPOTrainer")
    assert hasattr(alignment, "step_dpo_loss")


def test_existing_alignment_entries_intact():
    # Previously-exposed symbols must remain reachable.
    for name in (
        "bradley_terry_loss",
        "margin_ranking_loss",
        "dpo_pair_loss",
        "KTOv2Loss",
        "kto_v2_loss_functional",
    ):
        assert hasattr(alignment, name), f"missing {name}"


def test_end_to_end_small_training_reduces_loss():
    """Train a tiny logprob head to prefer chosen over rejected steps."""
    torch.manual_seed(0)

    # Toy: two "steps" represented by feature vectors; a tiny linear scorer
    # produces log-probs. We freeze a copy as the reference.
    feat_dim = 4
    n_examples = 6

    feats_chosen = torch.randn(n_examples, feat_dim)
    feats_rejected = torch.randn(n_examples, feat_dim)

    scorer = nn.Linear(feat_dim, 1, bias=False)
    ref_scorer = nn.Linear(feat_dim, 1, bias=False)
    ref_scorer.load_state_dict(scorer.state_dict())
    for p in ref_scorer.parameters():
        p.requires_grad_(False)

    def build_batch():
        examples = []
        policy_c = scorer(feats_chosen).squeeze(-1)
        policy_r = scorer(feats_rejected).squeeze(-1)
        with torch.no_grad():
            ref_c = ref_scorer(feats_chosen).squeeze(-1)
            ref_r = ref_scorer(feats_rejected).squeeze(-1)
        for i in range(n_examples):
            examples.append(
                StepPreferenceExample(
                    prefix_logprobs=torch.tensor(0.0),
                    chosen_step_logprobs=policy_c[i],
                    rejected_step_logprobs=policy_r[i],
                    chosen_step_ref_logprobs=ref_c[i],
                    rejected_step_ref_logprobs=ref_r[i],
                )
            )
        return examples

    trainer = StepDPOTrainer(beta=0.5)
    opt = torch.optim.SGD(scorer.parameters(), lr=0.5)

    initial_loss = trainer.compute_loss(build_batch()).item()

    for _ in range(50):
        batch = build_batch()
        trainer.step(opt, batch)

    final_loss = trainer.compute_loss(build_batch()).item()
    assert final_loss < initial_loss - 1e-3, (
        f"loss did not decrease: {initial_loss} -> {final_loss}"
    )
    # After training, reward margin should be positive on average.
    _, metrics = trainer.compute_loss_and_metrics(build_batch())
    assert metrics["reward_margin"] > 0.0


def test_functional_loss_matches_manual_formula():
    c = torch.tensor([0.4])
    r = torch.tensor([-0.1])
    cr = torch.tensor([0.1])
    rr = torch.tensor([0.0])
    beta = 0.3
    margin = (c - cr) - (r - rr)
    manual = torch.nn.functional.softplus(-beta * margin).mean()
    got = step_dpo_loss(c, r, cr, rr, beta=beta)
    assert torch.allclose(got, manual)


def test_callable_policy_and_ref_fns():
    """Trainer supports callables that compute log-probs on the fly."""
    theta = torch.tensor([1.0], requires_grad=True)

    def policy_fn(ex):
        return {
            "chosen": theta * ex.chosen_step_logprobs,
            "rejected": theta * ex.rejected_step_logprobs,
        }

    def ref_fn(ex):
        return {
            "chosen": ex.chosen_step_ref_logprobs,
            "rejected": ex.rejected_step_ref_logprobs,
        }

    trainer = StepDPOTrainer(
        policy_logprob_fn=policy_fn, ref_logprob_fn=ref_fn, beta=0.2
    )
    batch = [
        StepPreferenceExample(
            prefix_logprobs=torch.tensor(0.0),
            chosen_step_logprobs=torch.tensor(0.5),
            rejected_step_logprobs=torch.tensor(-0.5),
            chosen_step_ref_logprobs=torch.tensor(0.0),
            rejected_step_ref_logprobs=torch.tensor(0.0),
        )
    ]
    loss = trainer.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert theta.grad is not None and math.isfinite(theta.grad.item())
