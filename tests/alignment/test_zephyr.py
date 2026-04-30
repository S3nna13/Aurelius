"""Tests for src/alignment/zephyr.py"""

import torch
import torch.nn as nn

from aurelius.alignment.zephyr import (
    AIFeedbackScorer,
    PreferenceDataBuilder,
    ZephyrTrainer,
    dDPOLoss,
)

SEED = 42
B = 4


# ---------------------------------------------------------------------------
# AIFeedbackScorer
# ---------------------------------------------------------------------------


def test_score_batch_output_shape():
    scorer = AIFeedbackScorer()
    lp_list = [torch.randn(5), torch.randn(3), torch.randn(7)]
    scores = scorer.score_batch(lp_list)
    assert scores.shape == (3,)


def test_score_batch_higher_log_prob_higher_score():
    scorer = AIFeedbackScorer()
    # Completion A has higher (less negative) mean log-prob
    lp_a = torch.full((5,), -0.5)
    lp_b = torch.full((5,), -2.0)
    scores = scorer.score_batch([lp_a, lp_b])
    assert scores[0] > scores[1]


def test_score_batch_single_token():
    scorer = AIFeedbackScorer()
    scores = scorer.score_batch([torch.tensor([-0.3])])
    assert scores.shape == (1,)
    assert torch.isfinite(scores[0])


# ---------------------------------------------------------------------------
# PreferenceDataBuilder
# ---------------------------------------------------------------------------


def test_build_pairs_chosen_higher_score():
    scorer = AIFeedbackScorer()
    builder = PreferenceDataBuilder(scorer, margin=0.01)
    lp_good = torch.full((5,), -0.2)
    lp_bad = torch.full((5,), -3.0)
    chosen, rejected = builder.build_pairs([lp_good, lp_bad])
    assert chosen == 0
    assert rejected == 1


def test_build_pairs_margin_returns_none():
    scorer = AIFeedbackScorer()
    builder = PreferenceDataBuilder(scorer, margin=10.0)  # huge margin
    lp_a = torch.full((5,), -1.0)
    lp_b = torch.full((5,), -1.01)
    chosen, rejected = builder.build_pairs([lp_a, lp_b])
    assert chosen is None and rejected is None


def test_build_pairs_single_completion_returns_none():
    scorer = AIFeedbackScorer()
    builder = PreferenceDataBuilder(scorer)
    chosen, rejected = builder.build_pairs([torch.randn(3)])
    assert chosen is None and rejected is None


def test_build_batch_pairs_length():
    scorer = AIFeedbackScorer()
    builder = PreferenceDataBuilder(scorer, margin=0.01)
    batch = [[torch.randn(5), torch.randn(5) + 2.0] for _ in range(3)]
    pairs = builder.build_batch_pairs(batch)
    assert len(pairs) == 3


# ---------------------------------------------------------------------------
# dDPOLoss
# ---------------------------------------------------------------------------


def test_ddpo_loss_scalar():
    loss_fn = dDPOLoss(beta=0.1)
    pi_w = torch.randn(B)
    pi_l = torch.randn(B)
    ref_w = torch.zeros(B)
    ref_l = torch.zeros(B)
    loss, _ = loss_fn(pi_w, pi_l, ref_w, ref_l)
    assert loss.shape == ()


def test_ddpo_loss_finite():
    loss_fn = dDPOLoss()
    pi_w, pi_l = torch.randn(B), torch.randn(B)
    loss, _ = loss_fn(pi_w, pi_l, torch.zeros(B), torch.zeros(B))
    assert torch.isfinite(loss)


def test_ddpo_accuracy_in_range():
    loss_fn = dDPOLoss()
    loss, metrics = loss_fn(torch.randn(B), torch.randn(B), torch.zeros(B), torch.zeros(B))
    assert 0.0 <= metrics["accuracy"].item() <= 1.0


def test_ddpo_correct_direction_accuracy():
    # pi_lp_w >> pi_lp_l with equal ref → all pairs correct
    loss_fn = dDPOLoss(beta=1.0)
    pi_w = torch.full((B,), 5.0)
    pi_l = torch.full((B,), -5.0)
    _, metrics = loss_fn(pi_w, pi_l, torch.zeros(B), torch.zeros(B))
    assert metrics["accuracy"].item() == 1.0


def test_ddpo_gradient_flows():
    loss_fn = dDPOLoss()
    pi_w = torch.randn(B, requires_grad=True)
    pi_l = torch.randn(B, requires_grad=True)
    loss, _ = loss_fn(pi_w, pi_l, torch.zeros(B), torch.zeros(B))
    loss.backward()
    assert pi_w.grad is not None


# ---------------------------------------------------------------------------
# ZephyrTrainer
# ---------------------------------------------------------------------------


def test_dsft_step_finite():
    torch.manual_seed(SEED)
    model = nn.Linear(16, 16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = dDPOLoss()
    trainer = ZephyrTrainer(model, nn.Linear(16, 16), opt, loss_fn)
    # Logits must be computed through the model so backward() works
    x = torch.randn(2, 5, 16)
    logits = model(x)
    token_ids = torch.randint(0, 16, (2, 5))
    loss = trainer.dsft_step(logits, token_ids)
    assert torch.isfinite(loss)


def test_ddpo_step_returns_correct_keys():
    model = nn.Linear(4, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = dDPOLoss()
    trainer = ZephyrTrainer(model, nn.Linear(4, 4), opt, loss_fn)
    pi_w, pi_l = torch.randn(B, requires_grad=True), torch.randn(B, requires_grad=True)
    loss, metrics = trainer.ddpo_step(pi_w, pi_l, torch.zeros(B), torch.zeros(B))
    assert "reward_chosen" in metrics
    assert "reward_rejected" in metrics
    assert "accuracy" in metrics


def test_freeze_ref_freezes_all():
    model = nn.Linear(4, 4)
    ref_model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = ZephyrTrainer(model, ref_model, opt, dDPOLoss())
    trainer.freeze_ref()
    for p in ref_model.parameters():
        assert not p.requires_grad


def test_dsft_gradient_flows_to_model():
    torch.manual_seed(SEED)
    model = nn.Linear(8, 8)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = ZephyrTrainer(model, nn.Linear(8, 8), opt, dDPOLoss())
    W_before = model.weight.data.clone()
    x = torch.randn(2, 6, 8)
    logits = model(x)
    token_ids = torch.randint(0, 8, (2, 6))
    trainer.dsft_step(logits, token_ids)
    assert not torch.allclose(model.weight.data, W_before)
