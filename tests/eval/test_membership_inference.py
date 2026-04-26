"""Tests for src/eval/membership_inference.py.

Covers all required test cases (15 total):
 1.  LossAttack score shape (B,)
 2.  LossAttack: lower loss → higher member score
 3.  MinKProbAttack score shape (B,)
 4.  MinKProbAttack: k_percent=0.2 uses bottom-20% of tokens
 5.  MinKProbAttack: highly confident (peaked) text scores higher than flat
 6.  LikelihoodRatioAttack score shape (B,)
 7.  MembershipInferenceEvaluator.evaluate returns dict with required keys
 8.  AUC in [0, 1]
 9.  AUC ≈ 1.0 when members have much higher scores than non-members
10.  AUC ≈ 0.5 when scores are indistinguishable
11.  GradientNoiseDefense: output shapes match input
12.  GradientNoiseDefense: noised grad differs from original
13.  GradientNoiseDefense: noise_multiplier=0 → output ≈ clipped grad (no noise)
14.  Determinism under torch.manual_seed
15.  No NaN/Inf on normal inputs
"""

import math

import torch

from src.eval.membership_inference import (
    GradientNoiseDefense,
    LikelihoodRatioAttack,
    LossAttack,
    MembershipInferenceEvaluator,
    MinKProbAttack,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_log_probs(B: int = 4, T: int = 10, seed: int = 42) -> torch.Tensor:
    """Return a (B, T) tensor of plausible log-probabilities in (-5, 0)."""
    torch.manual_seed(seed)
    return -torch.rand(B, T) * 5.0  # values in (-5, 0)


# ---------------------------------------------------------------------------
# 1. LossAttack score shape
# ---------------------------------------------------------------------------


def test_loss_attack_shape():
    attack = LossAttack()
    log_probs = make_log_probs(B=4, T=10)
    scores = attack.score(log_probs)
    assert scores.shape == (4,), f"Expected (4,), got {scores.shape}"


# ---------------------------------------------------------------------------
# 2. LossAttack: lower loss → higher member score
# ---------------------------------------------------------------------------


def test_loss_attack_direction():
    """A sequence with higher (less-negative) log-probs has lower loss
    and should therefore receive a HIGHER membership score.

    score = mean(log_probs): members have higher log-probs → higher score.
    """
    attack = LossAttack()
    # low_loss_seq: log-probs close to 0  → low loss → score ≈ -0.1
    low_loss = torch.tensor([[-0.1, -0.1, -0.1, -0.1, -0.1]])
    # high_loss_seq: log-probs far from 0 → high loss → score ≈ -4.9
    high_loss = torch.tensor([[-4.9, -4.9, -4.9, -4.9, -4.9]])

    score_low = attack.score(low_loss)
    score_high = attack.score(high_loss)

    assert score_low.item() > score_high.item(), (
        "LossAttack: lower-loss sequence should score higher (more likely member)"
    )


# ---------------------------------------------------------------------------
# 3. MinKProbAttack score shape
# ---------------------------------------------------------------------------


def test_mink_attack_shape():
    attack = MinKProbAttack(k_percent=0.2)
    log_probs = make_log_probs(B=5, T=20)
    scores = attack.score(log_probs)
    assert scores.shape == (5,), f"Expected (5,), got {scores.shape}"


# ---------------------------------------------------------------------------
# 4. MinKProbAttack: k_percent=0.2 uses exactly bottom-20% tokens
# ---------------------------------------------------------------------------


def test_mink_attack_k_tokens():
    """Manually verify that MinKProbAttack with k=0.2 picks the bottom 20%
    of tokens and matches a hand-computed reference score."""
    T = 10
    max(1, math.ceil(T * 0.2))  # = 2

    # Construct log_probs so the 2 worst are clearly identifiable.
    log_probs_row = torch.tensor([-0.1, -0.2, -0.3, -0.4, -4.9, -5.0, -0.5, -0.6, -0.7, -0.8])
    log_probs = log_probs_row.unsqueeze(0)  # (1, 10)

    attack = MinKProbAttack(k_percent=0.2)
    score = attack.score(log_probs)

    # Expected: mean of the k=2 smallest log-probs: (-5.0 + -4.9) / 2
    expected = torch.tensor([-5.0, -4.9]).mean()
    assert torch.isclose(score[0], expected, atol=1e-5), (
        f"Expected {expected.item():.5f}, got {score[0].item():.5f}"
    )


# ---------------------------------------------------------------------------
# 5. MinKProbAttack: peaked text scores higher than flat text
# ---------------------------------------------------------------------------


def test_mink_peaked_vs_flat():
    """Peaked (high-confidence) text: most tokens have high log-prob,
    so the min-K% tokens are less negative → higher score.
    Flat text: all tokens have medium-low log-prob → lower score."""
    attack = MinKProbAttack(k_percent=0.2)

    T = 10
    # Peaked: 8 tokens near 0, 2 tokens at -1.0  → min-K% ≈ -1.0
    peaked = torch.cat(
        [
            torch.full((1, 8), -0.05),
            torch.full((1, 2), -1.0),
        ],
        dim=1,
    )

    # Flat: all tokens at -3.0  → min-K% ≈ -3.0
    flat = torch.full((1, T), -3.0)

    score_peaked = attack.score(peaked)
    score_flat = attack.score(flat)

    assert score_peaked.item() > score_flat.item(), (
        "MinKProbAttack: peaked text should score higher than flat text"
    )


# ---------------------------------------------------------------------------
# 6. LikelihoodRatioAttack score shape
# ---------------------------------------------------------------------------


def test_lr_attack_shape():
    attack = LikelihoodRatioAttack()
    log_probs = make_log_probs(B=6, T=15)
    scores = attack.score(log_probs)
    assert scores.shape == (6,), f"Expected (6,), got {scores.shape}"


# ---------------------------------------------------------------------------
# 7. MembershipInferenceEvaluator returns required keys
# ---------------------------------------------------------------------------


def test_evaluator_keys():
    evaluator = MembershipInferenceEvaluator()
    member_scores = torch.tensor([1.0, 2.0, 3.0])
    nonmember_scores = torch.tensor([0.1, 0.2, 0.3])
    result = evaluator.evaluate(member_scores, nonmember_scores)

    required_keys = {"auc", "tpr_at_fpr_0.1", "accuracy"}
    assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"


# ---------------------------------------------------------------------------
# 8. AUC in [0, 1]
# ---------------------------------------------------------------------------


def test_evaluator_auc_range():
    evaluator = MembershipInferenceEvaluator()
    torch.manual_seed(0)
    member_scores = torch.randn(50) + 1.0
    nonmember_scores = torch.randn(50)
    result = evaluator.evaluate(member_scores, nonmember_scores)
    assert 0.0 <= result["auc"] <= 1.0, f"AUC out of range: {result['auc']}"


# ---------------------------------------------------------------------------
# 9. AUC ≈ 1.0 when members have much higher scores
# ---------------------------------------------------------------------------


def test_evaluator_auc_near_one():
    evaluator = MembershipInferenceEvaluator()
    member_scores = torch.linspace(10.0, 20.0, 100)
    nonmember_scores = torch.linspace(-10.0, -1.0, 100)
    result = evaluator.evaluate(member_scores, nonmember_scores)
    assert result["auc"] >= 0.99, f"Expected AUC ≈ 1.0, got {result['auc']}"


# ---------------------------------------------------------------------------
# 10. AUC ≈ 0.5 when scores are indistinguishable
# ---------------------------------------------------------------------------


def test_evaluator_auc_near_half():
    """When members and non-members have the same score distribution,
    AUC should be near 0.5 (random chance)."""
    torch.manual_seed(123)
    evaluator = MembershipInferenceEvaluator()
    # Exact same distribution.
    scores = torch.randn(200)
    member_scores = scores[:100]
    nonmember_scores = scores[100:]
    result = evaluator.evaluate(member_scores, nonmember_scores)
    assert 0.3 <= result["auc"] <= 0.7, f"Expected AUC ≈ 0.5, got {result['auc']}"


# ---------------------------------------------------------------------------
# 11. GradientNoiseDefense: output shapes match input
# ---------------------------------------------------------------------------


def test_defense_shapes():
    defense = GradientNoiseDefense(noise_multiplier=1.0)
    torch.manual_seed(7)
    grads = [torch.randn(10, 20), torch.randn(5), torch.randn(3, 4, 4)]
    noised = defense.clip_and_noise(grads, max_norm=1.0)
    assert len(noised) == len(grads)
    for orig, out in zip(grads, noised):
        assert orig.shape == out.shape, f"Shape mismatch: {orig.shape} vs {out.shape}"


# ---------------------------------------------------------------------------
# 12. GradientNoiseDefense: noised grad differs from original
# ---------------------------------------------------------------------------


def test_defense_adds_noise():
    defense = GradientNoiseDefense(noise_multiplier=1.0)
    torch.manual_seed(99)
    grads = [torch.ones(100)]
    noised = defense.clip_and_noise(grads, max_norm=1.0)
    assert not torch.allclose(grads[0], noised[0]), (
        "Expected noised gradient to differ from original"
    )


# ---------------------------------------------------------------------------
# 13. noise_multiplier=0 → output ≈ clipped grad (no noise)
# ---------------------------------------------------------------------------


def test_defense_no_noise_when_zero_multiplier():
    """With noise_multiplier=0 the output should equal the clipped gradient."""
    defense = GradientNoiseDefense(noise_multiplier=0.0)
    torch.manual_seed(11)
    grad = torch.randn(50) * 10.0  # large norm to trigger clipping
    max_norm = 1.0
    noised = defense.clip_and_noise([grad], max_norm=max_norm)

    # Manually compute clipped gradient.
    norm = grad.norm(2)
    expected = grad * min(max_norm / (norm.item() + 1e-12), 1.0)

    assert torch.allclose(noised[0], expected, atol=1e-6), (
        "With noise_multiplier=0 output should equal clipped gradient"
    )


# ---------------------------------------------------------------------------
# 14. Determinism under torch.manual_seed
# ---------------------------------------------------------------------------


def test_defense_determinism():
    defense = GradientNoiseDefense(noise_multiplier=0.5)
    grad = torch.ones(20)

    torch.manual_seed(42)
    out1 = defense.clip_and_noise([grad.clone()], max_norm=1.0)

    torch.manual_seed(42)
    out2 = defense.clip_and_noise([grad.clone()], max_norm=1.0)

    assert torch.allclose(out1[0], out2[0]), (
        "Defense output is not deterministic under the same manual seed"
    )


# ---------------------------------------------------------------------------
# 15. No NaN/Inf on normal inputs
# ---------------------------------------------------------------------------


def test_no_nan_inf():
    torch.manual_seed(55)
    log_probs = make_log_probs(B=8, T=16)

    for AttackCls, kwargs in [
        (LossAttack, {}),
        (MinKProbAttack, {"k_percent": 0.3}),
        (LikelihoodRatioAttack, {}),
    ]:
        attack = AttackCls(**kwargs)
        scores = attack.score(log_probs)
        assert not torch.isnan(scores).any(), f"{AttackCls.__name__} produced NaN"
        assert not torch.isinf(scores).any(), f"{AttackCls.__name__} produced Inf"

    evaluator = MembershipInferenceEvaluator()
    result = evaluator.evaluate(scores[:4], scores[4:])
    for k, v in result.items():
        assert math.isfinite(v), f"Evaluator metric '{k}' is not finite: {v}"

    defense = GradientNoiseDefense(noise_multiplier=1.0)
    grad = torch.randn(32)
    out = defense.clip_and_noise([grad], max_norm=1.0)
    assert not torch.isnan(out[0]).any(), "Defense produced NaN"
    assert not torch.isinf(out[0]).any(), "Defense produced Inf"
