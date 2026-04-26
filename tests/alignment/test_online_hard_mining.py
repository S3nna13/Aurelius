"""Tests for Online Hard Example Mining (OHEM) for alignment training."""

from __future__ import annotations

import torch

from src.alignment.online_hard_mining import (
    HardExample,
    HardExampleBuffer,
    MiningConfig,
    OnlineHardMiner,
    compute_dpo_difficulty,
    compute_reward_difficulty,
    dpo_loss_with_mining,
    select_hard_examples,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B = 8  # batch size
T = 16  # sequence length

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_logprobs(size: int = B, low: float = -5.0, high: float = -0.1) -> torch.Tensor:
    return torch.empty(size).uniform_(low, high)


def _make_ids(batch: int = B, seq: int = T) -> torch.Tensor:
    return torch.randint(0, 512, (batch, seq))


# ---------------------------------------------------------------------------
# 1. compute_dpo_difficulty — shape and range
# ---------------------------------------------------------------------------


def test_compute_dpo_difficulty_shape_and_range():
    """compute_dpo_difficulty returns (B,) tensor with values in [0, 1]."""
    pc = _random_logprobs()
    pr = _random_logprobs()
    rc = _random_logprobs()
    rr = _random_logprobs()
    diff = compute_dpo_difficulty(pc, pr, rc, rr)
    assert diff.shape == (B,), f"Expected shape ({B},), got {diff.shape}"
    assert diff.min() >= 0.0, "Difficulty must be >= 0"
    assert diff.max() <= 1.0, "Difficulty must be <= 1"


# ---------------------------------------------------------------------------
# 2. compute_dpo_difficulty — chosen >> rejected → low difficulty
# ---------------------------------------------------------------------------


def test_compute_dpo_difficulty_easy_examples():
    """When chosen log-probs >> rejected, difficulty should be low (< 0.5)."""
    # Policy strongly prefers chosen.
    policy_chosen = torch.full((B,), -0.5)
    policy_rejected = torch.full((B,), -5.0)
    # Reference is neutral.
    ref_chosen = torch.full((B,), -2.0)
    ref_rejected = torch.full((B,), -2.0)
    diff = compute_dpo_difficulty(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    assert (diff < 0.5).all(), f"Expected all difficulties < 0.5, got {diff}"


# ---------------------------------------------------------------------------
# 3. compute_dpo_difficulty — chosen ≈ rejected → high difficulty (~0.5)
# ---------------------------------------------------------------------------


def test_compute_dpo_difficulty_hard_examples():
    """When chosen ≈ rejected log-probs, difficulty should be near 0.5."""
    lp = torch.full((B,), -2.0)
    diff = compute_dpo_difficulty(lp, lp, lp, lp)
    assert torch.allclose(diff, torch.full((B,), 0.5), atol=1e-5), (
        f"Expected difficulties ~0.5, got {diff}"
    )


# ---------------------------------------------------------------------------
# 4. compute_reward_difficulty — shape and range
# ---------------------------------------------------------------------------


def test_compute_reward_difficulty_shape_and_range():
    """compute_reward_difficulty returns (B,) tensor with values in [0, 1]."""
    rc = torch.randn(B)
    rr = torch.randn(B)
    diff = compute_reward_difficulty(rc, rr)
    assert diff.shape == (B,), f"Expected shape ({B},), got {diff.shape}"
    assert diff.min() >= 0.0
    assert diff.max() <= 1.0


# ---------------------------------------------------------------------------
# 5. compute_reward_difficulty — large gap → low difficulty
# ---------------------------------------------------------------------------


def test_compute_reward_difficulty_large_gap():
    """A large reward gap (chosen >> rejected) should give difficulty < 0.5."""
    chosen = torch.full((B,), 10.0)
    rejected = torch.full((B,), -10.0)
    diff = compute_reward_difficulty(chosen, rejected, normalize=False)
    assert (diff < 0.5).all(), f"Expected all difficulties < 0.5, got {diff}"


# ---------------------------------------------------------------------------
# 6. select_hard_examples "hardest" — returns top-k by difficulty
# ---------------------------------------------------------------------------


def test_select_hard_examples_hardest_correct_indices():
    """'hardest' strategy should return the indices with highest difficulty."""
    difficulties = torch.tensor([0.1, 0.9, 0.4, 0.8, 0.2, 0.7, 0.3, 0.6])
    k = 4  # top_k_ratio=0.5, B=8
    indices = select_hard_examples(difficulties, strategy="hardest", top_k_ratio=0.5)
    selected_difficulties = difficulties[indices]
    # All selected should be in the top-k.
    threshold = torch.topk(difficulties, k=k).values.min()
    assert (selected_difficulties >= threshold).all(), (
        f"Not all top-{k} difficulties selected. Got: {selected_difficulties}"
    )


# ---------------------------------------------------------------------------
# 7. select_hard_examples "semi-hard" — mid-range difficulty
# ---------------------------------------------------------------------------


def test_select_hard_examples_semi_hard_range():
    """'semi-hard' strategy should return indices with difficulty in [0.3, 0.7]."""
    difficulties = torch.tensor([0.05, 0.35, 0.45, 0.55, 0.65, 0.92, 0.1, 0.5])
    indices = select_hard_examples(difficulties, strategy="semi-hard", top_k_ratio=0.5)
    selected = difficulties[indices]
    # All selected should be in [0.3, 0.7] (or fallback top-k if none exist).
    mid_range = (selected >= 0.3) & (selected <= 0.7)
    assert mid_range.all(), f"Semi-hard examples out of range [0.3, 0.7]: {selected}"


# ---------------------------------------------------------------------------
# 8. select_hard_examples — correct count
# ---------------------------------------------------------------------------


def test_select_hard_examples_correct_count():
    """select_hard_examples should return exactly int(B * top_k_ratio) indices."""
    difficulties = torch.rand(B)
    for ratio in [0.25, 0.5, 0.75]:
        expected_k = max(1, int(B * ratio))
        indices = select_hard_examples(difficulties, strategy="hardest", top_k_ratio=ratio)
        assert indices.shape[0] == expected_k, (
            f"Expected {expected_k} indices for ratio={ratio}, got {indices.shape[0]}"
        )


# ---------------------------------------------------------------------------
# 9. HardExampleBuffer.add — increases length
# ---------------------------------------------------------------------------


def test_hard_example_buffer_add_increases_length():
    """Adding examples to buffer should increase its length."""
    buf = HardExampleBuffer(max_size=100)
    assert len(buf) == 0
    examples = [
        HardExample(
            chosen_ids=torch.randint(0, 512, (T,)),
            rejected_ids=torch.randint(0, 512, (T,)),
            difficulty=float(i) / 10,
        )
        for i in range(5)
    ]
    buf.add(examples)
    assert len(buf) == 5, f"Expected 5 examples, got {len(buf)}"
    buf.add(examples[:3])
    assert len(buf) == 8, f"Expected 8 examples, got {len(buf)}"


# ---------------------------------------------------------------------------
# 10. HardExampleBuffer — never exceeds max_size
# ---------------------------------------------------------------------------


def test_hard_example_buffer_max_size():
    """Buffer should never exceed max_size (evicts easiest on overflow)."""
    max_size = 6
    buf = HardExampleBuffer(max_size=max_size)
    examples = [
        HardExample(
            chosen_ids=torch.randint(0, 512, (T,)),
            rejected_ids=torch.randint(0, 512, (T,)),
            difficulty=float(i) / 20,
        )
        for i in range(15)
    ]
    buf.add(examples)
    assert len(buf) <= max_size, f"Buffer exceeded max_size={max_size}: got {len(buf)}"


# ---------------------------------------------------------------------------
# 11. HardExampleBuffer.sample — returns n examples
# ---------------------------------------------------------------------------


def test_hard_example_buffer_sample_count():
    """Buffer.sample(n) should return exactly n examples."""
    buf = HardExampleBuffer(max_size=50)
    examples = [
        HardExample(
            chosen_ids=torch.randint(0, 512, (T,)),
            rejected_ids=torch.randint(0, 512, (T,)),
            difficulty=float(i) / 10,
        )
        for i in range(10)
    ]
    buf.add(examples)
    sampled = buf.sample(4)
    assert len(sampled) == 4, f"Expected 4 samples, got {len(sampled)}"


# ---------------------------------------------------------------------------
# 12. OnlineHardMiner.mine — returns subset with correct shape
# ---------------------------------------------------------------------------


def test_online_hard_miner_mine_shape():
    """mine() should return (k, T) tensors where k < B."""
    miner = OnlineHardMiner(config=MiningConfig(top_k_ratio=0.5))
    chosen = _make_ids()
    rejected = _make_ids()
    policy_c = _random_logprobs()
    policy_r = _random_logprobs()
    ref_c = _random_logprobs()
    ref_r = _random_logprobs()

    chosen_hard, rejected_hard = miner.mine(chosen, rejected, policy_c, policy_r, ref_c, ref_r)
    expected_k = max(1, int(B * 0.5))
    assert chosen_hard.shape == (expected_k, T), (
        f"Expected chosen_hard shape ({expected_k}, {T}), got {chosen_hard.shape}"
    )
    assert rejected_hard.shape == (expected_k, T), (
        f"Expected rejected_hard shape ({expected_k}, {T}), got {rejected_hard.shape}"
    )


# ---------------------------------------------------------------------------
# 13. OnlineHardMiner.get_sample_weights — (B,) summing to B
# ---------------------------------------------------------------------------


def test_online_hard_miner_get_sample_weights():
    """get_sample_weights should return (B,) tensor summing to B."""
    miner = OnlineHardMiner()
    difficulties = torch.rand(B)
    weights = miner.get_sample_weights(difficulties)
    assert weights.shape == (B,), f"Expected shape ({B},), got {weights.shape}"
    assert torch.isclose(weights.sum(), torch.tensor(float(B)), atol=1e-4), (
        f"Weights should sum to {B}, got {weights.sum().item():.4f}"
    )


# ---------------------------------------------------------------------------
# 14. OnlineHardMiner.curriculum_difficulty_threshold — increases with step
# ---------------------------------------------------------------------------


def test_curriculum_difficulty_threshold_increases():
    """curriculum_difficulty_threshold should grow linearly with step count."""
    cfg = MiningConfig(strategy="curriculum", curriculum_warmup_steps=10)
    miner = OnlineHardMiner(config=cfg)

    thresholds = []
    chosen = _make_ids()
    rejected = _make_ids()
    policy_c = _random_logprobs()
    policy_r = _random_logprobs()

    for _ in range(5):
        thresholds.append(miner.curriculum_difficulty_threshold())
        miner.mine(chosen, rejected, policy_c, policy_r)

    # Threshold should be non-decreasing.
    for i in range(1, len(thresholds)):
        assert thresholds[i] >= thresholds[i - 1], (
            f"Threshold decreased at step {i}: {thresholds[i - 1]:.4f} → {thresholds[i]:.4f}"
        )
    # Should be strictly increasing (not all equal) during warmup.
    assert thresholds[-1] > thresholds[0], "Threshold did not increase over training steps."


# ---------------------------------------------------------------------------
# 15. dpo_loss_with_mining — returns scalar loss
# ---------------------------------------------------------------------------


def test_dpo_loss_with_mining_scalar():
    """dpo_loss_with_mining should return a scalar (0-dim) tensor loss."""
    pc = _random_logprobs()
    pr = _random_logprobs()
    rc = _random_logprobs()
    rr = _random_logprobs()
    loss, _ = dpo_loss_with_mining(pc, pr, rc, rr, beta=0.1)
    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert torch.isfinite(loss), "Loss must be finite"


# ---------------------------------------------------------------------------
# 16. dpo_loss_with_mining — metrics has 'mean_difficulty' key
# ---------------------------------------------------------------------------


def test_dpo_loss_with_mining_metrics_keys():
    """dpo_loss_with_mining metrics dict must contain 'mean_difficulty'."""
    pc = _random_logprobs()
    pr = _random_logprobs()
    rc = _random_logprobs()
    rr = _random_logprobs()
    _, metrics = dpo_loss_with_mining(pc, pr, rc, rr)
    assert "mean_difficulty" in metrics, (
        f"'mean_difficulty' not found in metrics keys: {list(metrics.keys())}"
    )
    assert "n_hard_examples" in metrics, (
        f"'n_hard_examples' not found in metrics keys: {list(metrics.keys())}"
    )
    assert 0.0 <= metrics["mean_difficulty"] <= 1.0, (
        f"mean_difficulty out of [0,1]: {metrics['mean_difficulty']}"
    )
