"""
Tests for reward_model_training.py.
Tiny configs: d_model=16, vocab=16, seq_len=8, batch=2, K=3.
All tests run forward and/or backward passes -- no instantiation-only tests.
"""

import torch
import torch.nn as nn

from src.training.reward_model_training import (
    BradleyTerryLoss,
    ListwiseRankingLoss,
    PreferenceDataset,
    RewardModel,
    RewardModelTrainer,
    RewardNormalizer,
)

# ---------------------------------------------------------------------------
# Tiny config constants
# ---------------------------------------------------------------------------
D_MODEL = 16
VOCAB = 16
SEQ_LEN = 8
BATCH = 2
K = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class EmbeddingBackbone(nn.Module):
    """Simple backbone: nn.Embedding returns (B, T, D) hidden states."""

    def __init__(self, vocab: int = VOCAB, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)  # (B, T, D)


def make_reward_model() -> RewardModel:
    backbone = EmbeddingBackbone()
    return RewardModel(backbone, d_model=D_MODEL)


def make_ids(batch: int = BATCH, seq_len: int = SEQ_LEN) -> torch.Tensor:
    return torch.randint(0, VOCAB, (batch, seq_len))


def make_trainer(reward_model=None):
    if reward_model is None:
        reward_model = make_reward_model()
    optimizer = torch.optim.SGD(reward_model.parameters(), lr=1e-2)
    bt_loss = BradleyTerryLoss()
    normalizer = RewardNormalizer()
    return RewardModelTrainer(reward_model, optimizer, bt_loss, normalizer)


# ---------------------------------------------------------------------------
# Test 1: RewardModel output shape
# ---------------------------------------------------------------------------


def test_reward_model_output_shape():
    model = make_reward_model()
    ids = make_ids()
    out = model(ids)
    assert out.shape == (BATCH,), f"Expected ({BATCH},), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2: RewardModel gradient flows to backbone and reward head
# ---------------------------------------------------------------------------


def test_reward_model_grad_flows():
    model = make_reward_model()
    ids = make_ids()
    out = model(ids)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum().item() > 0, f"Zero gradient for {name}"


# ---------------------------------------------------------------------------
# Test 3: BradleyTerryLoss: loss scalar finite, accuracy in [0,1]
# ---------------------------------------------------------------------------


def test_bt_loss_scalar_finite_accuracy_range():
    bt = BradleyTerryLoss()
    cr = torch.randn(BATCH)
    rr = torch.randn(BATCH)
    loss, acc = bt(cr, rr)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss).item()
    assert 0.0 <= acc.item() <= 1.0


# ---------------------------------------------------------------------------
# Test 4: BradleyTerryLoss: when chosen >> rejected, accuracy=1.0, loss~0
# ---------------------------------------------------------------------------


def test_bt_loss_chosen_dominates():
    bt = BradleyTerryLoss()
    cr = torch.full((BATCH,), 100.0)
    rr = torch.full((BATCH,), -100.0)
    loss, acc = bt(cr, rr)
    assert acc.item() == 1.0
    assert loss.item() < 1e-3


# ---------------------------------------------------------------------------
# Test 5: BradleyTerryLoss: margin > 0 gives higher loss than margin=0
#         when difference is small
# ---------------------------------------------------------------------------


def test_bt_loss_margin_increases_loss():
    cr = torch.tensor([0.1, 0.2])
    rr = torch.tensor([0.0, 0.0])
    bt0 = BradleyTerryLoss(margin=0.0)
    bt1 = BradleyTerryLoss(margin=1.0)
    loss0, _ = bt0(cr, rr)
    loss1, _ = bt1(cr, rr)
    assert loss1.item() > loss0.item()


# ---------------------------------------------------------------------------
# Test 6: ListwiseRankingLoss: scalar, finite, gradient flows
# ---------------------------------------------------------------------------


def test_listwise_loss_scalar_finite_grad():
    lrl = ListwiseRankingLoss()
    rewards = torch.randn(BATCH, K, requires_grad=True)
    prefs = torch.randn(BATCH, K)
    loss = lrl(rewards, prefs)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss).item()
    loss.backward()
    assert rewards.grad is not None
    assert rewards.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Test 7: ListwiseRankingLoss: uniform preference_labels -> higher loss
#         than correct (peaked) ordering
# ---------------------------------------------------------------------------


def test_listwise_loss_uniform_higher_than_correct():
    lrl = ListwiseRankingLoss()
    # Perfect prediction: rewards match the preference order
    rewards_good = torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    prefs_correct = torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    prefs_uniform = torch.zeros(BATCH, K)  # uniform = maximum entropy target

    loss_good = lrl(rewards_good, prefs_correct)
    loss_uniform = lrl(rewards_good, prefs_uniform)
    # Uniform preference spreads probability mass -> higher cross-entropy
    # against a peaked predicted distribution
    assert loss_uniform.item() >= loss_good.item()


# ---------------------------------------------------------------------------
# Test 8: RewardNormalizer.update + normalize: mean ~0 after many updates
# ---------------------------------------------------------------------------


def test_normalizer_mean_near_zero_after_updates():
    norm = RewardNormalizer(momentum=0.9)
    # Feed many batches so EMA converges
    rng = torch.Generator()
    rng.manual_seed(42)
    for _ in range(200):
        batch = torch.randn(8, generator=rng) * 3.0 + 5.0
        norm.update(batch)

    # Normalize a fresh batch from the same distribution
    test_batch = torch.randn(64, generator=rng) * 3.0 + 5.0
    normalized = norm.normalize(test_batch)
    # Mean should be pulled toward zero (within reasonable tolerance)
    assert abs(normalized.mean().item()) < 2.0


# ---------------------------------------------------------------------------
# Test 9: RewardNormalizer.denormalize: round-trips correctly
# ---------------------------------------------------------------------------


def test_normalizer_denormalize_roundtrip():
    norm = RewardNormalizer(momentum=0.5)
    data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    norm.update(data)
    normed = norm.normalize(data)
    recovered = norm.denormalize(normed)
    assert torch.allclose(recovered, data, atol=1e-5), f"Round-trip failed: {recovered} vs {data}"


# ---------------------------------------------------------------------------
# Test 10: RewardNormalizer.stats: all 3 keys present
# ---------------------------------------------------------------------------


def test_normalizer_stats_keys():
    norm = RewardNormalizer()
    norm.update(torch.randn(4))
    s = norm.stats()
    for key in ("mean", "std", "n_samples"):
        assert key in s, f"Missing key '{key}' in stats()"


# ---------------------------------------------------------------------------
# Test 11: PreferenceDataset: len correct, getitem returns 3-tuple
# ---------------------------------------------------------------------------


def test_preference_dataset_len_and_getitem():
    n = 5
    chosen = [torch.randint(0, VOCAB, (SEQ_LEN,)) for _ in range(n)]
    rejected = [torch.randint(0, VOCAB, (SEQ_LEN,)) for _ in range(n)]
    ds = PreferenceDataset(chosen, rejected)
    assert len(ds) == n
    item = ds[0]
    assert len(item) == 3  # (chosen_ids, rejected_ids, margin_label)


# ---------------------------------------------------------------------------
# Test 12: PreferenceDataset.split: sizes sum to total, no size-overlap
# ---------------------------------------------------------------------------


def test_preference_dataset_split_sizes():
    n = 10
    chosen = [torch.randint(0, VOCAB, (SEQ_LEN,)) for _ in range(n)]
    rejected = [torch.randint(0, VOCAB, (SEQ_LEN,)) for _ in range(n)]
    ds = PreferenceDataset(chosen, rejected)
    train_ds, val_ds = ds.split(ratio=0.8)
    assert len(train_ds) + len(val_ds) == n
    # Correct sizes
    assert len(train_ds) == 8
    assert len(val_ds) == 2


# ---------------------------------------------------------------------------
# Test 13: RewardModelTrainer.train_step: all keys present, loss finite
# ---------------------------------------------------------------------------


def test_trainer_train_step_keys_and_finite():
    trainer = make_trainer()
    chosen_ids = make_ids()
    rejected_ids = make_ids()
    result = trainer.train_step(chosen_ids, rejected_ids)
    for key in ("loss", "accuracy", "chosen_mean", "rejected_mean", "reward_margin"):
        assert key in result, f"Missing key '{key}'"
    assert torch.isfinite(torch.tensor(result["loss"])).item()


# ---------------------------------------------------------------------------
# Test 14: RewardModelTrainer.evaluate: accuracy and consistency_rate in [0,1]
# ---------------------------------------------------------------------------


def test_trainer_evaluate_ranges():
    trainer = make_trainer()
    chosen_ids = make_ids()
    rejected_ids = make_ids()
    result = trainer.evaluate(chosen_ids, rejected_ids)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["consistency_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Test 15: reward_margin > 0 after 5 training steps on clear preference signal
# ---------------------------------------------------------------------------


def test_trainer_reward_margin_positive_after_training():
    torch.manual_seed(0)
    model = make_reward_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    bt_loss = BradleyTerryLoss()
    normalizer = RewardNormalizer()
    trainer = RewardModelTrainer(model, optimizer, bt_loss, normalizer)

    # Fixed token ids: distinct chosen vs rejected tokens -> learnable separation
    chosen_ids = torch.randint(VOCAB // 2, VOCAB, (BATCH, SEQ_LEN))
    rejected_ids = torch.randint(0, VOCAB // 2, (BATCH, SEQ_LEN))

    for _ in range(5):
        result = trainer.train_step(chosen_ids, rejected_ids)

    assert result["reward_margin"] > 0, f"Expected reward_margin > 0, got {result['reward_margin']}"
