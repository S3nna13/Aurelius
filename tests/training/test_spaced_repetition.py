"""Tests for the spaced repetition learning scheduler."""

from __future__ import annotations

import pytest
import torch
from torch.optim import SGD

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.spaced_repetition import (
    CardState,
    SM2Config,
    SM2Scheduler,
    SpacedRepetitionDataset,
    SpacedRepetitionTrainer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    return SM2Config()


@pytest.fixture
def scheduler(cfg):
    return SM2Scheduler(cfg)


@pytest.fixture
def small_model():
    torch.manual_seed(42)
    model_cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    return AureliusTransformer(model_cfg)


def _make_examples(n: int = 6, seq_len: int = 8) -> list[tuple[str, list[int]]]:
    return [(f"card_{i}", list(range(1, seq_len + 1))) for i in range(n)]


def _make_dataset(n: int = 6, cfg: SM2Config | None = None) -> SpacedRepetitionDataset:
    examples = _make_examples(n)
    return SpacedRepetitionDataset(examples, cfg or SM2Config())


# ---------------------------------------------------------------------------
# 1. SM2Config defaults
# ---------------------------------------------------------------------------


def test_sm2config_defaults():
    cfg = SM2Config()
    assert cfg.initial_interval == 1
    assert cfg.initial_easiness == pytest.approx(2.5)
    assert cfg.min_easiness == pytest.approx(1.3)
    assert cfg.max_interval == 365
    assert cfg.performance_threshold == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 2. CardState defaults
# ---------------------------------------------------------------------------


def test_cardstate_defaults():
    card = CardState(card_id="x")
    assert card.card_id == "x"
    assert card.interval == 1
    assert card.easiness == pytest.approx(2.5)
    assert card.repetitions == 0
    assert card.last_loss == float("inf")
    assert card.next_review == 0


# ---------------------------------------------------------------------------
# 3. SM2Scheduler.register creates card
# ---------------------------------------------------------------------------


def test_scheduler_register_creates_card(scheduler):
    scheduler.register("card_a")
    assert "card_a" in scheduler._cards
    card = scheduler._cards["card_a"]
    assert isinstance(card, CardState)
    assert card.card_id == "card_a"


def test_scheduler_register_idempotent(scheduler):
    """Registering the same card twice should not reset its state."""
    scheduler.register("card_a")
    scheduler._cards["card_a"].repetitions = 5
    scheduler.register("card_a")  # second call
    assert scheduler._cards["card_a"].repetitions == 5


# ---------------------------------------------------------------------------
# 4. SM2Scheduler.update increases repetitions on good performance
# ---------------------------------------------------------------------------


def test_update_increases_repetitions_on_good_performance(scheduler):
    scheduler.register("card_a")
    # Loss well below threshold (0.6 * 0.5 = 0.3) → quality = 5
    scheduler.update("card_a", loss=0.1)
    assert scheduler._cards["card_a"].repetitions == 1


# ---------------------------------------------------------------------------
# 5. SM2Scheduler.update resets on bad performance
# ---------------------------------------------------------------------------


def test_update_resets_on_bad_performance(scheduler):
    scheduler.register("card_a")
    # First make it have some repetitions
    scheduler.update("card_a", loss=0.1)
    scheduler.update("card_a", loss=0.1)
    assert scheduler._cards["card_a"].repetitions == 2
    # Now a bad performance (loss > threshold)
    scheduler.update("card_a", loss=1.0)
    assert scheduler._cards["card_a"].repetitions == 0
    assert scheduler._cards["card_a"].interval == 1


# ---------------------------------------------------------------------------
# 6. SM2Scheduler.update interval grows for repeated good performance
# ---------------------------------------------------------------------------


def test_update_interval_grows_with_good_performance(scheduler):
    scheduler.register("card_a")
    # rep=0 → interval=1
    scheduler.update("card_a", loss=0.1)
    assert scheduler._cards["card_a"].interval == 1
    # rep=1 → interval=6
    scheduler.update("card_a", loss=0.1)
    assert scheduler._cards["card_a"].interval == 6
    # rep=2 → interval = int(6 * easiness)
    easiness_before = scheduler._cards["card_a"].easiness
    scheduler.update("card_a", loss=0.1)
    expected = int(6 * easiness_before)
    assert scheduler._cards["card_a"].interval == expected


# ---------------------------------------------------------------------------
# 7. SM2Scheduler.due returns cards with next_review <= step
# ---------------------------------------------------------------------------


def test_due_returns_cards_at_current_step(scheduler):
    scheduler.register("card_a")
    scheduler.register("card_b")
    # Both registered at step 0 with next_review=0 → both due
    due = scheduler.due()
    assert "card_a" in due
    assert "card_b" in due


def test_due_excludes_future_cards(scheduler):
    scheduler.register("card_a")
    # Update with good loss → next_review pushed into future
    scheduler.update("card_a", loss=0.1)
    # Still at step 0 → next_review = 0 + 1 = 1, so not due at step 0
    due = scheduler.due()
    assert "card_a" not in due


def test_due_respects_n_limit(scheduler):
    for i in range(5):
        scheduler.register(f"card_{i}")
    due = scheduler.due(n=3)
    assert len(due) <= 3


# ---------------------------------------------------------------------------
# 8. SM2Scheduler.step increments counter
# ---------------------------------------------------------------------------


def test_scheduler_step_increments(scheduler):
    assert scheduler._step == 0
    scheduler.step()
    assert scheduler._step == 1
    scheduler.step()
    assert scheduler._step == 2


# ---------------------------------------------------------------------------
# 9. SM2Scheduler.stats returns required keys
# ---------------------------------------------------------------------------


def test_scheduler_stats_keys(scheduler):
    scheduler.register("card_a")
    stats = scheduler.stats()
    assert "n_cards" in stats
    assert "n_due" in stats
    assert "mean_interval" in stats
    assert "mean_easiness" in stats


def test_scheduler_stats_values(scheduler):
    scheduler.register("card_a")
    scheduler.register("card_b")
    stats = scheduler.stats()
    assert stats["n_cards"] == 2
    # Both are due at step 0
    assert stats["n_due"] == 2
    assert isinstance(stats["mean_interval"], float)
    assert isinstance(stats["mean_easiness"], float)


# ---------------------------------------------------------------------------
# 10. SpacedRepetitionDataset.get_batch returns correct count
# ---------------------------------------------------------------------------


def test_dataset_get_batch_correct_count():
    ds = _make_dataset(n=6)
    batch = ds.get_batch(batch_size=4)
    assert len(batch) == 4


def test_dataset_get_batch_returns_tuples():
    ds = _make_dataset(n=4)
    batch = ds.get_batch(batch_size=2)
    for item in batch:
        assert len(item) == 2
        card_id, token_ids = item
        assert isinstance(card_id, str)
        assert isinstance(token_ids, list)


# ---------------------------------------------------------------------------
# 11. SpacedRepetitionDataset.record_losses updates scheduler
# ---------------------------------------------------------------------------


def test_dataset_record_losses_updates_scheduler():
    ds = _make_dataset(n=4)
    batch = ds.get_batch(batch_size=2)
    card_ids = [item[0] for item in batch]
    initial_reps = {cid: ds.scheduler._cards[cid].repetitions for cid in card_ids}

    # Record good losses (below threshold)
    ds.record_losses(card_ids, [0.1, 0.1])

    for cid in card_ids:
        assert ds.scheduler._cards[cid].repetitions > initial_reps[cid]


def test_dataset_record_losses_updates_last_loss():
    ds = _make_dataset(n=2)
    card_ids = ["card_0", "card_1"]
    ds.record_losses(card_ids, [0.42, 0.99])
    assert ds.scheduler._cards["card_0"].last_loss == pytest.approx(0.42)
    assert ds.scheduler._cards["card_1"].last_loss == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# 12. SpacedRepetitionDataset.get_batch prefers due cards
# ---------------------------------------------------------------------------


def test_dataset_get_batch_prefers_due_cards():
    """When some cards are due and others are not, due cards should appear first."""
    examples = _make_examples(n=4)
    ds = SpacedRepetitionDataset(examples, SM2Config())

    # Push card_0 and card_1 far into the future (not due)
    ds.scheduler.update("card_0", loss=0.01)  # good → reps=1, interval=1, review=1
    ds.scheduler.update("card_1", loss=0.01)

    # Advance scheduler step so those cards have future next_review
    # Force next_review to be large
    ds.scheduler._cards["card_0"].next_review = 9999
    ds.scheduler._cards["card_1"].next_review = 9999

    # card_2 and card_3 should still be due (next_review=0)
    due = ds.scheduler.due()
    assert "card_2" in due
    assert "card_3" in due

    batch = ds.get_batch(batch_size=2)
    batch_ids = [item[0] for item in batch]
    # Both selected should be due cards
    assert "card_2" in batch_ids or "card_3" in batch_ids


# ---------------------------------------------------------------------------
# 13. SpacedRepetitionTrainer.train_step returns required keys
# ---------------------------------------------------------------------------


def test_trainer_train_step_returns_required_keys(small_model):
    ds = _make_dataset(n=6)
    optimizer = SGD(small_model.parameters(), lr=1e-3)
    trainer = SpacedRepetitionTrainer(small_model, optimizer, ds, SM2Config())

    result = trainer.train_step()

    assert "loss" in result
    assert "n_due" in result
    assert "n_examples" in result
    assert "mean_interval" in result


# ---------------------------------------------------------------------------
# 14. SpacedRepetitionTrainer.train_step loss is positive float
# ---------------------------------------------------------------------------


def test_trainer_train_step_loss_is_positive_float(small_model):
    ds = _make_dataset(n=6)
    optimizer = SGD(small_model.parameters(), lr=1e-3)
    trainer = SpacedRepetitionTrainer(small_model, optimizer, ds, SM2Config())

    result = trainer.train_step()

    assert isinstance(result["loss"], float)
    assert result["loss"] > 0.0
    assert not (result["loss"] != result["loss"])  # not NaN


# ---------------------------------------------------------------------------
# 15. SpacedRepetitionTrainer.get_hard_examples returns list of strings
# ---------------------------------------------------------------------------


def test_trainer_get_hard_examples_returns_strings(small_model):
    ds = _make_dataset(n=6)
    optimizer = SGD(small_model.parameters(), lr=1e-3)
    trainer = SpacedRepetitionTrainer(small_model, optimizer, ds, SM2Config())

    # Run a step so scheduler has some data
    trainer.train_step()

    hard = trainer.get_hard_examples(n=3)
    assert isinstance(hard, list)
    assert len(hard) <= 3
    for item in hard:
        assert isinstance(item, str)


def test_trainer_get_hard_examples_sorted_by_difficulty(small_model):
    """Hard examples should have the lowest easiness scores."""
    ds = _make_dataset(n=6)
    optimizer = SGD(small_model.parameters(), lr=1e-3)
    trainer = SpacedRepetitionTrainer(small_model, optimizer, ds, SM2Config())

    # Manually manipulate easiness to create clear ordering
    for i, cid in enumerate(["card_0", "card_1", "card_2", "card_3", "card_4", "card_5"]):
        ds.scheduler._cards[cid].easiness = float(i + 1)  # 1.0, 2.0, 3.0, ...

    hard = trainer.get_hard_examples(n=2)
    assert hard[0] == "card_0"  # lowest easiness
    assert hard[1] == "card_1"
