"""Tests for data echoing helpers."""

import pytest
import torch

from src.training.data_echoing import (
    DataEchoBuffer,
    EchoConfig,
    EchoExample,
    EchoTrainer,
    echo_probabilities,
    echo_score,
    select_echoes,
    update_echo_metadata,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

_SMALL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)


def _make_model():
    """Return a fresh small AureliusTransformer in eval mode."""
    model = AureliusTransformer(_SMALL_CFG)
    model.train()
    return model


def _make_batch(batch_size: int = 4, seq_len: int = 16):
    """Return (input_ids, labels) with random token ids."""
    input_ids = torch.randint(0, _SMALL_CFG.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, _SMALL_CFG.vocab_size, (batch_size, seq_len))
    return input_ids, labels


def _make_data_iter(n_batches: int = 8, batch_size: int = 4, seq_len: int = 16):
    """Return a list of (input_ids, labels) batches acting as a data iterator."""
    return [_make_batch(batch_size, seq_len) for _ in range(n_batches)]


def make_examples() -> list[EchoExample]:
    return [
        EchoExample("easy", loss=0.2, difficulty=0.1, last_seen_step=9),
        EchoExample("hard", loss=1.2, difficulty=0.9, last_seen_step=2),
        EchoExample("medium", loss=0.6, difficulty=0.5, last_seen_step=5),
    ]


def test_echo_score_increases_with_loss_and_difficulty():
    low = echo_score(torch.tensor([0.2]), torch.tensor([0.1]), torch.tensor([1.0]))
    high = echo_score(torch.tensor([1.0]), torch.tensor([0.8]), torch.tensor([1.0]))
    assert high.item() > low.item()


def test_echo_probabilities_sum_to_one():
    probs = echo_probabilities(make_examples(), current_step=10)
    assert probs.sum().item() == pytest.approx(1.0)


def test_echo_probabilities_prioritize_harder_examples():
    examples = make_examples()
    probs = echo_probabilities(examples, current_step=10)
    assert probs[1].item() == pytest.approx(probs.max().item())


def test_select_echoes_returns_top_examples():
    selected = select_echoes(make_examples(), current_step=10, n_select=2)
    assert [example.example_id for example in selected] == ["hard", "medium"]


def test_update_echo_metadata_increments_seen_examples():
    examples = make_examples()
    update_echo_metadata(examples, ["hard"], current_step=11)
    hard = next(example for example in examples if example.example_id == "hard")
    assert hard.echo_count == 1
    assert hard.last_seen_step == 11


def test_select_echoes_handles_zero_selection():
    assert select_echoes(make_examples(), current_step=10, n_select=0) == []


def test_echo_probabilities_reject_bad_temperature():
    with pytest.raises(ValueError):
        echo_probabilities(make_examples(), current_step=10, temperature=0.0)


# ---------------------------------------------------------------------------
# Batch-repeat data echoing tests (Choi et al. 2019)
# ---------------------------------------------------------------------------


def test_echo_buffer_repeats_batch():
    """Each batch should be returned echo_factor times before the next is fetched."""
    echo_factor = 3
    data = _make_data_iter(n_batches=4)
    cfg = EchoConfig(echo_factor=echo_factor, shuffle_echoes=False)
    buf = DataEchoBuffer(data, cfg)

    # Collect first two full rounds (6 echoes → 2 unique batches)
    results = [next(buf) for _ in range(echo_factor * 2)]

    # First echo_factor results should all share the same underlying data
    first_ids = results[0][0]
    for ids, _ in results[:echo_factor]:
        assert torch.equal(ids, first_ids), "Expected same batch for all echoes in first group"

    # Next echo_factor results should all share a *different* batch
    second_ids = results[echo_factor][0]
    assert not torch.equal(first_ids, second_ids), "Expected a new batch after echo_factor repeats"
    for ids, _ in results[echo_factor : echo_factor * 2]:
        assert torch.equal(ids, second_ids), "Expected same batch for all echoes in second group"


def test_echo_buffer_shuffles_when_configured():
    """With shuffle_echoes=True consecutive echoes should (usually) differ in order."""
    torch.manual_seed(0)
    # Use a batch large enough that a random shuffle almost certainly changes something.
    input_ids, labels = _make_batch(batch_size=16, seq_len=8)
    data = [(input_ids, labels)] * 10  # infinite-ish supply of same batch
    cfg = EchoConfig(echo_factor=10, shuffle_echoes=True)
    buf = DataEchoBuffer(data, cfg)

    first_ids, _ = next(buf)
    found_different = False
    for _ in range(9):
        ids, _ = next(buf)
        if not torch.equal(ids, first_ids):
            found_different = True
            break

    assert found_different, "Expected at least one shuffled echo to differ from the first"


def test_echo_buffer_no_shuffle():
    """With shuffle_echoes=False all echoes of the same batch must be identical."""
    input_ids, labels = _make_batch(batch_size=4, seq_len=8)
    data = [(input_ids, labels)] * 4
    cfg = EchoConfig(echo_factor=4, shuffle_echoes=False)
    buf = DataEchoBuffer(data, cfg)

    echoes = [next(buf) for _ in range(4)]
    for ids, lbls in echoes:
        assert torch.equal(ids, input_ids)
        assert torch.equal(lbls, labels)


def test_echo_trainer_train_on_batch_returns_k_metrics():
    """train_on_batch must return exactly echo_factor metric dicts."""
    echo_factor = 4
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    cfg = EchoConfig(echo_factor=echo_factor, shuffle_echoes=False)
    trainer = EchoTrainer(model, optimizer, cfg)

    input_ids, labels = _make_batch()
    metrics = trainer.train_on_batch(input_ids, labels)

    assert isinstance(metrics, list)
    assert len(metrics) == echo_factor
    for i, m in enumerate(metrics):
        assert "loss" in m
        assert "echo" in m
        assert m["echo"] == i


def test_echo_trainer_loss_values_positive():
    """All per-echo losses should be strictly positive finite values."""
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    cfg = EchoConfig(echo_factor=4, shuffle_echoes=False)
    trainer = EchoTrainer(model, optimizer, cfg)

    input_ids, labels = _make_batch()
    metrics = trainer.train_on_batch(input_ids, labels)

    for m in metrics:
        assert m["loss"] > 0.0, f"Expected positive loss, got {m['loss']}"
        assert torch.isfinite(torch.tensor(m["loss"])), "Loss must be finite"


def test_train_steps_echo_ratio():
    """With n_steps=8 and echo_factor=4, echo_ratio should be 4.0 and unique_batches 2."""
    n_steps = 8
    echo_factor = 4
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    cfg = EchoConfig(echo_factor=echo_factor, shuffle_echoes=False)
    trainer = EchoTrainer(model, optimizer, cfg)

    # Provide enough data so iterator doesn't exhaust prematurely.
    data = _make_data_iter(n_batches=16)
    result = trainer.train_steps(data, n_steps=n_steps)

    assert result["total_steps"] == n_steps
    assert result["unique_batches"] == n_steps // echo_factor
    assert result["echo_ratio"] == pytest.approx(float(echo_factor))


def test_echo_buffer_exhausts_data_iter():
    """StopIteration should propagate when the underlying iterator is exhausted."""
    data = _make_data_iter(n_batches=2)
    cfg = EchoConfig(echo_factor=1, shuffle_echoes=False)
    buf = DataEchoBuffer(data, cfg)

    # Consume exactly 2 batches
    next(buf)
    next(buf)

    with pytest.raises(StopIteration):
        next(buf)


def test_shuffle_batch_preserves_batch_size():
    """_shuffle_batch must return tensors with the same shape as the input."""
    input_ids, labels = _make_batch(batch_size=8, seq_len=12)
    cfg = EchoConfig(echo_factor=1, shuffle_echoes=True)
    buf = DataEchoBuffer(iter([]), cfg)  # empty iter; we call method directly

    shuffled_ids, shuffled_labels = buf._shuffle_batch(input_ids, labels)

    assert shuffled_ids.shape == input_ids.shape
    assert shuffled_labels.shape == labels.shape
