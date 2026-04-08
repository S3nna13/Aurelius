"""Tests for Multi-Token Prediction (MTP) module.

Project test config: AureliusConfig(
    n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
    head_dim=16, d_ff=128, vocab_size=256, max_seq_len=64
)
"""

import torch
import torch.nn.functional as F
import pytest

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.model.multi_token_prediction import (
    MTPConfig,
    MTPHead,
    MultiTokenPredictionModel,
    MTPTrainer,
    acceptance_rate_stats,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture
def base_model(cfg):
    model = AureliusTransformer(cfg)
    model.eval()
    return model


@pytest.fixture
def mtp_cfg():
    return MTPConfig(n_heads=4, shared_head=False, head_type="linear", detach_hidden=True)


@pytest.fixture
def mtp_model(base_model, mtp_cfg):
    return MultiTokenPredictionModel(base_model, mtp_cfg)


# ---------------------------------------------------------------------------
# test_mtp_head_output_shape
# ---------------------------------------------------------------------------

def test_mtp_head_output_shape(cfg):
    """MTPHead: (B, T, D) -> (B, T, V)."""
    B, T = 2, 16
    head = MTPHead(cfg, step=1, head_type="linear")
    hidden = torch.randn(B, T, cfg.d_model)
    logits = head(hidden)
    assert logits.shape == (B, T, cfg.vocab_size), (
        f"Expected ({B}, {T}, {cfg.vocab_size}), got {logits.shape}"
    )


# ---------------------------------------------------------------------------
# test_mtp_model_forward_no_labels
# ---------------------------------------------------------------------------

def test_mtp_model_forward_no_labels(mtp_model, cfg):
    """Without labels: returns (None, logits, pkv) tuple."""
    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        result = mtp_model(input_ids)
    assert isinstance(result, tuple) and len(result) == 3
    loss, logits, pkv = result
    assert loss is None
    assert logits.shape == (B, T, cfg.vocab_size)
    assert isinstance(pkv, list)


# ---------------------------------------------------------------------------
# test_mtp_model_forward_with_labels
# ---------------------------------------------------------------------------

def test_mtp_model_forward_with_labels(mtp_model, cfg):
    """With labels: loss > 0."""
    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    labels = input_ids.clone()
    with torch.no_grad():
        loss, logits, pkv = mtp_model(input_ids, labels=labels)
    assert loss is not None
    assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"
    assert logits.shape == (B, T, cfg.vocab_size)


# ---------------------------------------------------------------------------
# test_mtp_model_loss_is_mean_of_all_heads
# ---------------------------------------------------------------------------

def test_mtp_model_loss_is_mean_of_all_heads(base_model, cfg):
    """Total loss is bounded by the min and max of individual head losses."""
    mtp_config = MTPConfig(n_heads=3, shared_head=False, head_type="linear", detach_hidden=False)
    model = MultiTokenPredictionModel(base_model, mtp_config)
    model.eval()

    B, T = 2, 20
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    labels = input_ids.clone()

    with torch.no_grad():
        hidden, main_loss, logits, pkv = model._get_hidden_states_with_labels(input_ids, labels)

        individual_losses = [main_loss.item()]
        for k, head in enumerate(model.mtp_heads):
            step = k + 1
            if step >= hidden.shape[1]:
                continue
            h_slice = hidden[:, :-step, :]
            target = labels[:, step:]
            hl = head(h_slice)
            B2, Ts, V = hl.shape
            hl_loss = F.cross_entropy(hl.reshape(B2 * Ts, V), target.reshape(B2 * Ts))
            individual_losses.append(hl_loss.item())

        total_loss, _, _ = model(input_ids, labels=labels)

    min_loss = min(individual_losses)
    max_loss = max(individual_losses)
    total = total_loss.item()

    assert total >= min_loss - 1e-4, f"Total loss {total} below min {min_loss}"
    assert total <= max_loss + 1e-4, f"Total loss {total} above max {max_loss}"


# ---------------------------------------------------------------------------
# test_mtp_parallel_decode_length
# ---------------------------------------------------------------------------

def test_mtp_parallel_decode_length(mtp_model, cfg):
    """parallel_decode returns exactly n_tokens token ids."""
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    for n in [1, 3, 4, 7, 10]:
        generated, stats = mtp_model.parallel_decode(input_ids, n_tokens=n)
        assert len(generated) == n, f"Expected {n} tokens, got {len(generated)}"
        assert "n_forward_passes" in stats
        assert "tokens_per_pass" in stats


# ---------------------------------------------------------------------------
# test_mtp_trainer_step_keys
# ---------------------------------------------------------------------------

def test_mtp_trainer_step_keys(base_model, cfg):
    """MTPTrainer.train_step returns dict with 'loss', 'main_loss', 'aux_loss'."""
    mtp_config = MTPConfig(n_heads=2, shared_head=False, head_type="linear")
    optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-4)
    trainer = MTPTrainer(base_model, mtp_config, optimizer, max_seq_len=64)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    result = trainer.train_step(input_ids)

    assert "loss" in result, f"Missing 'loss' key, got: {list(result.keys())}"
    assert "main_loss" in result, "Missing 'main_loss' key"
    assert "aux_loss" in result, "Missing 'aux_loss' key"


# ---------------------------------------------------------------------------
# test_mtp_trainer_step_loss_positive
# ---------------------------------------------------------------------------

def test_mtp_trainer_step_loss_positive(base_model, cfg):
    """MTPTrainer.train_step: loss > 0."""
    mtp_config = MTPConfig(n_heads=2, shared_head=False, head_type="linear")
    optimizer = torch.optim.Adam(
        list(base_model.parameters()), lr=1e-4
    )
    trainer = MTPTrainer(base_model, mtp_config, optimizer, max_seq_len=64)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    result = trainer.train_step(input_ids)

    assert result["loss"] > 0, f"Expected positive loss, got {result['loss']}"
    assert result["main_loss"] > 0, f"Expected positive main_loss, got {result['main_loss']}"


# ---------------------------------------------------------------------------
# test_acceptance_rate_stats_keys
# ---------------------------------------------------------------------------

def test_acceptance_rate_stats_keys(cfg):
    """acceptance_rate_stats returns dict with head accuracy keys."""
    T, V = 20, cfg.vocab_size
    n_heads = 3

    base_logits = torch.randn(T, V)
    head_logits_list = [torch.randn(T, V) for _ in range(n_heads)]
    true_tokens = torch.randint(0, V, (T,))

    stats = acceptance_rate_stats(base_logits, head_logits_list, true_tokens)

    for k in range(1, n_heads + 1):
        key = f"head_{k}_accuracy"
        assert key in stats, f"Missing key {key!r} in stats"
        assert 0.0 <= stats[key] <= 1.0, f"Accuracy out of range: {stats[key]}"


# ---------------------------------------------------------------------------
# test_mtp_config_n_heads
# ---------------------------------------------------------------------------

def test_mtp_config_n_heads(base_model, cfg):
    """Model has the correct number of MTP heads created."""
    for n in [1, 3, 6]:
        mtp_config = MTPConfig(n_heads=n, shared_head=False, head_type="linear")
        model = MultiTokenPredictionModel(base_model, mtp_config)
        assert len(model.mtp_heads) == n, (
            f"Expected {n} MTP heads, got {len(model.mtp_heads)}"
        )


# ---------------------------------------------------------------------------
# test_mtp_shared_head
# ---------------------------------------------------------------------------

def test_mtp_shared_head(base_model, cfg):
    """With shared_head=True, all heads share the same parameters."""
    mtp_config = MTPConfig(n_heads=4, shared_head=True, head_type="linear")
    model = MultiTokenPredictionModel(base_model, mtp_config)

    first_head = model.mtp_heads[0]
    for i, head in enumerate(model.mtp_heads):
        assert head.proj.weight.data_ptr() == first_head.proj.weight.data_ptr(), (
            f"Head {i} does not share parameters with head 0"
        )
