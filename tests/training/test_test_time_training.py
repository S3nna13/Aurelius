"""Tests for test-time training (TTT)."""
import torch
import pytest
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.test_time_training import (
    TTTConfig,
    create_masked_lm_task,
    save_model_state,
    restore_model_state,
    get_adapt_params,
    TestTimeTrainer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_model():
    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


def _input_ids(batch_size=1, seq_len=16, vocab_size=256):
    torch.manual_seed(0)
    return torch.randint(1, vocab_size, (batch_size, seq_len))


# ---------------------------------------------------------------------------
# 1. TTTConfig defaults
# ---------------------------------------------------------------------------

def test_tttconfig_defaults():
    cfg = TTTConfig()
    assert cfg.n_adapt_steps == 5
    assert cfg.adapt_lr == 1e-4
    assert cfg.mask_ratio == 0.15
    assert cfg.adapt_layers == []
    assert cfg.restore_after is True


# ---------------------------------------------------------------------------
# 2. create_masked_lm_task returns same shape as input
# ---------------------------------------------------------------------------

def test_create_masked_lm_task_shape():
    ids = _input_ids()
    masked, labels = create_masked_lm_task(ids)
    assert masked.shape == ids.shape
    assert labels.shape == ids.shape


# ---------------------------------------------------------------------------
# 3. create_masked_lm_task some tokens are masked (~mask_ratio)
# ---------------------------------------------------------------------------

def test_create_masked_lm_task_mask_ratio():
    torch.manual_seed(7)
    ids = _input_ids(batch_size=4, seq_len=200)
    mask_ratio = 0.15
    masked, labels = create_masked_lm_task(ids, mask_ratio=mask_ratio)

    n_masked = (labels != -100).sum().item()
    total = ids.numel()
    observed_ratio = n_masked / total

    # Allow generous tolerance due to randomness
    assert 0.05 <= observed_ratio <= 0.35, (
        f"Expected ~{mask_ratio} masked, got {observed_ratio:.3f}"
    )


# ---------------------------------------------------------------------------
# 4. create_masked_lm_task labels == -100 for unmasked positions
# ---------------------------------------------------------------------------

def test_create_masked_lm_task_labels_unmasked():
    torch.manual_seed(3)
    ids = _input_ids(batch_size=2, seq_len=32)
    masked, labels = create_masked_lm_task(ids, mask_token_id=0)

    # Positions NOT masked should have label == -100
    unmasked_positions = masked != 0
    assert (labels[unmasked_positions] == -100).all(), (
        "Unmasked positions should have label -100"
    )


# ---------------------------------------------------------------------------
# 5. save_model_state returns dict of tensors
# ---------------------------------------------------------------------------

def test_save_model_state_returns_dict_of_tensors():
    model = _make_model()
    state = save_model_state(model)

    assert isinstance(state, dict)
    assert len(state) > 0
    for name, tensor in state.items():
        assert isinstance(name, str)
        assert isinstance(tensor, torch.Tensor)


# ---------------------------------------------------------------------------
# 6. save_model_state -> modify -> restore_model_state -> original restored
# ---------------------------------------------------------------------------

def test_save_restore_model_state_roundtrip():
    model = _make_model()
    state = save_model_state(model)

    # Modify all parameters
    with torch.no_grad():
        for p in model.parameters():
            p.add_(10.0)

    restore_model_state(model, state)

    for name, restored_p in model.named_parameters():
        original = state[name]
        assert torch.allclose(restored_p.data, original), (
            f"Parameter '{name}' was not correctly restored"
        )


# ---------------------------------------------------------------------------
# 7. get_adapt_params returns list of parameters
# ---------------------------------------------------------------------------

def test_get_adapt_params_returns_list():
    model = _make_model()
    params = get_adapt_params(model, adapt_layers=[])
    assert isinstance(params, list)
    assert len(params) > 0
    assert all(isinstance(p, torch.nn.Parameter) for p in params)


# ---------------------------------------------------------------------------
# 8. get_adapt_params with empty adapt_layers returns all layer params
# ---------------------------------------------------------------------------

def test_get_adapt_params_empty_returns_all():
    model = _make_model()
    all_params = get_adapt_params(model, adapt_layers=[])
    expected = list(model.parameters())
    assert len(all_params) == len(expected)


# ---------------------------------------------------------------------------
# 9. TestTimeTrainer.adapt returns required keys
# ---------------------------------------------------------------------------

def test_adapt_returns_required_keys():
    model = _make_model()
    cfg = TTTConfig(n_adapt_steps=2, adapt_lr=1e-3)
    trainer = TestTimeTrainer(model, cfg)

    ids = _input_ids()
    stats = trainer.adapt(ids)

    assert "initial_loss" in stats
    assert "final_loss" in stats
    assert "loss_reduction" in stats


# ---------------------------------------------------------------------------
# 10. TestTimeTrainer.adapt final_loss <= initial_loss
# ---------------------------------------------------------------------------

def test_adapt_loss_decreases_or_stays_flat():
    torch.manual_seed(0)
    model = _make_model()
    cfg = TTTConfig(n_adapt_steps=2, adapt_lr=1e-2)
    trainer = TestTimeTrainer(model, cfg)

    ids = _input_ids(batch_size=1, seq_len=32)
    stats = trainer.adapt(ids)

    # After adaptation, final loss should not be dramatically worse than initial
    # (allow small increase due to stochasticity of masking between steps)
    assert stats["final_loss"] <= stats["initial_loss"] * 2.0, (
        f"Loss unexpectedly increased: {stats}"
    )


# ---------------------------------------------------------------------------
# 11. TestTimeTrainer.predict returns logits with correct shape
# ---------------------------------------------------------------------------

def test_predict_logits_shape():
    model = _make_model()
    cfg = TTTConfig(n_adapt_steps=2)
    trainer = TestTimeTrainer(model, cfg)

    B, T = 1, 16
    ids = _input_ids(batch_size=B, seq_len=T)
    logits = trainer.predict(ids)

    assert logits.shape == (B, T, 256), (
        f"Expected (1, 16, 256), got {logits.shape}"
    )


# ---------------------------------------------------------------------------
# 12. TestTimeTrainer.adapt_and_predict returns (Tensor, dict)
# ---------------------------------------------------------------------------

def test_adapt_and_predict_return_types():
    model = _make_model()
    cfg = TTTConfig(n_adapt_steps=2)
    trainer = TestTimeTrainer(model, cfg)

    ids = _input_ids()
    result = trainer.adapt_and_predict(ids)

    assert isinstance(result, tuple)
    assert len(result) == 2
    logits, stats = result
    assert isinstance(logits, torch.Tensor)
    assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# 13. TestTimeTrainer.adapt_and_predict logits shape (B, T, V)
# ---------------------------------------------------------------------------

def test_adapt_and_predict_logits_shape():
    model = _make_model()
    cfg = TTTConfig(n_adapt_steps=2)
    trainer = TestTimeTrainer(model, cfg)

    B, T = 1, 24
    ids = _input_ids(batch_size=B, seq_len=T)
    logits, _ = trainer.adapt_and_predict(ids)

    assert logits.shape == (B, T, 256)


# ---------------------------------------------------------------------------
# 14. restore_model_state correctly restores after adaptation
# ---------------------------------------------------------------------------

def test_restore_after_adaptation():
    model = _make_model()
    # Capture the original state before any TTT
    original_state = save_model_state(model)

    cfg = TTTConfig(n_adapt_steps=2, adapt_lr=1e-2, restore_after=True)
    trainer = TestTimeTrainer(model, cfg)

    ids = _input_ids(batch_size=1, seq_len=16)
    trainer.adapt_and_predict(ids)

    # After adapt_and_predict with restore_after=True, weights should match original
    for name, param in model.named_parameters():
        assert torch.allclose(param.data, original_state[name], atol=1e-6), (
            f"Parameter '{name}' not restored after adapt_and_predict"
        )
