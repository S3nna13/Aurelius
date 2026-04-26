"""Tests for src/training/training_config.py"""

from src.training.training_config import (
    DEFAULT_TRAINING_CONFIG,
    TrainingConfig,
)

# ---------------------------------------------------------------------------
# Tests: defaults
# ---------------------------------------------------------------------------


def test_default_vocab_size():
    cfg = TrainingConfig()
    assert cfg.vocab_size == 50257


def test_default_n_layers():
    cfg = TrainingConfig()
    assert cfg.n_layers == 12


def test_default_d_model():
    cfg = TrainingConfig()
    assert cfg.d_model == 768


def test_default_learning_rate():
    cfg = TrainingConfig()
    assert cfg.learning_rate == 6e-4


def test_default_weight_decay():
    cfg = TrainingConfig()
    assert cfg.weight_decay == 0.1


def test_default_dtype():
    cfg = TrainingConfig()
    assert cfg.dtype == "float32"


def test_default_n_kv_heads_is_none():
    cfg = TrainingConfig()
    assert cfg.n_kv_heads is None


def test_default_compile_false():
    cfg = TrainingConfig()
    assert cfg.compile is False


# ---------------------------------------------------------------------------
# Tests: presets
# ---------------------------------------------------------------------------


def test_gpt2_small_preset():
    cfg = TrainingConfig.gpt2_small()
    assert cfg.vocab_size == 50257
    assert cfg.n_layers == 12
    assert cfg.n_heads == 12
    assert cfg.d_model == 768


def test_gpt2_medium_preset():
    cfg = TrainingConfig.gpt2_medium()
    assert cfg.n_layers == 24
    assert cfg.n_heads == 16
    assert cfg.d_model == 1024


def test_aurelius_1b_preset():
    cfg = TrainingConfig.aurelius_1b()
    assert cfg.vocab_size == 128000
    assert cfg.d_model == 2048
    assert cfg.n_kv_heads == 8
    assert cfg.max_seq_len == 8192


# ---------------------------------------------------------------------------
# Tests: to_dict() / from_dict()
# ---------------------------------------------------------------------------


def test_to_dict_returns_dict():
    cfg = TrainingConfig()
    d = cfg.to_dict()
    assert isinstance(d, dict)


def test_to_dict_contains_all_fields():
    cfg = TrainingConfig()
    d = cfg.to_dict()
    assert "vocab_size" in d
    assert "learning_rate" in d
    assert "n_kv_heads" in d


def test_round_trip_to_from_dict():
    original = TrainingConfig(n_layers=8, d_model=512, learning_rate=1e-3)
    d = original.to_dict()
    restored = TrainingConfig.from_dict(d)
    assert restored.n_layers == 8
    assert restored.d_model == 512
    assert restored.learning_rate == 1e-3


def test_from_dict_ignores_unknown_keys():
    d = TrainingConfig().to_dict()
    d["unknown_key"] = "should_be_ignored"
    cfg = TrainingConfig.from_dict(d)
    assert cfg.vocab_size == 50257


def test_from_dict_partial_override():
    d = {"n_layers": 6}
    cfg = TrainingConfig.from_dict(d)
    assert cfg.n_layers == 6
    assert cfg.d_model == 768  # default preserved


# ---------------------------------------------------------------------------
# Tests: effective_tokens_per_step()
# ---------------------------------------------------------------------------


def test_effective_tokens_per_step_default():
    cfg = TrainingConfig()  # batch=12, grad_accum=1, seq=1024
    assert cfg.effective_tokens_per_step() == 12 * 1 * 1024


def test_effective_tokens_per_step_with_accum():
    cfg = TrainingConfig(batch_size=4, gradient_accumulation_steps=8, max_seq_len=2048)
    assert cfg.effective_tokens_per_step() == 4 * 8 * 2048


# ---------------------------------------------------------------------------
# Tests: chinchilla_optimal_steps()
# ---------------------------------------------------------------------------


def test_chinchilla_optimal_steps_exact():
    cfg = TrainingConfig(batch_size=4, gradient_accumulation_steps=1, max_seq_len=1024)
    tokens_per_step = cfg.effective_tokens_per_step()  # 4096
    total = tokens_per_step * 500
    assert cfg.chinchilla_optimal_steps(total) == 500


def test_chinchilla_optimal_steps_returns_int():
    cfg = TrainingConfig()
    result = cfg.chinchilla_optimal_steps(10_000_000)
    assert isinstance(result, int)


# ---------------------------------------------------------------------------
# Tests: module-level singleton
# ---------------------------------------------------------------------------


def test_default_config_singleton():
    assert isinstance(DEFAULT_TRAINING_CONFIG, TrainingConfig)
    assert DEFAULT_TRAINING_CONFIG.vocab_size == 50257
