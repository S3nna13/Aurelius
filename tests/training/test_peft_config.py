"""Tests for src.training.peft_config."""

from __future__ import annotations

import pytest

from src.training.peft_config import (
    PEFT_CONFIG_REGISTRY,
    PEFTConfig,
    PEFTMethod,
    TrainingMode,
    config_summary,
    validate_config,
)


def test_default_method():
    assert PEFTConfig().method == PEFTMethod.LORA


def test_default_rank():
    assert PEFTConfig().rank == 16


def test_default_alpha():
    assert PEFTConfig().alpha == 32.0


def test_default_target_modules():
    assert PEFTConfig().target_modules == ["q_proj", "v_proj"]


def test_default_dropout():
    assert PEFTConfig().dropout == 0.05


def test_default_learning_rate():
    assert PEFTConfig().learning_rate == 2e-4


def test_default_num_epochs():
    assert PEFTConfig().num_epochs == 3


def test_default_batch_size():
    assert PEFTConfig().batch_size == 4


def test_default_max_seq_length():
    assert PEFTConfig().max_seq_length == 2048


def test_default_grad_accum():
    assert PEFTConfig().gradient_accumulation_steps == 8


def test_default_warmup_ratio():
    assert PEFTConfig().warmup_ratio == 0.05


def test_default_weight_decay():
    assert PEFTConfig().weight_decay == 0.01


def test_default_training_mode():
    assert PEFTConfig().training_mode == TrainingMode.SFT


def test_default_output_dir():
    assert PEFTConfig().output_dir == "./outputs"


def test_custom_values():
    cfg = PEFTConfig(rank=32, alpha=64.0, dropout=0.1, learning_rate=1e-4)
    assert cfg.rank == 32
    assert cfg.alpha == 64.0
    assert cfg.dropout == 0.1
    assert cfg.learning_rate == 1e-4


def test_target_modules_independent_instances():
    a = PEFTConfig()
    b = PEFTConfig()
    a.target_modules.append("k_proj")
    assert "k_proj" not in b.target_modules


def test_validate_config_ok():
    assert validate_config(PEFTConfig()) == []


def test_validate_config_bad_rank():
    errors = validate_config(PEFTConfig(rank=0))
    assert any("rank" in e for e in errors)


def test_validate_config_negative_rank():
    errors = validate_config(PEFTConfig(rank=-1))
    assert any("rank" in e for e in errors)


def test_validate_config_bad_alpha():
    errors = validate_config(PEFTConfig(alpha=0.0))
    assert any("alpha" in e for e in errors)


def test_validate_config_bad_dropout_high():
    errors = validate_config(PEFTConfig(dropout=1.0))
    assert any("dropout" in e for e in errors)


def test_validate_config_bad_dropout_negative():
    errors = validate_config(PEFTConfig(dropout=-0.01))
    assert any("dropout" in e for e in errors)


def test_validate_config_bad_learning_rate():
    errors = validate_config(PEFTConfig(learning_rate=0.0))
    assert any("learning_rate" in e for e in errors)


def test_validate_config_bad_num_epochs():
    errors = validate_config(PEFTConfig(num_epochs=0))
    assert any("num_epochs" in e for e in errors)


def test_validate_config_bad_batch_size():
    errors = validate_config(PEFTConfig(batch_size=0))
    assert any("batch_size" in e for e in errors)


def test_validate_config_multiple_errors():
    errors = validate_config(PEFTConfig(rank=0, alpha=0.0, batch_size=0))
    assert len(errors) >= 3


def test_config_summary_returns_dict():
    s = config_summary(PEFTConfig())
    assert isinstance(s, dict)


def test_config_summary_has_keys():
    s = config_summary(PEFTConfig())
    expected = {
        "method",
        "rank",
        "alpha",
        "target_modules",
        "dropout",
        "learning_rate",
        "num_epochs",
        "batch_size",
        "max_seq_length",
        "gradient_accumulation_steps",
        "warmup_ratio",
        "weight_decay",
        "training_mode",
        "output_dir",
    }
    assert expected.issubset(s.keys())


def test_config_summary_enum_values_serialised():
    s = config_summary(PEFTConfig())
    assert s["method"] == "lora"
    assert s["training_mode"] == "sft"


def test_peft_method_members():
    assert {m.name for m in PEFTMethod} == {"LORA", "P_TUNING", "ADAPTER", "IA3"}


def test_training_mode_members():
    assert {m.name for m in TrainingMode} == {"SFT", "DPO", "GRPO", "SIMPO", "ORPO"}


def test_training_mode_values():
    assert TrainingMode.SFT.value == "sft"
    assert TrainingMode.DPO.value == "dpo"
    assert TrainingMode.GRPO.value == "grpo"
    assert TrainingMode.SIMPO.value == "simpo"
    assert TrainingMode.ORPO.value == "orpo"


def test_registry_has_default():
    assert "default" in PEFT_CONFIG_REGISTRY
    assert PEFT_CONFIG_REGISTRY["default"] is PEFTConfig


def test_registry_instance():
    cfg = PEFT_CONFIG_REGISTRY["default"]()
    assert isinstance(cfg, PEFTConfig)


def test_training_mode_docstring_mentions_dpo():
    assert "DPO" in (TrainingMode.__doc__ or "")


@pytest.mark.parametrize("method", list(PEFTMethod))
def test_peft_method_roundtrip(method):
    assert PEFTMethod(method.value) is method
