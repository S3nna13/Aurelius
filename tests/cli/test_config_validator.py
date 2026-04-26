"""Tests for src/cli/config_validator.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.cli.config_validator import ConfigValidator


class TestValidateDict:
    def test_valid_config(self):
        cfg = {
            "model": {
                "d_model": 2048,
                "n_layers": 24,
                "n_heads": 16,
                "vocab_size": 128000,
                "max_seq_len": 8192,
            },
            "training": {"seed": 42},
        }
        assert ConfigValidator.validate_dict(cfg) == []

    def test_valid_config_minimal(self):
        cfg = {
            "model": {
                "d_model": 64,
                "n_layers": 1,
                "n_heads": 1,
            },
        }
        assert ConfigValidator.validate_dict(cfg) == []

    def test_missing_required_keys(self):
        cfg = {
            "model": {
                "d_model": 64,
            },
        }
        errors = ConfigValidator.validate_dict(cfg)
        assert "missing required key in model: n_heads" in errors
        assert "missing required key in model: n_layers" in errors

    def test_no_model_section(self):
        cfg = {"training": {"seed": 42}}
        errors = ConfigValidator.validate_dict(cfg)
        assert any("missing or invalid 'model' section" in e for e in errors)

    def test_d_model_not_multiple_of_64(self):
        cfg = {
            "model": {
                "d_model": 100,
                "n_layers": 1,
                "n_heads": 1,
            },
        }
        errors = ConfigValidator.validate_dict(cfg)
        assert any("d_model must be a positive multiple of 64" in e for e in errors)

    def test_d_model_zero(self):
        cfg = {
            "model": {
                "d_model": 0,
                "n_layers": 1,
                "n_heads": 1,
            },
        }
        errors = ConfigValidator.validate_dict(cfg)
        assert any("d_model must be a positive multiple of 64" in e for e in errors)

    def test_n_layers_too_low(self):
        cfg = {
            "model": {
                "d_model": 64,
                "n_layers": 0,
                "n_heads": 1,
            },
        }
        errors = ConfigValidator.validate_dict(cfg)
        assert any("n_layers must be an integer in [1, 128]" in e for e in errors)

    def test_n_layers_too_high(self):
        cfg = {
            "model": {
                "d_model": 64,
                "n_layers": 129,
                "n_heads": 1,
            },
        }
        errors = ConfigValidator.validate_dict(cfg)
        assert any("n_layers must be an integer in [1, 128]" in e for e in errors)

    def test_vocab_size_too_low(self):
        cfg = {
            "model": {
                "d_model": 64,
                "n_layers": 1,
                "n_heads": 1,
                "vocab_size": 255,
            },
        }
        errors = ConfigValidator.validate_dict(cfg)
        assert any("vocab_size must be an integer in [256, 512000]" in e for e in errors)

    def test_vocab_size_too_high(self):
        cfg = {
            "model": {
                "d_model": 64,
                "n_layers": 1,
                "n_heads": 1,
                "vocab_size": 512_001,
            },
        }
        errors = ConfigValidator.validate_dict(cfg)
        assert any("vocab_size must be an integer in [256, 512000]" in e for e in errors)

    def test_max_seq_len_too_low(self):
        cfg = {
            "model": {
                "d_model": 64,
                "n_layers": 1,
                "n_heads": 1,
                "max_seq_len": 63,
            },
        }
        errors = ConfigValidator.validate_dict(cfg)
        assert any("max_seq_len must be an integer in [64, 1000000]" in e for e in errors)

    def test_max_seq_len_too_high(self):
        cfg = {
            "model": {
                "d_model": 64,
                "n_layers": 1,
                "n_heads": 1,
                "max_seq_len": 1_000_001,
            },
        }
        errors = ConfigValidator.validate_dict(cfg)
        assert any("max_seq_len must be an integer in [64, 1000000]" in e for e in errors)

    def test_unknown_top_level_key(self):
        cfg = {
            "model": {
                "d_model": 64,
                "n_layers": 1,
                "n_heads": 1,
            },
            "optimizer": {"type": "adamw"},
        }
        errors = ConfigValidator.validate_dict(cfg)
        assert any("unknown top-level key: optimizer" in e for e in errors)

    def test_multiple_unknown_keys(self):
        cfg = {
            "model": {
                "d_model": 64,
                "n_layers": 1,
                "n_heads": 1,
            },
            "scheduler": {"type": "cosine"},
            "checkpoint": {"enabled": True},
        }
        errors = ConfigValidator.validate_dict(cfg)
        assert "unknown top-level key: checkpoint" in errors
        assert "unknown top-level key: scheduler" in errors


class TestValidateFile:
    def test_valid_file(self):
        content = """
model:
  d_model: 128
  n_layers: 2
  n_heads: 2
training:
  seed: 42
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            path = f.name
        assert ConfigValidator.validate_file(path) == []
        Path(path).unlink()

    def test_file_not_found(self):
        errors = ConfigValidator.validate_file("/nonexistent/path/config.yaml")
        assert any("path does not exist" in e for e in errors)

    def test_yaml_error(self):
        content = "model: {unclosed"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            path = f.name
        errors = ConfigValidator.validate_file(path)
        assert any("YAML parse error" in e for e in errors)
        Path(path).unlink()

    def test_yaml_root_not_mapping(self):
        content = "- item1\n- item2\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            path = f.name
        errors = ConfigValidator.validate_file(path)
        assert any("YAML root must be a mapping" in e for e in errors)
        Path(path).unlink()

    def test_file_is_directory(self, tmp_path: Path):
        d = tmp_path / "config_dir"
        d.mkdir()
        errors = ConfigValidator.validate_file(str(d))
        assert any("path is not a file" in e for e in errors)
