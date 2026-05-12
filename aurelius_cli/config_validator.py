"""Configuration validator for Aurelius YAML configs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigValidator:
    """Validate Aurelius configuration files and dictionaries."""

    _WHITELIST: set[str] = {
        "model",
        "training",
        "inference",
        "serving",
        "agent",
        "safety",
        "data",
    }

    _REQUIRED_MODEL_KEYS: set[str] = {"d_model", "n_layers", "n_heads"}

    @classmethod
    def validate_file(cls, path: str) -> list[str]:
        """Validate a YAML configuration file at *path*.

        Returns a list of human-readable error strings. An empty list means
        the file is valid.
        """
        p = Path(path)
        if not p.exists():
            return [f"path does not exist: {path}"]
        if not p.is_file():
            return [f"path is not a file: {path}"]
        if not os.access(p, os.R_OK):
            return [f"path is not readable: {path}"]

        try:
            with open(p, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            return [f"YAML parse error: {exc}"]

        if not isinstance(raw, dict):
            return ["YAML root must be a mapping"]

        return cls.validate_dict(raw)

    @classmethod
    def validate_dict(cls, cfg: dict[str, Any]) -> list[str]:
        """Validate a configuration dictionary in memory.

        Returns a list of human-readable error strings. An empty list means
        the configuration is valid.
        """
        errors: list[str] = []

        # Top-level key whitelist
        unknown_keys = set(cfg.keys()) - cls._WHITELIST
        for key in sorted(unknown_keys):
            errors.append(f"unknown top-level key: {key}")

        model = cfg.get("model")
        if not isinstance(model, dict):
            errors.append("missing or invalid 'model' section")
            # Cannot proceed with model-level checks
            return errors

        # Required model keys
        missing = cls._REQUIRED_MODEL_KEYS - set(model.keys())
        for key in sorted(missing):
            errors.append(f"missing required key in model: {key}")

        # d_model: positive multiple of 64
        d_model = model.get("d_model")
        if d_model is not None:
            if not isinstance(d_model, int) or d_model <= 0 or d_model % 64 != 0:
                errors.append(f"d_model must be a positive multiple of 64, got {d_model}")

        # n_layers: [1, 128]
        n_layers = model.get("n_layers")
        if n_layers is not None:
            if not isinstance(n_layers, int) or not (1 <= n_layers <= 128):
                errors.append(f"n_layers must be an integer in [1, 128], got {n_layers}")

        # vocab_size: [256, 512_000]
        vocab_size = model.get("vocab_size")
        if vocab_size is not None:
            if not isinstance(vocab_size, int) or not (256 <= vocab_size <= 512_000):
                errors.append(f"vocab_size must be an integer in [256, 512000], got {vocab_size}")

        # max_seq_len: [64, 1_000_000]
        max_seq_len = model.get("max_seq_len")
        if max_seq_len is not None:
            if not isinstance(max_seq_len, int) or not (64 <= max_seq_len <= 1_000_000):
                errors.append(f"max_seq_len must be an integer in [64, 1000000], got {max_seq_len}")

        return errors
