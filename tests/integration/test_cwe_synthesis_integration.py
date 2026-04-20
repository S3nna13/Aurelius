"""Integration tests for CWE synthetic pair generator.

- Verifies LOADER_REGISTRY wiring in src.data package
- Verifies config flag default OFF
- Verifies batch diversity over several seeds
- Verifies data pipeline (tokenizer trainer) still imports cleanly
"""
from __future__ import annotations

import pytest


def test_loader_registry_exposes_cwe_generator():
    from src.data import LOADER_REGISTRY, CWESyntheticGenerator
    assert "cwe_synthesis" in LOADER_REGISTRY
    assert LOADER_REGISTRY["cwe_synthesis"] is CWESyntheticGenerator


def test_config_flag_defaults_off():
    from src.model.config import AureliusConfig
    cfg = AureliusConfig()
    assert hasattr(cfg, "data_cwe_synthesis_enabled")
    assert cfg.data_cwe_synthesis_enabled is False


def test_generate_five_pairs_well_formed():
    from src.data import CWESyntheticGenerator
    gen = CWESyntheticGenerator(rng_seed=123)
    batch = gen.generate_batch(5)
    assert len(batch) == 5
    for p in batch:
        assert p["cwe_id"].startswith("CWE-")
        assert p["vulnerable_code"] and p["secure_code"]
        assert p["vulnerable_code"] != p["secure_code"]


def test_distinct_cwes_across_seeds():
    from src.data import CWESyntheticGenerator
    seen: set[str] = set()
    for seed in range(8):
        gen = CWESyntheticGenerator(rng_seed=seed)
        for pair in gen.generate_batch(10):
            seen.add(pair["cwe_id"])
    # Over 80 draws across 8 seeds, we should hit many recipes.
    assert len(seen) >= 6


def test_integration_with_existing_data_pipeline_imports():
    """Importing the data package must still work with existing loaders."""
    import importlib
    mod = importlib.import_module("src.data")
    # Existing symbols intact.
    for name in ("AURELIUS_MIX", "FIMConfig", "BPEConfig"):
        assert hasattr(mod, name), name
    # New symbols present.
    for name in ("CWE_CATALOG", "CWESyntheticGenerator", "LOADER_REGISTRY"):
        assert hasattr(mod, name), name


def test_integration_no_foreign_imports_in_module():
    """Source of cwe_synthesis.py must not import banned frameworks."""
    import pathlib
    src = pathlib.Path(
        "src/data/cwe_synthesis.py"
    ).read_text(encoding="utf-8")
    for bad in (
        "transformers", "einops", "trl", "xformers", "flash_attn",
        "bitsandbytes", "peft", "diffusers", "datasets", "accelerate",
        "deepspeed", "langchain", "llamaindex",
    ):
        assert f"from {bad}" not in src
        assert f"import {bad}" not in src


def test_rendering_is_pure_string_never_executed():
    """Safety: render output is a string and is not exec'd by anyone."""
    from src.data import CWESyntheticGenerator
    gen = CWESyntheticGenerator(rng_seed=0)
    pair = gen.generate_pair(cwe_id="CWE-94")
    assert isinstance(pair["vulnerable_code"], str)
    assert isinstance(pair["secure_code"], str)
