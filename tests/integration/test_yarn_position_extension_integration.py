"""Integration tests for the YaRN position-extension utility surface.

Verify:
* The utility is exposed via ``src.longcontext`` (not registered in
  ``LONGCONTEXT_STRATEGY_REGISTRY`` — it's a helper, not a strategy).
* No existing registry entries are perturbed.
* Importing ``src.longcontext`` (and touching the YaRN helpers) does not
  import ``src.model`` — hermetic subprocess check.
"""

from __future__ import annotations

import subprocess
import sys

import torch

import src.longcontext as lc


def test_yarn_surface_exposed_on_package() -> None:
    assert hasattr(lc, "YarnConfig")
    assert hasattr(lc, "build_yarn_rotary_cache")
    assert hasattr(lc, "yarn_inv_freq")
    assert hasattr(lc, "yarn_linear_ramp_mask")
    assert hasattr(lc, "yarn_mscale")
    assert hasattr(lc, "yarn_apply_rotary")


def test_yarn_not_registered_as_strategy() -> None:
    # These are reusable helpers, not a KV/attention strategy.
    for key in ("yarn", "yarn_position_extension", "yarn_rotary"):
        assert key not in lc.LONGCONTEXT_STRATEGY_REGISTRY


def test_existing_registry_entries_preserved() -> None:
    # Ensure our additive __init__ edit didn't drop any previously
    # registered entry.
    expected = {
        "kv_int8",
        "attention_sinks",
        "ring_attention",
        "context_compaction",
        "kv_kivi_int4",
        "infini",
    }
    assert expected.issubset(set(lc.LONGCONTEXT_STRATEGY_REGISTRY.keys()))


def test_end_to_end_usage() -> None:
    cfg = lc.YarnConfig(
        head_dim=16,
        rope_theta=10000.0,
        original_max_seq_len=128,
        scaling_factor=4.0,
    )
    cos, sin = lc.build_yarn_rotary_cache(cfg, seq_len=512)
    assert cos.shape == (512, 16)
    x = torch.randn(2, 4, 512, 16)
    y = lc.yarn_apply_rotary(x, cos, sin)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_importing_longcontext_does_not_import_model() -> None:
    # Hermetic subprocess check mirroring sibling integration tests.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.longcontext as lc; import sys; "
            "assert 'src.model' not in sys.modules, list(sys.modules); "
            "cfg = lc.YarnConfig(head_dim=16); "
            "cos, sin = lc.build_yarn_rotary_cache(cfg, seq_len=64); "
            "assert cos.shape == (64, 16); "
            "assert 'src.model' not in sys.modules, list(sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
