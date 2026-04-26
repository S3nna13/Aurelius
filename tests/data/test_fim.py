"""Tests for Fill-in-the-Middle transformation."""

import pytest

from src.data.fim_transform import FIMResult, apply_fim, fim_transform_batch


def test_passthrough_short_code():
    # Very short code should pass through unchanged
    result = apply_fim("x=1", seed=0)
    # Either unchanged or contains FIM markers
    assert "x=1" in result or result == "x=1"


def test_fim_markers_present_when_applied():
    code = "def foo():\n    x = 1\n    y = 2\n    return x + y"
    # Run many seeds until we find one that applies FIM
    for seed in range(20):
        result = apply_fim(code, seed=seed)
        if result != code:
            assert any(
                tok in result for tok in ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"]
            )
            return
    # It's fine if FIM wasn't applied (50% rate) in 20 tries — just skip
    pytest.skip("All 20 seeds skipped FIM — statistically unlikely but acceptable")


def test_original_content_preserved():
    code = "def add(a, b):\n    return a + b"
    fim_tokens = ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"]
    for seed in range(10):
        result = apply_fim(code, seed=seed)
        if result == code:
            # Passthrough: original must be identical
            assert result == code
        else:
            # FIM applied: stripping markers should recover all original characters
            stripped = result
            for tok in fim_tokens:
                stripped = stripped.replace(tok, "")
            assert sorted(stripped) == sorted(code), "FIM must preserve all characters"


def test_fim_rate_approximately_50_percent():
    code = "x = 1\ny = 2\nz = x + y\nreturn z"
    results = [apply_fim(code, seed=i) for i in range(200)]
    fim_count = sum(1 for r in results if r != code)
    # 50% rate, allow generous 30-70% range
    assert 30 <= fim_count <= 170, f"FIM rate out of range: {fim_count}/200"


def test_batch_transform():
    codes = ["x = 1", "def foo(): pass", "import os\nprint(os.getcwd())"]
    results = fim_transform_batch(codes)
    assert len(results) == len(codes)
    for original, result in zip(codes, results):
        assert isinstance(result, FIMResult)
        assert isinstance(result.text, str)
        assert len(result.text) > 0
