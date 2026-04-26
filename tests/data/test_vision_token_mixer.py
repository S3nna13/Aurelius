"""Unit tests for src/data/vision_token_mixer.py.

Covers all 15 required test cases (IDs 1-15).
Pure Python / PyTorch only — no transformers, einops, scipy, sklearn.
"""

from __future__ import annotations

from src.data.vision_token_mixer import VisionTokenMixer, VisionTokenMixerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mixer(**kwargs) -> VisionTokenMixer:
    cfg = VisionTokenMixerConfig(**kwargs)
    return VisionTokenMixer(cfg)


def _make_text(n: int, start: int = 100) -> list[int]:
    """Return n distinct text token ids starting at `start`."""
    return list(range(start, start + n))


def _make_vision(n: int, start: int = 2) -> list[int]:
    """Return n distinct vision token ids starting at `start` (avoid 0 and 1)."""
    return list(range(start, start + n))


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = VisionTokenMixerConfig()
    assert cfg.vision_ratio == 0.1
    assert cfg.pad_token_id == 0
    assert cfg.vision_token_id == 1
    assert cfg.max_seq_len == 8192


# ---------------------------------------------------------------------------
# 2. test_mix_empty_vision
# ---------------------------------------------------------------------------


def test_mix_empty_vision():
    mixer = _make_mixer()
    text = _make_text(50)
    result = mixer.mix(text, [])
    assert result == text, "Empty vision_ids should return text_ids unchanged"


# ---------------------------------------------------------------------------
# 3. test_mix_ratio_achieved
# ---------------------------------------------------------------------------


def test_mix_ratio_achieved():
    mixer = _make_mixer(vision_ratio=0.1)
    text = _make_text(90)
    vision = _make_vision(20)
    result = mixer.mix(text, vision)
    vision_set = set(vision)
    actual_ratio = mixer.compute_ratio(result, vision_set)
    assert abs(actual_ratio - 0.1) <= 0.02, f"Expected ratio ~0.1, got {actual_ratio:.4f}"


# ---------------------------------------------------------------------------
# 4. test_mix_vision_tokens_distributed
# ---------------------------------------------------------------------------


def test_mix_vision_tokens_distributed():
    """Vision tokens should NOT all appear at the very front or very back."""
    mixer = _make_mixer(vision_ratio=0.1)
    text = _make_text(90)
    vision = _make_vision(10)
    result = mixer.mix(text, vision)

    vision_set = set(vision)
    positions = [i for i, t in enumerate(result) if t in vision_set]
    n = len(result)

    # At least one vision token should be in the first half and the second half
    first_half = [p for p in positions if p < n // 2]
    second_half = [p for p in positions if p >= n // 2]
    assert len(first_half) > 0, "No vision tokens in first half — not distributed"
    assert len(second_half) > 0, "No vision tokens in second half — not distributed"


# ---------------------------------------------------------------------------
# 5. test_mix_no_vision_duplicates
# ---------------------------------------------------------------------------


def test_mix_no_vision_duplicates():
    mixer = _make_mixer(vision_ratio=0.1)
    text = _make_text(90)
    vision = _make_vision(10)
    result = mixer.mix(text, vision)

    vision_set = set(vision)
    vision_tokens_in_result = [t for t in result if t in vision_set]
    assert len(vision_tokens_in_result) == len(set(vision_tokens_in_result)), (
        "Vision tokens appear more than once in output"
    )


# ---------------------------------------------------------------------------
# 6. test_mix_text_preserved
# ---------------------------------------------------------------------------


def test_mix_text_preserved():
    """All original text tokens appear in the output and their relative order is preserved."""
    mixer = _make_mixer(vision_ratio=0.1)
    text = _make_text(90)
    vision = _make_vision(10)
    result = mixer.mix(text, vision)

    vision_set = set(vision)
    text_in_result = [t for t in result if t not in vision_set]
    assert text_in_result == text, "Text tokens are not fully preserved (order or content mismatch)"


# ---------------------------------------------------------------------------
# 7. test_mix_excess_vision_truncated
# ---------------------------------------------------------------------------


def test_mix_excess_vision_truncated():
    """When vision_ids supplies more tokens than the ratio allows, they are truncated."""
    mixer = _make_mixer(vision_ratio=0.1)
    text = _make_text(90)
    # Supply 50 vision tokens — ratio only permits ~10
    vision = _make_vision(50)
    result = mixer.mix(text, vision)

    vision_set = set(vision)
    n_vision_in_result = sum(1 for t in result if t in vision_set)
    # Should be ~10, definitely not 50
    assert n_vision_in_result < 20, (
        f"Expected truncation to ~10, got {n_vision_in_result} vision tokens"
    )


# ---------------------------------------------------------------------------
# 8. test_mix_empty_text
# ---------------------------------------------------------------------------


def test_mix_empty_text():
    """When text_ids is empty, mix() should return vision_ids without raising."""
    mixer = _make_mixer(vision_ratio=0.1)
    vision = _make_vision(5)
    result = mixer.mix([], vision)
    # Should not raise; result is either vision_ids or []
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 9. test_mix_batch
# ---------------------------------------------------------------------------


def test_mix_batch():
    mixer = _make_mixer(vision_ratio=0.1)
    batch = [
        {"text_ids": _make_text(50, 100), "vision_ids": _make_vision(5, 2)},
        {"text_ids": _make_text(80, 200), "vision_ids": _make_vision(8, 20)},
        {"text_ids": _make_text(60, 300), "vision_ids": []},
    ]
    results = mixer.mix_batch(batch)
    assert len(results) == 3
    for item in results:
        assert "mixed_ids" in item, "Output dict missing 'mixed_ids'"
        assert "vision_mask" in item, "Output dict missing 'vision_mask'"
        assert isinstance(item["mixed_ids"], list)
        assert isinstance(item["vision_mask"], list)
        assert len(item["mixed_ids"]) == len(item["vision_mask"])


# ---------------------------------------------------------------------------
# 10. test_vision_mask_correct
# ---------------------------------------------------------------------------


def test_vision_mask_correct():
    """vision_mask should be 1 at every vision token position and 0 elsewhere."""
    mixer = _make_mixer(vision_ratio=0.1)
    text = _make_text(90, 100)
    vision = _make_vision(10, 2)
    batch = [{"text_ids": text, "vision_ids": vision}]
    result = mixer.mix_batch(batch)[0]

    mixed = result["mixed_ids"]
    mask = result["vision_mask"]
    vision_set = set(vision)

    for pos, (token, m) in enumerate(zip(mixed, mask)):
        expected = 1 if token in vision_set else 0
        assert m == expected, (
            f"Position {pos}: token={token}, expected mask={expected}, got mask={m}"
        )


# ---------------------------------------------------------------------------
# 11. test_compute_ratio
# ---------------------------------------------------------------------------


def test_compute_ratio():
    mixer = _make_mixer()
    # Known sequence: 2 vision tokens (ids 2, 3) out of 10 total => ratio = 0.2
    known_seq = [100, 2, 101, 102, 3, 103, 104, 105, 106, 107]
    vision_set = {2, 3}
    ratio = mixer.compute_ratio(known_seq, vision_set)
    assert abs(ratio - 0.2) < 1e-9, f"Expected 0.2, got {ratio}"


def test_compute_ratio_empty():
    mixer = _make_mixer()
    assert mixer.compute_ratio([], {1, 2}) == 0.0


# ---------------------------------------------------------------------------
# 12. test_determinism
# ---------------------------------------------------------------------------


def test_determinism():
    """Same inputs must always produce the same output."""
    mixer = _make_mixer(vision_ratio=0.1)
    text = _make_text(90)
    vision = _make_vision(15)
    result1 = mixer.mix(text, vision)
    result2 = mixer.mix(text, vision)
    assert result1 == result2, "mix() is not deterministic"


# ---------------------------------------------------------------------------
# 13. test_zero_ratio
# ---------------------------------------------------------------------------


def test_zero_ratio():
    mixer = _make_mixer(vision_ratio=0.0)
    text = _make_text(50)
    vision = _make_vision(10)
    result = mixer.mix(text, vision)
    assert result == text, "vision_ratio=0.0 should insert no vision tokens"


# ---------------------------------------------------------------------------
# 14. test_full_ratio
# ---------------------------------------------------------------------------


def test_full_ratio():
    mixer = _make_mixer(vision_ratio=1.0)
    text = _make_text(50)
    vision = _make_vision(10)
    result = mixer.mix(text, vision)
    # With ratio=1.0 the output should consist only of vision tokens
    assert result == vision, "vision_ratio=1.0 should return vision_ids directly"


# ---------------------------------------------------------------------------
# 15. test_large_sequence
# ---------------------------------------------------------------------------


def test_large_sequence():
    mixer = _make_mixer(vision_ratio=0.1)
    text = _make_text(1000, start=1000)
    vision = _make_vision(200, start=2)
    result = mixer.mix(text, vision)

    assert isinstance(result, list)
    assert len(result) > len(text), "Large sequence: output should be longer than input text"
    vision_set = set(vision)
    actual_ratio = mixer.compute_ratio(result, vision_set)
    assert abs(actual_ratio - 0.1) <= 0.02, (
        f"Large sequence ratio expected ~0.1, got {actual_ratio:.4f}"
    )
