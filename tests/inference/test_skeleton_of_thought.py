"""Tests for src/inference/skeleton_of_thought.py.

Covers:
  - SoTConfig defaults
  - SkeletonParser.parse / count_points (empty, two-point, excess separators)
  - SkeletonOfThoughtDecoder:
      generate_skeleton (EOS stop, length limit)
      expand_point (shape, EOS stop)
      expand_all (return type, total_tokens, points count)
      speedup_estimate (positive, single-point)
  - Integration test: full expand_all pipeline with a stub model_fn
"""
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.inference.skeleton_of_thought import (
    SkeletonOfThoughtDecoder,
    SkeletonParser,
    SoTConfig,
    SoTResult,
    SkeletonPoint,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SEP = 10       # arbitrary separator token id
EOS = 2        # EOS token id
VOCAB = 50     # small vocab for tests


def _fixed_logits(token_id: int, vocab: int = VOCAB) -> Tensor:
    """Return a logit tensor that greedily picks *token_id*."""
    logits = torch.full((vocab,), -1e9)
    logits[token_id] = 1e9
    return logits


def _model_fn_always(token_id: int):
    """model_fn that always returns logits favouring *token_id*."""
    def fn(context):  # noqa: ARG001
        return _fixed_logits(token_id)
    return fn


def _make_decoder(**kwargs) -> SkeletonOfThoughtDecoder:
    cfg = SoTConfig(**kwargs)
    return SkeletonOfThoughtDecoder(config=cfg, sep_token_id=SEP, eos_token_id=EOS)


# ===========================================================================
# 1. Config defaults
# ===========================================================================

def test_config_defaults():
    cfg = SoTConfig()
    assert cfg.max_skeleton_tokens == 128
    assert cfg.max_point_tokens == 256
    assert cfg.max_points == 8
    assert cfg.skeleton_sep == "||"
    assert cfg.expand_in_parallel is True


# ===========================================================================
# 2. SkeletonParser — empty input
# ===========================================================================

def test_parse_skeleton_empty():
    """No separators → single point (or empty if no tokens at all)."""
    parser = SkeletonParser(sep_token_id=SEP, eos_token_id=EOS)
    # Completely empty sequence
    starts = parser.parse([], vocab_size=VOCAB)
    assert starts == []

    # Non-empty sequence with no separator → one point start at 0
    starts = parser.parse([5, 6, 7], vocab_size=VOCAB)
    assert starts == [0]


# ===========================================================================
# 3. SkeletonParser — two points
# ===========================================================================

def test_parse_skeleton_two_points():
    """One separator → two point start indices."""
    parser = SkeletonParser(sep_token_id=SEP, eos_token_id=EOS)
    tokens = [3, 4, SEP, 7, 8]
    starts = parser.parse(tokens, vocab_size=VOCAB)
    assert len(starts) == 2
    assert starts[0] == 0
    assert starts[1] == 3  # index after the separator


# ===========================================================================
# 4. SkeletonParser — max_points cap
# ===========================================================================

def test_parse_skeleton_max_points():
    """Excess separators should be capped at max_points by the decoder."""
    decoder = _make_decoder(max_points=3)
    # Build a sequence with 5 separators → would give 6 segments
    tokens = [1, SEP, 2, SEP, 3, SEP, 4, SEP, 5, SEP, 6]
    segments = decoder.parse_skeleton(tokens)
    assert len(segments) <= decoder.config.max_points


# ===========================================================================
# 5. SkeletonParser.count_points
# ===========================================================================

def test_count_points_correct():
    parser = SkeletonParser(sep_token_id=SEP, eos_token_id=EOS)
    assert parser.count_points([]) == 0
    assert parser.count_points([5, 6, 7]) == 1                  # no sep → 1 point
    assert parser.count_points([5, SEP, 7]) == 2                # one sep → 2 points
    assert parser.count_points([1, SEP, 2, SEP, 3]) == 3        # two seps → 3 points


# ===========================================================================
# 6. generate_skeleton — EOS stops generation early
# ===========================================================================

def test_generate_skeleton_eos_stops():
    """When the model always returns EOS the output should be empty."""
    decoder = _make_decoder(max_skeleton_tokens=100)
    result = decoder.generate_skeleton(
        prompt_tokens=[1, 2, 3],
        model_fn=_model_fn_always(EOS),
    )
    assert result == []


# ===========================================================================
# 7. generate_skeleton — honours length limit
# ===========================================================================

def test_generate_skeleton_length_limit():
    """With a non-EOS model_fn output is capped at max_skeleton_tokens."""
    limit = 10
    decoder = _make_decoder(max_skeleton_tokens=limit)
    # model always picks token 5 (not EOS, not SEP)
    result = decoder.generate_skeleton(
        prompt_tokens=[1],
        model_fn=_model_fn_always(5),
    )
    assert len(result) == limit
    assert all(t == 5 for t in result)


# ===========================================================================
# 8. expand_point — returns list of ints
# ===========================================================================

def test_expand_point_shape():
    decoder = _make_decoder(max_point_tokens=5)
    expansion = decoder.expand_point(
        prompt_tokens=[1, 2],
        point_tokens=[3],
        model_fn=_model_fn_always(7),   # token 7, not EOS
    )
    assert isinstance(expansion, list)
    assert len(expansion) > 0
    assert all(isinstance(t, int) for t in expansion)


# ===========================================================================
# 9. expand_point — stops at EOS
# ===========================================================================

def test_expand_point_stops_at_eos():
    """EOS as first token means zero-length expansion."""
    decoder = _make_decoder(max_point_tokens=50)
    expansion = decoder.expand_point(
        prompt_tokens=[1],
        point_tokens=[3],
        model_fn=_model_fn_always(EOS),
    )
    assert expansion == []


# ===========================================================================
# 10. expand_all — returns SoTResult
# ===========================================================================

def test_expand_all_result_type():
    decoder = _make_decoder(max_skeleton_tokens=8, max_point_tokens=4, max_points=2)
    # Two-point skeleton: [5, SEP, 6]
    skeleton_toks = [5, SEP, 6]
    result = decoder.expand_all(
        prompt_tokens=[1],
        skeleton_tokens=skeleton_toks,
        model_fn=_model_fn_always(7),
    )
    assert isinstance(result, SoTResult)


# ===========================================================================
# 11. expand_all — total_tokens accounting
# ===========================================================================

def test_expand_all_total_tokens():
    """total_tokens == len(skeleton_tokens) + sum of expansion lengths."""
    decoder = _make_decoder(max_skeleton_tokens=6, max_point_tokens=3, max_points=4)
    skeleton_toks = [5, SEP, 6]  # two points

    result = decoder.expand_all(
        prompt_tokens=[1],
        skeleton_tokens=skeleton_toks,
        model_fn=_model_fn_always(7),
    )
    expected = len(skeleton_toks) + sum(len(p.expanded) for p in result.points)
    assert result.total_tokens == expected


# ===========================================================================
# 12. expand_all — points list length matches parse output
# ===========================================================================

def test_expand_all_points_count():
    decoder = _make_decoder(max_points=4)
    # Three points separated by two SEPs
    skeleton_toks = [3, SEP, 4, SEP, 5]
    result = decoder.expand_all(
        prompt_tokens=[1],
        skeleton_tokens=skeleton_toks,
        model_fn=_model_fn_always(7),
    )
    # parse_skeleton returns up to max_points segments
    expected_count = min(3, decoder.config.max_points)
    assert len(result.points) == expected_count


# ===========================================================================
# 13. speedup_estimate — always positive
# ===========================================================================

def test_speedup_estimate_positive():
    decoder = _make_decoder(max_point_tokens=10)
    # Build a dummy result
    pts = [SkeletonPoint(index=0, text=[1], expanded=[7, 8, 9]),
           SkeletonPoint(index=1, text=[2], expanded=[7, 7])]
    result = SoTResult(
        skeleton_tokens=[1, SEP, 2],
        points=pts,
        expanded_text="<7> <8> <9> <7> <7>",
        total_tokens=3 + 3 + 2,
    )
    speedup = decoder.speedup_estimate(result)
    assert speedup > 0


# ===========================================================================
# 14. speedup_estimate — single point gives meaningful value
# ===========================================================================

def test_speedup_single_point():
    """With one point and skeleton=0 tokens the speedup == 1 if total equals max_point_tokens."""
    max_pt = 20
    decoder = _make_decoder(max_point_tokens=max_pt)
    pts = [SkeletonPoint(index=0, text=[], expanded=list(range(max_pt)))]
    result = SoTResult(
        skeleton_tokens=[],
        points=pts,
        expanded_text=" ".join(f"<{t}>" for t in range(max_pt)),
        total_tokens=max_pt,
    )
    # total_tokens / max_point_tokens == 1.0
    assert decoder.speedup_estimate(result) == pytest.approx(1.0)


# ===========================================================================
# Integration test — full pipeline with stub model_fn
# ===========================================================================

def _integration_model_fn(context):
    """Stub model_fn for integration test.

    Emits:
      - SEP (10) for the first 3 tokens generated (to create skeleton points)
      - then token 7 indefinitely
    Called-token count is tracked via a mutable counter in a closure.
    """
    # We control what to return based on how many tokens context has grown
    # prompt=[1,2,3] → length 3; each generated token increments by 1
    # So for context lengths 3, 4, 5 emit SEP; afterwards emit 7
    L = len(context)
    if L <= 5:          # first two generated tokens during skeleton → SEP
        return _fixed_logits(SEP)
    return _fixed_logits(7)


def test_integration_expand_all():
    """Full SoT pipeline: skeleton → parse → expand → SoTResult."""
    cfg = SoTConfig(
        max_skeleton_tokens=6,
        max_point_tokens=4,
        max_points=4,
    )
    decoder = SkeletonOfThoughtDecoder(config=cfg, sep_token_id=SEP, eos_token_id=EOS)
    prompt = [1, 2, 3]

    # Generate skeleton
    skeleton = decoder.generate_skeleton(
        prompt_tokens=prompt,
        model_fn=_integration_model_fn,
        temperature=1.0,
    )
    assert isinstance(skeleton, list)
    assert len(skeleton) > 0

    # expand_all
    result = decoder.expand_all(
        prompt_tokens=prompt,
        skeleton_tokens=skeleton,
        model_fn=_integration_model_fn,
        temperature=1.0,
    )

    # Type checks
    assert isinstance(result, SoTResult)
    assert isinstance(result.skeleton_tokens, list)
    assert isinstance(result.points, list)
    assert isinstance(result.expanded_text, str)
    assert isinstance(result.total_tokens, int)

    # Structural consistency
    assert result.total_tokens == len(result.skeleton_tokens) + sum(
        len(p.expanded) for p in result.points
    )
    assert 1 <= len(result.points) <= cfg.max_points

    # Every point is a SkeletonPoint
    for pt in result.points:
        assert isinstance(pt, SkeletonPoint)
        assert isinstance(pt.expanded, list)

    # Speedup is positive
    assert decoder.speedup_estimate(result) > 0

    # Registry entry
    from src.inference import DECODER_REGISTRY
    assert "skeleton_of_thought" in DECODER_REGISTRY
