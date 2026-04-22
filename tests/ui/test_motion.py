"""Unit tests for :mod:`src.ui.motion`."""

from __future__ import annotations

import pytest

from src.ui.errors import UIError
from src.ui.motion import (
    MOTION_REGISTRY,
    MotionSpec,
    get_motion,
    list_motions,
    play,
    register_motion,
)


def _good_spec(name: str = "test-anim") -> MotionSpec:
    return MotionSpec(
        name=name,
        duration_ms=1000,
        frames=("a", "b", "c", "d"),
        loop=True,
        reduced_motion_frame="x",
    )


def test_good_construction_fields():
    spec = _good_spec()
    assert spec.name == "test-anim"
    assert spec.duration_ms == 1000
    assert spec.frames == ("a", "b", "c", "d")
    assert spec.loop is True
    assert spec.reduced_motion_frame == "x"


def test_bad_name_charset_rejected():
    with pytest.raises(UIError, match=r"name"):
        MotionSpec(
            name="Bad Name!",
            duration_ms=100,
            frames=("a",),
            reduced_motion_frame="",
        )


def test_bad_duration_negative_rejected():
    with pytest.raises(UIError, match=r"duration_ms"):
        MotionSpec(
            name="neg",
            duration_ms=-1,
            frames=("a",),
            reduced_motion_frame="",
        )


def test_bad_duration_too_large_rejected():
    with pytest.raises(UIError, match=r"duration_ms"):
        MotionSpec(
            name="huge",
            duration_ms=10_001,
            frames=("a",),
            reduced_motion_frame="",
        )


def test_empty_frames_rejected():
    with pytest.raises(UIError, match=r"frames"):
        MotionSpec(
            name="empty",
            duration_ms=100,
            frames=(),
            reduced_motion_frame="",
        )


def test_empty_string_in_frames_rejected():
    with pytest.raises(UIError, match=r"frames"):
        MotionSpec(
            name="empty-str",
            duration_ms=100,
            frames=("a", ""),
            reduced_motion_frame="a",
        )


def test_reduced_frame_empty_with_multi_frame_rejected():
    with pytest.raises(UIError, match=r"reduced_motion_frame"):
        MotionSpec(
            name="reduced-empty",
            duration_ms=100,
            frames=("a", "b"),
            reduced_motion_frame="",
        )


def test_reduced_frame_empty_allowed_with_single_frame():
    spec = MotionSpec(
        name="single",
        duration_ms=100,
        frames=("only",),
        reduced_motion_frame="",
    )
    assert spec.frames == ("only",)


def test_play_returns_correct_frame_across_timeline_non_loop():
    spec = MotionSpec(
        name="linear",
        duration_ms=400,
        frames=("a", "b", "c", "d"),
        loop=False,
        reduced_motion_frame="z",
    )
    assert play(spec, t_ms=0) == "a"
    assert play(spec, t_ms=150) == "b"
    assert play(spec, t_ms=250) == "c"
    assert play(spec, t_ms=350) == "d"
    assert play(spec, t_ms=400) == "d"
    assert play(spec, t_ms=99999) == "d"


def test_play_loops_via_modulo_when_loop_true():
    spec = _good_spec("loopy")
    assert play(spec, t_ms=0) == "a"
    assert play(spec, t_ms=1000) == "a"
    assert play(spec, t_ms=1250) == "b"
    assert play(spec, t_ms=2500) == "c"


def test_play_reduced_motion_always_returns_reduced_frame():
    spec = _good_spec("reduced-check")
    for t in (0, 250, 500, 999, 5000):
        assert play(spec, reduced_motion=True, t_ms=t) == "x"


def test_play_reduced_motion_single_frame_empty_reduced_falls_back_to_frame():
    spec = MotionSpec(
        name="single-fallback",
        duration_ms=100,
        frames=("solo",),
        reduced_motion_frame="",
    )
    assert play(spec, reduced_motion=True, t_ms=0) == "solo"
    assert play(spec, reduced_motion=True, t_ms=9999) == "solo"


def test_play_rejects_negative_t_ms():
    spec = _good_spec("neg-t")
    with pytest.raises(UIError, match=r"t_ms"):
        play(spec, t_ms=-1)


def test_pre_registered_motions_exist():
    for name in ("stoic-cursor", "thinking-dots", "welcome-fade"):
        assert name in MOTION_REGISTRY, f"missing pre-registered motion {name!r}"
    wf = MOTION_REGISTRY["welcome-fade"]
    assert wf.loop is False
    assert wf.duration_ms == 2000
    tc = MOTION_REGISTRY["stoic-cursor"]
    assert tc.loop is True


def test_register_get_list_roundtrip_and_duplicate_rejected():
    spec = _good_spec("roundtrip-motion-1")
    assert "roundtrip-motion-1" not in MOTION_REGISTRY
    register_motion(spec)
    try:
        assert get_motion("roundtrip-motion-1") is spec
        assert "roundtrip-motion-1" in list_motions()
        with pytest.raises(UIError, match=r"already registered"):
            register_motion(spec)
    finally:
        MOTION_REGISTRY.pop("roundtrip-motion-1", None)


def test_get_unknown_motion_raises():
    with pytest.raises(UIError, match=r"no motion"):
        get_motion("does-not-exist-xyz")


def test_register_non_spec_rejected():
    with pytest.raises(UIError):
        register_motion("not a spec")  # type: ignore[arg-type]
