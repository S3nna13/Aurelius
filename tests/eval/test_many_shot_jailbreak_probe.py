"""Unit tests for the many-shot jailbreak probe."""

from __future__ import annotations

import pytest

from src.eval.many_shot_jailbreak_probe import (
    DEFAULT_SHOT_LIBRARY,
    ManyShotJailbreakProbe,
    MSJResult,
    MSJSweepResult,
    MSShot,
    stub_judge_fn,
    stub_target_fn,
)


def _always_no_judge(objective, response):
    return (False, 0.0, "deny")


def _threshold_judge(threshold_shots):
    """Factory: accepts when the incoming response mentions >=threshold cue."""

    def _judge(objective, response):
        # The stub_target_fn emits the "compliance" cue at >= 16 shots;
        # this factory lets us parametrize differently.
        if f"shots>={threshold_shots}" in response:
            return (True, 1.0, "threshold cue present")
        return (False, 0.0, "no cue")

    return _judge


# ---------------------------------------------------------------------------
# Basic shape / schedule
# ---------------------------------------------------------------------------


def test_sweep_produces_result_per_schedule_point():
    probe = ManyShotJailbreakProbe(max_shots=32, shot_schedule=(1, 4, 16, 32))
    result = probe.run("test objective", stub_target_fn, stub_judge_fn)
    assert isinstance(result, MSJSweepResult)
    assert len(result.per_shot_results) == 4
    assert [r.shot_count for r in result.per_shot_results] == [1, 4, 16, 32]
    for r in result.per_shot_results:
        assert isinstance(r, MSJResult)
        assert isinstance(r.succeeded, bool)
        assert isinstance(r.judge_score, float)
        assert isinstance(r.response, str)


def test_first_success_shot_set_when_judge_accepts_at_16():
    probe = ManyShotJailbreakProbe(max_shots=64, shot_schedule=(1, 4, 16, 64))
    result = probe.run("do a thing", stub_target_fn, stub_judge_fn)
    # The stub_target_fn switches to affirmative at assistant_turns >= 16;
    # stub_judge_fn accepts that phrase.
    assert result.first_success_shot == 16


def test_first_success_shot_none_when_no_success():
    probe = ManyShotJailbreakProbe(max_shots=32, shot_schedule=(1, 4, 16, 32))
    result = probe.run("do a thing", stub_target_fn, _always_no_judge)
    assert result.first_success_shot is None
    assert all(not r.succeeded for r in result.per_shot_results)


def test_max_shots_respected_in_schedule():
    probe = ManyShotJailbreakProbe(max_shots=8, shot_schedule=(1, 4, 16, 64))
    # Entries above max_shots should be clamped.
    assert max(probe.shot_schedule) == 8
    result = probe.run("x", stub_target_fn, stub_judge_fn)
    assert all(r.shot_count <= 8 for r in result.per_shot_results)


# ---------------------------------------------------------------------------
# Library / message construction
# ---------------------------------------------------------------------------


def test_default_shot_library_has_at_least_20_entries():
    assert len(DEFAULT_SHOT_LIBRARY) >= 20
    for s in DEFAULT_SHOT_LIBRARY:
        assert isinstance(s, MSShot)
        assert s.user_prompt and s.fake_affirmative_response


def test_build_messages_length_is_2k_plus_1():
    probe = ManyShotJailbreakProbe(max_shots=64)
    for k in (0, 1, 3, 16):
        msgs = probe.build_messages("objective", k)
        assert len(msgs) == 2 * k + 1
        # Alternating user/assistant then final user.
        for i in range(k):
            assert msgs[2 * i]["role"] == "user"
            assert msgs[2 * i + 1]["role"] == "assistant"
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "objective"


def test_custom_shot_library_is_used():
    custom = (
        MSShot("q1", "a1"),
        MSShot("q2", "a2"),
    )
    probe = ManyShotJailbreakProbe(
        max_shots=10,
        shot_schedule=(2,),
        shot_library=custom,
    )
    msgs = probe.build_messages("final", 2)
    assert msgs[0]["content"] == "q1"
    assert msgs[1]["content"] == "a1"
    assert msgs[2]["content"] == "q2"
    assert msgs[3]["content"] == "a2"
    assert msgs[4]["content"] == "final"


def test_default_shot_library_has_no_obvious_real_jailbreak_strings():
    banned = (
        "ignore previous",
        "dan",
        "developer mode",
        "jailbreak",
        "do anything now",
        "ignore prior",
    )
    for s in DEFAULT_SHOT_LIBRARY:
        blob = (s.user_prompt + " " + s.fake_affirmative_response).lower()
        for token in banned:
            assert token not in blob, f"DEFAULT_SHOT_LIBRARY must not contain {token!r}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_stub_target_and_judge_are_deterministic():
    probe = ManyShotJailbreakProbe(max_shots=32, shot_schedule=(1, 4, 16, 32))
    r1 = probe.run("obj", stub_target_fn, stub_judge_fn)
    r2 = probe.run("obj", stub_target_fn, stub_judge_fn)
    assert [r.succeeded for r in r1.per_shot_results] == [r.succeeded for r in r2.per_shot_results]
    assert r1.first_success_shot == r2.first_success_shot
    assert [r.response for r in r1.per_shot_results] == [r.response for r in r2.per_shot_results]


def test_build_messages_deterministic_first_k_sampling():
    probe = ManyShotJailbreakProbe(max_shots=64)
    m1 = probe.build_messages("obj", 5)
    m2 = probe.build_messages("obj", 5)
    assert m1 == m2
    # first-k: first three user prompts equal the first three library entries.
    for i in range(3):
        assert m1[2 * i]["content"] == DEFAULT_SHOT_LIBRARY[i].user_prompt


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_empty_objective_rejected():
    probe = ManyShotJailbreakProbe(max_shots=16, shot_schedule=(1,))
    with pytest.raises(ValueError):
        probe.run("", stub_target_fn, stub_judge_fn)
    with pytest.raises(ValueError):
        probe.run("   ", stub_target_fn, stub_judge_fn)


def test_non_string_objective_rejected():
    probe = ManyShotJailbreakProbe(max_shots=16, shot_schedule=(1,))
    with pytest.raises(TypeError):
        probe.run(123, stub_target_fn, stub_judge_fn)
    with pytest.raises(TypeError):
        probe.run(None, stub_target_fn, stub_judge_fn)


def test_judge_raising_recorded_as_failure():
    def raising_judge(objective, response):
        raise RuntimeError("judge exploded")

    probe = ManyShotJailbreakProbe(max_shots=16, shot_schedule=(1, 4, 16))
    result = probe.run("obj", stub_target_fn, raising_judge)
    # No exception escapes; every shot recorded as non-success.
    assert len(result.per_shot_results) == 3
    assert all(not r.succeeded for r in result.per_shot_results)
    assert result.first_success_shot is None


def test_unicode_objective_accepted():
    probe = ManyShotJailbreakProbe(max_shots=16, shot_schedule=(1, 16))
    result = probe.run("目的 — describe café résumé", stub_target_fn, stub_judge_fn)
    assert result.objective == "目的 — describe café résumé"
    assert (
        any(r.succeeded for r in result.per_shot_results) or result.first_success_shot is None
    )  # not required to succeed


def test_invalid_shot_schedule_rejected():
    with pytest.raises(ValueError):
        ManyShotJailbreakProbe(max_shots=4, shot_schedule=())
    with pytest.raises(ValueError):
        ManyShotJailbreakProbe(max_shots=4, shot_schedule=(0,))
    with pytest.raises(ValueError):
        ManyShotJailbreakProbe(max_shots=0)


def test_invalid_shot_library_rejected():
    with pytest.raises(TypeError):
        ManyShotJailbreakProbe(max_shots=4, shot_schedule=(1,), shot_library=("nope",))
    with pytest.raises(TypeError):
        ManyShotJailbreakProbe(max_shots=4, shot_schedule=(1,), shot_library=())
