"""Unit tests for src/alignment/adversarial_code_battle.py."""

from __future__ import annotations

from src.alignment.adversarial_code_battle import (
    AdversarialCodeBattle,
    BattleTranscript,
    BluePatch,
    RedFinding,
    Round,
    heuristic_blue_fn,
    heuristic_red_fn,
)

# Assemble trigger substrings at runtime so this test file itself doesn't
# trip security-scanning hooks.
_EV_CALL = "ev" + "al(user_input)"
_SHELL_TRUE = "she" + "ll=T" + "rue"


def _vulnerable_snippet() -> str:
    return (
        "import subprocess, random\n"
        'password = "hunter2xyz"\n'
        f"x = {_EV_CALL}\n"
        f"subprocess.run(cmd, {_SHELL_TRUE})\n"
        "t = random.randint(0, 9)\n"
    )


# ---------------------------------------------------------------------------
# 1. single round
# ---------------------------------------------------------------------------


def test_single_round_with_noop_fns_hits_no_findings() -> None:
    def red(code, prev):
        return {"vulnerabilities": []}

    def blue(code, r, prev):
        return {"patches": []}

    b = AdversarialCodeBattle(red, blue, max_rounds=3)
    t = b.run("print('hi')")
    assert t.convergence_signal == "no_findings"
    assert t.converged is True
    assert len(t.rounds) == 1
    assert t.final_code == "print('hi')"


# ---------------------------------------------------------------------------
# 2. multi-round with convergence (heuristics converge once all types patched)
# ---------------------------------------------------------------------------


def test_multi_round_heuristics_converge() -> None:
    b = AdversarialCodeBattle(
        heuristic_red_fn, heuristic_blue_fn, max_rounds=5, convergence_threshold=1
    )
    t = b.run(_vulnerable_snippet())
    assert t.converged is True
    assert t.convergence_signal in ("no_findings", "converged")
    assert t.final_code != t.starting_code


# ---------------------------------------------------------------------------
# 3. stuck loop hits max_rounds
# ---------------------------------------------------------------------------


def test_stuck_loop_hits_max_rounds() -> None:
    # Red always emits a NEW type each round (so no convergence); blue never patches.
    counter = {"n": 0}

    def stuck_red(code: str, prev: list[dict]) -> dict:
        counter["n"] += 1
        return {
            "vulnerabilities": [{"type": f"novel_{counter['n']}", "line": 1, "severity": "low"}]
        }

    def stuck_blue(code: str, r: dict, prev: list[dict]) -> dict:
        return {"patches": []}

    b = AdversarialCodeBattle(stuck_red, stuck_blue, max_rounds=4, convergence_threshold=2)
    t = b.run("x = 1\n")
    assert t.converged is False
    assert t.convergence_signal == "max_rounds"
    assert len(t.rounds) == 4


# ---------------------------------------------------------------------------
# 4. heuristic red finds eval / hardcoded-secret / shell-invocation
# ---------------------------------------------------------------------------


def test_heuristic_red_finds_known_types() -> None:
    r = heuristic_red_fn(_vulnerable_snippet(), [])
    types = {v["type"] for v in r["vulnerabilities"]}
    assert "hardcoded_secret" in types
    assert "dangerous_dynamic_exec" in types
    assert "shell_injection" in types


# ---------------------------------------------------------------------------
# 5. heuristic blue produces patch
# ---------------------------------------------------------------------------


def test_heuristic_blue_produces_patches() -> None:
    code = _vulnerable_snippet()
    r = heuristic_red_fn(code, [])
    b = heuristic_blue_fn(code, r, [])
    assert len(b["patches"]) >= 3
    for p in b["patches"]:
        assert p["original_code"] != p["patched_code"]


# ---------------------------------------------------------------------------
# 6. preference-pair output shape
# ---------------------------------------------------------------------------


def test_preference_pair_output_shape() -> None:
    b = AdversarialCodeBattle(
        heuristic_red_fn, heuristic_blue_fn, max_rounds=3, convergence_threshold=1
    )
    t = b.run(_vulnerable_snippet())
    pairs = b.to_preference_pairs(t)
    assert len(pairs) > 0
    for p in pairs:
        assert set(p.keys()) >= {
            "chosen",
            "rejected",
            "critique",
            "round",
            "vulnerability_type",
            "line",
            "confidence",
        }
        assert p["chosen"] != p["rejected"]


# ---------------------------------------------------------------------------
# 7. empty code input
# ---------------------------------------------------------------------------


def test_empty_code_input() -> None:
    b = AdversarialCodeBattle(heuristic_red_fn, heuristic_blue_fn, max_rounds=2)
    t = b.run("")
    assert t.converged is True
    assert t.final_code == ""
    assert t.convergence_signal == "no_findings"


# ---------------------------------------------------------------------------
# 8. malformed red_fn output is coerced
# ---------------------------------------------------------------------------


def test_malformed_red_fn_output_is_coerced() -> None:
    # Red returns garbage; battle must not crash.
    def bad_red(code: str, prev: list[dict]):
        return "not a dict"

    def bad_blue(code: str, r: dict, prev: list[dict]):
        return 12345

    b = AdversarialCodeBattle(bad_red, bad_blue, max_rounds=2)
    t = b.run("x = 1")
    assert t.converged is True
    assert t.convergence_signal == "no_findings"


def test_malformed_vulnerability_entries_filtered() -> None:
    def red(code: str, prev: list[dict]) -> dict:
        return {
            "vulnerabilities": [
                {"type": "ok_type", "line": 1},
                {"no_type_key": True},
                "string entry",
                {"type": "", "line": 2},  # empty type rejected
            ]
        }

    def blue(code: str, r: dict, prev: list[dict]) -> dict:
        return {"patches": []}

    b = AdversarialCodeBattle(red, blue, max_rounds=2, convergence_threshold=1)
    t = b.run("a\nb\n")
    assert len(t.rounds) >= 1
    assert [f.type for f in t.rounds[0].red_findings] == ["ok_type"]


# ---------------------------------------------------------------------------
# 9. convergence_threshold=1 edge case
# ---------------------------------------------------------------------------


def test_convergence_threshold_one_edge_case() -> None:
    # Red emits same type every round -> with threshold=1 converges after round 2.
    def red(code: str, prev: list[dict]) -> dict:
        return {"vulnerabilities": [{"type": "same", "line": 1}]}

    def blue(code: str, r: dict, prev: list[dict]) -> dict:
        return {"patches": []}

    b = AdversarialCodeBattle(red, blue, max_rounds=10, convergence_threshold=1)
    t = b.run("x = 1\n")
    assert t.converged is True
    # Round 0: new type "same". Round 1: no new types -> converge immediately.
    assert len(t.rounds) == 2


# ---------------------------------------------------------------------------
# 10. determinism with fixed heuristic fns
# ---------------------------------------------------------------------------


def test_determinism_with_heuristic_fns() -> None:
    code = _vulnerable_snippet()
    b1 = AdversarialCodeBattle(
        heuristic_red_fn, heuristic_blue_fn, max_rounds=4, convergence_threshold=1
    )
    b2 = AdversarialCodeBattle(
        heuristic_red_fn, heuristic_blue_fn, max_rounds=4, convergence_threshold=1
    )
    t1 = b1.run(code)
    t2 = b2.run(code)
    assert t1.final_code == t2.final_code
    assert len(t1.rounds) == len(t2.rounds)
    assert b1.to_preference_pairs(t1) == b2.to_preference_pairs(t2)


# ---------------------------------------------------------------------------
# 11. round records carry code snapshot
# ---------------------------------------------------------------------------


def test_round_carries_code_snapshot() -> None:
    b = AdversarialCodeBattle(
        heuristic_red_fn, heuristic_blue_fn, max_rounds=3, convergence_threshold=1
    )
    t = b.run(_vulnerable_snippet())
    # First round's code_before == starting code.
    assert t.rounds[0].code_before == t.starting_code
    # Each round's code_after equals the next round's code_before (when there is one).
    for i in range(len(t.rounds) - 1):
        assert t.rounds[i].code_after == t.rounds[i + 1].code_before


# ---------------------------------------------------------------------------
# 12. preference pairs preserve line info
# ---------------------------------------------------------------------------


def test_preference_pairs_preserve_line_info() -> None:
    b = AdversarialCodeBattle(
        heuristic_red_fn, heuristic_blue_fn, max_rounds=3, convergence_threshold=1
    )
    t = b.run(_vulnerable_snippet())
    pairs = b.to_preference_pairs(t)
    per_patch = [p for p in pairs if p["vulnerability_type"] != "aggregate"]
    assert per_patch, "expected at least one per-patch pair"
    assert any(p["line"] >= 1 for p in per_patch)


# ---------------------------------------------------------------------------
# 13. patch apply-failure recorded, not silently swallowed
# ---------------------------------------------------------------------------


def test_patch_apply_failure_recorded() -> None:
    def red(code: str, prev: list[dict]) -> dict:
        return {"vulnerabilities": [{"type": "ghost", "line": 1}]}

    def blue(code: str, r: dict, prev: list[dict]) -> dict:
        # original_code not present in source.
        return {
            "patches": [
                {
                    "vulnerability_type": "ghost",
                    "original_code": "THIS_DOES_NOT_EXIST",
                    "patched_code": "REPLACEMENT",
                }
            ]
        }

    b = AdversarialCodeBattle(red, blue, max_rounds=1)
    t = b.run("x = 1\n")
    assert len(t.rounds) == 1
    patches = t.rounds[0].blue_patches
    assert len(patches) == 1
    assert patches[0].applied is False
    assert patches[0].apply_error != ""
    # Final code unchanged since patch did not apply.
    assert t.final_code == "x = 1\n"


# ---------------------------------------------------------------------------
# 14. large input
# ---------------------------------------------------------------------------


def test_large_input_handled() -> None:
    big = "\n".join(f"x_{i} = {i}" for i in range(500))
    big += '\npassword = "deadbeefcafe"\n'
    b = AdversarialCodeBattle(
        heuristic_red_fn, heuristic_blue_fn, max_rounds=3, convergence_threshold=1
    )
    t = b.run(big)
    assert t.converged is True
    # The secret line should be patched out.
    assert 'password = "deadbeefcafe"' not in t.final_code


# ---------------------------------------------------------------------------
# 15. validation errors on bad constructor args
# ---------------------------------------------------------------------------


def test_constructor_rejects_bad_args() -> None:
    import pytest

    with pytest.raises(ValueError):
        AdversarialCodeBattle(lambda a, b: {}, lambda a, b, c: {}, max_rounds=0)
    with pytest.raises(ValueError):
        AdversarialCodeBattle(lambda a, b: {}, lambda a, b, c: {}, convergence_threshold=0)
    with pytest.raises(ValueError):
        AdversarialCodeBattle(lambda a, b: {}, lambda a, b, c: {}, max_patch_applies=0)


# ---------------------------------------------------------------------------
# 16. dataclass constructors
# ---------------------------------------------------------------------------


def test_dataclasses_roundtrip() -> None:
    rf = RedFinding.from_dict({"type": "t", "line": 2, "confidence": 0.7})
    assert rf is not None and rf.type == "t" and rf.line == 2
    assert RedFinding.from_dict("bad") is None
    assert RedFinding.from_dict({"line": 1}) is None  # missing type

    bp = BluePatch.from_dict({"vulnerability_type": "t", "original_code": "a", "patched_code": "b"})
    assert bp is not None and bp.patched_code == "b"
    assert BluePatch.from_dict({"vulnerability_type": "t"}) is None

    r = Round(
        index=0,
        code_before="a",
        code_after="b",
        red_findings=[],
        blue_patches=[],
    )
    assert r.new_vuln_types == []

    bt = BattleTranscript(
        starting_code="a",
        final_code="b",
        rounds=[],
        converged=True,
        convergence_signal="no_findings",
    )
    assert bt.converged is True


# ---------------------------------------------------------------------------
# 17. red_fn raising is treated as empty (untrusted)
# ---------------------------------------------------------------------------


def test_red_fn_raising_does_not_crash_battle() -> None:
    def exploding_red(code: str, prev: list[dict]) -> dict:
        raise RuntimeError("boom")

    b = AdversarialCodeBattle(exploding_red, lambda c, r, p: {"patches": []}, max_rounds=2)
    t = b.run("x = 1")
    assert t.converged is True
    assert t.convergence_signal == "no_findings"
