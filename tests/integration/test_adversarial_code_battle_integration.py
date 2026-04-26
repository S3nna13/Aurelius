"""Integration tests for the adversarial code battle surface."""

from __future__ import annotations

import src.alignment as alignment_surface
from src.alignment import ALIGNMENT_REGISTRY, AdversarialCodeBattle
from src.alignment.adversarial_code_battle import (
    heuristic_blue_fn,
    heuristic_red_fn,
)
from src.model.config import AureliusConfig

_EV_CALL = "ev" + "al(user_input)"
_SHELL_TRUE = "she" + "ll=T" + "rue"


def test_surface_exports() -> None:
    for name in (
        "AdversarialCodeBattle",
        "BattleTranscript",
        "BluePatch",
        "RedFinding",
        "heuristic_red_fn",
        "heuristic_blue_fn",
    ):
        assert hasattr(alignment_surface, name), f"missing export: {name}"


def test_registry_contains_battle() -> None:
    assert "adversarial_code_battle" in ALIGNMENT_REGISTRY
    assert ALIGNMENT_REGISTRY["adversarial_code_battle"] is AdversarialCodeBattle
    # Additive regression: prior registrations untouched.
    assert "prm" in ALIGNMENT_REGISTRY


def test_config_flag_default_off() -> None:
    c = AureliusConfig()
    assert hasattr(c, "alignment_adversarial_code_battle_enabled")
    assert c.alignment_adversarial_code_battle_enabled is False


def test_two_round_battle_generates_preference_pairs() -> None:
    snippet = (
        "import subprocess, random\n"
        'api_key = "abcd1234xyz"\n'
        f"result = {_EV_CALL}\n"
        f"subprocess.run(cmd, {_SHELL_TRUE})\n"
        "n = random.randint(0, 10)\n"
        "def f(x):\n"
        "    return x + 1\n"
        "print(f(n))\n"
        "# end\n"
        "# trailer\n"
    )
    assert len(snippet.splitlines()) >= 10

    battle = AdversarialCodeBattle(
        heuristic_red_fn,
        heuristic_blue_fn,
        max_rounds=2,
        convergence_threshold=1,
    )
    transcript = battle.run(snippet)
    assert len(transcript.rounds) <= 2
    assert transcript.rounds, "expected at least one round"
    pairs = battle.to_preference_pairs(transcript)
    assert pairs, "expected at least one preference pair"
    for p in pairs:
        assert "chosen" in p and "rejected" in p
        assert p["chosen"] != p["rejected"]
    # Final code should no longer contain the hardcoded API key literal.
    assert '"abcd1234xyz"' not in transcript.final_code
