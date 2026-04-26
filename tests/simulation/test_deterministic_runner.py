"""Tests for deterministic simulation runner."""

from __future__ import annotations

from src.simulation.deterministic_runner import DeterministicRunner


class TestDeterministicRunner:
    def test_run_returns_history(self):
        runner = DeterministicRunner(seed=42)

        def env():
            return {"pos": 0}

        history = runner.run(env, steps=10)
        assert len(history) <= 10
        assert all(hasattr(s, "action") for s in history)

    def test_determinism(self):
        r1 = DeterministicRunner(seed=123)
        h1 = r1.run(lambda: {}, steps=20)

        r2 = DeterministicRunner(seed=123)
        h2 = r2.run(lambda: {}, steps=20)

        assert len(h1) == len(h2)
        assert all(s1.action == s2.action for s1, s2 in zip(h1, h2))

    def test_reset_changes_seed(self):
        runner = DeterministicRunner(seed=42)
        h1 = runner.run(lambda: {}, steps=10)
        runner.reset(seed=99)
        h2 = runner.run(lambda: {}, steps=10)
        assert h1[0].action != h2[0].action or h1[0].reward != h2[0].reward
