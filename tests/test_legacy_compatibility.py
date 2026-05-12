"""Regression tests for legacy Aurelius compatibility surfaces."""

from __future__ import annotations

from importlib import import_module


def test_legacy_registry_snapshot_shape() -> None:
    api_registry = import_module("aurelius.api_registry")
    snapshot = api_registry.get_registry_snapshot()

    assert len(snapshot["agents"]) == 22
    assert len(snapshot["skills"]) == 36
    assert snapshot["agent_categories"] == [
        "coding",
        "research",
        "devops",
        "communication",
        "creative",
        "education",
        "data",
        "productivity",
    ]
    assert snapshot["skill_categories"] == [
        "coding",
        "research",
        "devops",
        "communication",
        "creative",
        "education",
        "data",
        "productivity",
        "meta",
    ]
    assert any(agent["id"] == "coding" for agent in snapshot["agents"])
    assert any(agent["id"] == "security" for agent in snapshot["agents"])
    assert any(skill["id"] == "workflow_automation" for skill in snapshot["skills"])
    assert any(skill["id"] == "prompt_engineering" for skill in snapshot["skills"])


def test_legacy_neural_brain_surface() -> None:
    neural_brain = import_module("aurelius.neural_brain")

    brain = neural_brain.NeuralBrain()
    ctx = brain.run("Hello Aurelius compatibility layer")

    assert ctx.state.startswith("ready:")
    assert len(ctx.plan) == 3
    assert len(ctx.reasoning) == 3
    assert len(ctx.actions) == 3
    assert len(ctx.verifications) == 2
    assert len(ctx.reflections) == 2
    assert "Processed" in ctx.output
    assert brain.get_stats()["runs"] == 1


def test_legacy_self_upgrade_surface() -> None:
    self_upgrade = import_module("aurelius.self_upgrade")

    upgrader = self_upgrade.SelfUpgradeSystem()
    upgrader.record_metric("eval_accuracy", 0.72, target=0.90)
    upgrader.record_metric("training_loss", 2.1, target=1.5)
    summary = upgrader.run_upgrade_cycle()

    assert summary["status"] == "completed"
    assert "improvement" in summary
    assert upgrader.get_summary()["cycle_count"] == 1
