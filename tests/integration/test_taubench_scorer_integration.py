"""Integration test: TauBenchScorer is registered and runs end-to-end.

Verifies:
- BENCHMARK_REGISTRY and METRIC_REGISTRY contain "taubench"
- All prior registry entries remain intact
- AureliusConfig exposes enable_taubench_eval and taubench_domain
- End-to-end scoring on synthetic tasks produces valid metric structure
- Domain breakdown is populated correctly
- Efficiency score is non-negative
"""

from __future__ import annotations

import pytest

from src import eval as eval_pkg
from src.eval.taubench_scorer import (
    TauBenchDataset,
    TauBenchScorer,
    TauBenchTask,
    TauBenchTrajectory,
)
from src.model.config import AureliusConfig
from src.runtime.feature_flags import FEATURE_FLAG_REGISTRY, FeatureFlag

# ---------------------------------------------------------------------------
# Registry presence
# ---------------------------------------------------------------------------


def test_prior_registry_entries_intact():
    reg = eval_pkg.METRIC_REGISTRY
    expected_prior_keys = [
        "niah",
        "ruler",
        "humaneval",
        "mbpp",
        "swebench_lite",
        "ifeval",
        "mtbench",
        "alpacaeval",
        "arena_hard",
        "gpqa",
        "livecodebench",
        "mmlu",
        "humaneval_plus",
    ]
    for key in expected_prior_keys:
        assert key in reg, f"prior registry entry '{key}' missing after taubench registration"


def test_taubench_in_metric_registry():
    assert "taubench" in eval_pkg.METRIC_REGISTRY, "taubench not in METRIC_REGISTRY"


def test_taubench_in_benchmark_registry():
    assert "taubench" in eval_pkg.BENCHMARK_REGISTRY, "taubench not in BENCHMARK_REGISTRY"


def test_benchmark_registry_entry_is_scorer_class():
    entry = eval_pkg.BENCHMARK_REGISTRY["taubench"]
    assert entry is TauBenchScorer, (
        f"BENCHMARK_REGISTRY['taubench'] should be TauBenchScorer, got {entry}"
    )


def test_metric_registry_entry_is_scorer_class():
    entry = eval_pkg.METRIC_REGISTRY["taubench"]
    assert entry is TauBenchScorer, (
        f"METRIC_REGISTRY['taubench'] should be TauBenchScorer, got {entry}"
    )


# ---------------------------------------------------------------------------
# AureliusConfig integration
# ---------------------------------------------------------------------------


def test_config_has_taubench_fields():
    cfg = AureliusConfig()
    assert hasattr(cfg, "enable_taubench_eval"), "AureliusConfig missing enable_taubench_eval"
    assert hasattr(cfg, "taubench_domain"), "AureliusConfig missing taubench_domain"


def test_config_defaults():
    cfg = AureliusConfig()
    assert cfg.enable_taubench_eval is False, "enable_taubench_eval should default to False"
    assert cfg.taubench_domain == "coding", "taubench_domain should default to 'coding'"


def test_config_fields_can_be_set():
    FEATURE_FLAG_REGISTRY.register(
        FeatureFlag(name="eval.taubench", enabled=True, metadata={"domain": "retail"})
    )
    cfg = AureliusConfig()
    assert cfg.enable_taubench_eval is True
    assert cfg.taubench_domain == "retail"


# ---------------------------------------------------------------------------
# End-to-end scoring on synthetic tasks
# ---------------------------------------------------------------------------


def test_end_to_end_coding_domain():
    FEATURE_FLAG_REGISTRY.register(
        FeatureFlag(name="eval.taubench", enabled=True, metadata={"domain": "coding"})
    )
    cfg = AureliusConfig()
    tasks = TauBenchDataset.get_sample_tasks(domain=cfg.taubench_domain, n=5)
    trajectories = [TauBenchDataset.make_successful_trajectory(t) for t in tasks]

    scorer = TauBenchScorer()
    metrics = scorer.score_batch(tasks, trajectories)

    # Metric structure
    assert "success_rate" in metrics
    assert "avg_turns" in metrics
    assert "tool_accuracy" in metrics
    assert "efficiency_score" in metrics
    assert "domain_breakdown" in metrics
    assert "per_task" in metrics
    assert "n_tasks" in metrics

    # Values
    assert metrics["n_tasks"] == 5
    assert metrics["success_rate"] == pytest.approx(1.0)
    assert metrics["avg_turns"] > 0
    assert 0.0 <= metrics["tool_accuracy"] <= 1.0
    assert metrics["efficiency_score"] >= 0.0

    # Domain breakdown
    assert "coding" in metrics["domain_breakdown"]
    coding_bd = metrics["domain_breakdown"]["coding"]
    assert coding_bd["success_rate"] == pytest.approx(1.0)
    assert coding_bd["n"] == 5


def test_end_to_end_mixed_success():
    tasks = TauBenchDataset.get_sample_tasks(domain="coding", n=4)
    # Alternate success/failure
    trajectories = []
    for i, task in enumerate(tasks):
        if i % 2 == 0:
            traj = TauBenchDataset.make_successful_trajectory(task)
        else:
            traj = TauBenchTrajectory(
                task_id=task.task_id,
                turns=[],
                final_state={},
                success=False,
                num_turns=0,
            )
        trajectories.append(traj)

    scorer = TauBenchScorer()
    metrics = scorer.score_batch(tasks, trajectories)

    assert metrics["success_rate"] == pytest.approx(0.5)
    assert metrics["efficiency_score"] >= 0.0


def test_end_to_end_retail_domain():
    FEATURE_FLAG_REGISTRY.register(
        FeatureFlag(name="eval.taubench", enabled=True, metadata={"domain": "retail"})
    )
    cfg = AureliusConfig()
    tasks = TauBenchDataset.get_sample_tasks(domain=cfg.taubench_domain, n=3)
    trajectories = [TauBenchDataset.make_successful_trajectory(t) for t in tasks]

    scorer = TauBenchScorer()
    metrics = scorer.score_batch(tasks, trajectories)

    assert metrics["n_tasks"] == 3
    assert "retail" in metrics["domain_breakdown"]


def test_scorer_instantiation_from_registry():
    """Construct scorer via registry (as production code would)."""
    scorer_cls = eval_pkg.BENCHMARK_REGISTRY["taubench"]
    scorer = scorer_cls()
    assert isinstance(scorer, TauBenchScorer)

    tasks = TauBenchDataset.get_sample_tasks(n=2)
    trajs = [TauBenchDataset.make_successful_trajectory(t) for t in tasks]
    result = scorer.score_batch(tasks, trajs)
    assert result["success_rate"] == pytest.approx(1.0)


def test_per_task_results_shape():
    tasks = TauBenchDataset.get_sample_tasks(n=3)
    trajs = [TauBenchDataset.make_successful_trajectory(t) for t in tasks]
    scorer = TauBenchScorer()
    metrics = scorer.score_batch(tasks, trajs)

    assert len(metrics["per_task"]) == 3
    for item in metrics["per_task"]:
        assert "success" in item
        assert "turns" in item
        assert "tool_accuracy" in item
        assert "partial_credit" in item
        assert isinstance(item["success"], bool)
        assert isinstance(item["turns"], int)
        assert 0.0 <= item["tool_accuracy"] <= 1.0
        assert 0.0 <= item["partial_credit"] <= 1.0


def test_adversarial_malformed_trajectories_no_crash():
    """Ensure scorer never crashes on malformed inputs."""
    task = TauBenchTask(
        task_id="bad-001",
        instruction="Do something.",
        tools=[],
        expected_actions=[],
        success_criteria={},
        domain="coding",
    )
    bad_traj = TauBenchTrajectory(
        task_id="bad-001",
        turns=[
            {"role": "assistant"},  # missing tool_calls key
            {"role": "tool", "tool_calls": None},  # None tool_calls
            {"role": "user", "tool_calls": 42},  # wrong type
        ],
        final_state={},
        success=False,
        num_turns=3,
    )
    scorer = TauBenchScorer()
    result = scorer.score_trajectory(task, bad_traj)
    assert isinstance(result, dict)
    assert isinstance(result["success"], bool)
