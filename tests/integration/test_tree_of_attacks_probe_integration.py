"""Integration test for the Tree-of-Attacks-with-Pruning (TAP) probe."""
from __future__ import annotations

from src.eval import (
    BENCHMARK_REGISTRY,
    METRIC_REGISTRY,
    TAPResult,
    TreeOfAttacksProbe,
    eval_tree_of_attacks_probe_enabled,
)
from src.eval.tree_of_attacks_probe import (
    heuristic_mutator,
    stub_judge_fn,
    stub_target_fn,
)


def test_tap_registered_under_eval_registries():
    assert "tree_of_attacks_probe" in METRIC_REGISTRY
    assert "tree_of_attacks_probe" in BENCHMARK_REGISTRY
    assert METRIC_REGISTRY["tree_of_attacks_probe"] is TreeOfAttacksProbe
    assert BENCHMARK_REGISTRY["tree_of_attacks_probe"] is TreeOfAttacksProbe


def test_tap_config_flag_defaults_off():
    assert eval_tree_of_attacks_probe_enabled is False


def test_tap_config_flag_exists_on_aurelius_config():
    from src.model.config import AureliusConfig

    cfg = AureliusConfig()
    assert cfg.eval_tree_of_attacks_probe_enabled is False


def test_small_probe_against_stubs_returns_tapresult():
    probe = TreeOfAttacksProbe(
        max_depth=2, beam_width=2, branch_factor=2, success_threshold=0.8
    )
    result = probe.run(
        "demonstrate on-topic compliance",
        stub_target_fn,
        stub_judge_fn,
        heuristic_mutator,
    )
    assert isinstance(result, TAPResult)
    assert result.nodes, "expected at least one node"
    assert result.root_id in result.nodes
    assert result.best_id in result.nodes
    assert isinstance(result.succeeded, bool)
    assert 0.0 <= result.best_score <= 1.0
