"""Unit tests for Tree-of-Attacks-with-Pruning (TAP) probe."""
from __future__ import annotations

import pytest

from src.eval.tree_of_attacks_probe import (
    MUTATION_STRATEGIES,
    TAPNode,
    TAPResult,
    TreeOfAttacksProbe,
    heuristic_mutator,
    stub_judge_fn,
    stub_target_fn,
)


# Deterministic target/judge/mutator fixtures --------------------------------

def _always_low_target(messages):
    return "brief overview only"


def _always_low_judge(objective, response):
    return (0.1, "low score")


def _always_high_judge(objective, response):
    return (0.95, "success")


def _identity_mutator(prompt, strategy):
    return prompt + f" [{strategy}]"


# Tests ----------------------------------------------------------------------

def test_mutation_strategies_non_empty():
    assert isinstance(MUTATION_STRATEGIES, tuple)
    assert len(MUTATION_STRATEGIES) >= 3
    assert all(isinstance(s, str) and s for s in MUTATION_STRATEGIES)


def test_small_depth_run_bounded_node_count():
    probe = TreeOfAttacksProbe(
        max_depth=2, beam_width=2, branch_factor=2, success_threshold=0.99
    )
    result = probe.run(
        "objective x",
        _always_low_target,
        _always_low_judge,
        _identity_mutator,
    )
    # Max possible nodes: 1 (root) + beam * branch at each depth.
    # d=1: 1*2 = 2 children from root; kept beam=2. d=2: 2*2 = 4 children.
    # Total: 1 + 2 + 4 = 7.
    assert len(result.nodes) <= 1 + 2 + 4
    assert len(result.nodes) >= 1


def test_success_at_depth_one():
    probe = TreeOfAttacksProbe(
        max_depth=3, beam_width=2, branch_factor=2, success_threshold=0.8
    )
    result = probe.run(
        "obj",
        _always_low_target,
        _always_high_judge,
        _identity_mutator,
    )
    # Root scores 0.95 >= 0.8, so succeeds immediately at depth 0.
    assert result.succeeded is True
    assert result.best_score >= 0.8


def test_max_depth_exhausted_without_success():
    probe = TreeOfAttacksProbe(
        max_depth=2, beam_width=2, branch_factor=2, success_threshold=0.99
    )
    result = probe.run(
        "obj",
        _always_low_target,
        _always_low_judge,
        _identity_mutator,
    )
    assert result.succeeded is False
    assert result.best_score < 0.99


def test_beam_width_one_is_linear_at_each_depth():
    probe = TreeOfAttacksProbe(
        max_depth=3, beam_width=1, branch_factor=1, success_threshold=0.999
    )
    # Can't hit threshold >1 so full expansion
    result = probe.run(
        "obj", _always_low_target, _always_low_judge, _identity_mutator
    )
    # 1 root + 1 at each depth = 1+3=4
    assert len(result.nodes) == 4


def test_branch_factor_one_linear():
    probe = TreeOfAttacksProbe(
        max_depth=3, beam_width=4, branch_factor=1, success_threshold=0.999
    )
    result = probe.run(
        "obj", _always_low_target, _always_low_judge, _identity_mutator
    )
    # branch_factor=1 means each beam spawns 1 child;
    # with beam always size<=1 since only 1 child per parent at each depth.
    # Depth 0:1 root; depth1:1; depth2:1; depth3:1 => 4 nodes
    assert len(result.nodes) == 4


def test_pruning_discards_low_score_siblings():
    # Judge returns score based on length: longer prompts score higher.
    def len_judge(objective, response):
        return (min(1.0, len(response) / 200.0), "len-based")

    def len_target(messages):
        return messages[-1]["content"] * 2  # longer prompt -> longer response

    probe = TreeOfAttacksProbe(
        max_depth=1, beam_width=1, branch_factor=4, success_threshold=0.999
    )
    result = probe.run(
        "obj", len_target, len_judge, _identity_mutator
    )
    # At depth 1, generated 4 children; beam keeps 1, prunes 3.
    assert result.pruned_count == 3


def test_empty_objective_rejected():
    probe = TreeOfAttacksProbe(max_depth=1)
    with pytest.raises(ValueError):
        probe.run(
            "", _always_low_target, _always_low_judge, _identity_mutator
        )
    with pytest.raises(ValueError):
        probe.run(
            "   ", _always_low_target, _always_low_judge, _identity_mutator
        )


def test_non_string_objective_rejects():
    probe = TreeOfAttacksProbe(max_depth=1)
    with pytest.raises(TypeError):
        probe.run(
            123, _always_low_target, _always_low_judge, _identity_mutator
        )


def test_deterministic_run_same_inputs_same_outputs():
    probe = TreeOfAttacksProbe(
        max_depth=2, beam_width=2, branch_factor=2, success_threshold=0.99
    )
    r1 = probe.run(
        "obj", stub_target_fn, stub_judge_fn, heuristic_mutator
    )
    r2 = probe.run(
        "obj", stub_target_fn, stub_judge_fn, heuristic_mutator
    )
    assert len(r1.nodes) == len(r2.nodes)
    assert r1.best_score == r2.best_score
    assert r1.succeeded == r2.succeeded


def test_best_id_points_to_highest_score():
    probe = TreeOfAttacksProbe(
        max_depth=2, beam_width=2, branch_factor=2, success_threshold=0.999
    )
    result = probe.run(
        "obj", stub_target_fn, stub_judge_fn, heuristic_mutator
    )
    best = result.nodes[result.best_id]
    max_score = max(n.score for n in result.nodes.values())
    assert best.score == max_score
    assert result.best_score == max_score


def test_parent_children_pointers_consistent():
    probe = TreeOfAttacksProbe(
        max_depth=2, beam_width=2, branch_factor=2, success_threshold=0.999
    )
    result = probe.run(
        "obj", stub_target_fn, stub_judge_fn, heuristic_mutator
    )
    for nid, node in result.nodes.items():
        for cid in node.children_ids:
            assert cid in result.nodes
            assert result.nodes[cid].parent_id == nid
        if node.parent_id is not None:
            parent = result.nodes[node.parent_id]
            assert nid in parent.children_ids
    # Root's parent is None.
    assert result.nodes[result.root_id].parent_id is None


def test_unicode_objective_accepted():
    probe = TreeOfAttacksProbe(
        max_depth=1, beam_width=1, branch_factor=1, success_threshold=0.999
    )
    result = probe.run(
        "目标 🎯 obj", _always_low_target, _always_low_judge, _identity_mutator
    )
    assert result.objective == "目标 🎯 obj"
    assert len(result.nodes) == 2


def test_seed_prompt_used_when_provided():
    captured = {}

    def record_target(messages):
        captured.setdefault("first", messages[-1]["content"])
        return "brief"

    probe = TreeOfAttacksProbe(
        max_depth=1, beam_width=1, branch_factor=1, success_threshold=0.999
    )
    probe.run(
        "obj",
        record_target,
        _always_low_judge,
        _identity_mutator,
        seed_prompt="custom seed",
    )
    assert captured["first"] == "custom seed"


def test_invalid_judge_return_raises():
    def bad_judge(objective, response):
        return "not a tuple"

    probe = TreeOfAttacksProbe(max_depth=1)
    with pytest.raises(TypeError):
        probe.run("obj", _always_low_target, bad_judge, _identity_mutator)


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        TreeOfAttacksProbe(max_depth=0)
    with pytest.raises(ValueError):
        TreeOfAttacksProbe(beam_width=0)
    with pytest.raises(ValueError):
        TreeOfAttacksProbe(branch_factor=0)
    with pytest.raises(ValueError):
        TreeOfAttacksProbe(success_threshold=1.5)


def test_heuristic_mutator_all_strategies_deterministic():
    for s in MUTATION_STRATEGIES:
        a = heuristic_mutator("hello", s)
        b = heuristic_mutator("hello", s)
        assert a == b
        assert isinstance(a, str) and a


def test_result_is_tapresult_and_nodes_are_tapnode():
    probe = TreeOfAttacksProbe(
        max_depth=1, beam_width=1, branch_factor=1, success_threshold=0.999
    )
    result = probe.run(
        "obj", _always_low_target, _always_low_judge, _identity_mutator
    )
    assert isinstance(result, TAPResult)
    for node in result.nodes.values():
        assert isinstance(node, TAPNode)
