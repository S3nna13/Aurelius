"""Tests for MCTS reasoning (src/inference/mcts_reasoning.py)."""

from __future__ import annotations

import math

import pytest
import torch

from src.inference.mcts_reasoning import (
    MCTSConfig,
    MCTSNode,
    MCTSReasoner,
    backpropagate,
    evaluate_state,
    expand_node,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    torch.manual_seed(42)
    model = AureliusTransformer(cfg)
    model.eval()
    return model


def _encode(text: str) -> list[int]:
    """Trivial byte-level tokenizer."""
    return [b for b in text.encode("utf-8")][:16] or [0]


def _decode(ids: list[int]) -> str:
    """Trivial byte-level detokenizer."""
    return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# MCTSConfig tests
# ---------------------------------------------------------------------------


def test_mcts_config_defaults():
    cfg = MCTSConfig()
    assert cfg.n_simulations == 50
    assert cfg.c_puct == pytest.approx(1.4)
    assert cfg.max_depth == 10
    assert cfg.temperature == pytest.approx(1.0)
    assert cfg.value_discount == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# MCTSNode tests
# ---------------------------------------------------------------------------


def test_mcts_node_value_unvisited():
    node = MCTSNode(state=[1, 2, 3], parent=None)
    assert node.value() == pytest.approx(0.0)


def test_mcts_node_value_after_updates():
    node = MCTSNode(state=[1, 2, 3], parent=None)
    node.visit_count = 4
    node.value_sum = 2.0
    assert node.value() == pytest.approx(0.5)


def test_mcts_node_ucb_score_returns_float():
    node = MCTSNode(state=[1], parent=None, prior=0.5)
    score = node.ucb_score(parent_visits=10, c_puct=1.4)
    assert isinstance(score, float)
    assert math.isfinite(score)


def test_mcts_node_ucb_score_higher_for_unvisited():
    parent = MCTSNode(state=[1], parent=None, visit_count=10)
    visited = MCTSNode(state=[1, 2], parent=parent, prior=0.5, visit_count=5, value_sum=2.0)
    unvisited = MCTSNode(state=[1, 3], parent=parent, prior=0.5, visit_count=0, value_sum=0.0)
    assert unvisited.ucb_score(10, 1.4) > visited.ucb_score(10, 1.4)


def test_mcts_node_is_leaf_true_no_children():
    node = MCTSNode(state=[1, 2], parent=None)
    assert node.is_leaf() is True


def test_mcts_node_is_leaf_false_with_children():
    parent = MCTSNode(state=[1], parent=None)
    child = MCTSNode(state=[1, 2], parent=parent)
    parent.children.append(child)
    assert parent.is_leaf() is False


# ---------------------------------------------------------------------------
# evaluate_state tests
# ---------------------------------------------------------------------------


def test_evaluate_state_returns_float_in_range(small_model):
    state = [1, 2, 3, 4, 5]
    value = evaluate_state(small_model, state)
    assert isinstance(value, float)
    assert -1.0 <= value <= 1.0


def test_evaluate_state_single_token(small_model):
    value = evaluate_state(small_model, [42])
    assert isinstance(value, float)
    assert -1.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# expand_node tests
# ---------------------------------------------------------------------------


def test_expand_node_creates_children(small_model):
    node = MCTSNode(state=[1, 2, 3], parent=None)
    expand_node(small_model, node, top_k=5)
    assert len(node.children) == 5


def test_expand_node_children_have_valid_action_tokens(small_model):
    node = MCTSNode(state=[10, 20], parent=None)
    expand_node(small_model, node, top_k=3)
    for child in node.children:
        assert child.action is not None
        assert isinstance(child.action, int)
        assert 0 <= child.action < 256  # within vocab_size


def test_expand_node_children_priors_are_valid(small_model):
    node = MCTSNode(state=[5, 6, 7], parent=None)
    expand_node(small_model, node, top_k=4)
    for child in node.children:
        assert 0.0 <= child.prior <= 1.0


# ---------------------------------------------------------------------------
# backpropagate tests
# ---------------------------------------------------------------------------


def test_backpropagate_updates_visit_count():
    root = MCTSNode(state=[1], parent=None)
    child = MCTSNode(state=[1, 2], parent=root)
    root.children.append(child)
    backpropagate(child, value=0.8, discount=0.95)
    assert child.visit_count == 1
    assert root.visit_count == 1


def test_backpropagate_updates_value_sum():
    root = MCTSNode(state=[1], parent=None)
    child = MCTSNode(state=[1, 2], parent=root)
    root.children.append(child)
    backpropagate(child, value=1.0, discount=0.5)
    # child gets 1.0, root gets 1.0 * 0.5 = 0.5
    assert child.value_sum == pytest.approx(1.0)
    assert root.value_sum == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# MCTSReasoner tests
# ---------------------------------------------------------------------------


def test_mcts_reasoner_search_returns_str_float(small_model):
    cfg = MCTSConfig(n_simulations=3, max_depth=4)
    reasoner = MCTSReasoner(small_model, cfg, _encode, _decode)
    result = reasoner.search("hi")
    assert isinstance(result, tuple)
    assert len(result) == 2
    continuation, value = result
    assert isinstance(continuation, str)
    assert isinstance(value, float)


def test_mcts_reasoner_get_best_path_returns_list_of_ints(small_model):
    cfg = MCTSConfig(n_simulations=3, max_depth=4)
    reasoner = MCTSReasoner(small_model, cfg, _encode, _decode)
    prompt_ids = _encode("ab")
    root = MCTSNode(state=prompt_ids, parent=None)
    expand_node(small_model, root, top_k=3)
    path = reasoner.get_best_path(root)
    assert isinstance(path, list)
    assert all(isinstance(t, int) for t in path)


def test_mcts_multiple_simulations_increase_visit_counts(small_model):
    cfg = MCTSConfig(n_simulations=3, max_depth=3)
    reasoner = MCTSReasoner(small_model, cfg, _encode, _decode)
    prompt_ids = _encode("test")
    root = MCTSNode(state=prompt_ids, parent=None)

    for _ in range(cfg.n_simulations):
        leaf = reasoner._select(root)
        expand_node(small_model, leaf, top_k=3)
        if leaf.children:
            leaf = leaf.children[0]
        value = reasoner._simulate(leaf)
        backpropagate(leaf, value, cfg.value_discount)

    assert root.visit_count >= 1
