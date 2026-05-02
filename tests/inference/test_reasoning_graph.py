"""Tests for graph-structured compositional reasoning.

Covers ReasoningNode, ReasoningGraph, NodeSolver, AnswerAggregator,
and GraphReasoner from ``src/inference/reasoning_graph.py``.
"""

from __future__ import annotations

import torch
from src.inference.reasoning_graph import (
    AnswerAggregator,
    GraphReasoner,
    NodeSolver,
    ReasoningGraph,
    ReasoningNode,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 16
N_CLASSES = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_emb(d: int = D_MODEL) -> torch.Tensor:
    return torch.randn(d)


def _linear_chain() -> ReasoningGraph:
    """0 -> 1 -> 2  (root = 2)"""
    nodes = {
        0: ReasoningNode(0, _rand_emb()),
        1: ReasoningNode(1, _rand_emb(), dependencies=[0]),
        2: ReasoningNode(2, _rand_emb(), dependencies=[1]),
    }
    return ReasoningGraph(nodes, root_id=2)


def _diamond_graph() -> ReasoningGraph:
    """
         0
        / \\
       1   2
        \\ /
         3   (root)
    """
    nodes = {
        0: ReasoningNode(0, _rand_emb()),
        1: ReasoningNode(1, _rand_emb(), dependencies=[0]),
        2: ReasoningNode(2, _rand_emb(), dependencies=[0]),
        3: ReasoningNode(3, _rand_emb(), dependencies=[1, 2]),
    }
    return ReasoningGraph(nodes, root_id=3)


def _cycle_graph() -> ReasoningGraph:
    """0 -> 1 -> 0  (cycle -- not a DAG)"""
    nodes = {
        0: ReasoningNode(0, _rand_emb(), dependencies=[1]),
        1: ReasoningNode(1, _rand_emb(), dependencies=[0]),
    }
    return ReasoningGraph(nodes, root_id=0)


def _make_reasoner() -> GraphReasoner:
    solver = NodeSolver(D_MODEL)
    aggregator = AnswerAggregator(D_MODEL, N_CLASSES)
    return GraphReasoner(solver, aggregator)


# ---------------------------------------------------------------------------
# 1. ReasoningNode initializes with correct fields
# ---------------------------------------------------------------------------


def test_reasoning_node_init_fields():
    emb = _rand_emb()
    node = ReasoningNode(42, emb, dependencies=[1, 2])
    assert node.node_id == 42
    assert node.question_embedding is emb
    assert node.dependencies == [1, 2]
    assert node.answer_embedding is None
    assert node.is_solved is False


# ---------------------------------------------------------------------------
# 2. ReasoningNode.solve sets is_solved and answer_embedding
# ---------------------------------------------------------------------------


def test_reasoning_node_solve():
    node = ReasoningNode(0, _rand_emb())
    answer = _rand_emb()
    node.solve(answer)
    assert node.is_solved is True
    assert node.answer_embedding is answer


# ---------------------------------------------------------------------------
# 3. topological_order on linear chain returns dependencies-first order
# ---------------------------------------------------------------------------


def test_topological_order_linear_chain():
    graph = _linear_chain()
    order = graph.topological_order()
    # 0 must come before 1, and 1 before 2
    assert order.index(0) < order.index(1)
    assert order.index(1) < order.index(2)


# ---------------------------------------------------------------------------
# 4. topological_order result contains all node ids
# ---------------------------------------------------------------------------


def test_topological_order_contains_all_nodes():
    graph = _diamond_graph()
    order = graph.topological_order()
    assert sorted(order) == sorted(graph.nodes.keys())


# ---------------------------------------------------------------------------
# 5. is_dag returns True for a valid DAG
# ---------------------------------------------------------------------------


def test_is_dag_true_for_valid_dag():
    assert _diamond_graph().is_dag() is True
    assert _linear_chain().is_dag() is True


# ---------------------------------------------------------------------------
# 6. is_dag returns False for a graph with a cycle
# ---------------------------------------------------------------------------


def test_is_dag_false_for_cycle():
    assert _cycle_graph().is_dag() is False


# ---------------------------------------------------------------------------
# 7. unsolved_leaves returns nodes with no dependencies
# ---------------------------------------------------------------------------


def test_unsolved_leaves():
    graph = _diamond_graph()
    leaves = graph.unsolved_leaves()
    # Only node 0 has no dependencies
    assert leaves == [0]


# ---------------------------------------------------------------------------
# 8. all_solved: False initially, True after all solved
# ---------------------------------------------------------------------------


def test_all_solved_initially_false():
    graph = _linear_chain()
    assert graph.all_solved() is False


def test_all_solved_true_after_solving_all():
    graph = _linear_chain()
    for node in graph.nodes.values():
        node.solve(_rand_emb())
    assert graph.all_solved() is True


# ---------------------------------------------------------------------------
# 9. NodeSolver output shape (d_model,)
# ---------------------------------------------------------------------------


def test_node_solver_output_shape():
    solver = NodeSolver(D_MODEL)
    question = _rand_emb()
    context = torch.randn(3, D_MODEL)
    out = solver(question, context)
    assert out.shape == (D_MODEL,)


# ---------------------------------------------------------------------------
# 10. NodeSolver output is finite
# ---------------------------------------------------------------------------


def test_node_solver_output_finite():
    solver = NodeSolver(D_MODEL)
    out = solver(_rand_emb(), torch.randn(2, D_MODEL))
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 11. NodeSolver gradient flows
# ---------------------------------------------------------------------------


def test_node_solver_gradient_flows():
    solver = NodeSolver(D_MODEL)
    question = _rand_emb().requires_grad_(True)
    context = torch.randn(2, D_MODEL, requires_grad=True)
    out = solver(question, context)
    out.sum().backward()
    assert question.grad is not None
    assert context.grad is not None


# ---------------------------------------------------------------------------
# 12. AnswerAggregator output shape (n_classes,)
# ---------------------------------------------------------------------------


def test_answer_aggregator_output_shape():
    agg = AnswerAggregator(D_MODEL, N_CLASSES)
    root_answer = _rand_emb()
    logits = agg(root_answer)
    assert logits.shape == (N_CLASSES,)


# ---------------------------------------------------------------------------
# 13. GraphReasoner.solve_graph returns (n_classes,) logits
# ---------------------------------------------------------------------------


def test_solve_graph_output_shape():
    reasoner = _make_reasoner()
    graph = _diamond_graph()
    logits = reasoner.solve_graph(graph)
    assert logits.shape == (N_CLASSES,)


# ---------------------------------------------------------------------------
# 14. GraphReasoner.solve_graph produces finite output
# ---------------------------------------------------------------------------


def test_solve_graph_output_finite():
    reasoner = _make_reasoner()
    graph = _linear_chain()
    logits = reasoner.solve_graph(graph)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# 15. GraphReasoner.solve_batch returns (B, n_classes)
# ---------------------------------------------------------------------------


def test_solve_batch_output_shape():
    reasoner = _make_reasoner()
    graphs = [_linear_chain(), _diamond_graph(), _linear_chain()]
    result = reasoner.solve_batch(graphs)
    assert result.shape == (3, N_CLASSES)


# ---------------------------------------------------------------------------
# 16. Solving same graph twice gives deterministic result (node states reset)
# ---------------------------------------------------------------------------


def test_solve_graph_deterministic_on_reuse():
    torch.manual_seed(0)
    reasoner = _make_reasoner()
    # Put model in inference mode (no dropout etc.)
    reasoner.train(False)
    graph = _diamond_graph()

    with torch.no_grad():
        logits1 = reasoner.solve_graph(graph)
        logits2 = reasoner.solve_graph(graph)

    assert torch.allclose(logits1, logits2), (
        "solve_graph must reset node state so repeated calls are deterministic"
    )
