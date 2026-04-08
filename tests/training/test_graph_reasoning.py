"""Tests for GraphReasoningTrainer and related utilities."""
from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.graph_reasoning import (
    GraphConfig,
    GraphProblem,
    GraphReasoningTrainer,
    GraphTrainerConfig,
    extract_answer_nodes,
    generate_graph_dataset,
    generate_graph_problem,
    generate_random_graph,
    graph_f1,
    solve_bfs,
    solve_parents,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture
def small_model(small_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


def _simple_encode(text: str) -> list[int]:
    """Deterministic byte-level tokenizer staying within vocab_size=256."""
    return list(text.encode("utf-8", errors="replace"))


@pytest.fixture
def default_graph_cfg():
    return GraphConfig(n_nodes=10, edge_density=0.3, problem_type="bfs", bfs_depth=1, seed=42)


# ── Graph generation tests ────────────────────────────────────────────────────


def test_generate_random_graph_node_count(default_graph_cfg):
    graph = generate_random_graph(default_graph_cfg)
    assert len(graph) == default_graph_cfg.n_nodes, (
        f"Expected {default_graph_cfg.n_nodes} nodes, got {len(graph)}"
    )


def test_generate_random_graph_directed():
    cfg = GraphConfig(n_nodes=10, edge_density=0.5, seed=7)
    graph = generate_random_graph(cfg)
    # Check that edges are directed: build reverse lookup
    # At least some A->B should not have B->A  (with density=0.5, highly likely)
    asymmetric_found = False
    for node, neighbors in graph.items():
        for nb in neighbors:
            if node not in graph.get(nb, []):
                asymmetric_found = True
                break
        if asymmetric_found:
            break
    assert asymmetric_found, "With 50% edge density, expect at least one asymmetric edge"


def test_solve_bfs_depth_1():
    # Hand-crafted simple graph: A -> B -> C, A -> C
    graph = {"A": ["B", "C"], "B": ["C"], "C": []}
    result = solve_bfs(graph, "A", depth=1)
    assert result == ["B", "C"]


def test_solve_bfs_excludes_start():
    graph = {"A": ["B", "C"], "B": ["A"], "C": []}
    result = solve_bfs(graph, "A", depth=1)
    assert "A" not in result


def test_solve_bfs_exact_depth():
    # A -> B -> C; depth=2 from A should give [C], not [B]
    graph = {"A": ["B"], "B": ["C"], "C": []}
    result_d1 = solve_bfs(graph, "A", depth=1)
    result_d2 = solve_bfs(graph, "A", depth=2)
    assert "B" in result_d1
    assert "B" not in result_d2
    assert "C" in result_d2


def test_solve_parents_basic():
    # A -> C, B -> C, D -> E
    graph = {"A": ["C"], "B": ["C"], "C": [], "D": ["E"], "E": []}
    parents = solve_parents(graph, "C")
    assert sorted(parents) == ["A", "B"]


def test_solve_parents_excludes_self():
    # Even if C -> C existed (it shouldn't with our generator), exclude self
    graph = {"A": ["C"], "C": ["C"]}
    parents = solve_parents(graph, "C")
    assert "C" not in parents
    assert "A" in parents


# ── GraphProblem generation tests ─────────────────────────────────────────────


def test_generate_graph_problem_fields():
    cfg = GraphConfig(n_nodes=8, edge_density=0.3, problem_type="bfs", bfs_depth=1, seed=1)
    problem = generate_graph_problem(cfg)
    assert isinstance(problem.graph_edges, list)
    assert isinstance(problem.query_node, str) and len(problem.query_node) > 0
    assert problem.problem_type in ("bfs", "parents")
    assert isinstance(problem.answer_nodes, list)
    assert isinstance(problem.prompt, str) and len(problem.prompt) > 0
    assert "Final Answer:" in problem.prompt


def test_generate_graph_problem_answer_valid():
    cfg = GraphConfig(n_nodes=10, edge_density=0.3, problem_type="bfs", bfs_depth=1, seed=2)
    problem = generate_graph_problem(cfg)
    graph = generate_random_graph(cfg)
    all_nodes = set(graph.keys())
    for node in problem.answer_nodes:
        assert node in all_nodes, f"Answer node {node!r} not in graph"


def test_generate_graph_dataset_count():
    cfg = GraphConfig(n_nodes=8, edge_density=0.2, seed=10)
    dataset = generate_graph_dataset(15, cfg, seed=42)
    assert len(dataset) == 15


def test_generate_graph_dataset_reproducible():
    cfg = GraphConfig(n_nodes=8, edge_density=0.2, seed=10)
    ds1 = generate_graph_dataset(5, cfg, seed=99)
    ds2 = generate_graph_dataset(5, cfg, seed=99)
    for p1, p2 in zip(ds1, ds2):
        assert p1.query_node == p2.query_node
        assert p1.answer_nodes == p2.answer_nodes
        assert p1.prompt == p2.prompt


# ── F1 scoring tests ──────────────────────────────────────────────────────────


def test_graph_f1_perfect():
    assert graph_f1(["a", "b"], ["a", "b"]) == 1.0


def test_graph_f1_empty_both():
    assert graph_f1([], []) == 1.0


def test_graph_f1_no_overlap():
    assert graph_f1(["a", "b"], ["c", "d"]) == 0.0


# ── extract_answer_nodes tests ────────────────────────────────────────────────


def test_extract_answer_nodes_valid():
    text = "some text\nFinal Answer: [abc123, def456]"
    result = extract_answer_nodes(text)
    assert "abc123" in result
    assert "def456" in result


def test_extract_answer_nodes_empty():
    text = "no pattern here"
    result = extract_answer_nodes(text)
    assert result == []


# ── Trainer tests ─────────────────────────────────────────────────────────────


def test_graph_trainer_train_step_metrics(small_model):
    graph_cfg = GraphConfig(n_nodes=4, edge_density=0.6, problem_type="parents", seed=3)
    trainer_cfg = GraphTrainerConfig(max_seq_len=512, batch_size=2)
    optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-4)
    trainer = GraphReasoningTrainer(
        model=small_model,
        optimizer=optimizer,
        graph_cfg=graph_cfg,
        trainer_cfg=trainer_cfg,
        tokenizer_encode=_simple_encode,
    )
    result = trainer.train_step()
    assert "loss" in result
    assert isinstance(result["loss"], float)
    assert result["loss"] > 0
    assert "n_problems" in result
    assert result["n_problems"] == trainer_cfg.batch_size


def test_graph_trainer_evaluate_metrics(small_model):
    graph_cfg = GraphConfig(n_nodes=6, edge_density=0.4, problem_type="parents")
    trainer_cfg = GraphTrainerConfig(max_seq_len=128, batch_size=2)
    trainer = GraphReasoningTrainer(
        model=small_model,
        optimizer=None,
        graph_cfg=graph_cfg,
        trainer_cfg=trainer_cfg,
        tokenizer_encode=_simple_encode,
    )
    metrics = trainer.evaluate(n_eval=3)
    assert "mean_f1" in metrics
    assert "exact_match_rate" in metrics
    assert 0.0 <= metrics["mean_f1"] <= 1.0
    assert 0.0 <= metrics["exact_match_rate"] <= 1.0
