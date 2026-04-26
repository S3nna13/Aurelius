"""Synthetic graph reasoning data generator and SFT trainer.

Generates BFS-reachability and parent-finding problems over random directed
graphs (GraphWalks-inspired). No external data or network access needed.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class GraphConfig:
    """Configuration for graph problem generation."""

    n_nodes: int = 20  # number of nodes per graph
    edge_density: float = 0.15  # probability of each directed edge existing
    problem_type: str = "bfs"  # "bfs" | "parents" | "mixed"
    bfs_depth: int = 2  # BFS depth for reachability problems
    node_id_length: int = 6  # hex node id length
    seed: int | None = None


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class GraphProblem:
    """A single graph reasoning problem."""

    graph_edges: list[tuple[str, str]]  # (from_node, to_node) directed edges
    query_node: str  # node to query about
    problem_type: str  # "bfs" or "parents"
    answer_nodes: list[str]  # correct answer (sorted)
    prompt: str  # full natural-language prompt

    def format_answer(self) -> str:
        return f"Final Answer: [{', '.join(sorted(self.answer_nodes))}]"


# ── Graph generation ──────────────────────────────────────────────────────────


def generate_random_graph(cfg: GraphConfig) -> dict[str, list[str]]:
    """Return adjacency list: {node_id: [neighbor_ids]}.

    Node IDs are random hex strings of cfg.node_id_length chars.
    Each directed edge exists with probability cfg.edge_density.
    Self-loops are excluded.
    """
    rng = random.Random(cfg.seed)

    nodes: list[str] = []
    seen: set[str] = set()
    while len(nodes) < cfg.n_nodes:
        nid = "".join(rng.choices("0123456789abcdef", k=cfg.node_id_length))
        if nid not in seen:
            seen.add(nid)
            nodes.append(nid)

    adjacency: dict[str, list[str]] = {n: [] for n in nodes}
    for src in nodes:
        for dst in nodes:
            if src != dst and rng.random() < cfg.edge_density:
                adjacency[src].append(dst)

    return adjacency


def solve_bfs(graph: dict[str, list[str]], start: str, depth: int) -> list[str]:
    """Return all nodes reachable at EXACTLY `depth` hops from `start`.

    Excludes the start node and nodes reachable at other depths.
    Returns a sorted list.
    """
    if depth == 0:
        return []

    current_level: set[str] = {start}
    visited: set[str] = {start}

    for d in range(depth):
        next_level: set[str] = set()
        for node in current_level:
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    next_level.add(neighbor)
        if not next_level:
            return []
        visited |= next_level
        current_level = next_level
        if d + 1 == depth:
            return sorted(current_level)

    return sorted(current_level)


def solve_parents(graph: dict[str, list[str]], target: str) -> list[str]:
    """Return all nodes that have a directed edge TO `target`.

    Excludes `target` itself. Returns a sorted list.
    """
    parents = [node for node, neighbors in graph.items() if target in neighbors and node != target]
    return sorted(parents)


def _format_prompt(edges: list[tuple[str, str]], problem_description: str) -> str:
    edge_lines = "\n".join(f"{a} -> {b}" for a, b in edges)
    return (
        f"You will be given a directed graph as a list of edges.\n"
        f"Edges:\n{edge_lines}\n\n"
        f"Query: {problem_description}\n"
        f"Final Answer: "
    )


def generate_graph_problem(
    cfg: GraphConfig,
    rng: random.Random | None = None,
) -> GraphProblem:
    """Generate a random graph problem (BFS or parents)."""
    if rng is None:
        rng = random.Random(cfg.seed)

    graph = generate_random_graph(cfg)
    nodes = list(graph.keys())
    query_node = rng.choice(nodes)

    if cfg.problem_type == "mixed":
        ptype = rng.choice(["bfs", "parents"])
    else:
        ptype = cfg.problem_type

    edges: list[tuple[str, str]] = [
        (src, dst) for src, neighbors in graph.items() for dst in neighbors
    ]

    if ptype == "bfs":
        answer_nodes = solve_bfs(graph, query_node, cfg.bfs_depth)
        problem_description = (
            f"Starting from node {query_node}, which nodes are reachable in "
            f"exactly {cfg.bfs_depth} hop(s)?"
        )
    else:
        answer_nodes = solve_parents(graph, query_node)
        problem_description = f"Which nodes have a directed edge to node {query_node}?"

    prompt = _format_prompt(edges, problem_description)

    return GraphProblem(
        graph_edges=edges,
        query_node=query_node,
        problem_type=ptype,
        answer_nodes=answer_nodes,
        prompt=prompt,
    )


def generate_graph_dataset(
    n_problems: int,
    cfg: GraphConfig,
    seed: int = 42,
) -> list[GraphProblem]:
    """Generate n_problems with a fixed seed for reproducibility."""
    rng = random.Random(seed)
    return [generate_graph_problem(cfg, rng=rng) for _ in range(n_problems)]


# ── F1 scoring ────────────────────────────────────────────────────────────────


def graph_f1(predicted: list[str], gold: list[str]) -> float:
    """Set-based F1 score. Both empty -> 1.0."""
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    overlap = len(set(predicted) & set(gold))
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def extract_answer_nodes(text: str) -> list[str]:
    """Extract nodes from 'Final Answer: [node1, node2, ...]' pattern.

    Returns a sorted list; empty list if not found.
    """
    match = re.search(r"Final Answer:\s*\[([^\]]*)\]", text)
    if not match:
        return []
    content = match.group(1).strip()
    if not content:
        return []
    nodes = [n.strip() for n in content.split(",") if n.strip()]
    return sorted(nodes)


# ── SFT Trainer ───────────────────────────────────────────────────────────────


@dataclass
class GraphTrainerConfig:
    """Configuration for the GraphReasoningTrainer."""

    max_seq_len: int = 512
    batch_size: int = 4
    learning_rate: float = 2e-5
    loss_only_on_answer: bool = True


class GraphReasoningTrainer:
    """SFT trainer for graph reasoning problems.

    Generates synthetic graph problems on-the-fly (no external data needed).
    """

    def __init__(
        self,
        model,
        optimizer,
        graph_cfg: GraphConfig,
        trainer_cfg: GraphTrainerConfig,
        tokenizer_encode: Callable[[str], list[int]],
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.graph_cfg = graph_cfg
        self.trainer_cfg = trainer_cfg
        self.tokenizer_encode = tokenizer_encode
        self._rng = random.Random()

    def generate_batch(self, n: int) -> list[GraphProblem]:
        """Generate n fresh graph problems."""
        return [generate_graph_problem(self.graph_cfg, rng=self._rng) for _ in range(n)]

    def tokenize_problem(self, problem: GraphProblem) -> tuple[Tensor, Tensor]:
        """Tokenize one GraphProblem and build labels.

        Returns:
            (input_ids, labels) where:
            - input_ids: full prompt + answer tokenized, shape (T,)
            - labels: -100 on prompt tokens if loss_only_on_answer, else same as input_ids
        """
        prompt_ids = self.tokenizer_encode(problem.prompt)
        answer_suffix = f"[{', '.join(sorted(problem.answer_nodes))}]"
        answer_ids = self.tokenizer_encode(answer_suffix)

        full_ids = (prompt_ids + answer_ids)[: self.trainer_cfg.max_seq_len]
        prompt_len = min(len(prompt_ids), len(full_ids))

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()

        if self.trainer_cfg.loss_only_on_answer:
            labels[:prompt_len] = -100

        return input_ids, labels

    def train_step(self) -> dict[str, float]:
        """Generate batch, tokenize, forward, loss, backward, step.

        Returns:
            {"loss": float, "n_problems": int}
        """
        problems = self.generate_batch(self.trainer_cfg.batch_size)
        tokenized = [self.tokenize_problem(p) for p in problems]

        max_len = max(ids.shape[0] for ids, _ in tokenized)
        input_ids_list, labels_list = [], []
        for ids, lbls in tokenized:
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                ids = F.pad(ids, (0, pad_len), value=0)
                lbls = F.pad(lbls, (0, pad_len), value=-100)
            input_ids_list.append(ids)
            labels_list.append(lbls)

        input_ids = torch.stack(input_ids_list)
        labels = torch.stack(labels_list)

        if self.optimizer is not None:
            self.optimizer.zero_grad()

        _loss, logits, _pkv = self.model(input_ids)
        B, T, V = logits.shape

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(B, T - 1)

        mask = (shift_labels != -100).float()
        n_active = mask.sum().clamp(min=1.0)
        loss = (per_token_loss * mask).sum() / n_active

        if self.optimizer is not None:
            loss.backward()
            self.optimizer.step()

        return {"loss": loss.item(), "n_problems": len(problems)}

    def run_eval(self, n_eval: int = 50) -> dict[str, float]:
        """Generate n_eval problems, run greedy generation, compute F1.

        Returns:
            {"mean_f1": float, "exact_match_rate": float}
        """
        problems = self.generate_batch(n_eval)
        f1_scores: list[float] = []
        exact_matches: list[bool] = []

        self.model.eval()
        with torch.no_grad():
            for problem in problems:
                prompt_ids = self.tokenizer_encode(problem.prompt)
                prompt_ids = prompt_ids[: self.trainer_cfg.max_seq_len]

                max_new = max(1, self.trainer_cfg.max_seq_len - len(prompt_ids))
                generated = list(prompt_ids)
                for _ in range(max_new):
                    inp = torch.tensor(
                        generated[: self.trainer_cfg.max_seq_len],
                        dtype=torch.long,
                    ).unsqueeze(0)
                    _, logits, _ = self.model(inp)
                    next_token = int(logits[0, -1, :].argmax().item())
                    generated.append(next_token)
                    try:
                        tail = bytes([t % 256 for t in generated[len(prompt_ids) :]]).decode(
                            "utf-8", errors="replace"
                        )
                        if "]" in tail:
                            break
                    except Exception:
                        break

                try:
                    full_text = bytes([t % 256 for t in generated]).decode(
                        "utf-8", errors="replace"
                    )
                except Exception:
                    full_text = ""

                predicted = extract_answer_nodes(full_text)
                gold = problem.answer_nodes
                f1 = graph_f1(predicted, gold)
                f1_scores.append(f1)
                exact_matches.append(sorted(predicted) == sorted(gold))

        self.model.train()

        mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        exact_match_rate = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
        return {"mean_f1": mean_f1, "exact_match_rate": exact_match_rate}

    # Keep `evaluate` as an alias so tests can call either name
    def evaluate(self, n_eval: int = 50) -> dict[str, float]:
        """Alias for run_eval."""
        return self.run_eval(n_eval)
