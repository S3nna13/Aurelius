"""Graph-structured compositional reasoning for LLMs.

Decomposes complex questions into a DAG of sub-problems.
Nodes are solved bottom-up (dependencies before dependents).
Answers are aggregated from leaf to root via a learned combiner.

Inspired by: Khot et al. 2023 (Decomposed Prompting)
             Yao et al. 2023 (Tree of Thoughts)
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# ReasoningNode
# ---------------------------------------------------------------------------


class ReasoningNode:
    """A single node in the reasoning DAG.

    Parameters
    ----------
    node_id:
        Unique integer identifier for this node.
    question_embedding:
        Shape ``(d_model,)`` — embedding of the sub-question this node answers.
    dependencies:
        List of ``node_id`` values whose answers must be available before this
        node can be solved.  Defaults to an empty list (leaf node).
    """

    def __init__(
        self,
        node_id: int,
        question_embedding: Tensor,
        dependencies: list[int] | None = None,
    ) -> None:
        self.node_id: int = node_id
        self.question_embedding: Tensor = question_embedding
        self.dependencies: list[int] = dependencies if dependencies is not None else []
        self.answer_embedding: Tensor | None = None
        self.is_solved: bool = False

    def solve(self, answer: Tensor) -> None:
        """Store *answer* and mark this node as solved."""
        self.answer_embedding = answer
        self.is_solved = True


# ---------------------------------------------------------------------------
# ReasoningGraph
# ---------------------------------------------------------------------------


class ReasoningGraph:
    """Directed acyclic graph of :class:`ReasoningNode` objects.

    Parameters
    ----------
    nodes:
        Mapping from ``node_id`` to :class:`ReasoningNode`.
    root_id:
        The node whose answer represents the final answer to the original
        question.
    """

    def __init__(self, nodes: dict[int, ReasoningNode], root_id: int) -> None:
        self.nodes = nodes
        self.root_id = root_id

    # ------------------------------------------------------------------
    # Graph-structure helpers
    # ------------------------------------------------------------------

    def topological_order(self) -> list[int]:
        """Return node ids in a valid topological order (Kahn's algorithm).

        Dependencies always appear before the nodes that depend on them.

        Raises
        ------
        ValueError
            If the graph contains a cycle.
        """
        # Build in-degree count and adjacency list (edge: dep → dependent)
        in_degree: dict[int, int] = {nid: 0 for nid in self.nodes}
        children: dict[int, list[int]] = {nid: [] for nid in self.nodes}

        for nid, node in self.nodes.items():
            for dep in node.dependencies:
                children[dep].append(nid)
                in_degree[nid] += 1

        # Start with all nodes that have no incoming edges
        queue: deque[int] = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order: list[int] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for child in children[nid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            raise ValueError("Graph contains a cycle; topological order is not defined.")

        return order

    def is_dag(self) -> bool:
        """Return ``True`` if the graph is acyclic (DFS visited/in-stack check)."""
        visited: set[int] = set()
        in_stack: set[int] = set()

        def _dfs(nid: int) -> bool:
            """Return True if a cycle is found."""
            visited.add(nid)
            in_stack.add(nid)
            for dep in self.nodes[nid].dependencies:
                if dep not in visited:
                    if _dfs(dep):
                        return True
                elif dep in in_stack:
                    return True
            in_stack.discard(nid)
            return False

        for nid in self.nodes:
            if nid not in visited:
                if _dfs(nid):
                    return False
        return True

    def unsolved_leaves(self) -> list[int]:
        """Return ids of nodes that have no dependencies and are not yet solved."""
        return [
            nid for nid, node in self.nodes.items() if not node.dependencies and not node.is_solved
        ]

    def all_solved(self) -> bool:
        """Return ``True`` when every node in the graph has been solved."""
        return all(node.is_solved for node in self.nodes.values())


# ---------------------------------------------------------------------------
# NodeSolver
# ---------------------------------------------------------------------------


class NodeSolver(nn.Module):
    """Produces an answer embedding for a single reasoning node.

    Architecture
    ------------
    ``Linear(d_model * 2, d_model)`` applied to
    ``[question, mean(context)]`` concatenated, followed by ``nn.GELU()``.

    Parameters
    ----------
    d_model:
        Dimension of all embedding vectors.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(d_model * 2, d_model)
        self.activation = nn.GELU()

    def forward(self, question: Tensor, context: Tensor) -> Tensor:
        """Compute an answer embedding for one node.

        Parameters
        ----------
        question:
            Shape ``(d_model,)``.
        context:
            Shape ``(n_deps, d_model)`` — answer embeddings of dependency nodes.
            Pass a ``(1, d_model)`` zero tensor when there are no dependencies.

        Returns
        -------
        Tensor
            Shape ``(d_model,)`` answer embedding.
        """
        if context.shape[0] == 0:
            context = torch.zeros(1, self.d_model, dtype=question.dtype, device=question.device)
        context_mean = context.mean(dim=0)  # (d_model,)
        combined = torch.cat([question, context_mean], dim=0)  # (d_model * 2,)
        out = self.linear(combined)
        return self.activation(out)  # (d_model,)


# ---------------------------------------------------------------------------
# AnswerAggregator
# ---------------------------------------------------------------------------


class AnswerAggregator(nn.Module):
    """Projects the root node answer embedding to class logits.

    Parameters
    ----------
    d_model:
        Dimension of the root answer embedding.
    n_classes:
        Number of output classes.
    """

    def __init__(self, d_model: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, root_answer: Tensor) -> Tensor:
        """Map a root answer embedding to logits.

        Parameters
        ----------
        root_answer:
            Shape ``(d_model,)``.

        Returns
        -------
        Tensor
            Shape ``(n_classes,)`` logits.
        """
        return self.linear(root_answer)


# ---------------------------------------------------------------------------
# GraphReasoner
# ---------------------------------------------------------------------------


class GraphReasoner(nn.Module):
    """Orchestrates bottom-up solving of a :class:`ReasoningGraph`.

    Parameters
    ----------
    node_solver:
        Shared :class:`NodeSolver` used for every node.
    aggregator:
        :class:`AnswerAggregator` that converts the root answer to logits.
    """

    def __init__(self, node_solver: NodeSolver, aggregator: AnswerAggregator) -> None:
        super().__init__()
        self.node_solver = node_solver
        self.aggregator = aggregator

    def _reset_graph(self, graph: ReasoningGraph) -> None:
        """Clear all solved state so the same graph object can be reused."""
        for node in graph.nodes.values():
            node.answer_embedding = None
            node.is_solved = False

    def solve_graph(self, graph: ReasoningGraph) -> Tensor:
        """Solve *graph* bottom-up and return logits for the root node.

        The graph's node state is reset before solving, so the same
        :class:`ReasoningGraph` instance can be passed multiple times.

        Parameters
        ----------
        graph:
            A :class:`ReasoningGraph` whose nodes carry question embeddings.

        Returns
        -------
        Tensor
            Shape ``(n_classes,)`` logits produced by the aggregator.
        """
        self._reset_graph(graph)
        order = graph.topological_order()

        for nid in order:
            node = graph.nodes[nid]
            if node.dependencies:
                dep_answers = torch.stack(
                    [graph.nodes[dep].answer_embedding for dep in node.dependencies], dim=0
                )  # (n_deps, d_model)
            else:
                d_model = node.question_embedding.shape[0]
                dep_answers = torch.zeros(
                    1,
                    d_model,
                    dtype=node.question_embedding.dtype,
                    device=node.question_embedding.device,
                )

            answer = self.node_solver.forward(node.question_embedding, dep_answers)
            node.solve(answer)

        root_answer = graph.nodes[graph.root_id].answer_embedding
        return self.aggregator.forward(root_answer)

    def solve_batch(self, graphs: list[ReasoningGraph]) -> Tensor:
        """Solve a list of graphs independently and stack the results.

        Parameters
        ----------
        graphs:
            List of :class:`ReasoningGraph` objects.

        Returns
        -------
        Tensor
            Shape ``(len(graphs), n_classes)``.
        """
        results = [self.solve_graph(g) for g in graphs]
        return torch.stack(results, dim=0)
