"""Tree-of-Thoughts beam-search planner for Aurelius.

This module implements a Tree-of-Thoughts (ToT) style planner
(Yao et al. 2023, arXiv:2305.10601). The planner separates *planning*
from *execution*: a pluggable ``generate_fn`` proposes K candidate
sub-steps at each node, a pluggable ``scorer_fn`` assigns a scalar
utility to every proposal, and a breadth-first beam search keeps the
top-K partial plans at each depth. The winning root-to-leaf path is
converted into an execution queue consumable by
:class:`src.agent.react_loop.ReActLoop`.

Design notes
------------
* Pure stdlib: ``dataclasses``, ``typing``, ``uuid``, ``math``, ``logging``.
* ``generate_fn`` and ``scorer_fn`` are **untrusted**. Malformed output
  is *dropped with a warning* (never silently coerced, never fatal).
* ``nan``/``inf`` scores are clamped to ``-inf`` so they never win the
  beam; this preserves total ordering without silent data loss.
* ``max_depth=0`` is a legal input: it returns a lonely root with no
  expansions, matching the "plan a no-op" degenerate case.
* Cycle protection: the planner keys on the per-node ``depth`` counter
  rather than pointer equality, so even if ``generate_fn`` returns a
  payload that references an existing node, a fresh ``PlanNode`` is
  constructed each time and the depth cap terminates the search.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

__all__ = ["PlanNode", "BeamPlanner"]


_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class PlanNode:
    """One node in the plan tree.

    Attributes
    ----------
    id:
        Globally unique identifier (uuid4 hex by default).
    parent_id:
        ``None`` for the synthetic root; otherwise the ``id`` of the
        parent node. Used for tree serialisation/round-trip.
    depth:
        Distance from the root. The root has depth 0.
    description:
        Natural-language description of the sub-step.
    expected_tool:
        Name of the tool the executor is expected to call at this
        step, or ``None`` for pure reasoning steps.
    expected_effect:
        Short description of the effect the caller anticipates (used
        as a post-condition during reflection).
    score:
        Scalar utility. Higher is better. Non-finite values are
        clamped to ``-inf`` by :class:`BeamPlanner`.
    children:
        Direct descendants. Empty by default.
    """

    id: str
    parent_id: Optional[str]
    depth: int
    description: str
    expected_tool: Optional[str]
    expected_effect: str
    score: float = 0.0
    children: list["PlanNode"] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


_REQUIRED_KEYS = ("description", "expected_tool", "expected_effect")


class BeamPlanner:
    """Beam-search over a Tree-of-Thoughts plan tree.

    Parameters
    ----------
    generate_fn:
        ``(task, path_from_root) -> list[dict]``. Each dict must
        contain ``description``, ``expected_tool`` and
        ``expected_effect``; malformed entries are dropped.
    scorer_fn:
        ``(PlanNode) -> float``. Higher is better. ``nan``/``inf`` are
        clamped to ``-inf`` (a warning is logged).
    beam_width:
        Number of candidates kept at each depth. ``beam_width=1``
        degenerates to greedy search.
    max_depth:
        Plan-tree depth cap. ``0`` means "no expansion; return a bare
        root node".
    """

    def __init__(
        self,
        generate_fn: Callable[[str, list[PlanNode]], list[dict]],
        scorer_fn: Callable[[PlanNode], float],
        beam_width: int = 4,
        max_depth: int = 5,
    ) -> None:
        if not callable(generate_fn):
            raise TypeError("generate_fn must be callable")
        if not callable(scorer_fn):
            raise TypeError("scorer_fn must be callable")
        if not isinstance(beam_width, int) or beam_width < 1:
            raise ValueError("beam_width must be a positive int")
        if not isinstance(max_depth, int) or max_depth < 0:
            raise ValueError("max_depth must be a non-negative int")
        self._generate = generate_fn
        self._scorer = scorer_fn
        self.beam_width = beam_width
        self.max_depth = max_depth

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def expand(self, node: PlanNode, task: str) -> list[PlanNode]:
        """Generate and score at most ``beam_width`` children for ``node``.

        Malformed proposals (non-dict, missing keys, non-string fields)
        are dropped with a warning. The children are scored in order of
        generation and attached to ``node.children``.
        """
        if not isinstance(node, PlanNode):
            raise TypeError("node must be a PlanNode")
        if not isinstance(task, str):
            raise TypeError("task must be a str")

        path = self._path_from_root(node)
        try:
            proposals = self._generate(task, path)
        except Exception as exc:  # noqa: BLE001 - untrusted generator
            _LOG.warning(
                "generate_fn raised %s: %s; treating as no proposals",
                type(exc).__name__,
                exc,
            )
            return []

        if not isinstance(proposals, list):
            _LOG.warning(
                "generate_fn returned %s, expected list; dropping",
                type(proposals).__name__,
            )
            return []

        children: list[PlanNode] = []
        for raw in proposals[: self.beam_width]:
            child = self._coerce_proposal(raw, parent=node)
            if child is None:
                continue
            child.score = self._safe_score(child)
            children.append(child)
        node.children.extend(children)
        return children

    # ------------------------------------------------------------------
    # Top-level search
    # ------------------------------------------------------------------

    def plan(self, task: str) -> PlanNode:
        """Run beam search and return the root of the fully built tree."""
        if not isinstance(task, str):
            raise TypeError("task must be a str")

        root = PlanNode(
            id=uuid.uuid4().hex,
            parent_id=None,
            depth=0,
            description="<root>",
            expected_tool=None,
            expected_effect=f"plan for: {task}",
            score=0.0,
        )
        if self.max_depth == 0:
            return root

        frontier: list[PlanNode] = [root]
        for _ in range(self.max_depth):
            next_frontier: list[PlanNode] = []
            for node in frontier:
                next_frontier.extend(self.expand(node, task))
            if not next_frontier:
                # No-one could expand; terminate gracefully.
                break
            # Prune to top-K by score; ties broken by insertion order
            # (Python's sort is stable).
            next_frontier.sort(key=lambda n: n.score, reverse=True)
            frontier = next_frontier[: self.beam_width]
        return root

    # ------------------------------------------------------------------
    # Path selection
    # ------------------------------------------------------------------

    def best_path(self, root: PlanNode) -> list[PlanNode]:
        """Return the chain of nodes from ``root`` to its highest-scoring leaf.

        "Highest-scoring leaf" is the leaf whose cumulative path score
        (sum of node scores excluding the synthetic root) is maximal.
        Ties are broken by depth (deeper wins) then by id (stable).
        """
        if not isinstance(root, PlanNode):
            raise TypeError("root must be a PlanNode")

        best_leaf: Optional[PlanNode] = None
        best_total = -math.inf
        best_depth = -1
        stack: list[tuple[PlanNode, float]] = [(root, 0.0)]
        while stack:
            node, running = stack.pop()
            if not node.children:
                # Exclude the synthetic root from the sum.
                total = running if node.parent_id is not None else 0.0
                tiebreak_depth = node.depth
                if (
                    total > best_total
                    or (total == best_total and tiebreak_depth > best_depth)
                ):
                    best_leaf = node
                    best_total = total
                    best_depth = tiebreak_depth
                continue
            for child in node.children:
                stack.append((child, running + child.score))

        if best_leaf is None:
            return [root]

        # Reconstruct the chain from the leaf back up to the root.
        index: dict[str, PlanNode] = {}
        self._index_tree(root, index)
        chain: list[PlanNode] = []
        cur: Optional[PlanNode] = best_leaf
        while cur is not None:
            chain.append(cur)
            if cur.parent_id is None:
                break
            cur = index.get(cur.parent_id)
        chain.reverse()
        return chain

    # ------------------------------------------------------------------
    # Executor hand-off
    # ------------------------------------------------------------------

    def to_execution_queue(self, path: list[PlanNode]) -> list[dict]:
        """Render ``path`` as a list of ReActLoop-compatible observations.

        The synthetic root (``parent_id is None``) is skipped so the
        queue contains only actionable steps.
        """
        if not isinstance(path, list):
            raise TypeError("path must be a list")
        queue: list[dict] = []
        for step_idx, node in enumerate(path):
            if not isinstance(node, PlanNode):
                raise TypeError("path must contain PlanNode instances")
            if node.parent_id is None:
                continue
            queue.append(
                {
                    "step": step_idx,
                    "plan_id": node.id,
                    "depth": node.depth,
                    "description": node.description,
                    "expected_tool": node.expected_tool,
                    "expected_effect": node.expected_effect,
                    "score": node.score,
                }
            )
        return queue

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _path_from_root(node: PlanNode) -> list[PlanNode]:
        # Without an index we can only return the node itself; callers
        # that want ancestry should walk the tree separately. This is
        # passed to generate_fn purely as local context.
        return [node]

    def _coerce_proposal(
        self, raw: Any, parent: PlanNode
    ) -> Optional[PlanNode]:
        if not isinstance(raw, dict):
            _LOG.warning(
                "dropping proposal of type %s (expected dict)", type(raw).__name__
            )
            return None
        missing = [k for k in _REQUIRED_KEYS if k not in raw]
        if missing:
            _LOG.warning("dropping proposal missing keys %s", missing)
            return None
        description = raw.get("description")
        effect = raw.get("expected_effect")
        tool = raw.get("expected_tool")
        if not isinstance(description, str) or not isinstance(effect, str):
            _LOG.warning(
                "dropping proposal with non-string description/effect"
            )
            return None
        if tool is not None and not isinstance(tool, str):
            _LOG.warning("dropping proposal with non-string expected_tool")
            return None
        return PlanNode(
            id=uuid.uuid4().hex,
            parent_id=parent.id,
            depth=parent.depth + 1,
            description=description,
            expected_tool=tool,
            expected_effect=effect,
            score=0.0,
        )

    def _safe_score(self, node: PlanNode) -> float:
        try:
            value = self._scorer(node)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "scorer_fn raised %s: %s; clamping to -inf",
                type(exc).__name__,
                exc,
            )
            return -math.inf
        try:
            value = float(value)
        except (TypeError, ValueError):
            _LOG.warning(
                "scorer_fn returned non-numeric %r; clamping to -inf", value
            )
            return -math.inf
        if math.isnan(value) or math.isinf(value):
            _LOG.warning("scorer_fn returned non-finite %r; clamping to -inf", value)
            return -math.inf
        return value

    def _index_tree(self, node: PlanNode, index: dict[str, PlanNode]) -> None:
        index[node.id] = node
        for child in node.children:
            self._index_tree(child, index)
