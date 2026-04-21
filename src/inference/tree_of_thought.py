"""Tree of Thought (ToT) Decoder — Yao et al. 2023.

Reference: "Tree of Thoughts: Deliberate Problem Solving with Large Language
Models" (arXiv:2305.10601).

ToT frames problem solving as a search over a tree of intermediate reasoning
steps ("thoughts").  At each node the model proposes several candidate next
thoughts, evaluates each one with a heuristic value in [0, 1], and then
either expands the best candidates (BFS) or dives deeper with backtracking
on dead-ends (DFS).

Search modes
------------
* **BFS** — maintain a frontier of at most *breadth_limit* best-valued nodes
  at each depth level.  Expand all frontier nodes, score the resulting
  children, prune to the top-b again, and repeat until *max_depth* is reached
  or a terminal node is found.
* **DFS** — explore depth-first; backtrack whenever all children fall below
  *value_threshold* or when a terminal node is found.

Both modes return the full :class:`ToTTree` and the best (highest-value)
leaf node found.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ToTConfig:
    """Hyper-parameters for Tree-of-Thought search.

    Attributes
    ----------
    search_mode:
        ``"bfs"`` or ``"dfs"``.
    breadth_limit:
        BFS frontier width *b* — the number of best-valued nodes kept at
        each depth level.
    max_depth:
        Maximum depth of the search tree.
    n_candidates:
        Number of candidate thoughts (*k*) requested from ``propose_fn``
        per expansion.
    value_threshold:
        DFS pruning threshold.  A child node whose value is strictly below
        this threshold is not expanded further.
    aggregation:
        How to aggregate a node's value when multiple evaluations are
        possible in the future.  Currently ``"max"`` or ``"mean"``; stored
        for downstream use.
    """

    search_mode: str = "bfs"
    breadth_limit: int = 5
    max_depth: int = 6
    n_candidates: int = 3
    value_threshold: float = 0.3
    aggregation: str = "max"


# ---------------------------------------------------------------------------
# Tree data structures
# ---------------------------------------------------------------------------

@dataclass
class ThoughtNode:
    """A single node in the Tree-of-Thought search tree.

    Attributes
    ----------
    state:
        Accumulated reasoning text up to this node.
    depth:
        Distance from the root (root has depth 0).
    value:
        Estimated quality of this node in [0, 1].
    parent_id:
        ``node_id`` of the parent node, or ``None`` for the root.
    node_id:
        Unique integer identifier assigned at insertion time.
    children_ids:
        ``node_id`` values of all direct children.
    is_terminal:
        ``True`` when ``terminal_fn(state)`` returned ``True`` for this node.
    """

    state: str
    depth: int
    value: float
    parent_id: Optional[int]
    node_id: int
    children_ids: List[int] = field(default_factory=list)
    is_terminal: bool = False


class ToTTree:
    """Mutable tree data structure for Tree-of-Thought search.

    Nodes are identified by integer ids assigned sequentially starting
    from 0.  The first node added (id 0) is treated as the root.
    """

    def __init__(self) -> None:
        self._nodes: Dict[int, ThoughtNode] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(
        self,
        state: str,
        depth: int,
        value: float,
        parent_id: Optional[int],
    ) -> int:
        """Insert a new node and return its ``node_id``.

        If *parent_id* is not ``None`` the parent's ``children_ids`` list
        is updated automatically.

        Parameters
        ----------
        state:
            Accumulated reasoning text for this node.
        depth:
            Depth level of this node (root = 0).
        value:
            Heuristic quality score in [0, 1].
        parent_id:
            ``node_id`` of the parent, or ``None`` for the root.

        Returns
        -------
        int
            The ``node_id`` assigned to the newly created node.
        """
        nid = self._next_id
        self._next_id += 1
        node = ThoughtNode(
            state=state,
            depth=depth,
            value=value,
            parent_id=parent_id,
            node_id=nid,
        )
        self._nodes[nid] = node
        if parent_id is not None and parent_id in self._nodes:
            self._nodes[parent_id].children_ids.append(nid)
        return nid

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_node(self, node_id: int) -> ThoughtNode:
        """Return the node with the given *node_id*.

        Raises
        ------
        KeyError
            If *node_id* does not exist in the tree.
        """
        return self._nodes[node_id]

    def children(self, node_id: int) -> List[ThoughtNode]:
        """Return all direct children of *node_id*.

        Parameters
        ----------
        node_id:
            Id of the parent node.

        Returns
        -------
        list of ThoughtNode
            Direct children in insertion order.
        """
        parent = self._nodes[node_id]
        return [self._nodes[cid] for cid in parent.children_ids]

    def path_to_root(self, node_id: int) -> List[ThoughtNode]:
        """Return the path from the root down to *node_id* (inclusive).

        Parameters
        ----------
        node_id:
            Id of the target node.

        Returns
        -------
        list of ThoughtNode
            Nodes ordered from root (index 0) to *node_id* (last index).
        """
        path: List[ThoughtNode] = []
        current_id: Optional[int] = node_id
        while current_id is not None:
            node = self._nodes[current_id]
            path.append(node)
            current_id = node.parent_id
        path.reverse()
        return path

    def best_leaf(self) -> ThoughtNode:
        """Return the leaf node with the highest value.

        A leaf is any node with no children.  In the degenerate case where
        all nodes have children (impossible after at least one expansion),
        the node with maximum value across all nodes is returned.

        Returns
        -------
        ThoughtNode
            The leaf with the highest ``value``.

        Raises
        ------
        ValueError
            If the tree is empty.
        """
        if not self._nodes:
            raise ValueError("Tree is empty — no best leaf.")
        leaves = [n for n in self._nodes.values() if not n.children_ids]
        if not leaves:
            leaves = list(self._nodes.values())
        return max(leaves, key=lambda n: n.value)

    def size(self) -> int:
        """Return the total number of nodes in the tree."""
        return len(self._nodes)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TreeOfThoughtDecoder:
    """BFS / DFS search over a tree of intermediate reasoning steps.

    The decoder is model-agnostic: it accepts *callable* hooks for thought
    proposal, state evaluation, and terminal detection, making it trivial to
    stub out in tests.

    Parameters
    ----------
    config:
        :class:`ToTConfig` controlling all search hyper-parameters.
    """

    def __init__(self, config: ToTConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def generate_thoughts(
        self,
        state: str,
        propose_fn: Callable[[str], List[str]],
    ) -> List[str]:
        """Generate up to *n_candidates* candidate next thoughts.

        Parameters
        ----------
        state:
            The current accumulated reasoning state.
        propose_fn:
            Callable that takes *state* and returns a list of candidate
            thought strings.

        Returns
        -------
        list of str
            Up to ``config.n_candidates`` candidate thoughts (order
            preserved from *propose_fn*).
        """
        candidates = propose_fn(state)
        return candidates[: self.config.n_candidates]

    def evaluate_state(
        self,
        state: str,
        value_fn: Callable[[str], float],
    ) -> float:
        """Score *state* using *value_fn*.

        Parameters
        ----------
        state:
            Accumulated reasoning text to evaluate.
        value_fn:
            Callable that returns a float in [0, 1] for *state*.

        Returns
        -------
        float
            The value returned by *value_fn*.
        """
        return value_fn(state)

    # ------------------------------------------------------------------
    # BFS
    # ------------------------------------------------------------------

    def bfs(
        self,
        initial_state: str,
        propose_fn: Callable[[str], List[str]],
        value_fn: Callable[[str], float],
        terminal_fn: Callable[[str], bool],
    ) -> ToTTree:
        """Breadth-first search over the thought tree.

        At each depth level the algorithm:

        1. Expands every node in the current frontier.
        2. Evaluates all newly created children.
        3. Marks terminal children (``terminal_fn`` returns ``True``).
        4. Prunes to the top ``config.breadth_limit`` children by value to
           form the next frontier.

        Search stops when *max_depth* is reached, the frontier is empty, or
        a terminal node is found.

        Parameters
        ----------
        initial_state:
            Text for the root node.
        propose_fn:
            State → list of candidate thought strings.
        value_fn:
            State → scalar value in [0, 1].
        terminal_fn:
            State → bool; ``True`` means this is a solution/stop node.

        Returns
        -------
        ToTTree
            The fully built search tree.
        """
        tree = ToTTree()
        root_value = self.evaluate_state(initial_state, value_fn)
        root_id = tree.add_node(
            state=initial_state,
            depth=0,
            value=root_value,
            parent_id=None,
        )
        root = tree.get_node(root_id)
        root.is_terminal = terminal_fn(initial_state)

        if root.is_terminal:
            return tree

        frontier: List[int] = [root_id]

        for _depth in range(self.config.max_depth):
            if not frontier:
                break

            next_candidates: List[Tuple[int, float]] = []  # (node_id, value)

            for parent_id in frontier:
                parent_node = tree.get_node(parent_id)
                thoughts = self.generate_thoughts(parent_node.state, propose_fn)

                for thought in thoughts:
                    new_state = parent_node.state + " " + thought
                    val = self.evaluate_state(new_state, value_fn)
                    child_depth = parent_node.depth + 1
                    child_id = tree.add_node(
                        state=new_state,
                        depth=child_depth,
                        value=val,
                        parent_id=parent_id,
                    )
                    child = tree.get_node(child_id)
                    child.is_terminal = terminal_fn(new_state)
                    next_candidates.append((child_id, val))

                    if child.is_terminal:
                        # Return as soon as any terminal is found
                        return tree

            # Keep the breadth_limit best candidates for the next frontier
            next_candidates.sort(key=lambda x: x[1], reverse=True)
            frontier = [nid for nid, _ in next_candidates[: self.config.breadth_limit]]

        return tree

    # ------------------------------------------------------------------
    # DFS
    # ------------------------------------------------------------------

    def dfs(
        self,
        initial_state: str,
        propose_fn: Callable[[str], List[str]],
        value_fn: Callable[[str], float],
        terminal_fn: Callable[[str], bool],
        path: Optional[List[int]] = None,
        tree: Optional[ToTTree] = None,
    ) -> ToTTree:
        """Depth-first search with value-threshold pruning.

        Recursively expands the current node's children in order of
        decreasing value.  A child is skipped if its value is strictly
        below ``config.value_threshold``.  Backtracking occurs naturally
        via recursion when all children are pruned or max depth is reached.

        Parameters
        ----------
        initial_state:
            Text for the root node (only used on the outermost call).
        propose_fn:
            State → list of candidate thought strings.
        value_fn:
            State → scalar value in [0, 1].
        terminal_fn:
            State → bool; ``True`` means this is a solution/stop node.
        path:
            Internal — list of node ids on the current DFS path.  Pass
            ``None`` on the first call.
        tree:
            Internal — shared :class:`ToTTree` instance.  Pass ``None``
            on the first call.

        Returns
        -------
        ToTTree
            The fully built search tree after DFS completes.
        """
        # Initialise on first call
        if tree is None:
            tree = ToTTree()
            root_value = self.evaluate_state(initial_state, value_fn)
            root_id = tree.add_node(
                state=initial_state,
                depth=0,
                value=root_value,
                parent_id=None,
            )
            root = tree.get_node(root_id)
            root.is_terminal = terminal_fn(initial_state)
            path = [root_id]
            if root.is_terminal:
                return tree

        assert path is not None  # always set after first call

        current_id = path[-1]
        current_node = tree.get_node(current_id)

        if current_node.is_terminal or current_node.depth >= self.config.max_depth:
            return tree

        thoughts = self.generate_thoughts(current_node.state, propose_fn)

        # Build children and sort by value descending
        child_entries: List[Tuple[int, float]] = []
        for thought in thoughts:
            new_state = current_node.state + " " + thought
            val = self.evaluate_state(new_state, value_fn)
            child_depth = current_node.depth + 1
            child_id = tree.add_node(
                state=new_state,
                depth=child_depth,
                value=val,
                parent_id=current_id,
            )
            child = tree.get_node(child_id)
            child.is_terminal = terminal_fn(new_state)
            child_entries.append((child_id, val))

        child_entries.sort(key=lambda x: x[1], reverse=True)

        for child_id, val in child_entries:
            if val < self.config.value_threshold:
                # Prune — skip this subtree entirely
                continue
            child_node = tree.get_node(child_id)
            if child_node.is_terminal:
                return tree
            self.dfs(
                initial_state=initial_state,
                propose_fn=propose_fn,
                value_fn=value_fn,
                terminal_fn=terminal_fn,
                path=path + [child_id],
                tree=tree,
            )

        return tree

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    def search(
        self,
        initial_state: str,
        propose_fn: Callable[[str], List[str]],
        value_fn: Callable[[str], float],
        terminal_fn: Optional[Callable[[str], bool]] = None,
    ) -> Tuple["ToTTree", "ThoughtNode"]:
        """Run ToT search and return the tree plus the best node found.

        Dispatches to :meth:`bfs` or :meth:`dfs` according to
        ``config.search_mode``.

        Parameters
        ----------
        initial_state:
            Text for the root node.
        propose_fn:
            State → list of candidate thought strings.
        value_fn:
            State → scalar value in [0, 1].
        terminal_fn:
            Optional predicate; defaults to ``lambda s: False`` (no early
            stopping).

        Returns
        -------
        tuple of (ToTTree, ThoughtNode)
            The completed search tree and the leaf node with the highest
            value.

        Raises
        ------
        ValueError
            If ``config.search_mode`` is not ``"bfs"`` or ``"dfs"``.
        """
        if terminal_fn is None:
            terminal_fn = lambda s: False  # noqa: E731

        if self.config.search_mode == "bfs":
            tree = self.bfs(initial_state, propose_fn, value_fn, terminal_fn)
        elif self.config.search_mode == "dfs":
            tree = self.dfs(initial_state, propose_fn, value_fn, terminal_fn)
        else:
            raise ValueError(
                f"Unknown search_mode {self.config.search_mode!r}; "
                "expected 'bfs' or 'dfs'."
            )

        best = tree.best_leaf()
        return tree, best


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from src.inference import DECODER_REGISTRY  # noqa: E402

DECODER_REGISTRY["tree_of_thought"] = TreeOfThoughtDecoder

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ToTConfig",
    "ThoughtNode",
    "ToTTree",
    "TreeOfThoughtDecoder",
]
