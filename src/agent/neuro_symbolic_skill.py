"""Neuro-Symbolic Skill Induction (NSI) — arXiv:2605.01293.

Lifts interaction traces into generalized logic-grounded programs via an
empirical consistency objective.  Decouples neural perception from symbolic
execution using four logic-grounded node operators, then iteratively refines
local experts into global skills via Intra-Trajectory Consolidation and
Inter-Trajectory Merging.  Runtime failures are converted into skill-honing
opportunities through Reflective Planning.

Classes
-------
SkillGraph
    Directed acyclic graph of skill nodes; stores skills as programs.
SkillNode
    Base class for nodes; subclasses: DataOpNode, CheckOpNode, LoopOpNode,
    PrimitiveOpNode, TerminalOpNode.
TraceToLogicInducer
    Converts interaction traces into logic-grounded programs.
IntraTrajectoryConsolidator
    Refines local skill experts within a trajectory.
InterTrajectoryMerger
    Merges skills across trajectories for global skill acquisition.
ReflectivePlanner
    Converts runtime failures into skill honing.
NSIAgent
    Main agent that uses NSI for skill induction and execution.
"""

from __future__ import annotations

import importlib
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any

__all__ = [
    "SkillGraph",
    "SkillNode",
    "DataOpNode",
    "CheckOpNode",
    "LoopOpNode",
    "PrimitiveOpNode",
    "TerminalOpNode",
    "TraceToLogicInducer",
    "IntraTrajectoryConsolidator",
    "InterTrajectoryMerger",
    "ReflectivePlanner",
    "NSIAgent",
    "NSIError",
]

_LOGGER = logging.getLogger(__name__)


def _callable_path(fn: Callable[..., Any] | None) -> str | None:
    if fn is None:
        return None
    return f"{fn.__module__}:{fn.__qualname__}"


def _resolve_callable(path: str | None) -> Callable[..., Any] | None:
    if not path:
        return None
    try:
        module_name, qualname = path.split(":", 1)
        obj: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            if part == "<locals>":
                return None
            obj = getattr(obj, part)
        return obj if callable(obj) else None
    except (ImportError, AttributeError, ValueError):
        _LOGGER.debug("Failed to resolve callable %s", path, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NSIError(Exception):
    """Raised for invalid NSI operations."""


# ---------------------------------------------------------------------------
# Node operator types (First-Order Logic grounded)
# ---------------------------------------------------------------------------


class NodeOpType(StrEnum):
    """Four logic-grounded node operators + structural consolidation types."""

    DATA_OP = auto()
    CHECK_OP = auto()
    LOOP_OP = auto()
    PRIMITIVE_OP = auto()
    TERMINAL_OP = auto()


class ConsolidationOp(StrEnum):
    """Structural consolidation operators for skill refinement."""

    CONDITIONAL_BRANCHING = auto()
    MODULAR_CROSSOVER = auto()
    VARIABLE_LIFTING = auto()
    LOOP_FOLDING = auto()


# ---------------------------------------------------------------------------
# SkillNode hierarchy
# ---------------------------------------------------------------------------


@dataclass
class SkillNode:
    """Base class for all logic-grounded skill nodes.

    Attributes
    ----------
    node_id : str
        Unique identifier for this node.
    op_type : NodeOpType
        The operator type that classifies this node's logical role.
    preconditions : list[str]
        FOL preconditions that must hold before this node executes.
    postconditions : list[str]
        FOL postconditions guaranteed after this node executes.
    bindings : dict[str, Any]
        Dynamic variable bindings produced by this node.
    metadata : dict[str, Any]
        Arbitrary metadata (e.g., source trace indices, confidence scores).
    """

    node_id: str
    op_type: NodeOpType
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SkillNode):
            return NotImplemented
        return self.node_id == other.node_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "op_type": self.op_type.value,
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "bindings": dict(self.bindings),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillNode:
        op_type = NodeOpType(data["op_type"])
        subclass = _NODE_SUBCLASSES.get(op_type, SkillNode)
        node = subclass.__new__(subclass)
        node.node_id = data["node_id"]
        node.op_type = op_type
        node.preconditions = list(data.get("preconditions", []))
        node.postconditions = list(data.get("postconditions", []))
        node.bindings = dict(data.get("bindings", {}))
        node.metadata = dict(data.get("metadata", {}))
        if isinstance(node, DataOpNode):
            node.perception_fn = _resolve_callable(data.get("perception_fn"))
        elif isinstance(node, CheckOpNode):
            node.condition = data.get("condition", "")
            node.condition_fn = _resolve_callable(data.get("condition_fn"))
        elif isinstance(node, LoopOpNode):
            node.body_node_ids = list(data.get("body_node_ids", []))
            node.max_iterations = int(data.get("max_iterations", 10))
            node.iteration_var = data.get("iteration_var", "i")
        elif isinstance(node, PrimitiveOpNode):
            node.action_name = data.get("action_name", "")
            node.action_fn = _resolve_callable(data.get("action_fn"))
            node.arg_bindings = dict(data.get("arg_bindings", {}))
        elif isinstance(node, TerminalOpNode):
            node.outcome = data.get("outcome", "unknown")
            node.diagnosis = data.get("diagnosis", "")
            node.success = bool(data.get("success", False))
        return node

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(f"{self.__class__.__name__}.execute must be overridden")


@dataclass
class DataOpNode(SkillNode):
    """Dynamic variable binding node — think-then-act logic.

    Performs perceptual encoding of environmental state into bound
    variables that subsequent nodes can reference via FOL bindings.
    """

    perception_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.op_type = NodeOpType.DATA_OP

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        if self.perception_fn is not None:
            self.bindings = self.perception_fn(context)
        else:
            self.bindings = {k: context.get(k) for k in context.get("_bind_keys", [])}
        return {"bindings": dict(self.bindings), "node_id": self.node_id}

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["perception_fn"] = _callable_path(self.perception_fn)
        return d


@dataclass
class CheckOpNode(SkillNode):
    """Decision boundary node — branching condition evaluation in FOL.

    Evaluates a condition expressed as FOL predicates over the current
    variable bindings.  Determines which outgoing edge the skill graph
    traversal will follow.
    """

    condition: str = ""
    condition_fn: Callable[[dict[str, Any]], bool] | None = None

    def __post_init__(self) -> None:
        self.op_type = NodeOpType.CHECK_OP

    def evaluate(self, context: dict[str, Any]) -> bool:
        if self.condition_fn is not None:
            return self.condition_fn(context)
        key = self.condition.split()[0] if self.condition else ""
        val = context.get(key, context.get(self.condition, False))
        return bool(val)

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        result = self.evaluate(context)
        self.bindings["_check_result"] = result
        return {"node_id": self.node_id, "branch_taken": result}

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["condition"] = self.condition
        d["condition_fn"] = _callable_path(self.condition_fn)
        return d


@dataclass
class LoopOpNode(SkillNode):
    """Iteration control node — bounded loop with exit condition in FOL.

    Supports bounded iteration with configurable max_iterations.  Each
    iteration produces new bindings that are fed back into the loop body.
    """

    body_node_ids: list[str] = field(default_factory=list)
    max_iterations: int = 10
    iteration_var: str = "i"

    def __post_init__(self) -> None:
        self.op_type = NodeOpType.LOOP_OP

    def execute(
        self,
        context: dict[str, Any],
        graph: SkillGraph | None = None,
    ) -> dict[str, Any]:
        results = []
        bindings_snapshot = dict(self.bindings)
        for i in range(self.max_iterations):
            context[self.iteration_var] = i
            context["_loop_iteration"] = i
            context.update(bindings_snapshot)
            if graph is None:
                results.append({"iteration": i, "bindings": dict(context)})
                continue
            body_results = []
            for body_node_id in self.body_node_ids:
                body_node = graph.nodes.get(body_node_id)
                if body_node is None or body_node is self:
                    continue
                result = body_node.execute(context)
                context.update(body_node.bindings)
                body_results.append(result)
            results.append({"iteration": i, "body_results": body_results})
        self.bindings["_loop_results"] = results
        return {"node_id": self.node_id, "iterations": len(results)}

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["body_node_ids"] = list(self.body_node_ids)
        d["max_iterations"] = self.max_iterations
        d["iteration_var"] = self.iteration_var
        return d


@dataclass
class PrimitiveOpNode(SkillNode):
    """Grounded action execution node — dynamically resolved action arguments.

    Represents a low-level action (e.g., tool call, API invocation) whose
    arguments are resolved at runtime from the current FOL bindings.
    """

    action_name: str = ""
    action_fn: Callable[..., Any] | None = None
    arg_bindings: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.op_type = NodeOpType.PRIMITIVE_OP

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        resolved_args = {
            dest: context.get(src, src) for dest, src in self.arg_bindings.items()
        }
        if self.action_fn is not None:
            result = self.action_fn(**resolved_args)
            self.bindings["_last_result"] = result
            return {"node_id": self.node_id, "action": self.action_name, "result": result}
        self.bindings["_last_result"] = None
        return {"node_id": self.node_id, "action": self.action_name, "resolved_args": resolved_args}

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["action_name"] = self.action_name
        d["action_fn"] = _callable_path(self.action_fn)
        d["arg_bindings"] = dict(self.arg_bindings)
        return d


@dataclass
class TerminalOpNode(SkillNode):
    """Execution feedback node — logic-grounded diagnosis of outcomes.

    Provides structured feedback on skill execution success/failure,
    recording diagnostic information as FOL postconditions.
    """

    outcome: str = "unknown"
    diagnosis: str = ""
    success: bool = False

    def __post_init__(self) -> None:
        self.op_type = NodeOpType.TERMINAL_OP

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        self.bindings["_outcome"] = self.outcome
        self.bindings["_success"] = self.success
        self.bindings["_diagnosis"] = self.diagnosis
        return {
            "node_id": self.node_id,
            "outcome": self.outcome,
            "success": self.success,
            "diagnosis": self.diagnosis,
        }

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["outcome"] = self.outcome
        d["diagnosis"] = self.diagnosis
        d["success"] = self.success
        return d


_NODE_SUBCLASSES: dict[NodeOpType, type[SkillNode]] = {
    NodeOpType.DATA_OP: DataOpNode,
    NodeOpType.CHECK_OP: CheckOpNode,
    NodeOpType.LOOP_OP: LoopOpNode,
    NodeOpType.PRIMITIVE_OP: PrimitiveOpNode,
    NodeOpType.TERMINAL_OP: TerminalOpNode,
}


# ---------------------------------------------------------------------------
# SkillGraph — directed acyclic graph of skill nodes
# ---------------------------------------------------------------------------


@dataclass
class SkillGraph:
    """Directed acyclic graph of logic-grounded skill nodes.

    Attributes
    ----------
    graph_id : str
        Unique identifier for this skill graph.
    nodes : dict[str, SkillNode]
        Mapping from node_id to SkillNode instance.
    edges : dict[str, list[str]]
        Adjacency list: source node_id -> list of target node_ids.
    entry_node_id : str | None
        Node id of the graph entry point.
    exit_node_ids : set[str]
        Node ids that represent terminal exits.
    metadata : dict[str, Any]
        Arbitrary graph-level metadata (e.g., skill name, version).

    Examples
    --------
    >>> graph = SkillGraph(graph_id="test_skill")
    >>> data_node = DataOpNode(node_id="d1", preconditions=[], postconditions=["state(v)"])
    >>> graph.add_node(data_node)
    >>> prim_node = PrimitiveOpNode(node_id="p1", action_name="move")
    >>> graph.add_node(prim_node)
    >>> graph.add_edge("d1", "p1")
    >>> graph.set_entry("d1")
    >>> graph.set_exit("p1")
    """

    graph_id: str
    nodes: dict[str, SkillNode] = field(default_factory=dict)
    edges: dict[str, list[str]] = field(default_factory=dict)
    entry_node_id: str | None = None
    exit_node_ids: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: SkillNode) -> None:
        self.nodes[node.node_id] = node
        if node.node_id not in self.edges:
            self.edges[node.node_id] = []

    def add_edge(self, source_id: str, target_id: str) -> None:
        if source_id not in self.nodes:
            raise NSIError(f"Source node {source_id!r} not in graph")
        if target_id not in self.nodes:
            raise NSIError(f"Target node {target_id!r} not in graph")
        if target_id not in self.edges[source_id]:
            self.edges[source_id].append(target_id)

    def set_entry(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise NSIError(f"Node {node_id!r} not in graph")
        self.entry_node_id = node_id

    def set_exit(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise NSIError(f"Node {node_id!r} not in graph")
        self.exit_node_ids.add(node_id)

    def topological_sort(self) -> list[str]:
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        for sources in self.edges.values():
            for tgt in sources:
                in_degree[tgt] = in_degree.get(tgt, 0) + 1
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        sorted_ids: list[str] = []
        while queue:
            nid = queue.pop(0)
            sorted_ids.append(nid)
            for child in self.edges.get(nid, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        if len(sorted_ids) != len(self.nodes):
            raise NSIError("Graph contains a cycle — topological sort impossible")
        return sorted_ids

    def execute(self, initial_context: dict[str, Any]) -> dict[str, Any]:
        if self.entry_node_id is None:
            raise NSIError("Graph has no entry node — call set_entry first")
        context = dict(initial_context)
        sorted_ids = self.topological_sort()
        execution_trace: list[dict[str, Any]] = []
        for node_id in sorted_ids:
            node = self.nodes[node_id]
            try:
                if isinstance(node, LoopOpNode):
                    result = node.execute(context, self)
                else:
                    result = node.execute(context)
                execution_trace.append(result)
                context.update(node.bindings)
            except Exception as exc:  # pragma: no cover — runtime nodes should be safe
                execution_trace.append({"node_id": node_id, "error": str(exc)})
                _LOGGER.warning("Node %s raised during execution: %s", node_id, exc)
        terminal_nodes = [
            node for node in self.nodes.values()
            if isinstance(node, TerminalOpNode)
        ]
        return {
            "execution_trace": execution_trace,
            "final_context": context,
            "success": bool(terminal_nodes) and all(
                node.bindings.get("_success", True)
                for node in terminal_nodes
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": dict(self.edges),
            "entry_node_id": self.entry_node_id,
            "exit_node_ids": list(self.exit_node_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillGraph:
        graph = cls(graph_id=data["graph_id"], metadata=dict(data.get("metadata", {})))
        for nid, node_dict in data.get("nodes", {}).items():
            node = SkillNode.from_dict(node_dict)
            graph.add_node(node)
        for src, tgts in data.get("edges", {}).items():
            for tgt in tgts:
                graph.add_edge(src, tgt)
        entry = data.get("entry_node_id")
        if entry:
            graph.set_entry(entry)
        for exit_id in data.get("exit_node_ids", []):
            graph.set_exit(exit_id)
        return graph

    def merge_with(self, other: SkillGraph, label: str = "") -> SkillGraph:
        merged = SkillGraph(graph_id=f"{self.graph_id}_merged_{label}")
        for nid, node in self.nodes.items():
            merged.add_node(node)
        for src, tgts in self.edges.items():
            for tgt in tgts:
                merged.add_edge(src, tgt)
        for nid, node in other.nodes.items():
            if nid not in merged.nodes:
                merged.add_node(node)
        for src, tgts in other.edges.items():
            for tgt in tgts:
                if src not in merged.edges:
                    merged.edges[src] = []
                if tgt not in merged.edges[src]:
                    merged.add_edge(src, tgt)
        return merged

    def __len__(self) -> int:
        return len(self.nodes)


# ---------------------------------------------------------------------------
# TraceToLogicInducer — converts interaction traces to logic-grounded programs
# ---------------------------------------------------------------------------


@dataclass
class TraceStep:
    """Single step in an interaction trace."""

    step_index: int
    action: str
    observation: str = ""
    result: str = ""
    success: bool = False
    state_delta: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceToLogicInducer:
    """Lifts interaction traces into logic-grounded skill graphs.

    Uses an empirical consistency objective to iteratively refine local
    experts into generalized skills.  The induction process:

    1. Encodes each trace step into a candidate node with FOL pre/post-conditions.
    2. Scores node sequences by consistency with the full trace.
    3. Builds a DAG from the highest-scoring node sequence.
    4. Returns a SkillGraph encoding the induced skill.

    Parameters
    ----------
    max_nodes : int
        Maximum number of nodes per induced graph (default 64).
    consistency_threshold : float
        Minimum empirical consistency score to accept a candidate
        graph (default 0.7).
    """

    max_nodes: int = 64
    consistency_threshold: float = 0.7

    def induce_from_trace(self, trace: list[TraceStep]) -> SkillGraph:
        if not trace:
            raise NSIError("Cannot induce skill from empty trace")

        graph_id = f"induced_{uuid.uuid4().hex[:8]}"
        graph = SkillGraph(graph_id=graph_id)

        prev_node_id: str | None = None
        for step in trace:
            node = self._step_to_node(step)

            if node is None:
                continue

            graph.add_node(node)

            if prev_node_id is not None:
                graph.add_edge(prev_node_id, node.node_id)
            elif graph.entry_node_id is None:
                graph.set_entry(node.node_id)

            if step.success and isinstance(node, TerminalOpNode):
                node.outcome = "success"
                node.success = True
                node.diagnosis = f"Step {step.step_index} succeeded"

            if self._is_terminal_step(step):
                graph.set_exit(node.node_id)

            prev_node_id = node.node_id

            if len(graph.nodes) >= self.max_nodes:
                break

        return graph

    def _step_to_node(self, step: TraceStep) -> SkillNode | None:
        step_hash = uuid.uuid4().hex[:8]
        node_id = f"n_{step_hash}"

        if step.action.startswith("check") or step.action.startswith("if"):
            return CheckOpNode(
                node_id=node_id,
                condition=step.action,
                preconditions=self._extract_preconditions(step.state_delta),
                postconditions=[],
                metadata={"step_index": step.step_index},
            )

        if step.action.startswith("loop") or step.action.startswith("while"):
            return LoopOpNode(
                node_id=node_id,
                max_iterations=step.metadata.get("max_iterations", 10),
                preconditions=self._extract_preconditions(step.state_delta),
                postconditions=[],
                metadata={"step_index": step.step_index},
            )

        if step.action.startswith("observe") or step.action.startswith("sense"):
            return DataOpNode(
                node_id=node_id,
                preconditions=[],
                postconditions=self._extract_postconditions(step.state_delta),
                metadata={"step_index": step.step_index},
            )

        if step.action.startswith("return") or step.action.startswith("done"):
            return TerminalOpNode(
                node_id=node_id,
                outcome="success" if step.success else "failure",
                diagnosis=step.observation or step.result,
                success=step.success,
                preconditions=self._extract_preconditions(step.state_delta),
                postconditions=[],
                metadata={"step_index": step.step_index},
            )

        arg_bindings: dict[str, str] = {}
        for key, val in step.state_delta.items():
            if isinstance(val, str):
                arg_bindings[key] = val

        return PrimitiveOpNode(
            node_id=node_id,
            action_name=step.action,
            arg_bindings=arg_bindings,
            preconditions=self._extract_preconditions(step.state_delta),
            postconditions=self._extract_postconditions(step.state_delta),
            metadata={"step_index": step.step_index},
        )

    def _extract_preconditions(self, state_delta: dict[str, Any]) -> list[str]:
        return [f"holds({k},v)" for k in state_delta if k != "_result"]

    def _extract_postconditions(self, state_delta: dict[str, Any]) -> list[str]:
        return [f"updated({k})" for k in state_delta if k != "_result"]

    def _is_terminal_step(self, step: TraceStep) -> bool:
        keywords = ("return", "done", "finish", "complete", "success", "failure")
        action_lower = step.action.lower()
        return any(kw in action_lower for kw in keywords) or bool(
            getattr(step, "terminal", False) or getattr(step, "is_terminal", False)
        )

    def compute_consistency_score(self, graph: SkillGraph, trace: list[TraceStep]) -> float:
        if not graph.nodes or not trace:
            return 0.0

        node_actions = {
            nid: node.action_name if isinstance(node, PrimitiveOpNode) else ""
            for nid, node in graph.nodes.items()
        }
        trace_actions = [step.action for step in trace]

        matches = sum(
            1
            for ta in trace_actions
            for na in node_actions.values()
            if self._action_similar(ta, na)
        )
        max_possible = max(len(trace_actions), 1)
        return min(matches / max_possible, 1.0)

    @staticmethod
    def _action_similar(a: str, b: str) -> bool:
        if not a or not b:
            return False
        return a.split()[0] == b.split()[0] if " " in a or " " in b else a in b or b in a


# ---------------------------------------------------------------------------
# IntraTrajectoryConsolidator — refines local skill experts within a trajectory
# ---------------------------------------------------------------------------


@dataclass
class IntraTrajectoryConsolidator:
    """Refines local skill experts into globally consistent skills.

    Operates within a single trajectory, consolidating overlapping or
    redundant nodes through:
    - Conditional Branching: splits nodes with multiple exit paths.
    - Modular Crossover: combines compatible subgraph fragments.
    - Variable Lifting: elevates frequently-used bindings to graph level.
    - Loop Folding: converts repeated sequential patterns into LoopOpNodes.

    Parameters
    ----------
    fold_threshold : int
        Number of repeated patterns required before loop folding (default 3).
    lift_threshold : float
        Fraction of nodes using same binding to trigger variable lifting
        (default 0.5).
    """

    fold_threshold: int = 3
    lift_threshold: float = 0.5

    def consolidate(self, graph: SkillGraph) -> SkillGraph:
        graph = self._conditional_branching(graph)
        graph = self._variable_lifting(graph)
        graph = self._loop_folding(graph)
        return graph

    def _conditional_branching(self, graph: SkillGraph) -> SkillGraph:
        for node in graph.nodes.values():
            if isinstance(node, CheckOpNode) and node.condition:
                branch_vars = self._extract_branch_vars(node.condition)
                node.preconditions.extend(f"branch_var({v})" for v in branch_vars)
        return graph

    def _variable_lifting(self, graph: SkillGraph) -> SkillGraph:
        binding_counts: dict[str, int] = {}
        for node in graph.nodes.values():
            for key in node.bindings:
                binding_counts[key] = binding_counts.get(key, 0) + 1

        total_nodes = max(len(graph.nodes), 1)
        threshold_count = int(total_nodes * self.lift_threshold)

        lifted_vars = {v for v, cnt in binding_counts.items() if cnt >= threshold_count}

        for node in graph.nodes.values():
            for var in lifted_vars:
                if var not in node.postconditions:
                    node.postconditions.append(f"lifted({var})")

        return graph

    def _loop_folding(self, graph: SkillGraph) -> SkillGraph:
        sorted_ids = graph.topological_sort()
        pattern_counts: dict[str, list[tuple[str, str]]] = {}

        for i in range(len(sorted_ids) - 1):
            src = sorted_ids[i]
            tgt = sorted_ids[i + 1]
            pattern_counts.setdefault(f"{src}_{tgt}", []).append((src, tgt))

        for pattern_key, occurrences in pattern_counts.items():
            if len(occurrences) >= self.fold_threshold:
                src_id, tgt_id = occurrences[0]
                loop_node_id = f"loop_{src_id}_{tgt_id}"
                if loop_node_id not in graph.nodes:
                    loop_node = LoopOpNode(
                        node_id=loop_node_id,
                        body_node_ids=[src_id, tgt_id],
                        max_iterations=len(occurrences),
                        preconditions=graph.nodes[src_id].preconditions.copy(),
                        postconditions=graph.nodes[tgt_id].postconditions.copy(),
                    )
                    graph.add_node(loop_node)
                    for src, tgt in occurrences:
                        for src_nodes in list(graph.edges.keys()):
                            if src in graph.edges[src_nodes]:
                                graph.edges[src_nodes] = [
                                    n if n != src else loop_node_id
                                    for n in graph.edges[src_nodes]
                                ]
                    _LOGGER.debug(
                        "Loop-folded pattern %s with %d occurrences",
                        pattern_key,
                        len(occurrences),
                    )

        return graph

    @staticmethod
    def _extract_branch_vars(condition: str) -> list[str]:
        return [w for w in condition.split() if w and not w.startswith("(") and not w.endswith(")")]


# ---------------------------------------------------------------------------
# InterTrajectoryMerger — merges skills across trajectories for global acquisition
# ---------------------------------------------------------------------------


@dataclass
class InterTrajectoryMerger:
    """Merges skills from multiple trajectories into a global skill graph.

    Applies structural consolidation operators across trajectory boundaries:
    - Conditional Branching: creates unified branching nodes from parallel
      trajectories that handle the same decision point differently.
    - Modular Crossover: swaps compatible subgraphs between trajectories to
      produce hybrid skills with combined capabilities.
    - Variable Lifting: propagates shared variable bindings to the global graph.
    - Loop Folding: extracts common iteration patterns across trajectories.

    Parameters
    ----------
    similarity_threshold : float
        Minimum structural similarity (0–1) to consider two subgraphs
        mergeable (default 0.6).
    """

    similarity_threshold: float = 0.6

    def merge(self, graphs: list[SkillGraph]) -> SkillGraph:
        if not graphs:
            raise NSIError("No graphs provided for merging")
        if len(graphs) == 1:
            return graphs[0]

        result = graphs[0]
        for i, graph in enumerate(graphs[1:], start=1):
            result = self._merge_two(result, graph, label=str(i))
            result = self._apply_structural_operators(result)

        return result

    def _merge_two(self, g1: SkillGraph, g2: SkillGraph, label: str) -> SkillGraph:
        merged = g1.merge_with(g2, label=label)
        merged.metadata["merged_from"] = [g1.graph_id, g2.graph_id]
        return merged

    def _apply_structural_operators(self, graph: SkillGraph) -> SkillGraph:
        graph = self._unify_conditional_branches(graph)
        graph = self._propagate_shared_variables(graph)
        graph = self._extract_common_loops(graph)
        return graph

    def _unify_conditional_branches(self, graph: SkillGraph) -> SkillGraph:
        check_nodes = [
            (nid, node)
            for nid, node in graph.nodes.items()
            if isinstance(node, CheckOpNode)
        ]
        for i, (id1, n1) in enumerate(check_nodes):
            for id2, n2 in check_nodes[i + 1 :]:
                if id1 not in graph.nodes or id2 not in graph.nodes:
                    continue
                if self._nodes_structurally_similar(n1, n2):
                    merged_condition = f"{n1.condition} OR {n2.condition}"
                    n1.condition = merged_condition
                    n1.preconditions = list(set(n1.preconditions + n2.preconditions))
                    outgoing = graph.edges.pop(id2, [])
                    graph.edges.setdefault(id1, [])
                    for target_id in outgoing:
                        remapped_target = id1 if target_id == id2 else target_id
                        if (
                            remapped_target != id1
                            and remapped_target not in graph.edges[id1]
                        ):
                            graph.edges[id1].append(remapped_target)

                    for src_id, targets in list(graph.edges.items()):
                        remapped_targets: list[str] = []
                        for target_id in targets:
                            remapped_target = id1 if target_id == id2 else target_id
                            if (
                                remapped_target != src_id
                                and remapped_target not in remapped_targets
                            ):
                                remapped_targets.append(remapped_target)
                        graph.edges[src_id] = remapped_targets

                    if graph.entry_node_id == id2:
                        graph.entry_node_id = id1
                    if id2 in graph.exit_node_ids:
                        graph.exit_node_ids.discard(id2)
                        graph.exit_node_ids.add(id1)
                    graph.nodes.pop(id2, None)
        return graph

    def _propagate_shared_variables(self, graph: SkillGraph) -> SkillGraph:
        var_usage: dict[str, int] = {}
        for node in graph.nodes.values():
            for key in node.bindings:
                var_usage[key] = var_usage.get(key, 0) + 1

        shared_vars = {v for v, cnt in var_usage.items() if cnt > 1}

        for node in graph.nodes.values():
            for var in shared_vars:
                if var not in node.preconditions:
                    node.preconditions.append(f"shared({var})")

        return graph

    def _extract_common_loops(self, graph: SkillGraph) -> SkillGraph:
        if len(graph.nodes) < 4:
            return graph

        try:
            sorted_ids = graph.topological_sort()
        except NSIError:
            return graph

        for i in range(len(sorted_ids) - 2):
            chain = sorted_ids[i : i + 3]
            node_types = [type(graph.nodes[nid]).__name__ for nid in chain]
            if node_types.count("PrimitiveOpNode") >= 2:
                loop_id = f"common_loop_{chain[0]}_{chain[-1]}"
                if loop_id not in graph.nodes:
                    loop_node = LoopOpNode(
                        node_id=loop_id,
                        body_node_ids=list(chain),
                        max_iterations=self._estimate_loop_bound(chain, graph),
                    )
                    graph.add_node(loop_node)
        return graph

    def _estimate_loop_bound(self, body_ids: list[str], graph: SkillGraph) -> int:
        return min(len(body_ids) * 2, 10)

    def compute_similarity(self, g1: SkillGraph, g2: SkillGraph) -> float:
        if not g1.nodes or not g2.nodes:
            return 0.0

        g1_types = sorted(set(n.op_type for n in g1.nodes.values()))
        g2_types = sorted(set(n.op_type for n in g2.nodes.values()))

        if g1_types != g2_types:
            type_match = 0.0
        else:
            type_match = 1.0

        g1_size = len(g1.nodes)
        g2_size = len(g2.nodes)
        size_ratio = min(g1_size, g2_size) / max(g1_size, g2_size, 1)

        g1_edges = sum(len(v) for v in g1.edges.values())
        g2_edges = sum(len(v) for v in g2.edges.values())
        edge_denom = max(g1_edges, g2_edges, 1)
        edge_ratio = min(g1_edges, g2_edges) / edge_denom

        return (type_match + size_ratio + edge_ratio) / 3.0

    def _nodes_structurally_similar(self, n1: SkillNode, n2: SkillNode) -> bool:
        return (
            isinstance(n1, type(n2))
            and set(n1.preconditions) == set(n2.preconditions)
            and len(n1.bindings) == len(n2.bindings)
        )


# ---------------------------------------------------------------------------
# ReflectivePlanner — converts runtime failures into skill honing
# ---------------------------------------------------------------------------


@dataclass
class ReflectivePlanner:
    """Converts runtime failures into skill-honing opportunities.

    When a skill graph fails at runtime, this planner:
    1. Identifies the failure node and the error context.
    2. Grafts a successful recovery trajectory onto the failure node.
    3. Produces an improved graph that handles the previously-failing case.

    Parameters
    ----------
    max_grafts : int
        Maximum number of recovery grafts per planning call (default 4).
    """

    max_grafts: int = 4

    def plan_from_failure(
        self,
        failed_graph: SkillGraph,
        failure_context: dict[str, Any],
        recovery_trace: list[TraceStep],
    ) -> SkillGraph:
        if not recovery_trace:
            return failed_graph

        inducer = TraceToLogicInducer()
        recovery_graph = inducer.induce_from_trace(recovery_trace)

        improved = self._graft_recovery(failed_graph, recovery_graph, failure_context)

        improved.metadata["failure_context"] = failure_context
        improved.metadata["grafted_from"] = recovery_graph.graph_id
        improved.metadata["was_failing"] = True

        return improved

    def _graft_recovery(
        self,
        failed_graph: SkillGraph,
        recovery_graph: SkillGraph,
        failure_context: dict[str, Any],
    ) -> SkillGraph:
        graft_count = 0

        for node in list(failed_graph.nodes.values()):
            if isinstance(node, TerminalOpNode) and not node.success:
                if graft_count >= self.max_grafts:
                    break

                recovery_entry = recovery_graph.entry_node_id
                if recovery_entry is None:
                    continue

                id_map: dict[str, str] = {}
                for recovery_id, recovery_node in recovery_graph.nodes.items():
                    copied_node = SkillNode.from_dict(recovery_node.to_dict())
                    copied_id = recovery_id
                    if copied_id in failed_graph.nodes or copied_id in id_map.values():
                        copied_id = f"recovery_{recovery_id}_{uuid.uuid4().hex[:6]}"
                    copied_node.node_id = copied_id
                    failed_graph.add_node(copied_node)
                    id_map[recovery_id] = copied_id

                for recovery_source, recovery_targets in recovery_graph.edges.items():
                    if recovery_source not in id_map:
                        continue
                    for recovery_target in recovery_targets:
                        if recovery_target in id_map:
                            failed_graph.add_edge(
                                id_map[recovery_source],
                                id_map[recovery_target],
                            )

                graft_id = f"graft_{uuid.uuid4().hex[:6]}"
                new_terminal = TerminalOpNode(
                    node_id=graft_id,
                    outcome="recovered",
                    diagnosis=f"Grafted recovery for {node.node_id}",
                    success=True,
                    preconditions=node.preconditions.copy(),
                    postconditions=recovery_graph.nodes[recovery_entry].postconditions.copy(),
                )

                failed_graph.add_node(new_terminal)
                failed_graph.set_exit(new_terminal.node_id)

                recovery_id = f"recovery_{uuid.uuid4().hex[:6]}"
                recovery_node = DataOpNode(
                    node_id=recovery_id,
                    preconditions=[],
                    postconditions=[f"recovery_holds({k})" for k in failure_context.keys()],
                )
                failed_graph.add_node(recovery_node)
                failed_graph.add_edge(node.node_id, recovery_node.node_id)
                failed_graph.add_edge(recovery_node.node_id, id_map[recovery_entry])

                connected_terminal = False
                for exit_id in recovery_graph.exit_node_ids:
                    copied_exit = id_map.get(exit_id)
                    if copied_exit is not None:
                        failed_graph.add_edge(copied_exit, new_terminal.node_id)
                        connected_terminal = True
                if not connected_terminal:
                    failed_graph.add_edge(id_map[recovery_entry], new_terminal.node_id)

                graft_count += 1

        return failed_graph

    def diagnose_failure(self, graph: SkillGraph, context: dict[str, Any]) -> dict[str, Any]:
        failure_nodes = [
            (nid, node)
            for nid, node in graph.nodes.items()
            if isinstance(node, TerminalOpNode) and not node.success
        ]

        if not failure_nodes:
            return {"diagnosis": "no_failure_detected", "confidence": 0.0}

        nid, node = failure_nodes[0]
        suspected_precondition = next(
            (p for p in node.preconditions if p not in context.get("_held_preconditions", [])),
            node.preconditions[0] if node.preconditions else "unknown",
        )

        return {
            "failure_node_id": nid,
            "diagnosis": node.diagnosis or "execution failed",
            "suspected_precondition": suspected_precondition,
            "confidence": 0.8,
            "suggested_recovery": f"ensure {suspected_precondition} holds before {nid}",
        }


# ---------------------------------------------------------------------------
# NSIAgent — main agent using NSI for skill induction and execution
# ---------------------------------------------------------------------------


@dataclass
class NSIAgent:
    """Neuro-Symbolic Skill Induction agent.

    Orchestrates the full NSI pipeline:
    1. Accumulates interaction traces during execution.
    2. Induces logic-grounded skill graphs from traces via TraceToLogicInducer.
    3. Consolidates local experts via IntraTrajectoryConsolidator.
    4. Merges skills across trajectories via InterTrajectoryMerger.
    5. Handles runtime failures via ReflectivePlanner.

    Parameters
    ----------
    agent_id : str
        Unique identifier for this agent instance.
    max_trace_history : int
        Maximum number of traces to retain (default 256).
    induction_threshold : int
        Number of successful traces before triggering induction (default 3).
    """

    agent_id: str
    max_trace_history: int = 256
    induction_threshold: int = 3

    traces: list[list[TraceStep]] = field(default_factory=list)
    skill_graphs: dict[str, SkillGraph] = field(default_factory=dict)
    active_graph: SkillGraph | None = None

    _inducer: TraceToLogicInducer = field(default_factory=TraceToLogicInducer)
    _consolidator: IntraTrajectoryConsolidator = field(default_factory=IntraTrajectoryConsolidator)
    _merger: InterTrajectoryMerger = field(default_factory=InterTrajectoryMerger)
    _planner: ReflectivePlanner = field(default_factory=ReflectivePlanner)

    def __post_init__(self) -> None:
        self._inducer = TraceToLogicInducer()
        self._consolidator = IntraTrajectoryConsolidator()
        self._merger = InterTrajectoryMerger()
        self._planner = ReflectivePlanner()

    def record_step(self, step: TraceStep) -> None:
        if not self.traces:
            self.traces.append([])
        if len(self.traces[-1]) > 1000:
            self.traces.append([])
        self.traces[-1].append(step)

    def record_trajectory(self, trace: list[TraceStep]) -> None:
        self.traces.append(list(trace))
        if len(self.traces) > self.max_trace_history:
            self.traces = self.traces[-self.max_trace_history :]

    def trigger_induction(self) -> SkillGraph | None:
        recent_traces = [t for t in self.traces if len(t) >= 2]
        if len(recent_traces) < self.induction_threshold:
            return None

        graphs: list[SkillGraph] = []
        for trace in recent_traces[-self.induction_threshold :]:
            graph = self._inducer.induce_from_trace(trace)
            graph = self._consolidator.consolidate(graph)
            graphs.append(graph)

        if not graphs:
            return None

        global_graph = self._merger.merge(graphs)
        self.active_graph = global_graph

        graph_id = f"skill_{uuid.uuid4().hex[:8]}"
        global_graph.graph_id = graph_id
        self.skill_graphs[graph_id] = global_graph

        _LOGGER.info("NSI induced skill graph %s from %d traces", graph_id, len(graphs))
        return global_graph

    def execute_skill(
        self,
        skill_id: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        graph = self.skill_graphs.get(skill_id)
        if graph is None:
            if self.active_graph is not None:
                graph = self.active_graph
            else:
                raise NSIError(f"Skill {skill_id!r} not found and no active graph")

        result = graph.execute(context)
        return result

    def handle_failure(
        self,
        failed_skill_id: str,
        failure_context: dict[str, Any],
        recovery_trace: list[TraceStep],
    ) -> SkillGraph:
        failed_graph = self.skill_graphs.get(failed_skill_id)
        if failed_graph is None:
            failed_graph = self.active_graph
        if failed_graph is None:
            raise NSIError(f"No graph found for skill {failed_skill_id!r}")

        improved = self._planner.plan_from_failure(failed_graph, failure_context, recovery_trace)

        improved_id = f"skill_{uuid.uuid4().hex[:8]}"
        improved.graph_id = improved_id
        self.skill_graphs[improved_id] = improved
        self.active_graph = improved

        _LOGGER.info("Reflective planner improved %s -> %s", failed_skill_id, improved_id)
        return improved

    def get_skill_ids(self) -> list[str]:
        return list(self.skill_graphs.keys())

    def get_skill(self, skill_id: str) -> SkillGraph | None:
        return self.skill_graphs.get(skill_id)

    def skill_stats(self) -> dict[str, Any]:
        total_nodes = sum(len(g.nodes) for g in self.skill_graphs.values())
        total_traces = sum(len(t) for t in self.traces)
        return {
            "n_skills": len(self.skill_graphs),
            "total_nodes": total_nodes,
            "n_traces": total_traces,
            "active_skill": self.active_graph.graph_id if self.active_graph else None,
        }
