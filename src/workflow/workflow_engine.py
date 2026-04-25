import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class WorkflowState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Transition:
    from_state: WorkflowState
    to_state: WorkflowState
    trigger: str
    guard: Callable | None = None
    action: Callable | None = None


@dataclass
class WorkflowContext:
    workflow_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    state: WorkflowState = WorkflowState.IDLE
    data: dict = field(default_factory=dict)
    history: list[tuple[str, str]] = field(default_factory=list)


class WorkflowEngine:
    DEFAULT_TRANSITIONS: list[Transition] = [
        Transition(WorkflowState.IDLE, WorkflowState.RUNNING, "start"),
        Transition(WorkflowState.RUNNING, WorkflowState.PAUSED, "pause"),
        Transition(WorkflowState.PAUSED, WorkflowState.RUNNING, "resume"),
        Transition(WorkflowState.RUNNING, WorkflowState.COMPLETED, "complete"),
        Transition(WorkflowState.RUNNING, WorkflowState.FAILED, "fail"),
        Transition(WorkflowState.RUNNING, WorkflowState.CANCELLED, "cancel"),
        Transition(WorkflowState.COMPLETED, WorkflowState.IDLE, "reset"),
        Transition(WorkflowState.FAILED, WorkflowState.IDLE, "reset"),
        Transition(WorkflowState.CANCELLED, WorkflowState.IDLE, "reset"),
    ]

    def __init__(self, transitions: list[Transition] | None = None) -> None:
        self._transitions: list[Transition] = list(
            transitions if transitions is not None else self.DEFAULT_TRANSITIONS
        )

    def add_transition(self, t: Transition) -> None:
        self._transitions.append(t)

    def _match(self, ctx: WorkflowContext, trigger: str) -> Transition | None:
        for t in self._transitions:
            if t.from_state == ctx.state and t.trigger == trigger:
                if t.guard is None or t.guard(ctx):
                    return t
        return None

    def can_trigger(self, ctx: WorkflowContext, trigger: str) -> bool:
        return self._match(ctx, trigger) is not None

    def trigger(self, ctx: WorkflowContext, trigger: str, **kwargs) -> bool:
        t = self._match(ctx, trigger)
        if t is None:
            return False
        ctx.state = t.to_state
        ctx.history.append((trigger, t.to_state.name))
        if t.action is not None:
            t.action(ctx)
        return True

    def valid_triggers(self, ctx: WorkflowContext) -> list[str]:
        seen: list[str] = []
        for t in self._transitions:
            if t.from_state == ctx.state and t.trigger not in seen:
                if t.guard is None or t.guard(ctx):
                    seen.append(t.trigger)
        return seen


WORKFLOW_ENGINE_REGISTRY: dict[str, type[WorkflowEngine]] = {"default": WorkflowEngine}
