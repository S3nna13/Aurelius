from src.workflow.workflow_engine import (
    WorkflowEngine,
    WorkflowState,
    WorkflowContext,
    Transition,
    WORKFLOW_ENGINE_REGISTRY,
)


def _engine() -> WorkflowEngine:
    return WorkflowEngine()


def test_registry_has_default():
    assert "default" in WORKFLOW_ENGINE_REGISTRY
    assert WORKFLOW_ENGINE_REGISTRY["default"] is WorkflowEngine


def test_context_defaults():
    ctx = WorkflowContext()
    assert ctx.state == WorkflowState.IDLE
    assert ctx.data == {}
    assert ctx.history == []
    assert len(ctx.workflow_id) == 8


def test_context_ids_unique():
    a = WorkflowContext()
    b = WorkflowContext()
    assert a.workflow_id != b.workflow_id


def test_default_transitions_nonempty():
    assert len(WorkflowEngine.DEFAULT_TRANSITIONS) >= 5


def test_default_start_transition():
    eng = _engine()
    ctx = WorkflowContext()
    assert eng.can_trigger(ctx, "start")
    assert eng.trigger(ctx, "start") is True
    assert ctx.state == WorkflowState.RUNNING


def test_default_pause_resume():
    eng = _engine()
    ctx = WorkflowContext()
    eng.trigger(ctx, "start")
    assert eng.trigger(ctx, "pause")
    assert ctx.state == WorkflowState.PAUSED
    assert eng.trigger(ctx, "resume")
    assert ctx.state == WorkflowState.RUNNING


def test_default_complete_reset():
    eng = _engine()
    ctx = WorkflowContext()
    eng.trigger(ctx, "start")
    eng.trigger(ctx, "complete")
    assert ctx.state == WorkflowState.COMPLETED
    eng.trigger(ctx, "reset")
    assert ctx.state == WorkflowState.IDLE


def test_default_fail_reset():
    eng = _engine()
    ctx = WorkflowContext()
    eng.trigger(ctx, "start")
    eng.trigger(ctx, "fail")
    assert ctx.state == WorkflowState.FAILED
    eng.trigger(ctx, "reset")
    assert ctx.state == WorkflowState.IDLE


def test_invalid_trigger_returns_false():
    eng = _engine()
    ctx = WorkflowContext()
    assert eng.trigger(ctx, "complete") is False
    assert ctx.state == WorkflowState.IDLE


def test_can_trigger_negative():
    eng = _engine()
    ctx = WorkflowContext()
    assert eng.can_trigger(ctx, "pause") is False


def test_history_recorded():
    eng = _engine()
    ctx = WorkflowContext()
    eng.trigger(ctx, "start")
    eng.trigger(ctx, "complete")
    assert ctx.history == [("start", "RUNNING"), ("complete", "COMPLETED")]


def test_history_ignores_invalid():
    eng = _engine()
    ctx = WorkflowContext()
    eng.trigger(ctx, "bogus")
    assert ctx.history == []


def test_valid_triggers_from_idle():
    eng = _engine()
    ctx = WorkflowContext()
    assert "start" in eng.valid_triggers(ctx)


def test_valid_triggers_from_running():
    eng = _engine()
    ctx = WorkflowContext()
    eng.trigger(ctx, "start")
    vt = eng.valid_triggers(ctx)
    assert "pause" in vt
    assert "complete" in vt
    assert "fail" in vt


def test_add_transition_custom():
    eng = WorkflowEngine(transitions=[])
    eng.add_transition(Transition(WorkflowState.IDLE, WorkflowState.RUNNING, "go"))
    ctx = WorkflowContext()
    assert eng.trigger(ctx, "go")
    assert ctx.state == WorkflowState.RUNNING


def test_guard_blocks_transition():
    eng = WorkflowEngine(transitions=[])
    eng.add_transition(
        Transition(
            WorkflowState.IDLE,
            WorkflowState.RUNNING,
            "go",
            guard=lambda c: c.data.get("ok", False),
        )
    )
    ctx = WorkflowContext()
    assert eng.can_trigger(ctx, "go") is False
    assert eng.trigger(ctx, "go") is False
    ctx.data["ok"] = True
    assert eng.trigger(ctx, "go") is True


def test_guard_allows_valid_triggers():
    eng = WorkflowEngine(transitions=[])
    eng.add_transition(
        Transition(
            WorkflowState.IDLE,
            WorkflowState.RUNNING,
            "go",
            guard=lambda c: c.data.get("ok", False),
        )
    )
    ctx = WorkflowContext()
    assert "go" not in eng.valid_triggers(ctx)
    ctx.data["ok"] = True
    assert "go" in eng.valid_triggers(ctx)


def test_action_called_on_trigger():
    calls = []
    eng = WorkflowEngine(transitions=[])
    eng.add_transition(
        Transition(
            WorkflowState.IDLE,
            WorkflowState.RUNNING,
            "go",
            action=lambda c: calls.append(c.workflow_id),
        )
    )
    ctx = WorkflowContext()
    eng.trigger(ctx, "go")
    assert calls == [ctx.workflow_id]


def test_action_not_called_on_invalid():
    calls = []
    eng = WorkflowEngine(transitions=[])
    eng.add_transition(
        Transition(
            WorkflowState.IDLE,
            WorkflowState.RUNNING,
            "go",
            action=lambda c: calls.append(1),
        )
    )
    ctx = WorkflowContext()
    ctx.state = WorkflowState.RUNNING
    eng.trigger(ctx, "go")
    assert calls == []


def test_cancel_then_reset():
    eng = _engine()
    ctx = WorkflowContext()
    eng.trigger(ctx, "start")
    assert eng.trigger(ctx, "cancel")
    assert ctx.state == WorkflowState.CANCELLED
    assert eng.trigger(ctx, "reset")
    assert ctx.state == WorkflowState.IDLE


def test_valid_triggers_empty_when_no_match():
    eng = WorkflowEngine(transitions=[])
    ctx = WorkflowContext()
    assert eng.valid_triggers(ctx) == []


def test_transition_frozen():
    t = Transition(WorkflowState.IDLE, WorkflowState.RUNNING, "go")
    try:
        t.trigger = "nope"  # type: ignore
    except Exception:
        return
    assert False, "Transition should be frozen"


def test_state_enum_values():
    assert WorkflowState.IDLE.value == "idle"
    assert WorkflowState.RUNNING.value == "running"
    assert WorkflowState.PAUSED.value == "paused"
    assert WorkflowState.COMPLETED.value == "completed"
    assert WorkflowState.FAILED.value == "failed"
    assert WorkflowState.CANCELLED.value == "cancelled"


def test_multiple_triggers_same_from_state():
    eng = _engine()
    ctx = WorkflowContext()
    eng.trigger(ctx, "start")
    vt = eng.valid_triggers(ctx)
    assert len(set(vt)) == len(vt)


def test_history_length_matches_triggers():
    eng = _engine()
    ctx = WorkflowContext()
    eng.trigger(ctx, "start")
    eng.trigger(ctx, "pause")
    eng.trigger(ctx, "resume")
    eng.trigger(ctx, "complete")
    assert len(ctx.history) == 4


def test_custom_transitions_override_defaults():
    eng = WorkflowEngine(transitions=[
        Transition(WorkflowState.IDLE, WorkflowState.COMPLETED, "finish"),
    ])
    ctx = WorkflowContext()
    assert eng.can_trigger(ctx, "start") is False
    assert eng.trigger(ctx, "finish")
    assert ctx.state == WorkflowState.COMPLETED


def test_data_dict_modifiable():
    ctx = WorkflowContext()
    ctx.data["k"] = 1
    assert ctx.data == {"k": 1}
