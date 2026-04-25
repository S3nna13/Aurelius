"""Tests for src/chat/turn_planner.py."""

import pytest

from src.chat.turn_planner import TurnAction, TurnPlan, TurnPlanner


# ---------------------------------------------------------------------------
# Enum smoke tests
# ---------------------------------------------------------------------------

def test_turn_action_values():
    assert TurnAction.RESPOND == "respond"
    assert TurnAction.USE_TOOL == "use_tool"
    assert len(TurnAction) == 5


# ---------------------------------------------------------------------------
# plan() — USE_TOOL
# ---------------------------------------------------------------------------

def test_plan_use_tool_search():
    planner = TurnPlanner()
    plan = planner.plan([], "Please search for the latest papers on LLMs")
    assert plan.action == TurnAction.USE_TOOL


def test_plan_use_tool_calculate():
    planner = TurnPlanner()
    plan = planner.plan([], "Can you calculate 2 + 2?")
    assert plan.action == TurnAction.USE_TOOL


def test_plan_use_tool_execute():
    planner = TurnPlanner()
    plan = planner.plan([], "execute the test suite")
    assert plan.action == TurnAction.USE_TOOL


def test_plan_use_tool_run():
    planner = TurnPlanner()
    plan = planner.plan([], "run the evaluation script")
    assert plan.action == TurnAction.USE_TOOL


# ---------------------------------------------------------------------------
# plan() — ASK_CLARIFICATION
# ---------------------------------------------------------------------------

def test_plan_clarification_short_message():
    planner = TurnPlanner()
    plan = planner.plan([], "hi")
    assert plan.action == TurnAction.ASK_CLARIFICATION


def test_plan_clarification_question_mark():
    planner = TurnPlanner()
    plan = planner.plan([], "What is the capital of France?")
    assert plan.action == TurnAction.ASK_CLARIFICATION


# ---------------------------------------------------------------------------
# plan() — SUMMARIZE
# ---------------------------------------------------------------------------

def test_plan_summarize_long_history():
    planner = TurnPlanner()
    history = [{"role": "user", "content": f"msg {i}"} for i in range(21)]
    plan = planner.plan(history, "tell me more")
    assert plan.action == TurnAction.SUMMARIZE


def test_plan_summarize_exactly_at_threshold_plus_one():
    planner = TurnPlanner()
    history = [{}] * 21
    plan = planner.plan(history, "ok")
    assert plan.action == TurnAction.SUMMARIZE


# ---------------------------------------------------------------------------
# plan() — DEFER
# ---------------------------------------------------------------------------

def test_plan_defer_later():
    planner = TurnPlanner()
    plan = planner.plan([], "Do this later please")
    assert plan.action == TurnAction.DEFER


def test_plan_defer_remind():
    planner = TurnPlanner()
    plan = planner.plan([], "remind me about this tomorrow")
    assert plan.action == TurnAction.DEFER


# ---------------------------------------------------------------------------
# plan() — RESPOND (default)
# ---------------------------------------------------------------------------

def test_plan_respond_default():
    planner = TurnPlanner()
    plan = planner.plan([], "Explain the transformer architecture in detail.")
    assert plan.action == TurnAction.RESPOND


def test_plan_respond_high_confidence():
    planner = TurnPlanner()
    plan = planner.plan([], "Summarise the attention mechanism for me please.")
    assert plan.action == TurnAction.RESPOND
    assert plan.confidence >= 0.8


# ---------------------------------------------------------------------------
# TurnPlan dataclass
# ---------------------------------------------------------------------------

def test_turn_plan_has_rationale():
    planner = TurnPlanner()
    plan = planner.plan([], "search for something")
    assert isinstance(plan.rationale, str)
    assert len(plan.rationale) > 0


def test_turn_plan_metadata_is_dict():
    planner = TurnPlanner()
    plan = planner.plan([], "search for something")
    assert isinstance(plan.metadata, dict)


# ---------------------------------------------------------------------------
# estimate_response_length
# ---------------------------------------------------------------------------

def test_estimate_length_respond():
    planner = TurnPlanner()
    plan = TurnPlan(action=TurnAction.RESPOND, rationale="", confidence=0.9)
    length = planner.estimate_response_length(plan, context_tokens=512)
    assert length > 0


def test_estimate_length_use_tool_is_short():
    planner = TurnPlanner()
    plan_tool = TurnPlan(action=TurnAction.USE_TOOL, rationale="", confidence=0.9)
    plan_respond = TurnPlan(action=TurnAction.RESPOND, rationale="", confidence=0.9)
    assert planner.estimate_response_length(plan_tool, 512) <= planner.estimate_response_length(plan_respond, 512)


def test_estimate_length_defer_is_very_short():
    planner = TurnPlanner()
    plan = TurnPlan(action=TurnAction.DEFER, rationale="", confidence=0.8)
    assert planner.estimate_response_length(plan, 1024) <= 64
