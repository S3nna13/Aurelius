"""Tests for src/computer_use/action_planner.py — 10+ unit tests, no GPU."""

from __future__ import annotations

import pytest

from src.computer_use.action_planner import (
    COMPUTER_USE_REGISTRY,
    ActionPlan,
    ActionPlanner,
    ActionType,
    PlannedAction,
)


@pytest.fixture()
def planner() -> ActionPlanner:
    return ActionPlanner()


# ---------------------------------------------------------------------------
# ActionType inference
# ---------------------------------------------------------------------------


class TestActionTypeInference:
    def test_click_keyword(self, planner):
        plan = planner.plan("click the submit button", {})
        assert plan.steps[0].action_type == ActionType.CLICK

    def test_type_keyword(self, planner):
        plan = planner.plan("type hello world into the search box", {})
        assert plan.steps[0].action_type == ActionType.TYPE

    def test_scroll_keyword(self, planner):
        plan = planner.plan("scroll down the page", {})
        assert plan.steps[0].action_type == ActionType.SCROLL

    def test_screenshot_keyword(self, planner):
        plan = planner.plan("take a screenshot of the window", {})
        assert plan.steps[0].action_type == ActionType.SCREENSHOT

    def test_wait_keyword(self, planner):
        plan = planner.plan("wait 3 seconds for the page to load", {})
        assert plan.steps[0].action_type == ActionType.WAIT

    def test_drag_keyword(self, planner):
        plan = planner.plan("drag the slider to the right", {})
        assert plan.steps[0].action_type == ActionType.DRAG

    def test_hover_keyword(self, planner):
        plan = planner.plan("hover over the menu item", {})
        assert plan.steps[0].action_type == ActionType.HOVER

    def test_key_combo_keyword(self, planner):
        plan = planner.plan("press Ctrl+S to save", {})
        assert plan.steps[0].action_type == ActionType.KEY_COMBO

    def test_default_action_is_click(self, planner):
        # No matching keyword → defaults to CLICK
        plan = planner.plan("xyzzy unknown action florp", {})
        assert plan.steps[0].action_type == ActionType.CLICK


# ---------------------------------------------------------------------------
# Params extraction
# ---------------------------------------------------------------------------


class TestParamExtraction:
    def test_type_action_has_text_param(self, planner):
        plan = planner.plan('type "hello world" into the field', {})
        step = plan.steps[0]
        assert step.action_type == ActionType.TYPE
        assert "text" in step.params

    def test_scroll_direction_down(self, planner):
        plan = planner.plan("scroll down", {})
        assert plan.steps[0].params.get("direction") == "down"

    def test_scroll_direction_up(self, planner):
        plan = planner.plan("scroll up the sidebar", {})
        assert plan.steps[0].params.get("direction") == "up"

    def test_wait_duration_extracted(self, planner):
        plan = planner.plan("wait 5 seconds", {})
        assert plan.steps[0].params.get("duration") == 5.0

    def test_screenshot_region_from_context(self, planner):
        plan = planner.plan("take a screenshot", {"region": "header"})
        assert plan.steps[0].params.get("region") == "header"

    def test_target_from_context(self, planner):
        plan = planner.plan("click", {"focused_element": "login_btn"})
        assert plan.steps[0].target == "login_btn"

    def test_target_from_quoted_goal(self, planner):
        plan = planner.plan('click "search_field"', {})
        assert plan.steps[0].target == "search_field"


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_confidence_higher_with_context(self, planner):
        plan_ctx = planner.plan("click the button", {"focused_element": "btn"})
        plan_no_ctx = planner.plan("click the button")
        assert plan_ctx.confidence >= plan_no_ctx.confidence

    def test_confidence_in_range(self, planner):
        plan = planner.plan("type something", {})
        assert 0.0 <= plan.confidence <= 1.0


# ---------------------------------------------------------------------------
# validate_plan
# ---------------------------------------------------------------------------


class TestValidatePlan:
    def test_valid_plan_no_errors(self, planner):
        plan = planner.plan("click the button", {"focused_element": "btn"})
        errors = planner.validate_plan(plan)
        assert errors == []

    def test_empty_goal_error(self, planner):
        bad_plan = ActionPlan(steps=[], goal="", confidence=0.5)
        errors = planner.validate_plan(bad_plan)
        assert any("empty goal" in e.lower() for e in errors)

    def test_no_steps_error(self, planner):
        bad_plan = ActionPlan(steps=[], goal="click", confidence=0.5)
        errors = planner.validate_plan(bad_plan)
        assert any("no steps" in e.lower() for e in errors)

    def test_bad_confidence_error(self, planner):
        step = PlannedAction(ActionType.CLICK, "btn", {}, "test")
        bad_plan = ActionPlan(steps=[step], goal="click", confidence=1.5)
        errors = planner.validate_plan(bad_plan)
        assert any("confidence" in e.lower() for e in errors)

    def test_empty_target_error(self, planner):
        step = PlannedAction(ActionType.CLICK, "", {}, "test")
        bad_plan = ActionPlan(steps=[step], goal="click", confidence=0.8)
        errors = planner.validate_plan(bad_plan)
        assert any("target is empty" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def test_cost_positive(self, planner):
        plan = planner.plan("click the button", {})
        cost = planner.estimate_cost(plan)
        assert cost > 0.0

    def test_screenshot_more_expensive_than_wait(self, planner):
        screenshot_plan = planner.plan("take a screenshot", {})
        wait_plan = planner.plan("wait 2 seconds", {})
        assert planner.estimate_cost(screenshot_plan) > planner.estimate_cost(wait_plan)

    def test_empty_plan_cost_zero(self, planner):
        empty = ActionPlan(steps=[], goal="none", confidence=0.5)
        assert planner.estimate_cost(empty) == 0.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_contains_action_planner(self):
        assert "action_planner" in COMPUTER_USE_REGISTRY

    def test_registry_value_is_class(self):
        assert COMPUTER_USE_REGISTRY["action_planner"] is ActionPlanner
