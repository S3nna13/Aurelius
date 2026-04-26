"""Tests for conditional_branch — workflow branching logic."""

from __future__ import annotations

from src.workflow.conditional_branch import (
    BranchCondition,
    ConditionalBranch,
    ConditionOperator,
)


class TestBranchCondition:
    def test_equals(self):
        c = BranchCondition(field="status", operator=ConditionOperator.EQUALS, value="success")
        assert c.evaluate({"status": "success"})
        assert not c.evaluate({"status": "failed"})

    def test_greater_than(self):
        c = BranchCondition(field="score", operator=ConditionOperator.GT, value=0.5)
        assert c.evaluate({"score": 0.8})
        assert not c.evaluate({"score": 0.3})

    def test_contains(self):
        c = BranchCondition(field="tags", operator=ConditionOperator.CONTAINS, value="urgent")
        assert c.evaluate({"tags": ["urgent", "bug"]})
        assert not c.evaluate({"tags": ["feature"]})

    def test_missing_field(self):
        c = BranchCondition(field="nonexistent", operator=ConditionOperator.EQUALS, value="x")
        assert not c.evaluate({})


class TestConditionalBranch:
    def test_true_branch_executes(self):
        branch = ConditionalBranch()
        branch.add_condition(
            BranchCondition("score", ConditionOperator.GT, 0.5),
            true_action="approve",
            false_action="reject",
        )
        result = branch.evaluate({"score": 0.9})
        assert result.action == "approve"
        assert result.triggered

    def test_false_branch_executes(self):
        branch = ConditionalBranch()
        branch.add_condition(
            BranchCondition("score", ConditionOperator.GT, 0.5),
            true_action="approve",
            false_action="reject",
        )
        result = branch.evaluate({"score": 0.1})
        assert result.action == "reject"
        assert not result.triggered

    def test_multiple_conditions_all_and(self):
        branch = ConditionalBranch()
        branch.add_condition(
            BranchCondition("a", ConditionOperator.EQUALS, 1),
            true_action="a_pass",
            false_action="a_fail",
        )
        branch.add_condition(
            BranchCondition("b", ConditionOperator.EQUALS, 2),
            true_action="b_pass",
            false_action="b_fail",
        )
        r1 = branch.evaluate({"a": 1, "b": 2})
        assert r1.action == "b_pass"
        r2 = branch.evaluate({"a": 1, "b": 99})
        assert r2.action == "b_fail"

    def test_empty_conditions(self):
        branch = ConditionalBranch()
        result = branch.evaluate({"x": 1})
        assert result.action is None

    def test_condition_count(self):
        branch = ConditionalBranch()
        branch.add_condition(BranchCondition("x", ConditionOperator.EQUALS, 1), "t", "f")
        branch.add_condition(BranchCondition("y", ConditionOperator.EQUALS, 2), "t", "f")
        assert branch.condition_count() == 2
