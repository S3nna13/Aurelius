from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ConditionOperator(Enum):
    EQUALS = "=="
    GT = ">"
    LT = "<"
    CONTAINS = "in"
    NOT_EQUALS = "!="


@dataclass
class BranchCondition:
    field: str
    operator: ConditionOperator
    value: Any

    def evaluate(self, context: dict[str, Any]) -> bool:
        actual = context.get(self.field)
        if actual is None:
            return False
        if self.operator == ConditionOperator.EQUALS:
            return actual == self.value
        elif self.operator == ConditionOperator.NOT_EQUALS:
            return actual != self.value
        elif self.operator == ConditionOperator.GT:
            return actual > self.value
        elif self.operator == ConditionOperator.LT:
            return actual < self.value
        elif self.operator == ConditionOperator.CONTAINS:
            return self.value in actual if isinstance(actual, (list, str)) else False
        return False


@dataclass
class BranchResult:
    action: str | None
    triggered: bool = False
    condition_name: str | None = None


@dataclass
class _BranchRule:
    condition: BranchCondition
    true_action: str
    false_action: str
    name: str = ""


class ConditionalBranch:
    def __init__(self) -> None:
        self._rules: list[_BranchRule] = []

    def add_condition(
        self, condition: BranchCondition, true_action: str, false_action: str, name: str = ""
    ) -> None:
        self._rules.append(
            _BranchRule(
                condition=condition, true_action=true_action, false_action=false_action, name=name
            )
        )

    def evaluate(self, context: dict[str, Any]) -> BranchResult:
        result = BranchResult(action=None)
        for rule in self._rules:
            if rule.condition.evaluate(context):
                result = BranchResult(
                    action=rule.true_action, triggered=True, condition_name=rule.name
                )
            else:
                result = BranchResult(
                    action=rule.false_action, triggered=False, condition_name=rule.name
                )
        return result

    def condition_count(self) -> int:
        return len(self._rules)


CONDITIONAL_BRANCH = ConditionalBranch()
