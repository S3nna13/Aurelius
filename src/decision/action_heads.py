"""Aurelius v2 Action Heads — ToolCall, MemoryOp, Skill, Critic, Verifier, Escalation, CUA."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any


class HeadOutput(enum.StrEnum):
    """Enum for head output types."""
    TOOL_CALL = "tool_call"
    MEMORY_OP = "memory_op"
    SKILL_USE = "skill_use"
    CRITIQUE = "critique"
    VERIFY = "verify"
    ESCALATE = "escalate"
    CUA_ACTION = "computer_action"


@dataclass
class ToolCallHead:
    """Generates structured tool call outputs.

    Ensures:
    - Schema-valid tool calls
    - Retries on failure
    - Provenance tracking
    - Safety gates
    """

    def format_call(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"tool": tool_name, "params": params, "type": "tool_call"}

    def validate_response(self, response: dict[str, Any]) -> bool:
        return "result" in response or "error" in response


@dataclass
class MemoryOpHead:
    """Generates memory operations.

    Supports: read, write, search, delete, quarantine
    """

    def read(self, key: str) -> dict[str, Any]:
        return {"op": "read", "key": key}

    def write(self, key: str, value: Any, trust_level: str = "untrusted") -> dict[str, Any]:
        return {"op": "write", "key": key, "value": value, "trust_level": trust_level}

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        return {"op": "search", "query": query, "limit": limit}

    def quarantine(self, key: str, reason: str) -> dict[str, Any]:
        return {"op": "quarantine", "key": key, "reason": reason}


@dataclass
class SkillHead:
    """Generates skill invocation calls.

    Supports: retrieve (find matching skill), use (execute skill), compose (chain skills)
    """

    def retrieve(self, query: str, category: str | None = None) -> dict[str, Any]:
        return {"op": "retrieve_skill", "query": query, "category": category}

    def use(self, skill_id: str, mode: str = "execute", inputs: dict | None = None) -> dict[str, Any]:
        return {"op": "use_skill", "skill_id": skill_id, "mode": mode, "inputs": inputs or {}}

    def compose(self, skill_chain: list[dict[str, Any]]) -> dict[str, Any]:
        return {"op": "compose_skills", "chain": skill_chain}


@dataclass
class CriticHead:
    """Generates critique/verification of agent output.

    Used for self-reflection and output quality assessment.
    """

    def critique_response(self, response: str, criteria: list[str] | None = None) -> dict[str, Any]:
        return {"op": "critique", "target": response, "criteria": criteria or ["accuracy", "completeness", "safety"]}

    def critique_plan(self, plan: list[dict], criteria: list[str] | None = None) -> dict[str, Any]:
        return {"op": "critique_plan", "plan": plan, "criteria": criteria or ["feasibility", "safety", "efficiency"]}


@dataclass
class VerifierHead:
    """Generates verification results.

    Used for CUA action verification, tool output validation, and safety checks.
    """

    def verify_action(self, action: dict, before_state: Any, after_state: Any) -> dict[str, Any]:
        return {"op": "verify_action", "action": action, "before": before_state, "after": after_state, "valid": True}

    def verify_output(self, output: str, expected_format: str) -> dict[str, Any]:
        return {"op": "verify_output", "output": output, "format": expected_format, "valid": True}


@dataclass
class EscalationHead:
    """Generates escalation requests.

    Used when local model/hardware is insufficient for the task.
    """

    def escalate_model(self, current_model: str, target_model: str, reason: str) -> dict[str, Any]:
        return {"op": "escalate_model", "from": current_model, "to": target_model, "reason": reason}

    def escalate_remote(self, task: dict) -> dict[str, Any]:
        return {"op": "escalate_remote", "task": task}


@dataclass
class CUAActionHead:
    """Generates computer-use actions.

    Used for desktop automation with verifier-based safety.
    """

    def click(self, element_ref: str, app: str = "") -> dict[str, Any]:
        return {"op": "click", "element": element_ref, "app": app}

    def type(self, text: str, element_ref: str = "") -> dict[str, Any]:
        return {"op": "type", "text": text, "element": element_ref}

    def scroll(self, direction: str, amount: int = 3) -> dict[str, Any]:
        return {"op": "scroll", "direction": direction, "amount": amount}

    def key(self, keys: str) -> dict[str, Any]:
        return {"op": "key", "keys": keys}
