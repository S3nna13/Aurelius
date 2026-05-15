"""Aurelius v2 Decision Prompt Templates — structured prompt construction for decision heads.

Implements the shared protocol from Section 9 of the master plan:
- Control tokens for all output families
- Context channel separation (not collapsed into one blob)
- Runtime truth in responses
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OUTPUT_FAMILIES = [
    "text", "structured_output", "tool_call", "computer_action",
    "memory_op", "skill_use", "plan", "critique", "verify",
    "ask_user", "delegate", "escalate", "refuse_safe", "final",
]

CONTROL_TOKENS = {
    "text": "<|text|>", "tool_call": "<|tool_call|>",
    "computer_action": "<|computer_action|>", "memory_op": "<|memory_op|>",
    "skill_use": "<|skill_use|>", "plan": "<|plan|>",
    "critique": "<|critique|>", "verify": "<|verify|>",
    "ask_user": "<|ask_user|>", "delegate": "<|delegate|>",
    "escalate": "<|escalate|>", "refuse_safe": "<|refuse_safe|>",
    "final": "<|final|>", "screen": "<|screen|>",
    "ax_tree": "<|ax_tree|>", "file_context": "<|file_context|>",
    "web_context": "<|web_context|>", "memory_context": "<|memory_context|>",
    "skill_context": "<|skill_context|>",
}

PROMPT_CHANNELS = [
    "system_policy", "user_request", "mode", "tool_schemas",
    "native_skill_manifests", "runtime_observations", "memory_context",
    "file_web_screen_context", "private_agent_state", "audit_provenance",
]


@dataclass
class PromptContext:
    system_policy: str = ""
    user_request: str = ""
    mode: str = "chat"
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    skill_manifests: list[dict[str, Any]] = field(default_factory=list)
    runtime_observations: str = ""
    memory_context: str = ""
    file_web_screen_context: str = ""
    private_agent_state: dict[str, Any] = field(default_factory=dict)
    audit_provenance: str = ""
    max_tokens: int = 4096

    def build_prompt(self) -> str:
        parts = []
        if self.system_policy:
            parts.append(f"{CONTROL_TOKENS['text']}\n{self.system_policy}")
        if self.user_request:
            parts.append(f"\nUser: {self.user_request}")
        if self.memory_context:
            parts.append(f"\n{CONTROL_TOKENS['memory_context']}\n{self.memory_context}")
        if self.file_web_screen_context:
            parts.append(f"\n{CONTROL_TOKENS['file_context']}\n{self.file_web_screen_context}")
        if self.skill_manifests:
            parts.append(f"\n{CONTROL_TOKENS['skill_context']}\n{len(self.skill_manifests)} skills available")
        if self.tool_schemas:
            parts.append(f"\n{CONTROL_TOKENS['tool_call']}\n{len(self.tool_schemas)} tools available")
        if self.runtime_observations:
            parts.append(f"\nRuntime: {self.runtime_observations}")
        return "\n".join(parts)
