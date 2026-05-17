"""ToolGT — Guided-Structured Templates for Function Calling.

Template-based reasoning for tool use. Instead of free-form CoT, the model
fills deterministic XML-like templates. Each template is divided into *slots*;
when a slot is active, constrained decoding can force the structural prefix
and restrict the vocabulary to valid tokens for that slot.

Research basis: ToolGT (2025) shows +2.8/+1.7 over CoT on BFCLv2/Nexus for
small models. Template-based reasoning is more reliable than free-form CoT for
1.3B parameter models.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from src._compat import StrEnum
from typing import Any

# Re-use existing constrained-decoding types when available.
try:
    from ..inference.constrained_decoding import ConstraintConfig
except ImportError:  # pragma: no cover
    ConstraintConfig = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


class ToolGTTemplate(StrEnum):
    SEARCH_THEN_ANSWER = "search_then_answer"
    MULTI_TOOL_SEQUENCE = "multi_tool_sequence"
    COMPARE_THEN_SELECT = "compare_then_select"
    FREEFORM = "freeform"


@dataclass
class TemplateSlot:
    """A single fillable region inside a ToolGT template."""

    name: str
    prefix: str
    # Suffix that terminates the slot (used for parsing / validation).
    suffix: str
    # Optional human-readable hint shown in the prompt.
    hint: str = ""


@dataclass
class ToolGTSchema:
    """Concrete template schema with ordered slots."""

    template: ToolGTTemplate
    slots: list[TemplateSlot] = field(default_factory=list)
    # Static text inserted between slots (same length as slots - 1).
    delimiters: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Predefined schemas
# ---------------------------------------------------------------------------

_SEARCH_SCHEMA = ToolGTSchema(
    template=ToolGTTemplate.SEARCH_THEN_ANSWER,
    slots=[
        TemplateSlot(
            name="think",
            prefix="<think>",
            suffix="</think>",
            hint="Plan the search query.",
        ),
        TemplateSlot(
            name="tool_call",
            prefix="<tool_call>",
            suffix="</tool_call>",
            hint='Emit a JSON object: {"name": "...", "arguments": {...}}',
        ),
        TemplateSlot(
            name="observation",
            prefix="<observation>",
            suffix="</observation>",
            hint="Summarise the tool result.",
        ),
        TemplateSlot(
            name="answer",
            prefix="<answer>",
            suffix="</answer>",
            hint="Provide the final answer.",
        ),
    ],
)

_MULTI_SCHEMA = ToolGTSchema(
    template=ToolGTTemplate.MULTI_TOOL_SEQUENCE,
    slots=[
        TemplateSlot(
            name="plan",
            prefix="<plan>",
            suffix="</plan>",
            hint="List the tools you will call and why.",
        ),
        TemplateSlot(
            name="tool_call_1",
            prefix="<tool_call_1>",
            suffix="</tool_call_1>",
            hint='First JSON tool call: {"name": "...", "arguments": {...}}',
        ),
        TemplateSlot(
            name="observation_1",
            prefix="<observation_1>",
            suffix="</observation_1>",
            hint="Result of the first tool.",
        ),
        TemplateSlot(
            name="tool_call_2",
            prefix="<tool_call_2>",
            suffix="</tool_call_2>",
            hint='Second JSON tool call (optional): {"name": "...", "arguments": {...}}',
        ),
        TemplateSlot(
            name="observation_2",
            prefix="<observation_2>",
            suffix="</observation_2>",
            hint="Result of the second tool.",
        ),
        TemplateSlot(
            name="answer",
            prefix="<answer>",
            suffix="</answer>",
            hint="Synthesise the results into a final answer.",
        ),
    ],
)

_COMPARE_SCHEMA = ToolGTSchema(
    template=ToolGTTemplate.COMPARE_THEN_SELECT,
    slots=[
        TemplateSlot(
            name="options",
            prefix="<options>",
            suffix="</options>",
            hint="List the candidate options.",
        ),
        TemplateSlot(
            name="criteria",
            prefix="<criteria>",
            suffix="</criteria>",
            hint="State the comparison criteria.",
        ),
        TemplateSlot(
            name="tool_call",
            prefix="<tool_call>",
            suffix="</tool_call>",
            hint='Tool call to fetch extra data if needed: {"name": "...", "arguments": {...}}',
        ),
        TemplateSlot(
            name="observation",
            prefix="<observation>",
            suffix="</observation>",
            hint="Data returned by the tool.",
        ),
        TemplateSlot(
            name="comparison",
            prefix="<comparison>",
            suffix="</comparison>",
            hint="Compare each option against the criteria.",
        ),
        TemplateSlot(
            name="selection",
            prefix="<selection>",
            suffix="</selection>",
            hint="State the best option and justify briefly.",
        ),
    ],
)

_SCHEMA_REGISTRY: dict[ToolGTTemplate, ToolGTSchema] = {
    ToolGTTemplate.SEARCH_THEN_ANSWER: _SEARCH_SCHEMA,
    ToolGTTemplate.MULTI_TOOL_SEQUENCE: _MULTI_SCHEMA,
    ToolGTTemplate.COMPARE_THEN_SELECT: _COMPARE_SCHEMA,
}


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------


def classify_intent(query: str) -> ToolGTTemplate:
    """Simple keyword/heuristic intent classifier.

    Returns the most appropriate template for *query*.
    """
    text = query.lower()

    # Comparison / selection signals
    compare_keywords = [
        "compare",
        "versus",
        "vs",
        "choose",
        "select",
        "best",
        "better",
        "difference between",
        "pros and cons",
    ]
    if any(kw in text for kw in compare_keywords):
        return ToolGTTemplate.COMPARE_THEN_SELECT

    # Multi-tool chaining signals
    multi_keywords = [
        "and then",
        "followed by",
        "chain",
        "sequence",
        "step by step",
        "multiple",
        "first",
        "next",
    ]
    if any(kw in text for kw in multi_keywords):
        return ToolGTTemplate.MULTI_TOOL_SEQUENCE

    # Search / lookup signals (default for tool use)
    search_keywords = [
        "search",
        "look up",
        "lookup",
        "find",
        "query",
        "retrieve",
        "get",
        "fetch",
    ]
    if any(kw in text for kw in search_keywords):
        return ToolGTTemplate.SEARCH_THEN_ANSWER

    # Fallback: if the text mentions a tool, still use search_then_answer
    if "tool" in text or "call" in text:
        return ToolGTTemplate.SEARCH_THEN_ANSWER

    return ToolGTTemplate.FREEFORM


# ---------------------------------------------------------------------------
# ToolGTReasoner
# ---------------------------------------------------------------------------


class ToolGTReasoner:
    """Guided-structured template reasoner for function calling.

    Typical usage:

        reasoner = ToolGTReasoner()
        schema = reasoner.select_schema("Search for the capital of France")
        prompt = reasoner.format_prompt(schema, "Search for the capital of France")
        # During generation, use reasoner.constraint_for_slot(...) to obtain
        # a ConstraintConfig that forces the next structural prefix.
    """

    def __init__(self, schemas: dict[ToolGTTemplate, ToolGTSchema] | None = None) -> None:
        self.schemas = schemas or _SCHEMA_REGISTRY.copy()

    # -- template selection ---------------------------------------------------

    def select_schema(self, query: str) -> ToolGTSchema:
        """Pick the best schema for *query*."""
        intent = classify_intent(query)
        if intent in self.schemas:
            return self.schemas[intent]
        # Fallback to search_then_answer if freeform or missing.
        return self.schemas.get(ToolGTTemplate.SEARCH_THEN_ANSWER, _SEARCH_SCHEMA)

    # -- prompt formatting ----------------------------------------------------

    def format_prompt(self, schema: ToolGTSchema, query: str) -> str:
        """Return a prompt that instructs the model to fill *schema* slots."""
        lines: list[str] = [
            "You must follow the exact template below. Fill each slot with the requested content.",
            "",
        ]
        for slot in schema.slots:
            lines.append(f"{slot.prefix} ... {slot.suffix}  ({slot.hint})")
        lines.append("")
        lines.append(f"Question: {query}")
        return "\n".join(lines)

    def format_full_example(self, schema: ToolGTSchema, fillings: dict[str, str]) -> str:
        """Render a fully-filled template (useful for few-shot examples)."""
        parts: list[str] = []
        for slot in schema.slots:
            value = fillings.get(slot.name, "")
            parts.append(f"{slot.prefix}{value}{slot.suffix}")
        return "\n".join(parts)

    # -- constrained-decoding helpers -----------------------------------------

    def constraint_for_slot(
        self,
        schema: ToolGTSchema,
        slot_index: int,
        encode_fn: Any | None = None,
        max_new_tokens: int = 128,
    ) -> Any:
        """Build a :class:`ConstraintConfig` that forces the slot prefix.

        Args:
            schema: The active template schema.
            slot_index: Index of the slot that is about to be generated.
            encode_fn: Callable ``str -> list[int]`` used to turn the slot
                prefix into token ids. If ``None``, a text-only fallback is
                returned and the caller must tokenise.
            max_new_tokens: Generation budget for this slot.

        Returns:
            A ``ConstraintConfig`` (or ``None`` if the import failed).
        """
        if slot_index >= len(schema.slots):
            return None

        slot = schema.slots[slot_index]
        prefix_tokens: list[int] | None = None
        if encode_fn is not None:
            prefix_tokens = encode_fn(slot.prefix)

        if ConstraintConfig is None:
            return None

        return ConstraintConfig(
            prefix_tokens=prefix_tokens,
            max_new_tokens=max_new_tokens,
        )

    def slot_constraints_sequence(
        self,
        schema: ToolGTSchema,
        encode_fn: Any | None = None,
        max_new_tokens_per_slot: int = 128,
    ) -> list[Any]:
        """Return a list of :class:`ConstraintConfig`, one per slot."""
        return [
            self.constraint_for_slot(schema, i, encode_fn, max_new_tokens_per_slot)
            for i in range(len(schema.slots))
        ]

    # -- parsing --------------------------------------------------------------

    @staticmethod
    def parse_fillings(text: str, schema: ToolGTSchema) -> dict[str, str]:
        """Extract slot fillings from generated *text*.

        Returns a dict mapping slot name -> stripped content.
        Missing slots map to the empty string.
        """
        out: dict[str, str] = {}
        for slot in schema.slots:
            pattern = re.compile(
                re.escape(slot.prefix) + r"(.*?)" + re.escape(slot.suffix),
                re.DOTALL,
            )
            m = pattern.search(text)
            out[slot.name] = m.group(1).strip() if m else ""
        return out

    @staticmethod
    def extract_tool_calls(slot_text: str) -> list[dict[str, Any]]:
        """Extract JSON tool-call dicts from *slot_text*.

        Uses ``json.JSONDecoder.raw_decode`` so nested objects are handled
        correctly. Returns a list of dicts that contain a ``name`` key.
        """
        calls: list[dict[str, Any]] = []
        decoder = json.JSONDecoder()
        idx = 0
        text = slot_text.strip()
        while idx < len(text):
            try:
                parsed, end = decoder.raw_decode(text, idx)
                if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                    calls.append(parsed)
                idx = end
            except json.JSONDecodeError:
                idx += 1
        return calls

    # -- validation -----------------------------------------------------------

    @staticmethod
    def validate_completion(text: str, schema: ToolGTSchema) -> dict[str, bool]:
        """Check structural completeness of *text* against *schema*.

        Returns a dict with one key per slot (``True`` iff both prefix and
        suffix are present and non-overlapping).
        """
        out: dict[str, bool] = {}
        for slot in schema.slots:
            has_open = slot.prefix in text
            has_close = slot.suffix in text
            # Non-overlapping simple check
            if has_open and has_close:
                open_idx = text.find(slot.prefix)
                close_idx = text.find(slot.suffix)
                out[slot.name] = close_idx > open_idx
            else:
                out[slot.name] = False
        return out


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

TOOLGT_REASONER = ToolGTReasoner()


# ---------------------------------------------------------------------------
# Convenience registry
# ---------------------------------------------------------------------------

TOOLGT_REGISTRY: dict[str, ToolGTReasoner] = {
    "default": TOOLGT_REASONER,
}
