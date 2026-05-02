"""Structured output v2: OutputSchema, StructuredOutputParser, JSONModeDecoder.

V2 because structured_output.py exists with a different (PartialJSONValidator) API.
Provides schema-validated, grammar-constrained structured output generation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class OutputSchema:
    """Schema specification for structured JSON output."""

    schema_type: str = "json"
    required_keys: list[str] = field(default_factory=list)
    value_types: dict[str, str] = field(default_factory=dict)
    max_tokens: int = 512


_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
}


def is_valid_json(text: str) -> bool:
    """Return True if text is valid JSON, False otherwise."""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def extract_json_from_text(text: str) -> str | None:
    """Find first balanced JSON object or array in text using bracket counting.

    Returns extracted JSON string, or None if not found.
    """
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        idx = text.find(start_char)
        if idx == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[idx:], start=idx):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[idx : i + 1]
                    if is_valid_json(candidate):
                        return candidate
                    break
    return None


def validate_schema(data: dict, schema: OutputSchema) -> tuple[bool, list[str]]:
    """Validate parsed dict against OutputSchema.

    Returns (is_valid, list_of_error_messages).
    """
    errors: list[str] = []

    for key in schema.required_keys:
        if key not in data:
            errors.append(f"Missing required key: '{key}'")

    for key, type_name in schema.value_types.items():
        if key not in data:
            continue
        expected_type = _TYPE_MAP.get(type_name)
        if expected_type is None:
            errors.append(f"Unknown type '{type_name}' for key '{key}'")
            continue
        actual = data[key]
        # bool is a subclass of int in Python — check bool before int/float
        if type_name in ("int", "float") and isinstance(actual, bool):
            errors.append(f"Key '{key}': expected {type_name}, got bool")
            continue
        if not isinstance(actual, expected_type):
            errors.append(f"Key '{key}': expected {type_name}, got {type(actual).__name__}")

    return len(errors) == 0, errors


def _infer_json_state(s: str) -> str:
    """Infer current JSON grammar state from partial JSON string.

    Returns: 'in_string', 'expect_value', 'after_value', or 'unknown'.
    """
    if not s.strip():
        return "expect_value"

    in_string = False
    escape_next = False
    last_non_space = ""

    for ch in s:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            last_non_space = ch
            continue
        if in_string:
            continue
        if ch in ("{", "[", ":", ","):
            last_non_space = ch
        elif ch not in (" ", "\t", "\n", "\r"):
            last_non_space = ch

    if in_string:
        return "in_string"
    if last_non_space in (":", "[", "{", ",", ""):
        return "expect_value"
    if last_non_space in ('"', "}", "]") or last_non_space.isdigit():
        return "after_value"
    return "expect_value"


def build_json_grammar_logit_mask(current_json: str, vocab: list[str]) -> Tensor:
    """Return (vocab_size,) float mask: 0.0 = allowed, -inf = banned.

    Simplified grammar rules based on current JSON state:
    - in_string: ban tokens containing unescaped quotes.
    - expect_value: allow tokens starting with quote, digit, -, {, [, t, f, n.
    - after_value: allow tokens starting with ,, }, ].
    """
    vocab_size = len(vocab)
    mask = torch.zeros(vocab_size, dtype=torch.float32)
    NEG_INF = float("-inf")
    state = _infer_json_state(current_json)

    if state == "in_string":
        for i, tok in enumerate(vocab):
            if '"' in tok:
                mask[i] = NEG_INF
    elif state == "expect_value":
        allowed_starts = set('"0123456789-{[tfn')
        for i, tok in enumerate(vocab):
            if not tok or tok[0] not in allowed_starts:
                mask[i] = NEG_INF
    elif state == "after_value":
        allowed = {",", "}", "]"}
        for i, tok in enumerate(vocab):
            stripped = tok.strip()
            if not stripped or stripped[0] not in allowed:
                mask[i] = NEG_INF

    return mask


class StructuredOutputParser:
    """Parse and repair structured JSON output from model-generated text."""

    def __init__(self, schema: OutputSchema) -> None:
        self.schema = schema

    def parse(self, text: str) -> dict | None:
        """Extract, parse, and validate JSON from text; return dict or None."""
        extracted = extract_json_from_text(text)
        if extracted is None:
            if is_valid_json(text):
                extracted = text
            else:
                return None

        try:
            data = json.loads(extracted)
        except (json.JSONDecodeError, ValueError):
            return None

        if not isinstance(data, dict):
            return None

        is_valid, _ = validate_schema(data, self.schema)
        return data if is_valid else None

    def repair(self, text: str) -> str:
        """Attempt to repair malformed JSON by removing trailing commas and adding closers."""
        repaired = re.sub(r",\s*([}\]])", r"\1", text)
        open_brackets = repaired.count("[") - repaired.count("]")
        open_braces = repaired.count("{") - repaired.count("}")
        repaired += "]" * max(0, open_brackets)
        repaired += "}" * max(0, open_braces)
        return repaired

    def batch_parse(self, texts: list[str]) -> list[dict | None]:
        """Parse each text; return None for failures."""
        return [self.parse(t) for t in texts]


class JSONModeDecoder:
    """Greedy decoder that applies grammar masking to enforce JSON validity."""

    def __init__(
        self,
        model_fn: Callable,
        schema: OutputSchema,
        vocab: list[str],
        eos_token_id: int = 2,
    ) -> None:
        self.model_fn = model_fn
        self.schema = schema
        self.vocab = vocab
        self.eos_token_id = eos_token_id

    def decode(self, prompt_ids: Tensor, max_tokens: int = 100) -> str:
        """Greedy decode with grammar masking; stop at EOS or complete valid JSON."""
        input_ids = prompt_ids.clone()
        generated_text = ""

        for _ in range(max_tokens):
            logits = self.model_fn(input_ids)
            if logits.dim() > 1:
                logits = logits[-1]

            mask = build_json_grammar_logit_mask(generated_text, self.vocab)
            logits = logits + mask

            next_token_id = int(torch.argmax(logits).item())
            if next_token_id == self.eos_token_id:
                break

            token_str = self.vocab[next_token_id] if next_token_id < len(self.vocab) else ""
            generated_text += token_str
            input_ids = torch.cat([input_ids, torch.tensor([next_token_id], dtype=input_ids.dtype)])

            if is_valid_json(generated_text):
                break

        return generated_text
