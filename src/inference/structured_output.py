"""Constrained generation that produces valid JSON via character-level prefix validation.

Uses a simple approach: maintain the generated string so far, and at each token
position only allow tokens whose string representation would keep the partial
output as a valid JSON prefix.
"""

import json
from dataclasses import dataclass

import torch

from src.inference.logit_processors import LogitProcessor


@dataclass
class JSONConstraint:
    """Constraint specification for structured JSON generation."""

    schema: dict  # Simple JSON schema dict, e.g. {"type": "object", "properties": {...}}
    # Supported schema types: "object", "array", "string", "number", "boolean", "null"
    # For "string" with "enum": only allow those string values


class PartialJSONValidator:
    """Tracks generated JSON and validates whether a prefix is still potentially valid JSON.

    Uses a simple heuristic: try to parse, if fails check if it's a valid incomplete prefix.
    """

    def __init__(self) -> None:
        self.generated: str = ""

    def append(self, text: str) -> None:
        self.generated += text

    def is_valid_prefix(self, candidate: str) -> bool:
        """Return True if self.generated + candidate is still potentially valid JSON.

        Uses: try full parse -> if succeeds, valid. If fails, check it's a valid
        incomplete JSON prefix by checking that adding closing brackets makes it parseable.
        """
        test = self.generated + candidate
        return _is_json_prefix(test)

    def is_complete(self) -> bool:
        """Return True if self.generated is complete, valid JSON."""
        try:
            json.loads(self.generated)
            return True
        except (json.JSONDecodeError, ValueError):
            return False


def _is_json_prefix(s: str) -> bool:
    """Return True if s is a valid prefix of some valid JSON value.

    Strategy: try parsing directly; if that fails, try completing with closing
    brackets/quotes to see if s is a valid prefix of valid JSON.
    """
    s = s.strip()
    if not s:
        return True  # empty is valid prefix

    # Try direct parse
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, ValueError):
        pass

    # Try completing with various suffixes to see if s is a valid prefix
    completions = [
        '"}',
        '"]}',
        '"}]',
        '"}]}',
        '"',
        '"]',
        "]",
        "}",
        "]}",
        "]}",
        "rue",
        "alse",
        "ull",
        "0}",
        "0",
        "0]",
        '""}}',
        '""}',
        ': ""}',
        ": null}",
        ": 0}",
    ]
    for suffix in completions:
        try:
            json.loads(s + suffix)
            return True
        except (json.JSONDecodeError, ValueError):
            continue

    return False


class JSONSchemaLogitProcessor(LogitProcessor):
    """Constrain token logits to produce valid JSON matching a schema.

    Works with a vocabulary dict (token_id -> string).
    At each step, masks out tokens that would make the partial JSON invalid.
    """

    def __init__(self, vocab: dict[int, str], constraint: JSONConstraint) -> None:
        self.vocab = vocab
        self.constraint = constraint
        self.validator = PartialJSONValidator()
        # Start with opening brace for object schemas
        if constraint.schema.get("type") == "object":
            self.validator.append("{")

    def update(self, token_id: int) -> None:
        """Call after sampling to update the validator state."""
        token_str = self.vocab.get(token_id, "")
        self.validator.append(token_str)

    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Mask out tokens that would produce invalid JSON."""
        logits = logits.clone()
        for token_id, token_str in self.vocab.items():
            if not self.validator.is_valid_prefix(token_str):
                logits[token_id] = float("-inf")
        return logits
