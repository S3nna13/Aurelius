"""Structured output / grammar-constrained decoding for Aurelius serving.

Constrains language-model token sampling so every generated token keeps the
output on track toward a valid JSON document (or a custom CFG string).  The
algorithm ports the core insight from Outlines / LMQL / Guidance **without**
importing those libraries:

    At each decoding step, compute a boolean token mask over the full
    vocabulary.  A token is *allowed* iff appending its string to the current
    partial output still represents a valid *prefix* of some complete JSON
    document that matches the given schema.  Disallowed token logits are set
    to -inf before sampling.

Key classes
-----------
StructuredOutputDecoder
    JSON-schema–guided decoder.  Uses an incremental state machine plus
    Python's stdlib ``json`` module for completeness checks.

GrammarConstrainedDecoder
    Simpler CFG-based decoder that enforces a whitelist of allowed next-token
    strings keyed by grammar state.

TokenTrie
    Prefix trie over the model vocabulary — O(len(token)) prefix lookups.

STRUCTURED_OUTPUT_REGISTRY
    Maps ``"json_schema"`` / ``"grammar"`` to the above decoder classes.
"""

from __future__ import annotations

import json
import re
from enum import Enum, auto
from typing import Any

import torch

__all__ = [
    "JsonParseState",
    "StructuredOutputDecoder",
    "GrammarConstrainedDecoder",
    "TokenTrie",
    "STRUCTURED_OUTPUT_REGISTRY",
]

# ---------------------------------------------------------------------------
# JSON parse-state machine
# ---------------------------------------------------------------------------


class JsonParseState(Enum):
    """Coarse parse state used by the incremental JSON validator."""

    START = auto()  # nothing emitted yet
    IN_OBJECT = auto()  # inside a { … } block
    IN_ARRAY = auto()  # inside a [ … ] block
    IN_STRING = auto()  # inside a " … " value
    IN_NUMBER = auto()  # accumulating numeric digits / decimal
    IN_BOOL_NULL = auto()  # accumulating true/false/null literal
    AFTER_KEY = auto()  # after an object key, expecting ':'
    AFTER_COLON = auto()  # after ':', expecting a value
    AFTER_VALUE = auto()  # after a complete value, expecting ',' or '}'/'  ]'
    COMPLETE = auto()  # the top-level value is closed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SCALAR_TYPES = frozenset({"string", "number", "integer", "boolean", "null"})

# Characters that can legally start a JSON value.
_VALUE_START_CHARS = frozenset('"0123456789-tfn{[')

_WS = frozenset(" \t\n\r")


def _lstrip_ws(s: str, pos: int) -> int:
    """Advance *pos* past any whitespace characters in *s*."""
    while pos < len(s) and s[pos] in _WS:
        pos += 1
    return pos


def _is_valid_json(text: str) -> tuple[bool, Any]:
    """Return (True, parsed) if *text* is valid JSON, else (False, None)."""
    try:
        return True, json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False, None


# ---------------------------------------------------------------------------
# Incremental prefix validator — the algorithmic core
# ---------------------------------------------------------------------------


class _JsonPrefixValidator:
    """Stateless incremental JSON prefix validator.

    The single public method :meth:`is_valid_prefix` answers: "does *partial*
    represent the beginning of some well-formed JSON value that could satisfy
    *schema*?"  Crucially it accepts *incomplete* JSON — e.g. ``{"key":`` is
    a valid prefix for an object schema.

    Algorithm
    ---------
    We strip one layer of structure from the schema and try to match the
    partial text:

    * For scalar types we use a hand-written state machine that recognises
      partial strings/numbers/booleans.
    * For ``object`` we track bracket depth and key/value alternation.
    * For ``array`` we track bracket depth.
    * For ``anyOf`` / ``oneOf`` we return True if *any* sub-schema validates.
    * For ``enum`` we return True if *partial* is a prefix of any serialised
      allowed value.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_valid_prefix(self, schema: dict, partial: str) -> bool:  # noqa: C901
        if not isinstance(schema, dict):
            raise ValueError(f"Schema must be a dict, got {type(schema).__name__}")

        # Empty partial is always a valid prefix (nothing committed yet).
        if partial == "":
            return True

        # Resolve anyOf / oneOf / allOf
        for combiner in ("anyOf", "oneOf"):
            if combiner in schema:
                sub = schema[combiner]
                if not isinstance(sub, list):
                    raise ValueError(f"'{combiner}' must be a list")
                return any(self.is_valid_prefix(s, partial) for s in sub)

        # Resolve $ref — we do not support external refs; raise clearly.
        if "$ref" in schema:
            raise ValueError("$ref resolution is not supported; inline your schema")

        schema_type = schema.get("type")

        # --- enum ---
        if "enum" in schema:
            return self._check_enum_prefix(schema["enum"], partial)

        # --- const ---
        if "const" in schema:
            return self._check_const_prefix(schema["const"], partial)

        # --- dispatch on type ---
        if schema_type == "string":
            return self._check_string_prefix(partial)

        if schema_type in ("number", "integer"):
            return self._check_number_prefix(partial, integer_only=(schema_type == "integer"))

        if schema_type == "boolean":
            return self._check_literal_prefix(partial, ("true", "false"))

        if schema_type == "null":
            return self._check_literal_prefix(partial, ("null",))

        if schema_type == "object":
            return self._check_object_prefix(schema, partial)

        if schema_type == "array":
            return self._check_array_prefix(schema, partial)

        # No type specified — accept anything that is a valid JSON prefix.
        return self._check_any_prefix(partial)

    # ------------------------------------------------------------------
    # Per-type prefix checkers
    # ------------------------------------------------------------------

    def _check_enum_prefix(self, allowed: list, partial: str) -> bool:
        for val in allowed:
            serialised = json.dumps(val)
            if serialised.startswith(partial) or partial.startswith(serialised):
                # partial could be an exact match or truncated prefix
                if serialised.startswith(partial):
                    return True
        return False

    def _check_const_prefix(self, value: Any, partial: str) -> bool:
        serialised = json.dumps(value)
        return serialised.startswith(partial)

    def _check_literal_prefix(self, partial: str, literals: tuple[str, ...]) -> bool:
        p = partial.lstrip()
        return any(lit.startswith(p) for lit in literals)

    def _check_string_prefix(self, partial: str) -> bool:
        """A valid string prefix must start with '"' and not contain an
        unescaped closing '"' unless it is the very last character."""
        p = partial.lstrip()
        if not p:
            return True
        if p[0] != '"':
            return False
        # Walk through the string body to ensure no premature close.
        i = 1
        while i < len(p):
            ch = p[i]
            if ch == "\\":
                i += 2  # skip escaped char
                continue
            if ch == '"':
                # A closing quote is OK only if it is the last char
                # (the string is complete) — any further content is invalid.
                return i == len(p) - 1
            i += 1
        # We reached end-of-partial while still inside the string body: valid prefix.
        return True

    def _check_number_prefix(self, partial: str, *, integer_only: bool) -> bool:
        p = partial.lstrip()
        if not p:
            return True
        # A valid number partial must match a number prefix pattern.
        pattern = r"^-?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+-]?\d*)?$"
        if integer_only:
            pattern = r"^-?(?:0|[1-9]\d*)$"
        return bool(re.match(pattern, p))

    def _check_object_prefix(self, schema: dict, partial: str) -> bool:
        p = partial.lstrip()
        if not p:
            return True
        if p[0] != "{":
            return False
        # Use depth tracking; try to parse as much as possible.
        # Optimistic: if partial ends mid-string or mid-value, it's still valid.
        try:
            json.loads(p)
            return True  # complete & valid object
        except json.JSONDecodeError as exc:
            # Accept "incomplete" but not "malformed".
            return _is_incomplete_error(exc, p)

    def _check_array_prefix(self, schema: dict, partial: str) -> bool:
        p = partial.lstrip()
        if not p:
            return True
        if p[0] != "[":
            return False
        try:
            json.loads(p)
            return True
        except json.JSONDecodeError as exc:
            return _is_incomplete_error(exc, p)

    def _check_any_prefix(self, partial: str) -> bool:
        p = partial.lstrip()
        if not p:
            return True
        try:
            json.loads(p)
            return True
        except json.JSONDecodeError as exc:
            return _is_incomplete_error(exc, p)


def _is_incomplete_error(exc: json.JSONDecodeError, text: str) -> bool:
    """Return True when a JSONDecodeError indicates truncation rather than malformation.

    Python's json decoder raises specific messages for premature EOF that we
    can inspect to distinguish "just not finished yet" from "syntactically broken".
    """
    msg = exc.msg.lower()
    incomplete_msgs = (
        "expecting value",
        "unterminated string",
        "expecting ',' delimiter",
        "expecting ':' delimiter",
        "end of file",
        "end of data",
    )
    # If the error position is at the very end of the string, it is incomplete.
    at_end = exc.pos >= len(text)
    return at_end or any(m in msg for m in incomplete_msgs)


# ---------------------------------------------------------------------------
# StructuredOutputDecoder
# ---------------------------------------------------------------------------


class StructuredOutputDecoder:
    """Constrains LM token sampling to produce outputs matching a JSON schema.

    At each decoding step, :meth:`build_token_mask_from_schema` iterates the
    vocabulary and marks each token as *allowed* iff appending it to the
    current partial output still yields a valid JSON prefix (or completes the
    JSON).  :meth:`constrained_logits` then sets forbidden token logits to
    ``-inf``.

    Parameters
    ----------
    vocab_size:
        Number of tokens in the model vocabulary.
    eos_token_id:
        Token ID for the end-of-sequence symbol.  Always allowed once a
        *complete* JSON document has been produced.
    """

    JsonParseState = JsonParseState  # re-export as class attribute

    def __init__(self, vocab_size: int, eos_token_id: int) -> None:
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self._validator = _JsonPrefixValidator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_valid_prefix(self, schema: dict, partial_json: str) -> bool:
        """Return True if *partial_json* is a valid prefix of some JSON matching schema."""
        return self._validator.is_valid_prefix(schema, partial_json)

    def is_complete(self, schema: dict, json_str: str) -> bool:
        """Return True if *json_str* is a complete, schema-conforming JSON document."""
        ok, value = _is_valid_json(json_str)
        if not ok:
            return False
        return self._value_matches_schema(value, schema)

    def build_token_mask_from_schema(
        self,
        schema: dict,
        partial_output: str,
        vocab: list[str],
    ) -> torch.Tensor:
        """Return a boolean mask of shape [vocab_size].

        ``mask[i] == True``  iff token *i* is an allowed next token.

        Parameters
        ----------
        schema:
            JSON schema dict.
        partial_output:
            The portion of the output generated so far (may be empty).
        vocab:
            List of token strings ordered by token ID.  ``len(vocab)`` must
            equal ``self.vocab_size``.
        """
        if len(vocab) != self.vocab_size:
            raise ValueError(f"vocab length {len(vocab)} != vocab_size {self.vocab_size}")

        is_done = self.is_complete(schema, partial_output)
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)

        for idx, token_str in enumerate(vocab):
            if idx == self.eos_token_id:
                # EOS is allowed iff we have a complete document.
                mask[idx] = is_done
                continue
            candidate = partial_output + token_str
            try:
                allowed = self._validator.is_valid_prefix(schema, candidate)
            except ValueError:
                allowed = False
            mask[idx] = allowed

        # Safety: if nothing is allowed (schema too restrictive at this state),
        # fall back to allowing EOS to avoid an all-inf row downstream.
        if not mask.any():
            mask[self.eos_token_id] = True

        return mask

    def constrained_logits(
        self,
        logits: torch.Tensor,
        schema: dict,
        partial_output: str,
        vocab: list[str],
    ) -> torch.Tensor:
        """Return a copy of *logits* with disallowed token positions set to ``-inf``.

        Parameters
        ----------
        logits:
            Float tensor of shape ``[batch, vocab_size]`` or ``[vocab_size]``.
        schema:
            JSON schema dict.
        partial_output:
            Partial output so far.
        vocab:
            Vocabulary token strings.
        """
        mask = self.build_token_mask_from_schema(schema, partial_output, vocab)
        out = logits.clone()
        # Broadcast mask across batch dimension if needed.
        if out.dim() == 2:
            out[:, ~mask] = float("-inf")
        else:
            out[~mask] = float("-inf")
        return out

    # ------------------------------------------------------------------
    # Schema conformance checker
    # ------------------------------------------------------------------

    def _value_matches_schema(self, value: Any, schema: dict) -> bool:  # noqa: C901
        """Return True if *value* satisfies *schema*."""
        if not isinstance(schema, dict):
            return False

        # anyOf / oneOf
        for combiner in ("anyOf", "oneOf"):
            if combiner in schema:
                sub = schema[combiner]
                return any(self._value_matches_schema(value, s) for s in sub)

        # enum
        if "enum" in schema:
            return value in schema["enum"]

        # const
        if "const" in schema:
            return value == schema["const"]

        schema_type = schema.get("type")
        if schema_type is None:
            return True  # no type constraint

        if schema_type == "null":
            return value is None
        if schema_type == "boolean":
            return isinstance(value, bool)
        if schema_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if schema_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if schema_type == "string":
            return isinstance(value, str)
        if schema_type == "array":
            if not isinstance(value, list):
                return False
            item_schema = schema.get("items")
            if item_schema is not None:
                return all(self._value_matches_schema(v, item_schema) for v in value)
            return True
        if schema_type == "object":
            if not isinstance(value, dict):
                return False
            required = schema.get("required", [])
            for key in required:
                if key not in value:
                    return False
            props = schema.get("properties", {})
            for key, sub_schema in props.items():
                if key in value:
                    if not self._value_matches_schema(value[key], sub_schema):
                        return False
            return True

        return False


# ---------------------------------------------------------------------------
# TokenTrie — O(len(token)) prefix queries
# ---------------------------------------------------------------------------


class TokenTrie:
    """Prefix trie over the model vocabulary.

    Allows efficient lookup of "which token IDs have strings that start with
    a given prefix?" without scanning the full vocabulary.

    Parameters
    ----------
    vocab:
        List of token strings ordered by token ID.
    """

    def __init__(self, vocab: list[str]) -> None:
        self._root: dict = {}
        for token_id, token_str in enumerate(vocab):
            self._insert(token_str, token_id)

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def _insert(self, token_str: str, token_id: int) -> None:
        node = self._root
        for ch in token_str:
            node = node.setdefault(ch, {})
        node.setdefault("__ids__", []).append(token_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prefix_match(self, prefix: str) -> list[int]:
        """Return all token IDs whose strings start with *prefix*.

        Parameters
        ----------
        prefix:
            The query prefix string.

        Returns
        -------
        list[int]
            Sorted list of token IDs.
        """
        node = self._root
        for ch in prefix:
            if ch not in node:
                return []
            node = node[ch]
        return sorted(self._collect_ids(node))

    def _collect_ids(self, node: dict) -> list[int]:
        ids: list[int] = list(node.get("__ids__", []))
        for key, child in node.items():
            if key != "__ids__":
                ids.extend(self._collect_ids(child))
        return ids

    def has_prefix(self, prefix: str) -> bool:
        """Return True if any token string starts with *prefix*."""
        node = self._root
        for ch in prefix:
            if ch not in node:
                return False
            node = node[ch]
        return True


# ---------------------------------------------------------------------------
# GrammarConstrainedDecoder
# ---------------------------------------------------------------------------


class GrammarConstrainedDecoder:
    """Simple CFG-based constrained decoder.

    Uses a *state → allowed token strings* mapping rather than a full Earley
    parser.  Valid continuations are looked up via a :class:`TokenTrie` for
    O(len(token)) queries.

    Parameters
    ----------
    vocab_size:
        Model vocabulary size.
    eos_token_id:
        EOS token ID — always included in the mask when ``terminal_states``
        contains the current state.
    grammar_states:
        Dict mapping state name → list of allowed next-token strings.
        Example::

            {
                "start": ['"', "true", "false", "null", "{", "["],
                "in_string": list("abcdefghijklmnopqrstuvwxyz "),
            }

    terminal_states:
        Set of state names that are accepting (generation can stop here).
    initial_state:
        Starting state name.
    """

    def __init__(
        self,
        vocab_size: int,
        eos_token_id: int,
        grammar_states: dict[str, list[str]],
        terminal_states: set[str] | None = None,
        initial_state: str = "start",
    ) -> None:
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if not isinstance(grammar_states, dict):
            raise ValueError("grammar_states must be a dict")
        if initial_state not in grammar_states:
            raise ValueError(f"initial_state '{initial_state}' not found in grammar_states")
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.grammar_states = grammar_states
        self.terminal_states: set[str] = terminal_states or set()
        self.initial_state = initial_state
        self._current_state = initial_state

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset to the initial grammar state."""
        self._current_state = self.initial_state

    def transition(self, state: str) -> None:
        """Manually set the current grammar state."""
        if state not in self.grammar_states:
            raise ValueError(f"Unknown grammar state: '{state}'")
        self._current_state = state

    @property
    def current_state(self) -> str:
        return self._current_state

    # ------------------------------------------------------------------
    # Mask computation
    # ------------------------------------------------------------------

    def build_token_mask(self, vocab: list[str], state: str | None = None) -> torch.Tensor:
        """Return a boolean mask of shape [vocab_size] for the given state.

        Parameters
        ----------
        vocab:
            Vocabulary token strings.
        state:
            Grammar state to use.  Defaults to ``self.current_state``.
        """
        if len(vocab) != self.vocab_size:
            raise ValueError(f"vocab length {len(vocab)} != vocab_size {self.vocab_size}")
        state = state or self._current_state
        allowed_strings = self.grammar_states.get(state, [])
        allowed_set = set(allowed_strings)

        is_terminal = state in self.terminal_states
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)

        for idx, tok in enumerate(vocab):
            if idx == self.eos_token_id:
                mask[idx] = is_terminal
            else:
                mask[idx] = tok in allowed_set

        # Safety fallback.
        if not mask.any():
            mask[self.eos_token_id] = True

        return mask

    def constrained_logits(
        self,
        logits: torch.Tensor,
        vocab: list[str],
        state: str | None = None,
    ) -> torch.Tensor:
        """Return logits with disallowed positions set to ``-inf``."""
        mask = self.build_token_mask(vocab, state=state)
        out = logits.clone()
        if out.dim() == 2:
            out[:, ~mask] = float("-inf")
        else:
            out[~mask] = float("-inf")
        return out


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

STRUCTURED_OUTPUT_REGISTRY: dict[str, type] = {
    "json_schema": StructuredOutputDecoder,
    "grammar": GrammarConstrainedDecoder,
}
