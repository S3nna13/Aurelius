"""Grammar-constrained generation: regex FSM, JSON schema constraints, and logit masking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor


@dataclass
class GrammarConfig:
    """Configuration for grammar-constrained generation."""

    max_new_tokens: int = 64
    grammar_type: str = "regex"  # "regex" | "json_schema" | "free"
    pattern: str = ""            # regex pattern for "regex" mode
    allow_any_on_fail: bool = True  # if FSM gets stuck, allow any token


class RegexFSM:
    """Finite state machine for regex-constrained generation.

    Tracks the text generated so far and uses the compiled regex to
    determine which tokens are still reachable (partial-match check).
    """

    def __init__(self, pattern: str, vocab_size: int) -> None:
        self.pattern = re.compile(pattern)
        self._raw_pattern = pattern
        self.current_text: str = ""
        self._vocab_size = vocab_size

    def get_allowed_tokens(
        self,
        vocab_size: int,
        decode_fn: Callable[[int], str],
    ) -> Tensor:
        """Return binary mask of shape (vocab_size,) — 1 = allowed, 0 = blocked.

        A token is allowed if current_text + decode_fn(token) could still
        be a prefix of a string that fully matches the pattern.
        """
        mask = torch.zeros(vocab_size, dtype=torch.float32)
        partial_re = re.compile(self._raw_pattern + ".*", re.DOTALL)
        for token_id in range(vocab_size):
            candidate = self.current_text + decode_fn(token_id)
            if partial_re.match(candidate) is not None:
                mask[token_id] = 1.0
        return mask

    def advance(self, token_id: int, decode_fn: Callable[[int], str]) -> None:
        """Append the decoded token to current_text."""
        self.current_text += decode_fn(token_id)


# Characters considered structurally valid in JSON
_JSON_ALLOWED_CHARS = set('{}[]":,. \t\n\r') | set("abcdefghijklmnopqrstuvwxyz") | set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | set("0123456789") | set("_-+*/\\")


def json_schema_mask(
    schema: dict,
    current_json: str,
    vocab_size: int,
    decode_fn: Callable[[int], str],
) -> Tensor:
    """Return binary mask (vocab_size,) allowing tokens that keep JSON structurally valid.

    Uses a simplified check: tracks brace/bracket depth and allows characters
    that are legal in JSON ({, }, [, ], ", :, ,, digits, letters, space).
    """
    mask = torch.zeros(vocab_size, dtype=torch.float32)

    # Compute current brace/bracket depth from existing JSON
    depth = 0
    in_string = False
    escape_next = False
    for ch in current_json:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch in ("{", "["):
                depth += 1
            elif ch in ("}", "]"):
                depth -= 1

    for token_id in range(vocab_size):
        token_str = decode_fn(token_id)
        if not token_str:
            continue
        # Allow token if all characters are JSON-legal
        if all(ch in _JSON_ALLOWED_CHARS for ch in token_str):
            # Check that adding token_str would not produce negative depth
            test_depth = depth
            test_in_string = in_string
            test_escape = False
            valid = True
            for ch in token_str:
                if test_escape:
                    test_escape = False
                    continue
                if ch == "\\" and test_in_string:
                    test_escape = True
                    continue
                if ch == '"':
                    test_in_string = not test_in_string
                elif not test_in_string:
                    if ch in ("{", "["):
                        test_depth += 1
                    elif ch in ("}", "]"):
                        test_depth -= 1
                        if test_depth < 0:
                            valid = False
                            break
            if valid:
                mask[token_id] = 1.0

    return mask


def apply_grammar_mask(logits: Tensor, mask: Tensor) -> Tensor:
    """Set logits to -inf where mask == 0. Returns masked logits."""
    logits = logits.clone()
    logits[mask == 0] = float("-inf")
    return logits


def validate_against_pattern(text: str, pattern: str) -> bool:
    """Return True if text fully matches the regex pattern."""
    return re.fullmatch(pattern, text) is not None


class ConstrainedDecoder:
    """Autoregressive decoder with grammar constraints.

    Supports three grammar_type modes:
      - "regex":       use RegexFSM to mask invalid tokens at each step
      - "json_schema": use json_schema_mask at each step
      - "free":        no masking
    """

    def __init__(
        self,
        model,
        encode_fn: Callable[[str], list[int]],
        decode_fn: Callable[[int], str],
        config: GrammarConfig,
    ) -> None:
        self.model = model
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.config = config

    def decode_token(self, token_id: int) -> str:
        """Decode a single token id to string."""
        return self.decode_fn(token_id)

    def generate(self, prompt: str) -> tuple[str, dict]:
        """Generate text from prompt with grammar constraints.

        Returns:
            (generated_text, stats) where stats has:
                "n_tokens": total tokens generated
                "n_constrained_steps": steps where mask reduced token choices
        """
        config = self.config
        input_ids_list = self.encode_fn(prompt)
        input_ids = torch.tensor(input_ids_list, dtype=torch.long).unsqueeze(0)

        # Infer vocab_size from first forward pass
        with torch.no_grad():
            _, logits_init, _ = self.model(input_ids)
        vocab_size = logits_init.shape[-1]

        # Initialise FSM for regex mode
        fsm: RegexFSM | None = None
        if config.grammar_type == "regex":
            fsm = RegexFSM(config.pattern, vocab_size)

        generated_ids: list[int] = []
        current_json = ""
        n_constrained_steps = 0

        current_input = input_ids
        for _ in range(config.max_new_tokens):
            with torch.no_grad():
                _, logits, _ = self.model(current_input)

            # logits shape: (1, seq_len, vocab_size) — take last position
            next_logits = logits[0, -1, :]  # (vocab_size,)

            # Build and apply mask based on grammar_type
            mask: Tensor | None = None
            if config.grammar_type == "regex" and fsm is not None:
                mask = fsm.get_allowed_tokens(vocab_size, self.decode_fn)
            elif config.grammar_type == "json_schema":
                schema = {}
                mask = json_schema_mask(schema, current_json, vocab_size, self.decode_fn)

            if mask is not None:
                allowed_count = int(mask.sum().item())
                if allowed_count == 0 and config.allow_any_on_fail:
                    # FSM stuck — allow any token
                    pass
                else:
                    original_allowed = int((next_logits > float("-inf")).sum().item())
                    next_logits = apply_grammar_mask(next_logits, mask)
                    new_allowed = int((next_logits > float("-inf")).sum().item())
                    if new_allowed < original_allowed:
                        n_constrained_steps += 1

            # Greedy sampling
            next_token_id = int(next_logits.argmax().item())
            generated_ids.append(next_token_id)

            # Update FSM / json state
            if fsm is not None:
                fsm.advance(next_token_id, self.decode_fn)
            if config.grammar_type == "json_schema":
                current_json += self.decode_fn(next_token_id)

            # Append token to input sequence
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
            current_input = torch.cat([current_input, next_token_tensor], dim=1)

        generated_text = "".join(self.decode_fn(t) for t in generated_ids)
        stats = {
            "n_tokens": len(generated_ids),
            "n_constrained_steps": n_constrained_steps,
        }
        return generated_text, stats
