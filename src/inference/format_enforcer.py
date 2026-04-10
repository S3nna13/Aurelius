"""Token-level format enforcement via logit masks.

Provides logit-bias-based constrained decoding for various output formats.
This is distinct from structured_output.py (which uses schema-based JSON
prefix validation over string tokens) — here we operate at the raw byte
(token ID = byte value) level with explicit logit masks.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Callable, List

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# FormatSpec
# ---------------------------------------------------------------------------

@dataclass
class FormatSpec:
    """Specification for constrained format generation."""

    format_type: str  # "json" | "markdown" | "list" | "code" | "free"
    required_prefix: str = ""
    required_suffix: str = ""
    max_length: int = 256
    allowed_chars: str = ""  # empty = all chars allowed


# ---------------------------------------------------------------------------
# TokenMask
# ---------------------------------------------------------------------------

class TokenMask:
    """Maintains a boolean mask over the vocabulary and converts to logit biases."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        # True = allowed, False = blocked
        self._mask: list[bool] = [True] * vocab_size

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def allow_all(self) -> "TokenMask":
        """Set all tokens as allowed and return self."""
        self._mask = [True] * self.vocab_size
        return self

    def allow_only(self, token_ids: list[int]) -> "TokenMask":
        """Allow only the specified token IDs; block everything else."""
        self._mask = [False] * self.vocab_size
        for tid in token_ids:
            if 0 <= tid < self.vocab_size:
                self._mask[tid] = True
        return self

    def block(self, token_ids: list[int]) -> "TokenMask":
        """Block the specified token IDs."""
        for tid in token_ids:
            if 0 <= tid < self.vocab_size:
                self._mask[tid] = False
        return self

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_logit_bias(self, blocked_value: float = -1e9) -> Tensor:
        """Return a (vocab_size,) tensor: 0 for allowed, blocked_value for blocked."""
        bias = torch.zeros(self.vocab_size)
        for i, allowed in enumerate(self._mask):
            if not allowed:
                bias[i] = blocked_value
        return bias

    def apply_to_logits(self, logits: Tensor) -> Tensor:
        """Add the logit bias to logits of shape (vocab_size,) or (B, vocab_size)."""
        bias = self.to_logit_bias()
        if logits.dim() == 1:
            return logits + bias
        # (B, V)
        return logits + bias.unsqueeze(0)


# ---------------------------------------------------------------------------
# JsonStateMachine
# ---------------------------------------------------------------------------

_PRINTABLE_ASCII: list[int] = [b for b in range(256) if chr(b) in string.printable]


class JsonStateMachine:
    """Simplified state machine to track JSON structure and restrict next bytes."""

    def __init__(self) -> None:
        self._state: str = "start"
        self._depth: int = 0

    def update(self, token_byte: int) -> None:
        """Update state based on the decoded byte value."""
        ch = chr(token_byte) if 0 <= token_byte < 128 else ""

        if self._state == "start":
            if ch == "{":
                self._state = "after_key"
                self._depth = 1
            # else stay in start

        elif self._state == "after_key":
            if ch == "}":
                self._depth -= 1
                if self._depth == 0:
                    self._state = "done"
            elif ch == ":":
                self._state = "in_value"
            # other chars: stay in after_key

        elif self._state == "in_value":
            if ch == '"':
                self._state = "in_string"
            elif ch == "{":
                self._depth += 1
                self._state = "after_key"
            elif ch == "}":
                self._depth -= 1
                if self._depth == 0:
                    self._state = "done"
                else:
                    self._state = "after_key"
            elif ch == ",":
                self._state = "after_key"

        elif self._state == "in_string":
            if ch == '"':
                self._state = "after_key"
            # else stay in_string

        elif self._state == "done":
            pass  # no further updates

    def allowed_next_bytes(self) -> list[int]:
        """Return byte values (0-255) valid in current state."""
        if self._state == "start":
            return [ord("{")]
        if self._state == "done":
            return []
        # All other states: allow printable ASCII
        return list(_PRINTABLE_ASCII)


# ---------------------------------------------------------------------------
# FormatEnforcer
# ---------------------------------------------------------------------------

class FormatEnforcer:
    """Applies format constraints to logits at each generation step."""

    # Byte value used as a stand-in EOS token (vocab_size=256 has no true EOS)
    _EOS_BYTE: int = 0  # null byte treated as EOS

    def __init__(
        self,
        spec: FormatSpec,
        tokenizer_encode: Callable[[str], list[int]],
        vocab_size: int = 256,
    ) -> None:
        self.spec = spec
        self.tokenizer_encode = tokenizer_encode
        self.vocab_size = vocab_size

        # Pre-encode prefix / suffix bytes
        self._prefix_bytes: list[int] = [b for b in spec.required_prefix.encode("utf-8")]
        self._suffix_bytes: list[int] = [b for b in spec.required_suffix.encode("utf-8")]

        # Pre-compute allowed char set (as byte values)
        if spec.allowed_chars:
            self._allowed_byte_set: list[int] | None = [
                b for ch in spec.allowed_chars for b in ch.encode("utf-8")
            ]
        else:
            self._allowed_byte_set = None

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def get_allowed_tokens(self, generated_so_far: list[int]) -> TokenMask:
        """Return a TokenMask based on format constraints and current position."""
        mask = TokenMask(self.vocab_size)
        n = len(generated_so_far)
        prefix_len = len(self._prefix_bytes)
        suffix_len = len(self._suffix_bytes)

        # 1. Still building the required prefix
        if n < prefix_len:
            next_byte = self._prefix_bytes[n]
            mask.allow_only([next_byte])
            return mask

        # 2. Approaching max_length — force suffix or EOS
        remaining = self.spec.max_length - n
        if remaining <= suffix_len and suffix_len > 0:
            # Work out which suffix byte to force next
            suffix_pos = suffix_len - remaining
            if 0 <= suffix_pos < suffix_len:
                mask.allow_only([self._suffix_bytes[suffix_pos], self._EOS_BYTE])
                return mask
            else:
                mask.allow_only([self._EOS_BYTE])
                return mask

        if remaining <= 0:
            mask.allow_only([self._EOS_BYTE])
            return mask

        # 3. Allowed chars restriction
        if self._allowed_byte_set is not None:
            mask.allow_only(self._allowed_byte_set)
            return mask

        # 4. No restriction
        mask.allow_all()
        return mask

    def enforce_prefix(self, logits: Tensor, generated: list[int]) -> Tensor:
        """Apply the token mask for the current position to logits."""
        mask = self.get_allowed_tokens(generated)
        return mask.apply_to_logits(logits)

    def is_complete(self, generated: list[int]) -> bool:
        """Return True if the required suffix is present at the end of generated."""
        if not self._suffix_bytes:
            return True
        suf = self._suffix_bytes
        if len(generated) < len(suf):
            return False
        return generated[-len(suf):] == suf


# ---------------------------------------------------------------------------
# ConstrainedGenerator
# ---------------------------------------------------------------------------

class ConstrainedGenerator:
    """Wraps a model with format-constrained autoregressive generation."""

    def __init__(
        self,
        model,
        spec: FormatSpec,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
        vocab_size: int = 256,
    ) -> None:
        self.model = model
        self.spec = spec
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.vocab_size = vocab_size
        self._enforcer = FormatEnforcer(spec, tokenizer_encode, vocab_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_token(self, logits: Tensor) -> int:
        """Greedy sample from a (vocab_size,) logit vector."""
        return int(torch.argmax(logits).item())

    def _model_forward(self, input_ids: Tensor) -> Tensor:
        """Run the model and return the last-step logits, shape (vocab_size,)."""
        out = self.model(input_ids)
        # Model returns (loss, logits, pkv) tuple
        if isinstance(out, tuple):
            logits = out[1]
        else:
            logits = out
        # logits shape: (B, T, V) or (B, V) or (V,)
        if logits.dim() == 3:
            logits = logits[0, -1, :]
        elif logits.dim() == 2:
            logits = logits[0, -1] if logits.shape[0] == 1 else logits[-1]
        return logits  # (V,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        """Generate text with format enforcement applied at each step."""
        prompt_ids = self.tokenizer_encode(prompt)
        generated: list[int] = []

        input_ids = torch.tensor([prompt_ids], dtype=torch.long)

        for _ in range(max_new_tokens):
            logits = self._model_forward(input_ids)
            # Apply format mask
            logits = self._enforcer.enforce_prefix(logits, generated)
            next_token = self._sample_token(logits)
            generated.append(next_token)

            # Append to input_ids for next step
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], dtype=torch.long)], dim=1
            )

            # Stop if complete
            if self._enforcer.is_complete(generated):
                break

            # Stop on EOS byte (0x00)
            if next_token == 0:
                break

        return self.tokenizer_decode(generated)

    def generate_json(self, prompt: str) -> str:
        """Generate with JSON format enforcement."""
        json_spec = FormatSpec(
            format_type="json",
            required_prefix="{",
            required_suffix="}",
            max_length=self.spec.max_length,
            allowed_chars=self.spec.allowed_chars,
        )
        original_spec = self._enforcer.spec
        original_enforcer = self._enforcer
        self._enforcer = FormatEnforcer(json_spec, self.tokenizer_encode, self.vocab_size)
        result = self.generate(prompt, max_new_tokens=self.spec.max_length)
        self._enforcer = original_enforcer
        return result

    def generate_list(self, prompt: str, n_items: int = 3) -> list[str]:
        """Generate a bullet list and parse into individual items."""
        list_spec = FormatSpec(
            format_type="list",
            required_prefix="- ",
            max_length=self.spec.max_length,
        )
        original_enforcer = self._enforcer
        self._enforcer = FormatEnforcer(list_spec, self.tokenizer_encode, self.vocab_size)
        raw = self.generate(prompt, max_new_tokens=self.spec.max_length)
        self._enforcer = original_enforcer

        items: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("- "):
                items.append(line[2:].strip())
            elif line.startswith("* "):
                items.append(line[2:].strip())
            elif line:
                items.append(line)
            if len(items) >= n_items:
                break
        return items
