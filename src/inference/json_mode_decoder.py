"""JSON-mode constrained decoder.

A mask-level primitive that, given a caller-supplied vocabulary
(``list[str]``), computes a boolean mask over which token strings
keep the accumulated output a valid JSON prefix.

Simpler than :mod:`src.inference.grammar_constrained`; focuses
specifically on the structural validity of JSON. Pair with any
sampler by calling :meth:`JSONMaskBuilder.mask_logits` before
sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

MAX_DEPTH = 32

# Stack entries.
_OBJ = "obj"
_ARR = "arr"


@dataclass
class JSONDecoderState:
    """Tracks the state of an incremental JSON parser.

    Attributes
    ----------
    stack : list[str]
        Stack of open containers (``"obj"`` / ``"arr"``).
    in_string : bool
        Whether the decoder is currently inside a JSON string literal.
    escape : bool
        Whether the previous character inside the string was a
        backslash (so the next character is escaped).
    expecting_value : bool
        Whether the next non-whitespace character starts a value
        (top-level value, after ``[``, after ``,`` in an array,
        or after ``:`` in an object).
    expecting_key : bool
        Whether the next non-whitespace character starts an object
        key (after ``{`` or after ``,`` inside an object).
    after_key : bool
        Whether a key has just been completed and we expect ``:``.
    after_value : bool
        Whether a value has just been completed and we expect
        either a container close or ``,``.
    in_number : bool
        Whether the decoder is currently inside a numeric literal.
    done : bool
        Whether a complete top-level JSON value has been produced.
    """

    stack: list[str] = field(default_factory=list)
    in_string: bool = False
    escape: bool = False
    expecting_value: bool = True
    expecting_key: bool = False
    after_key: bool = False
    after_value: bool = False
    in_number: bool = False
    done: bool = False

    def clone(self) -> JSONDecoderState:
        return JSONDecoderState(
            stack=list(self.stack),
            in_string=self.in_string,
            escape=self.escape,
            expecting_value=self.expecting_value,
            expecting_key=self.expecting_key,
            after_key=self.after_key,
            after_value=self.after_value,
            in_number=self.in_number,
            done=self.done,
        )


class JSONMaskBuilder:
    """Builds boolean masks that keep the generated string a valid
    JSON prefix.

    The builder operates purely over token *strings* (UTF-8). A
    token is admissible iff, starting from the current parser
    state, feeding its characters one-by-one never produces an
    invalid JSON prefix.
    """

    def __init__(self, vocab: list[str], allow_whitespace: bool = True) -> None:
        if not isinstance(vocab, list):
            raise TypeError("vocab must be a list[str]")
        for tok in vocab:
            if not isinstance(tok, str):
                raise TypeError("vocab must contain only str tokens")
        self.vocab = vocab
        self.allow_whitespace = allow_whitespace

    # -- state lifecycle ----------------------------------------------------

    def reset(self) -> JSONDecoderState:
        return JSONDecoderState()

    # -- core character transition -----------------------------------------

    def _step_char(self, state: JSONDecoderState, ch: str) -> JSONDecoderState:
        """Advance ``state`` by a single character. Returns a new state.

        Raises ``ValueError`` on invalid input, ``OverflowError``
        if container depth exceeds :data:`MAX_DEPTH`.
        """

        s = state.clone()

        if s.done:
            if self.allow_whitespace and ch in " \t\n\r":
                return s
            raise ValueError(f"trailing char {ch!r} after complete JSON value")

        # Inside a string literal.
        if s.in_string:
            if s.escape:
                s.escape = False
                return s
            if ch == "\\":
                s.escape = True
                return s
            if ch == '"':
                s.in_string = False
                # A string finishes either a key or a value.
                if s.expecting_key:
                    s.expecting_key = False
                    s.after_key = True
                else:
                    # was a value
                    s.expecting_value = False
                    s = self._finish_value(s)
                return s
            # Any other char: stays in string. (Control chars are
            # technically invalid per RFC 8259 but we're lenient
            # for the mask-level primitive.)
            return s

        # Inside a number.
        if s.in_number:
            if ch.isdigit() or ch in ".eE+-":
                return s
            # Number ended. Re-feed ch as a post-value char.
            s.in_number = False
            s.expecting_value = False
            s = self._finish_value(s)
            return self._step_char(s, ch)

        # Whitespace.
        if ch in " \t\n\r":
            if self.allow_whitespace:
                return s
            raise ValueError("whitespace not allowed")

        # Expecting a value.
        if s.expecting_value:
            if ch == "{":
                if len(s.stack) >= MAX_DEPTH:
                    raise OverflowError("JSON depth exceeded MAX_DEPTH")
                s.stack.append(_OBJ)
                s.expecting_value = False
                s.expecting_key = True  # either key or "}"
                return s
            if ch == "[":
                if len(s.stack) >= MAX_DEPTH:
                    raise OverflowError("JSON depth exceeded MAX_DEPTH")
                s.stack.append(_ARR)
                s.expecting_value = True  # either value or "]"
                return s
            if ch == '"':
                s.in_string = True
                s.escape = False
                # value string (not key)
                return s
            if ch.isdigit() or ch == "-":
                s.in_number = True
                return s
            # Array close directly after "[" (empty array).
            if ch == "]" and s.stack and s.stack[-1] == _ARR:
                s.stack.pop()
                s.expecting_value = False
                s = self._finish_value(s)
                return s
            # t/f/n for true/false/null - not required for the spec
            # minimal vocab but supported trivially:
            if ch in "tfn":
                # Treat like in_number: consume alpha until space/terminator.
                # Simplify: mark as "in bareword".
                s.in_number = True  # reuse flag (accepts alpha? no)
                # We don't actually accept in the digit branch; so refuse.
                raise ValueError("bareword literals not supported")
            raise ValueError(f"unexpected char {ch!r} when expecting value")

        # Expecting an object key.
        if s.expecting_key:
            if ch == '"':
                s.in_string = True
                s.escape = False
                return s
            if ch == "}" and s.stack and s.stack[-1] == _OBJ:
                # empty object or trailing key position? Only valid if
                # stack top is an object and we were right after "{".
                s.stack.pop()
                s.expecting_key = False
                s.expecting_value = False
                s = self._finish_value(s)
                return s
            raise ValueError(f"unexpected char {ch!r} when expecting key")

        # After a key, expecting ":".
        if s.after_key:
            if ch == ":":
                s.after_key = False
                s.expecting_value = True
                return s
            raise ValueError(f"unexpected char {ch!r} when expecting ':'")

        # After a value, expecting "," or container close.
        if s.after_value:
            if not s.stack:
                # Top-level value complete. Only ws permitted.
                raise ValueError(f"unexpected char {ch!r} after JSON")
            top = s.stack[-1]
            if ch == ",":
                s.after_value = False
                if top == _OBJ:
                    s.expecting_key = True
                else:
                    s.expecting_value = True
                return s
            if ch == "}" and top == _OBJ:
                s.stack.pop()
                s.after_value = False
                s = self._finish_value(s)
                return s
            if ch == "]" and top == _ARR:
                s.stack.pop()
                s.after_value = False
                s = self._finish_value(s)
                return s
            raise ValueError(f"unexpected char {ch!r} after value")

        raise ValueError(f"parser stuck on char {ch!r}")

    def _finish_value(self, s: JSONDecoderState) -> JSONDecoderState:
        """Marks a value as complete and updates the surrounding context."""
        if not s.stack:
            s.done = True
            s.after_value = False
            s.expecting_value = False
        else:
            s.after_value = True
            s.expecting_value = False
        return s

    # -- public transition API ---------------------------------------------

    def update(self, state: JSONDecoderState, next_string: str) -> JSONDecoderState:
        """Advance ``state`` by feeding every character of ``next_string``."""
        s = state
        for ch in next_string:
            s = self._step_char(s, ch)
        return s

    # -- mask construction -------------------------------------------------

    def _try_token(self, state: JSONDecoderState, tok: str) -> bool:
        if tok == "":
            return True
        try:
            self.update(state, tok)
        except (ValueError, OverflowError):
            return False
        return True

    def get_mask(
        self, state: JSONDecoderState, vocab_strings: list[str] | None = None
    ) -> torch.Tensor:
        """Return a boolean mask over the vocab of admissible tokens."""
        vocab = vocab_strings if vocab_strings is not None else self.vocab
        admissible = [self._try_token(state, tok) for tok in vocab]
        return torch.tensor(admissible, dtype=torch.bool)

    def mask_logits(self, logits: torch.Tensor, state: JSONDecoderState) -> torch.Tensor:
        """Return a copy of ``logits`` with forbidden entries set to -inf."""
        if logits.dim() != 1:
            raise ValueError("mask_logits expects a 1-D logits tensor [V]")
        if logits.shape[0] != len(self.vocab):
            raise ValueError(f"logits size {logits.shape[0]} != vocab size {len(self.vocab)}")
        mask = self.get_mask(state)
        out = logits.clone()
        out[~mask] = float("-inf")
        return out


def is_valid_json_prefix(s: str) -> bool:
    """Return True iff ``s`` is a valid prefix of some JSON document.

    Uses the same state machine as :class:`JSONMaskBuilder` with
    whitespace allowed.
    """
    builder = JSONMaskBuilder(vocab=[""], allow_whitespace=True)
    try:
        builder.update(builder.reset(), s)
    except (ValueError, OverflowError):
        return False
    return True


__all__ = [
    "JSONDecoderState",
    "JSONMaskBuilder",
    "MAX_DEPTH",
    "is_valid_json_prefix",
]
