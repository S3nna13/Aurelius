"""Wait Token Forcer — S1-style wait-token injection for extended thinking.

Implements the "wait token injection" technique from the S1 (simple test-time
scaling) paper (2025).  When a model tries to end its thinking chain early by
producing an end-of-thinking token (e.g. </think>), this module detects the
token and replaces it with a special "Wait" continuation token, forcing the
model to reason further.

This is a *sequence-manipulation* approach distinct from the logit-bias
approach used in token_budget_forcing.py.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WaitTokenForcerConfig:
    """Configuration for the wait-token injection strategy.

    Args:
        end_think_token_id:  Token ID that signals end-of-thinking (e.g.
                             the ``</think>`` token).  Qwen2.5 default: 151645.
        wait_token_id:       Token ID for the "Wait" continuation token.
                             Qwen2.5 default: 151649.
        max_thinking_tokens: Hard upper limit on the thinking sequence length.
                             Wait tokens are never injected once this limit is
                             reached.
        max_wait_injections: Maximum number of wait tokens that may be injected
                             for a single sequence.
        min_thinking_tokens: Minimum sequence length before injection is
                             allowed.  Prevents injecting into very short
                             prefixes where the end-of-think token would be
                             spurious.
    """

    end_think_token_id: int = 151645  # </think>
    wait_token_id: int = 151649  # "Wait"
    max_thinking_tokens: int = 16384
    max_wait_injections: int = 8
    min_thinking_tokens: int = 64


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class WaitTokenForcer:
    """Inject wait tokens to prevent premature end-of-thinking.

    Usage (streaming / step-by-step)::

        forcer = WaitTokenForcer()
        token_ids = []
        for token in model.stream(prompt):
            token_ids.append(token)
            token_ids = forcer.inject_wait(token_ids)

    Usage (batch / full sequence)::

        forcer = WaitTokenForcer()
        processed, n = forcer.process_sequence(token_ids)

    Args:
        config: :class:`WaitTokenForcerConfig`.  Uses defaults when omitted.
    """

    def __init__(self, config: WaitTokenForcerConfig | None = None) -> None:
        self._config: WaitTokenForcerConfig = config or WaitTokenForcerConfig()
        self._injections_used: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def injections_used(self) -> int:
        """Number of wait tokens injected since the last :meth:`reset`."""
        return self._injections_used

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def should_inject_wait(self, token_ids: list[int]) -> bool:
        """Return ``True`` if a wait token should be injected right now.

        All four conditions must hold:

        1. The last token in *token_ids* is ``end_think_token_id``.
        2. ``len(token_ids) < max_thinking_tokens``.
        3. ``injections_used < max_wait_injections``.
        4. ``len(token_ids) >= min_thinking_tokens``.

        Args:
            token_ids: Current token sequence (including the latest token).

        Returns:
            ``True`` when injection is warranted, ``False`` otherwise.
        """
        cfg = self._config
        if not token_ids:
            return False
        if token_ids[-1] != cfg.end_think_token_id:
            return False
        if len(token_ids) >= cfg.max_thinking_tokens:
            return False
        if self._injections_used >= cfg.max_wait_injections:
            return False
        if len(token_ids) < cfg.min_thinking_tokens:
            return False
        return True

    def inject_wait(self, token_ids: list[int]) -> list[int]:
        """Conditionally replace a trailing end-think token with a wait token.

        If :meth:`should_inject_wait` returns ``True``:

        * Remove the last token (``end_think_token_id``).
        * Append ``wait_token_id``.
        * Increment the internal injection counter.

        Otherwise the list is returned unchanged (a copy is *not* made when no
        injection occurs).

        Args:
            token_ids: Current token sequence.

        Returns:
            Possibly modified token sequence.
        """
        if not self.should_inject_wait(token_ids):
            return token_ids

        result = token_ids[:-1] + [self._config.wait_token_id]
        self._injections_used += 1
        return result

    def process_sequence(self, token_ids: list[int]) -> tuple[list[int], int]:
        """Process a complete token sequence in a single pass.

        Scans *token_ids* left-to-right and replaces each occurrence of
        ``end_think_token_id`` with ``wait_token_id`` (up to the configured
        limits).  After the injection budget is exhausted, remaining
        end-think tokens are left in place.

        .. note::
            This method uses (and modifies) the instance's internal injection
            counter.  Call :meth:`reset` beforehand if a clean slate is needed.

        Args:
            token_ids: Input token sequence to process.

        Returns:
            A tuple ``(processed_ids, n_injections)`` where *n_injections* is
            the number of wait tokens injected during *this call*.
        """
        cfg = self._config
        result: list[int] = []
        injections_this_call: int = 0

        for tok in token_ids:
            result.append(tok)
            # Reuse should_inject_wait logic: check if end-think token is last
            # in the result built so far, within all budget limits.
            if (
                tok == cfg.end_think_token_id
                and len(result) >= cfg.min_thinking_tokens
                and len(result) < cfg.max_thinking_tokens
                and self._injections_used < cfg.max_wait_injections
            ):
                result[-1] = cfg.wait_token_id
                self._injections_used += 1
                injections_this_call += 1

        return result, injections_this_call

    def reset(self) -> None:
        """Reset the injection counter for a new sequence."""
        self._injections_used = 0

    def injection_stats(self) -> dict:
        """Return a snapshot of the current injection budget state.

        Returns:
            A dict with keys ``"injections_used"``, ``"max_allowed"``, and
            ``"budget_remaining"``.
        """
        used = self._injections_used
        maximum = self._config.max_wait_injections
        return {
            "injections_used": used,
            "max_allowed": maximum,
            "budget_remaining": maximum - used,
        }
