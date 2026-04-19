"""Message-truncation policies for long conversations.

Applied by serving/inference surfaces *before* a chat template is
rendered, when the accumulated multi-turn history would otherwise
exceed a context-window budget or a turn cap.

Strategies (explicit, no silent fallbacks):

- ``keep_last_n``
    Drop older turns, preserve the last ``max_turns``.
- ``keep_system_and_last_n``
    Always keep the leading ``system`` message(s); drop older
    non-system turns until ``max_turns`` non-system turns remain.
- ``token_budget_oldest_first``
    Iteratively drop the oldest non-system turn until the total
    token count (``token_counter`` applied to each content) is
    <= ``max_tokens``.
- ``token_budget_summarize_oldest``
    Like ``token_budget_oldest_first``, but the dropped prefix is
    replaced by a single synthetic ``system`` message produced by
    ``summarize_fn``.
- ``priority_score``
    Rank turns via ``priority_fn`` and keep the top ``max_turns``
    (stable on the original order for ties; leading system message
    is always preserved).

All strategies are pure, deterministic given their inputs, and use
only the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

__all__ = [
    "TruncatedResult",
    "MessageTruncationPolicy",
    "VALID_STRATEGIES",
]


VALID_STRATEGIES = frozenset(
    {
        "keep_last_n",
        "keep_system_and_last_n",
        "token_budget_oldest_first",
        "token_budget_summarize_oldest",
        "priority_score",
    }
)


def _default_token_counter(text: str) -> int:
    """Cheap default: ~4 characters per token. Used only when the
    caller hasn't supplied a real tokenizer."""
    if not isinstance(text, str):
        raise TypeError("token_counter received non-str content")
    return len(text) // 4


@dataclass
class TruncatedResult:
    """Outcome of applying a truncation policy.

    Attributes:
        messages: The surviving message list (in original order, with
            any synthetic summary prepended where applicable).
        dropped_count: Number of original messages that were removed.
        summary_added: True iff a synthetic summary message was
            inserted (only possible for ``token_budget_summarize_oldest``).
    """

    messages: List[dict]
    dropped_count: int
    summary_added: bool


class MessageTruncationPolicy:
    """Apply a named truncation strategy to a list of chat messages.

    Args:
        strategy: Name of the strategy. Must be one of
            :data:`VALID_STRATEGIES`; otherwise :class:`ValueError`.
        max_tokens: Token budget (required for the two token_budget_*
            strategies).
        max_turns: Turn cap (required for keep_last_n,
            keep_system_and_last_n, priority_score).
        token_counter: ``str -> int``. Defaults to ``len(text)//4``.
        summarize_fn: ``list[dict] -> str``. Required for
            ``token_budget_summarize_oldest``; rejected elsewhere is
            not enforced (callers may set and not use it).
        priority_fn: ``dict -> float``. Required for ``priority_score``.
    """

    def __init__(
        self,
        strategy: str,
        max_tokens: Optional[int] = None,
        max_turns: Optional[int] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        summarize_fn: Optional[Callable[[List[dict]], str]] = None,
        priority_fn: Optional[Callable[[dict], float]] = None,
    ) -> None:
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"unknown truncation strategy: {strategy!r}. "
                f"valid: {sorted(VALID_STRATEGIES)}"
            )
        if max_tokens is not None and max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        if max_turns is not None and max_turns < 0:
            raise ValueError("max_turns must be >= 0")

        self.strategy = strategy
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.token_counter = token_counter or _default_token_counter
        self.summarize_fn = summarize_fn
        self.priority_fn = priority_fn

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def apply(self, messages: List[dict]) -> TruncatedResult:
        """Apply the configured strategy. Returns a new list; does not
        mutate ``messages``."""
        if not isinstance(messages, list):
            raise TypeError("messages must be a list of dicts")
        if not messages:
            return TruncatedResult(messages=[], dropped_count=0, summary_added=False)
        for m in messages:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise ValueError(
                    "every message must be a dict with 'role' and 'content' keys"
                )

        if self.strategy == "keep_last_n":
            return self._keep_last_n(messages)
        if self.strategy == "keep_system_and_last_n":
            return self._keep_system_and_last_n(messages)
        if self.strategy == "token_budget_oldest_first":
            return self._token_budget_oldest_first(messages)
        if self.strategy == "token_budget_summarize_oldest":
            return self._token_budget_summarize_oldest(messages)
        if self.strategy == "priority_score":
            return self._priority_score(messages)
        # Unreachable: constructor validates.
        raise ValueError(f"unknown strategy: {self.strategy!r}")

    # ------------------------------------------------------------------
    # strategies
    # ------------------------------------------------------------------
    def _require_max_turns(self) -> int:
        if self.max_turns is None:
            raise ValueError(
                f"strategy {self.strategy!r} requires max_turns"
            )
        return self.max_turns

    def _require_max_tokens(self) -> int:
        if self.max_tokens is None:
            raise ValueError(
                f"strategy {self.strategy!r} requires max_tokens"
            )
        return self.max_tokens

    def _count_tokens(self, messages: List[dict]) -> int:
        total = 0
        for m in messages:
            total += self.token_counter(m["content"])
        return total

    def _keep_last_n(self, messages: List[dict]) -> TruncatedResult:
        n = self._require_max_turns()
        if n >= len(messages):
            return TruncatedResult(
                messages=list(messages), dropped_count=0, summary_added=False
            )
        kept = list(messages[-n:]) if n > 0 else []
        dropped = len(messages) - len(kept)
        return TruncatedResult(
            messages=kept, dropped_count=dropped, summary_added=False
        )

    def _keep_system_and_last_n(self, messages: List[dict]) -> TruncatedResult:
        n = self._require_max_turns()
        # Leading system messages are always preserved.
        leading_system: List[dict] = []
        i = 0
        while i < len(messages) and messages[i]["role"] == "system":
            leading_system.append(messages[i])
            i += 1
        rest = messages[i:]
        if n >= len(rest):
            kept = list(leading_system) + list(rest)
            return TruncatedResult(
                messages=kept,
                dropped_count=len(messages) - len(kept),
                summary_added=False,
            )
        tail = list(rest[-n:]) if n > 0 else []
        kept = list(leading_system) + tail
        dropped = len(messages) - len(kept)
        return TruncatedResult(
            messages=kept, dropped_count=dropped, summary_added=False
        )

    def _token_budget_oldest_first(
        self, messages: List[dict]
    ) -> TruncatedResult:
        budget = self._require_max_tokens()
        work = list(messages)
        dropped = 0
        # Preserve leading system messages: drop only from the
        # non-system prefix.
        sys_end = 0
        while sys_end < len(work) and work[sys_end]["role"] == "system":
            sys_end += 1

        while self._count_tokens(work) > budget:
            if sys_end >= len(work):
                # Only system messages remain but budget still exceeded;
                # keep dropping from the front of the system block as a
                # last resort.
                if not work:
                    break
                work.pop(0)
                if sys_end > 0:
                    sys_end -= 1
                dropped += 1
                continue
            # drop oldest non-system
            del work[sys_end]
            dropped += 1

        return TruncatedResult(
            messages=work, dropped_count=dropped, summary_added=False
        )

    def _token_budget_summarize_oldest(
        self, messages: List[dict]
    ) -> TruncatedResult:
        if self.summarize_fn is None:
            raise ValueError(
                "strategy 'token_budget_summarize_oldest' requires summarize_fn"
            )
        budget = self._require_max_tokens()

        # Leading system messages survive unchanged.
        sys_end = 0
        while sys_end < len(messages) and messages[sys_end]["role"] == "system":
            sys_end += 1
        leading_system = list(messages[:sys_end])
        tail = list(messages[sys_end:])

        # If already within budget, no-op.
        if self._count_tokens(leading_system + tail) <= budget:
            return TruncatedResult(
                messages=list(messages),
                dropped_count=0,
                summary_added=False,
            )

        # Drop oldest non-system turns one at a time, collecting them
        # for summarization. After each drop, re-check budget including
        # the synthetic summary (so we don't overshoot).
        to_summarize: List[dict] = []
        while tail:
            candidate_summary_text = self.summarize_fn(to_summarize) if to_summarize else ""
            synthetic = (
                [{"role": "system", "content": candidate_summary_text}]
                if to_summarize
                else []
            )
            current = leading_system + synthetic + tail
            if self._count_tokens(current) <= budget:
                break
            to_summarize.append(tail.pop(0))

        if not to_summarize:
            # Nothing was summarized; fall back semantically to no-op
            # (budget already satisfied by leading system + tail, or
            # tail exhausted with no non-system content).
            return TruncatedResult(
                messages=leading_system + tail,
                dropped_count=len(messages) - len(leading_system) - len(tail),
                summary_added=False,
            )

        summary_text = self.summarize_fn(to_summarize)
        if not isinstance(summary_text, str):
            raise TypeError("summarize_fn must return str")
        summary_msg = {"role": "system", "content": summary_text}
        kept = leading_system + [summary_msg] + tail
        return TruncatedResult(
            messages=kept,
            dropped_count=len(to_summarize),
            summary_added=True,
        )

    def _priority_score(self, messages: List[dict]) -> TruncatedResult:
        if self.priority_fn is None:
            raise ValueError(
                "strategy 'priority_score' requires priority_fn"
            )
        n = self._require_max_turns()

        # Always preserve leading system message(s) outside the ranking
        # so they cannot be evicted.
        sys_end = 0
        while sys_end < len(messages) and messages[sys_end]["role"] == "system":
            sys_end += 1
        leading_system = list(messages[:sys_end])
        rest = list(messages[sys_end:])

        if n >= len(rest):
            kept = leading_system + rest
            return TruncatedResult(
                messages=kept,
                dropped_count=len(messages) - len(kept),
                summary_added=False,
            )

        # Score, then pick top-N stably by score desc, original-index asc.
        scored = []
        for idx, m in enumerate(rest):
            score = self.priority_fn(m)
            if not isinstance(score, (int, float)):
                raise TypeError("priority_fn must return a real number")
            scored.append((float(score), idx, m))
        # Sort by score desc, then idx asc (stable tie-break).
        scored.sort(key=lambda t: (-t[0], t[1]))
        chosen = scored[:n] if n > 0 else []
        # Re-sort chosen by original index to preserve conversation
        # order.
        chosen.sort(key=lambda t: t[1])
        kept = leading_system + [t[2] for t in chosen]
        dropped = len(messages) - len(kept)
        return TruncatedResult(
            messages=kept, dropped_count=dropped, summary_added=False
        )
