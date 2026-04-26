"""Tool-error recovery strategies for agent loops.

Given a failed :class:`ToolInvocationResult` from
:mod:`src.agent.tool_registry_dispatcher`, this module classifies the
error, selects a recovery strategy, and (via
:class:`RecoveringDispatcher`) applies that strategy automatically up to
a policy-configured limit.

The five strategies are deliberately coarse so an agent policy layer
can reason about them:

* ``retry_backoff`` — transient failure, wait then reissue.
* ``retry_modified_args`` — schema/validation failure, let the agent
  edit arguments before retry.
* ``fallback_tool`` — this tool is broken or missing; try an
  alternative that implements the same capability.
* ``escalate`` — surface to a human/outer loop (permission, policy).
* ``abort`` — give up; unrecoverable or retry budget exhausted.

The naming and "reflect then retry" rhythm is informed by Reflexion
(Shinn 2023, arXiv:2303.11366), but the mechanics here are
deterministic and dependency-free — no LLM reflection is baked in, the
hook is left for a higher-level policy.

Standard library only.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from .tool_registry_dispatcher import ToolInvocationResult, ToolRegistryDispatcher

__all__ = [
    "RecoveryStrategy",
    "RecoveryDecision",
    "RecoveryPolicy",
    "ErrorClassifier",
    "decide",
    "RecoveringDispatcher",
]


# ---------------------------------------------------------------------------
# Strategy constants
# ---------------------------------------------------------------------------


class RecoveryStrategy:
    """Namespace of valid recovery strategy identifiers.

    Kept as string constants (rather than ``enum.Enum``) so they
    serialise trivially into tool traces and audit logs.
    """

    RETRY_BACKOFF = "retry_backoff"
    RETRY_MODIFIED_ARGS = "retry_modified_args"
    FALLBACK_TOOL = "fallback_tool"
    ESCALATE = "escalate"
    ABORT = "abort"

    ALL: frozenset[str] = frozenset(
        {
            "retry_backoff",
            "retry_modified_args",
            "fallback_tool",
            "escalate",
            "abort",
        }
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RecoveryDecision:
    """A concrete recovery action selected for a failed tool call."""

    strategy: str
    reason: str
    delay_seconds: float = 0.0
    modified_args: dict | None = None
    fallback_tool: str | None = None

    def __post_init__(self) -> None:
        if self.strategy not in RecoveryStrategy.ALL:
            raise ValueError(f"unknown recovery strategy: {self.strategy!r}")
        if self.delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative")


@dataclass
class RecoveryPolicy:
    """Configuration for :func:`decide` / :class:`RecoveringDispatcher`."""

    max_retries: int = 3
    backoff_base: float = 0.5
    backoff_factor: float = 2.0
    max_delay: float = 10.0
    fallback_map: dict[str, str] | None = None
    escalate_on: list[str] | None = None
    abort_on: list[str] | None = None

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.backoff_base < 0:
            raise ValueError("backoff_base must be non-negative")
        if self.backoff_factor < 1.0:
            raise ValueError("backoff_factor must be >= 1.0")
        if self.max_delay < 0:
            raise ValueError("max_delay must be non-negative")

    def compute_delay(self, attempt_count: int) -> float:
        """Exponential backoff, capped by ``max_delay``.

        ``attempt_count`` is the number of attempts already performed
        (1 after the first failure). The delay for attempt ``n`` is
        ``backoff_base * backoff_factor ** (n - 1)``.
        """

        n = max(1, int(attempt_count))
        delay = self.backoff_base * (self.backoff_factor ** (n - 1))
        return min(delay, self.max_delay)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class ErrorClassifier:
    """Classify dispatcher error strings into coarse categories.

    Categories: ``timeout``, ``validation``, ``rate_limit``,
    ``unknown_tool``, ``internal_error``, ``permission_denied``.

    Unknown errors fall through to ``internal_error`` — callers should
    treat that bucket as "safely retryable at most once then abort".
    """

    CATEGORIES: frozenset[str] = frozenset(
        {
            "timeout",
            "validation",
            "rate_limit",
            "unknown_tool",
            "internal_error",
            "permission_denied",
        }
    )

    @staticmethod
    def classify(error: str | None) -> str:
        if error is None:
            return "internal_error"
        e = error.lower()

        if "unknown_tool" in e:
            return "unknown_tool"
        if "timeout" in e or "timed out" in e:
            return "timeout"
        if "rate_limit" in e or "rate limited" in e or "ratelimit" in e:
            return "rate_limit"
        if "permission" in e or "forbidden" in e or "unauthorized" in e or "denied" in e:
            return "permission_denied"
        if (
            "schema_error" in e
            or "invalid_arguments" in e
            or "validation" in e
            or "tool_argument_error" in e
            or "type_mismatch" in e
            or "missing_required" in e
        ):
            return "validation"
        return "internal_error"


# ---------------------------------------------------------------------------
# Decision function
# ---------------------------------------------------------------------------


def decide(
    result: ToolInvocationResult,
    policy: RecoveryPolicy,
    attempt_count: int,
) -> RecoveryDecision:
    """Select a recovery strategy for a failed dispatch result.

    Parameters
    ----------
    result:
        The failing :class:`ToolInvocationResult`.
    policy:
        Active :class:`RecoveryPolicy`.
    attempt_count:
        Number of attempts already performed, including the one that
        produced ``result`` (so ``1`` after the very first failure).

    Raises
    ------
    ValueError
        If ``result`` represents a successful dispatch.
    """

    if result.ok:
        raise ValueError("decide() called on a successful ToolInvocationResult")
    if attempt_count < 1:
        raise ValueError("attempt_count must be >= 1")

    err = result.error or ""
    category = ErrorClassifier.classify(err)

    # Explicit policy overrides win before anything else.
    if policy.abort_on and any(tok in err for tok in policy.abort_on):
        return RecoveryDecision(
            strategy=RecoveryStrategy.ABORT,
            reason=f"abort_on matched in error: {err!r}",
        )
    if policy.escalate_on and any(tok in err for tok in policy.escalate_on):
        return RecoveryDecision(
            strategy=RecoveryStrategy.ESCALATE,
            reason=f"escalate_on matched in error: {err!r}",
        )

    # Retry budget exhausted -> abort (unless a non-retry path applies).
    budget_exhausted = attempt_count >= max(1, policy.max_retries)

    if category == "permission_denied":
        return RecoveryDecision(
            strategy=RecoveryStrategy.ESCALATE,
            reason="permission_denied requires human intervention",
        )

    if category == "unknown_tool":
        fb = (policy.fallback_map or {}).get(result.name)
        if fb is not None:
            return RecoveryDecision(
                strategy=RecoveryStrategy.FALLBACK_TOOL,
                reason=f"unknown tool {result.name!r}; falling back to {fb!r}",
                fallback_tool=fb,
            )
        return RecoveryDecision(
            strategy=RecoveryStrategy.ABORT,
            reason=f"unknown tool {result.name!r} and no fallback configured",
        )

    if category == "validation":
        if budget_exhausted:
            return RecoveryDecision(
                strategy=RecoveryStrategy.ABORT,
                reason="validation error after max_retries exhausted",
            )
        return RecoveryDecision(
            strategy=RecoveryStrategy.RETRY_MODIFIED_ARGS,
            reason=f"validation error: {err!r}",
        )

    if category in ("timeout", "rate_limit"):
        if budget_exhausted:
            return RecoveryDecision(
                strategy=RecoveryStrategy.ABORT,
                reason=f"{category} after max_retries exhausted",
            )
        return RecoveryDecision(
            strategy=RecoveryStrategy.RETRY_BACKOFF,
            reason=f"transient {category}",
            delay_seconds=policy.compute_delay(attempt_count),
        )

    # internal_error / unknown: one cautious retry then abort. Allow
    # retries while budget is available, but only with backoff.
    if budget_exhausted:
        return RecoveryDecision(
            strategy=RecoveryStrategy.ABORT,
            reason="internal_error after max_retries exhausted",
        )
    return RecoveryDecision(
        strategy=RecoveryStrategy.RETRY_BACKOFF,
        reason=f"internal_error, cautious retry: {err!r}",
        delay_seconds=policy.compute_delay(attempt_count),
    )


# ---------------------------------------------------------------------------
# Recovering dispatcher
# ---------------------------------------------------------------------------


class RecoveringDispatcher:
    """Wraps a :class:`ToolRegistryDispatcher` with automatic recovery.

    On dispatch failure, the wrapper consults :func:`decide` and either
    retries (possibly with different args, a fallback tool, or after a
    sleep) or returns the last failing result. ``escalate`` is returned
    to the caller verbatim — the outer agent is expected to surface it
    to a human.

    The wrapper does **not** modify the inner dispatcher. All budget
    accounting, rate limiting, and audit logging live in the inner
    dispatcher exactly as before; this class only decides whether to
    call ``dispatch`` again.

    Parameters
    ----------
    inner:
        The concrete :class:`ToolRegistryDispatcher` (or any object
        exposing a ``dispatch(name, arguments)`` method returning a
        :class:`ToolInvocationResult`).
    policy:
        Active :class:`RecoveryPolicy`.
    args_mutator:
        Optional callable ``(name, args, result) -> dict`` invoked when
        the chosen strategy is ``retry_modified_args``. If ``None``,
        such strategies degrade to abort (agent can't self-repair).
    sleep:
        Injection point for the clock; defaults to :func:`time.sleep`.
    """

    def __init__(
        self,
        inner: ToolRegistryDispatcher,
        policy: RecoveryPolicy,
        args_mutator: Callable[[str, dict, ToolInvocationResult], dict] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if inner is None:
            raise ValueError("inner dispatcher is required")
        if not isinstance(policy, RecoveryPolicy):
            raise TypeError("policy must be a RecoveryPolicy")
        if not callable(sleep):
            raise TypeError("sleep must be callable")
        self._inner = inner
        self._policy = policy
        self._args_mutator = args_mutator
        self._sleep = sleep
        self._history: list[tuple[ToolInvocationResult, RecoveryDecision]] = []

    @property
    def inner(self) -> ToolRegistryDispatcher:
        return self._inner

    @property
    def policy(self) -> RecoveryPolicy:
        return self._policy

    def recovery_history(
        self,
    ) -> list[tuple[ToolInvocationResult, RecoveryDecision]]:
        """Return the (result, decision) pairs recorded since construction."""

        return list(self._history)

    def dispatch(self, name: str, arguments: dict) -> ToolInvocationResult:
        attempt = 0
        cur_name = name
        cur_args = dict(arguments) if isinstance(arguments, dict) else arguments
        last_result: ToolInvocationResult | None = None

        while True:
            attempt += 1
            last_result = self._inner.dispatch(cur_name, cur_args)
            if last_result.ok:
                return last_result

            decision = decide(last_result, self._policy, attempt)
            self._history.append((last_result, decision))

            strategy = decision.strategy
            if strategy == RecoveryStrategy.ABORT:
                return last_result
            if strategy == RecoveryStrategy.ESCALATE:
                # Return the failing result as-is; the outer loop reads
                # ``recovery_history`` to see the ESCALATE verdict.
                return last_result
            if strategy == RecoveryStrategy.RETRY_BACKOFF:
                if decision.delay_seconds > 0:
                    self._sleep(decision.delay_seconds)
                continue
            if strategy == RecoveryStrategy.RETRY_MODIFIED_ARGS:
                if decision.modified_args is not None:
                    cur_args = decision.modified_args
                elif self._args_mutator is not None:
                    try:
                        new_args = self._args_mutator(cur_name, cur_args, last_result)
                    except Exception as exc:
                        # Mutator failure -> abort with an annotated result.
                        self._history.append(
                            (
                                last_result,
                                RecoveryDecision(
                                    strategy=RecoveryStrategy.ABORT,
                                    reason=f"args_mutator raised: {exc!r}",
                                ),
                            )
                        )
                        return last_result
                    if not isinstance(new_args, dict):
                        return last_result
                    cur_args = new_args
                else:
                    # No way to mutate args -> abort.
                    self._history.append(
                        (
                            last_result,
                            RecoveryDecision(
                                strategy=RecoveryStrategy.ABORT,
                                reason="no args_mutator configured for retry_modified_args",
                            ),
                        )
                    )
                    return last_result
                continue
            if strategy == RecoveryStrategy.FALLBACK_TOOL:
                if not decision.fallback_tool:
                    return last_result
                cur_name = decision.fallback_tool
                continue

            # Defensive: decide() must produce a known strategy. If
            # somehow not, raise loudly rather than silently absorbing.
            raise RuntimeError(f"unknown recovery strategy: {strategy!r}")
