"""Composable guardrail middleware for the Aurelius serving stack.

This module is intentionally distinct from :mod:`src.serving.guardrails` —
that module provides a monolithic, policy-driven pipeline; this one exposes
a *middleware* interface suitable for wrapping a raw ``generate_fn`` with
pre-request validation, structured logging, and post-response scanning.

Design goals
------------

* Pure Python stdlib plus ``src.safety`` internals. No third-party deps.
* Deterministic — same input always produces the same decision sequence.
* Fail-closed defaults — the default pre/post handlers wire up the heuristic
  jailbreak detector, indirect prompt-injection scanner, and the composed
  :class:`~src.safety.output_safety_filter.OutputSafetyFilter`.
* Never swallow exceptions raised by the wrapped ``generate_fn``; callers
  upstream must see (and record) model failures.
* Callers may override either handler with a custom ``Callable``. Passing
  ``enable_default_pre=False`` or ``enable_default_post=False`` disables the
  corresponding phase entirely (the middleware becomes pass-through for that
  phase).

Decision semantics
------------------

A :class:`MiddlewareDecision` carries four fields:

``allowed``
    ``True`` iff the request/response is permitted (optionally with a
    modified payload). ``False`` means the middleware is blocking.
``reason``
    Short human-readable reason string.
``modified_input`` / ``modified_output``
    When non-``None``, the middleware substitutes this string for the
    original payload. A block decision may also carry a modified payload
    which, by contract, callers are *not* expected to surface.

If pre blocks, the wrapped ``generate_fn`` is **not invoked**; the refusal
string ``"I can't help with that request."`` is returned instead. If post
blocks, the refusal string replaces the model output. If either handler
returns ``allowed=True`` with a ``modified_input``/``modified_output``,
subsequent stages see the modified text.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from src.safety.jailbreak_detector import JailbreakDetector
from src.safety.output_safety_filter import OutputSafetyFilter
from src.safety.prompt_injection_scanner import PromptInjectionScanner

__all__ = [
    "MiddlewareDecision",
    "GuardrailMiddleware",
    "REFUSAL_STRING",
]

REFUSAL_STRING: str = "I can't help with that request."

_LOGGER = logging.getLogger("aurelius.serving.guardrail_middleware")


# --------------------------------------------------------------------------- #
# Decision record
# --------------------------------------------------------------------------- #


@dataclass
class MiddlewareDecision:
    """Outcome of a single middleware phase (pre or post).

    Attributes
    ----------
    allowed:
        ``True`` iff the payload is permitted. ``False`` = block.
    reason:
        Short human-readable explanation.
    modified_input:
        When non-``None`` and produced by the ``pre`` phase, replaces the
        request string seen by downstream stages / the model.
    modified_output:
        When non-``None`` and produced by the ``post`` phase, replaces the
        response string returned to the caller.
    """

    allowed: bool
    reason: str
    modified_input: str | None = None
    modified_output: str | None = None


# --------------------------------------------------------------------------- #
# Middleware
# --------------------------------------------------------------------------- #


PreHandler = Callable[[str], MiddlewareDecision]
PostHandler = Callable[[str, str], MiddlewareDecision]


class GuardrailMiddleware:
    """Composable pre/post guardrail middleware.

    Parameters
    ----------
    pre_handler:
        Optional custom callable invoked as ``pre_handler(request)``. When
        supplied, it fully overrides the default pre check (the
        ``enable_default_pre`` flag is ignored).
    post_handler:
        Optional custom callable invoked as ``post_handler(request, response)``.
        When supplied, it fully overrides the default post check.
    enable_default_pre:
        If no ``pre_handler`` is supplied and this is ``True`` (default),
        the middleware wires up the default pre pipeline
        (:class:`~src.safety.jailbreak_detector.JailbreakDetector` +
        :class:`~src.safety.prompt_injection_scanner.PromptInjectionScanner`).
        If ``False``, the pre phase becomes a pass-through.
    enable_default_post:
        Mirror of ``enable_default_pre`` for the post phase, which by
        default runs the
        :class:`~src.safety.output_safety_filter.OutputSafetyFilter`.
    """

    def __init__(
        self,
        pre_handler: PreHandler | None = None,
        post_handler: PostHandler | None = None,
        enable_default_pre: bool = True,
        enable_default_post: bool = True,
    ) -> None:
        self._custom_pre = pre_handler
        self._custom_post = post_handler
        self._enable_default_pre = bool(enable_default_pre)
        self._enable_default_post = bool(enable_default_post)

        # Lazy default safety primitives — constructed eagerly so
        # ``pre_check`` / ``post_check`` stay O(n) in input length.
        if self._custom_pre is None and self._enable_default_pre:
            self._default_jailbreak = JailbreakDetector()
            self._default_injection = PromptInjectionScanner()
        else:
            self._default_jailbreak = None
            self._default_injection = None

        if self._custom_post is None and self._enable_default_post:
            self._default_output_filter = OutputSafetyFilter()
        else:
            self._default_output_filter = None

    # ------------------------------------------------------------------ #
    # Phase entry points
    # ------------------------------------------------------------------ #

    def pre_check(self, request: str) -> MiddlewareDecision:
        """Run the pre phase.

        Returns a :class:`MiddlewareDecision`. Empty / pass-through inputs
        always resolve to ``allowed=True``.
        """
        if not isinstance(request, str):
            raise TypeError(f"request must be str, got {type(request).__name__}")
        if self._custom_pre is not None:
            decision = self._custom_pre(request)
            self._validate_decision(decision, "pre_handler")
            _LOGGER.debug(
                "pre_check(custom): allowed=%s reason=%s",
                decision.allowed,
                decision.reason,
            )
            return decision

        if not self._enable_default_pre:
            return MiddlewareDecision(
                allowed=True,
                reason="pre check disabled",
            )

        # Default: jailbreak + prompt-injection heuristics.
        if not request:
            return MiddlewareDecision(
                allowed=True,
                reason="empty request",
            )

        jb = self._default_jailbreak.score(request)  # type: ignore[union-attr]
        if jb.is_jailbreak:
            _LOGGER.info(
                "pre_check blocked: jailbreak score=%.3f signals=%s",
                jb.score,
                jb.triggered_signals,
            )
            return MiddlewareDecision(
                allowed=False,
                reason=(
                    f"jailbreak detector fired "
                    f"(score={jb.score:.3f}, signals={jb.triggered_signals})"
                ),
            )

        inj = self._default_injection.scan(  # type: ignore[union-attr]
            request,
            source="user_input",
        )
        if inj.is_injection:
            _LOGGER.info(
                "pre_check blocked: prompt injection score=%.3f signals=%s",
                inj.score,
                inj.signals,
            )
            return MiddlewareDecision(
                allowed=False,
                reason=(
                    f"prompt injection scanner fired (score={inj.score:.3f}, signals={inj.signals})"
                ),
            )

        return MiddlewareDecision(allowed=True, reason="pre checks passed")

    def post_check(self, request: str, response: str) -> MiddlewareDecision:
        """Run the post phase on a completed ``response``."""
        if not isinstance(request, str):
            raise TypeError(f"request must be str, got {type(request).__name__}")
        if not isinstance(response, str):
            raise TypeError(f"response must be str, got {type(response).__name__}")
        if self._custom_post is not None:
            decision = self._custom_post(request, response)
            self._validate_decision(decision, "post_handler")
            _LOGGER.debug(
                "post_check(custom): allowed=%s reason=%s",
                decision.allowed,
                decision.reason,
            )
            return decision

        if not self._enable_default_post:
            return MiddlewareDecision(
                allowed=True,
                reason="post check disabled",
            )

        if not response:
            return MiddlewareDecision(
                allowed=True,
                reason="empty response",
            )

        fd = self._default_output_filter.filter(response)  # type: ignore[union-attr]
        if fd.action == "block":
            _LOGGER.info("post_check blocked: %s", fd.reason)
            return MiddlewareDecision(
                allowed=False,
                reason=f"output safety filter blocked: {fd.reason}",
            )
        if fd.action == "redact":
            _LOGGER.info("post_check redacted: %s", fd.reason)
            return MiddlewareDecision(
                allowed=True,
                reason=f"output redacted: {fd.reason}",
                modified_output=fd.redacted_text,
            )
        return MiddlewareDecision(allowed=True, reason="post checks passed")

    # ------------------------------------------------------------------ #
    # Generation wrapper
    # ------------------------------------------------------------------ #

    def wrap_generate(
        self, generate_fn: Callable[[str], str]
    ) -> Callable[[str], tuple[str, list[MiddlewareDecision]]]:
        """Return a wrapped generator that enforces pre + post guardrails.

        The returned callable has signature ``(request) -> (response,
        decisions)`` where ``decisions`` is an ordered list containing, in
        order, the pre decision and (if pre allowed) the post decision.

        Exceptions raised by ``generate_fn`` are **not** swallowed — they
        propagate to the caller so upstream error handling can record them.
        """
        if not callable(generate_fn):
            raise TypeError("generate_fn must be callable")

        def _wrapped(request: str) -> tuple[str, list[MiddlewareDecision]]:
            decisions: list[MiddlewareDecision] = []
            pre = self.pre_check(request)
            decisions.append(pre)
            if not pre.allowed:
                _LOGGER.info("wrap_generate: pre blocked: %s", pre.reason)
                return REFUSAL_STRING, decisions

            effective_request = pre.modified_input if pre.modified_input is not None else request
            # ``generate_fn`` exceptions intentionally propagate.
            response = generate_fn(effective_request)
            if not isinstance(response, str):
                raise TypeError(f"generate_fn must return str, got {type(response).__name__}")

            post = self.post_check(effective_request, response)
            decisions.append(post)
            if not post.allowed:
                _LOGGER.info("wrap_generate: post blocked: %s", post.reason)
                return REFUSAL_STRING, decisions

            effective_response = (
                post.modified_output if post.modified_output is not None else response
            )
            return effective_response, decisions

        return _wrapped

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_decision(decision: object, origin: str) -> None:
        if not isinstance(decision, MiddlewareDecision):
            raise TypeError(
                f"{origin} must return MiddlewareDecision, got {type(decision).__name__}"
            )
