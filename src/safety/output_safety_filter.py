"""Unified pre-emit output safety filter for Aurelius.

Composes the four existing heuristic safety classifiers into a single
streaming-aware filter applied before model output is emitted:

* :class:`~src.safety.jailbreak_detector.JailbreakDetector` — catches model
  outputs that reproduce / comply with a jailbreak template.
* :class:`~src.safety.prompt_injection_scanner.PromptInjectionScanner` —
  catches embedded instructions / role-break markers that should never appear
  in a well-formed assistant turn.
* :class:`~src.safety.harm_taxonomy_classifier.HarmTaxonomyClassifier` —
  Llama-Guard-style taxonomy classifier.
* :class:`~src.safety.pii_detector.PIIDetector` — regex + checksum PII
  detector with redactor.

The filter is deterministic, pure-stdlib (relies only on internal ``src.safety``
modules), and never raises on ill-typed input — it sits on the untrusted
boundary between the model and the user and must not become a crash vector.

Policy semantics
----------------

Decisions are produced by a strict priority: **block > redact > allow**. If
any detector requests a block action the filter blocks; otherwise if any
detector requests a redaction the filter returns redacted text; otherwise it
allows the text through unchanged.

Strict mode (``policy.mode == "strict"``) lowers the effective decision
thresholds of all four detectors by ``0.2`` (clamped to ``[0.0, 1.0]``). The
raw detector scores returned in ``signal_breakdown`` are unchanged — only
the ``action`` decision is affected — so downstream calibration and logging
remain meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.safety.harm_taxonomy_classifier import (
    HARM_CATEGORIES,
    HarmTaxonomyClassifier,
)
from src.safety.jailbreak_detector import JailbreakDetector
from src.safety.pii_detector import PIIDetector
from src.safety.prompt_injection_scanner import PromptInjectionScanner

# --------------------------------------------------------------------------- #
# Action / mode vocabularies
# --------------------------------------------------------------------------- #

_ACTIONS: frozenset[str] = frozenset({"allow", "redact", "block"})
_MODES: frozenset[str] = frozenset({"strict", "permissive"})
_STRICT_DELTA: float = 0.2


def _coerce_text(text: object) -> str:
    """Accept ``str``/``bytes``/``bytearray``/``memoryview``/``None``.

    Bytes are decoded as UTF-8 with ``errors="replace"``. ``None`` becomes
    the empty string. Anything else is routed through ``str()``.
    """
    if text is None:
        return ""
    if isinstance(text, (bytes, bytearray, memoryview)):
        try:
            return bytes(text).decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - belt-and-braces
            return ""
    if isinstance(text, str):
        return text
    try:
        return str(text)
    except Exception:  # pragma: no cover
        return ""


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FilterDecision:
    """Decision produced by :meth:`OutputSafetyFilter.filter`.

    Attributes
    ----------
    action:
        One of ``"allow"``, ``"redact"``, ``"block"``.
    reason:
        Human-readable explanation of why the decision was reached.
    redacted_text:
        The text that should be emitted in place of the original when
        ``action == "redact"``. ``None`` otherwise.
    signal_breakdown:
        Mapping containing one sub-entry per detector
        (``"jailbreak"``, ``"prompt_injection"``, ``"harm_taxonomy"``,
        ``"pii"``) plus the effective ``mode`` string. Useful for
        observability / audit trails.
    """

    action: str
    reason: str
    redacted_text: Optional[str]
    signal_breakdown: dict = field(default_factory=dict)


@dataclass
class OutputSafetyPolicy:
    """Policy controlling the :class:`OutputSafetyFilter`.

    Parameters
    ----------
    mode:
        ``"strict"`` or ``"permissive"``. In strict mode all detector
        thresholds are lowered by :data:`_STRICT_DELTA`.
    harm_threshold:
        Classification threshold applied uniformly to every category in the
        harm taxonomy. ``[0.0, 1.0]``.
    pii_action:
        Action to take when PII is detected. Must be one of ``"allow"``,
        ``"redact"`` or ``"block"``.
    jailbreak_action:
        Action to take when the jailbreak detector fires. Must be in
        ``{"allow", "redact", "block"}`` — ``"redact"`` is accepted but is
        equivalent to ``"block"`` for the jailbreak signal since there is
        no safe redaction (the whole message is suspect).
    injection_action:
        Action for the prompt injection scanner. Same vocabulary.
    max_buffer_chars:
        Upper bound on the streaming buffer size. :meth:`filter_chunk` will
        force-flush (i.e. run the full filter and reset) once the buffer
        grows beyond this many characters, so adversarial inputs cannot
        pin unbounded memory.
    """

    mode: str = "strict"
    harm_threshold: float = 0.5
    pii_action: str = "redact"
    jailbreak_action: str = "block"
    injection_action: str = "block"
    max_buffer_chars: int = 256

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(
                f"unknown mode {self.mode!r}; expected one of {sorted(_MODES)}"
            )
        if not isinstance(self.harm_threshold, (int, float)):
            raise TypeError("harm_threshold must be a real number")
        if not 0.0 <= float(self.harm_threshold) <= 1.0:
            raise ValueError("harm_threshold must lie in [0.0, 1.0]")
        self.harm_threshold = float(self.harm_threshold)
        for name in ("pii_action", "jailbreak_action", "injection_action"):
            val = getattr(self, name)
            if val not in _ACTIONS:
                raise ValueError(
                    f"unknown {name}={val!r}; expected one of {sorted(_ACTIONS)}"
                )
        if not isinstance(self.max_buffer_chars, int):
            raise TypeError("max_buffer_chars must be int")
        if self.max_buffer_chars <= 0:
            raise ValueError("max_buffer_chars must be positive")


# --------------------------------------------------------------------------- #
# Filter
# --------------------------------------------------------------------------- #


_ACTION_PRIORITY: dict[str, int] = {"allow": 0, "redact": 1, "block": 2}


class OutputSafetyFilter:
    """Composed pre-emit output safety filter.

    Parameters
    ----------
    policy:
        Optional :class:`OutputSafetyPolicy`. Defaults to
        ``OutputSafetyPolicy()`` (strict mode, redact-on-PII, block-on-
        jailbreak / injection).

    The filter is stateless after construction — one instance may be shared
    across threads and requests. :meth:`filter_chunk` is the only entry point
    that carries streaming state, and that state lives entirely in the caller-
    supplied ``buffer`` argument.
    """

    def __init__(self, policy: Optional[OutputSafetyPolicy] = None) -> None:
        if policy is None:
            policy = OutputSafetyPolicy()
        elif not isinstance(policy, OutputSafetyPolicy):
            raise TypeError(
                "policy must be an OutputSafetyPolicy or None, got "
                f"{type(policy).__name__}"
            )
        self.policy = policy

        # Effective thresholds under strict mode.
        if policy.mode == "strict":
            jb_thr = max(0.0, 0.5 - _STRICT_DELTA)
            inj_thr = max(0.0, 0.5 - _STRICT_DELTA)
            harm_thr = max(0.0, policy.harm_threshold - _STRICT_DELTA)
        else:
            jb_thr = 0.5
            inj_thr = 0.5
            harm_thr = policy.harm_threshold

        self._jailbreak = JailbreakDetector(threshold=jb_thr)
        self._injection = PromptInjectionScanner(
            threshold=inj_thr, strict_mode=False
        )
        # Apply harm threshold uniformly to every category.
        harm_overrides = {cat: harm_thr for cat in HARM_CATEGORIES}
        self._harm = HarmTaxonomyClassifier(thresholds=harm_overrides)
        # PII detector: use "placeholder" redaction so the emitted text still
        # carries a marker of what was removed.
        self._pii = PIIDetector(redaction_mode="placeholder")

        self._effective_thresholds = {
            "jailbreak": jb_thr,
            "injection": inj_thr,
            "harm": harm_thr,
        }

    # --------------------------------------------------------------------- #
    # Single-shot filter
    # --------------------------------------------------------------------- #

    def filter(self, text: object) -> FilterDecision:
        """Apply the full filter pipeline to a single string.

        Empty input short-circuits to ``allow``.
        """
        raw = _coerce_text(text)
        if not raw:
            return FilterDecision(
                action="allow",
                reason="empty input",
                redacted_text=None,
                signal_breakdown={
                    "jailbreak": {"score": 0.0, "triggered": False, "signals": []},
                    "prompt_injection": {
                        "score": 0.0, "triggered": False, "signals": []
                    },
                    "harm_taxonomy": {
                        "max_score": 0.0,
                        "flagged": False,
                        "triggered": [],
                        "top_category": None,
                    },
                    "pii": {"matches": [], "count": 0},
                    "mode": self.policy.mode,
                },
            )

        # --- Run all four detectors ---------------------------------------
        jb_result = self._jailbreak.score(raw)
        inj_result = self._injection.scan(raw, source="model_output")
        harm_result = self._harm.classify(raw)
        pii_matches = self._pii.detect(raw)

        # --- Collect candidate actions ------------------------------------
        candidates: list[Tuple[str, str]] = []  # (action, reason)

        if jb_result.is_jailbreak:
            action = self.policy.jailbreak_action
            if action != "allow":
                candidates.append(
                    (
                        action,
                        f"jailbreak detector fired (score={jb_result.score:.3f}, "
                        f"signals={jb_result.triggered_signals})",
                    )
                )

        if inj_result.is_injection:
            action = self.policy.injection_action
            if action != "allow":
                candidates.append(
                    (
                        action,
                        f"prompt injection scanner fired "
                        f"(score={inj_result.score:.3f}, "
                        f"signals={inj_result.signals})",
                    )
                )

        if harm_result.flagged:
            # Harm taxonomy always blocks — redaction of harmful content is
            # ill-defined. The policy has no per-harm action override because
            # frontier-tier models must not emit flagged harmful content.
            candidates.append(
                (
                    "block",
                    f"harm taxonomy fired "
                    f"(top={harm_result.top_category}, "
                    f"max_score={harm_result.max_score:.3f}, "
                    f"triggered={harm_result.triggered})",
                )
            )

        redacted_by_pii: Optional[str] = None
        if pii_matches:
            action = self.policy.pii_action
            if action == "redact":
                redacted_by_pii = self._pii.redact(raw).redacted_text
                candidates.append(
                    (
                        "redact",
                        f"PII redaction applied ({len(pii_matches)} match(es): "
                        f"{sorted({m.type for m in pii_matches})})",
                    )
                )
            elif action == "block":
                candidates.append(
                    (
                        "block",
                        f"PII detected and policy pii_action='block' "
                        f"({len(pii_matches)} match(es))",
                    )
                )
            # action == "allow" -> PII signal is ignored for decision.

        # --- Pick highest-priority action ---------------------------------
        signal_breakdown = self._build_breakdown(
            jb_result, inj_result, harm_result, pii_matches
        )

        if not candidates:
            return FilterDecision(
                action="allow",
                reason="no safety signals triggered",
                redacted_text=None,
                signal_breakdown=signal_breakdown,
            )

        # Block > Redact > Allow — pick the candidate with max priority; on
        # ties, keep the first (deterministic insertion order above).
        best = max(candidates, key=lambda c: _ACTION_PRIORITY[c[0]])
        if best[0] == "block":
            return FilterDecision(
                action="block",
                reason=best[1],
                redacted_text=None,
                signal_breakdown=signal_breakdown,
            )
        if best[0] == "redact":
            # Redacted text must come from the PII detector; jailbreak /
            # injection / harm never reach here as "redact" in our policy
            # defaults but if a caller configures jailbreak_action="redact"
            # we fall back to masking the entire message.
            text_out = redacted_by_pii if redacted_by_pii is not None else "<REDACTED>"
            return FilterDecision(
                action="redact",
                reason=best[1],
                redacted_text=text_out,
                signal_breakdown=signal_breakdown,
            )
        # Shouldn't happen — "allow" candidates are never appended — but keep
        # a defensive path rather than a silent fallthrough.
        return FilterDecision(
            action="allow",
            reason="no safety signals triggered",
            redacted_text=None,
            signal_breakdown=signal_breakdown,
        )

    # --------------------------------------------------------------------- #
    # Streaming filter
    # --------------------------------------------------------------------- #

    def filter_chunk(
        self, chunk: object, buffer: str = ""
    ) -> Tuple[FilterDecision, str]:
        """Buffered streaming-mode filter.

        The caller accumulates incoming chunks into ``buffer`` and invokes
        this method once per chunk. The filter runs on
        ``buffer + chunk`` and returns a ``(decision, new_buffer)`` tuple:

        * If ``action == "block"``: ``new_buffer`` is empty and the caller
          should stop streaming.
        * If ``action == "redact"``: ``new_buffer`` is empty — the redacted
          text has already been emitted into ``decision.redacted_text``.
        * If ``action == "allow"`` and the buffer is below
          ``policy.max_buffer_chars``: ``new_buffer`` carries the
          accumulated text forward so subsequent chunks can still be
          inspected as a whole. Once the cap is reached the buffer is
          force-flushed (``new_buffer == ""``) so memory remains bounded.
        """
        if not isinstance(buffer, str):
            raise TypeError(f"buffer must be str, got {type(buffer).__name__}")
        chunk_str = _coerce_text(chunk)
        combined = buffer + chunk_str

        decision = self.filter(combined)

        if decision.action == "block":
            return decision, ""
        if decision.action == "redact":
            return decision, ""
        # allow
        if len(combined) >= self.policy.max_buffer_chars:
            # Force-flush; do not keep growing.
            return decision, ""
        return decision, combined

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #

    def _build_breakdown(
        self, jb_result, inj_result, harm_result, pii_matches
    ) -> dict:
        return {
            "jailbreak": {
                "score": float(jb_result.score),
                "triggered": bool(jb_result.is_jailbreak),
                "signals": list(jb_result.triggered_signals),
            },
            "prompt_injection": {
                "score": float(inj_result.score),
                "triggered": bool(inj_result.is_injection),
                "signals": list(inj_result.signals),
            },
            "harm_taxonomy": {
                "max_score": float(harm_result.max_score),
                "flagged": bool(harm_result.flagged),
                "triggered": list(harm_result.triggered),
                "top_category": harm_result.top_category,
            },
            "pii": {
                "matches": [
                    {"type": m.type, "span": list(m.span),
                     "confidence": float(m.confidence)}
                    for m in pii_matches
                ],
                "count": len(pii_matches),
            },
            "mode": self.policy.mode,
            "effective_thresholds": dict(self._effective_thresholds),
        }


__all__ = [
    "FilterDecision",
    "OutputSafetyPolicy",
    "OutputSafetyFilter",
]
