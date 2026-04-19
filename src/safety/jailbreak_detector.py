"""Heuristic jailbreak detector.

A stdlib-only, deterministic classifier for detecting jailbreak / prompt
injection attempts in a single text turn. It ports algorithmic ideas from
Llama-Guard (arXiv:2312.06674) and the Garak probe suite (keyword catalogs,
chat-template role-confusion probes, imperative-override probes, repetition
burst heuristics) into a single weighted-sum scorer.

Design goals
------------

* Pure Python stdlib. No torch / transformers / sklearn / spacy.
* Never raises on malformed input. Bytes are decoded with ``errors="replace"``,
  control characters and BOMs are stripped prior to matching.
* Deterministic. Same input maps to the exact same score and signal set.
* O(n) in the length of the input. 100 KB of text scores in well under a
  second on commodity hardware (regex pre-compiled at class construction).
* Unicode-aware. Input is NFKC-normalised so fullwidth tricks such as
  ``ＤＡＮ`` or smart-quoted ``"ignore previous"`` still match.

Signal weights
--------------

The final score is a clamped weighted sum of five normalised sub-scores. The
weights sum to 1.0 so that saturating a single strong signal cannot on its
own push the output above threshold — this mirrors the multi-signal
requirement recommended by Garak's combined-probe scoring.

======================  =======  =========================================
Signal                  Weight   Rationale
======================  =======  =========================================
keyword                 0.40     Classic DAN/unrestricted catalog hits.
injection               0.35     Imperative overrides of the system persona.
role_confusion          0.15     Literal chat special tokens in user turn.
repetition              0.07     Same phrase spammed N+ times (low-prior).
custom_keyword          0.03     User-supplied extra keywords, small weight
                                 so they cannot dominate by themselves.
======================  =======  =========================================

A single bare keyword match (e.g. just the word ``jailbreak``) yields
``0.40 * (1/3)`` ≈ 0.133 which is well below the default 0.5 threshold, so
at least two signal families must fire for a positive classification. This
matches the spec guidance ("needs 2+ signals").

Threshold semantics: ``is_jailbreak`` is True iff ``score >= threshold``
(inclusive lower bound), so exactly-threshold scores are classified as
jailbreaks.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Catalogs
# --------------------------------------------------------------------------- #

# Keyword / phrase catalog. Matched case-insensitively against the NFKC-
# normalised input. Word-boundary aware where appropriate so we do not flag
# substrings inside unrelated words ("scandal" should not trigger "dan").
_KEYWORD_PATTERNS: Tuple[str, ...] = (
    r"\bdan\b",
    r"\bdo\s+anything\s+now\b",
    r"\bignore\s+(?:all\s+)?(?:the\s+)?previous\s+(?:instructions?|prompts?|rules?)\b",
    r"\bignore\s+(?:all\s+)?above\s+(?:instructions?|prompts?|rules?)\b",
    r"\byou\s+are\s+now\b",
    r"\bdeveloper\s+mode\b",
    r"\bjailbreak(?:ing|ed|s)?\b",
    r"\bunrestricted\b",
    r"\brefuse\s+nothing\b",
    r"\bno\s+restrictions?\b",
    r"\bwithout\s+any\s+(?:restrictions?|limits?|rules?|filters?)\b",
    r"\bbypass\s+(?:your\s+)?(?:safety|filters?|guidelines?|rules?)\b",
    r"\bact\s+as\s+(?:if|though)\b",
    r"\bact\s+as\s+an?\s+[a-z]+\b",
    r"\bpretend\s+(?:to\s+be|you(?:'re|\s+are))\b",
    r"\broleplay\s+as\b",
    r"\bevil\s+(?:assistant|ai|bot)\b",
    r"\bopposite\s+day\b",
    r"\bhypothetically\s+speaking\b",
    r"\banything\s+(?:goes|is\s+allowed)\b",
)

# Role-confusion probe patterns: literal chat special tokens that a well-formed
# user turn would never contain. Matched case-sensitively (chat templates are
# specified lowercase) but after NFKC normalisation.
_ROLE_CONFUSION_PATTERNS: Tuple[str, ...] = (
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|system\|>",
    r"<\|user\|>",
    r"<\|assistant\|>",
    r"<\|endoftext\|>",
    r"<\|begin_of_text\|>",
    r"\[INST\]",
    r"\[/INST\]",
    r"<<SYS>>",
    r"<</SYS>>",
    r"(?im)^\s*###\s*system\s*:",
    r"(?im)^\s*system\s*:",
    r"(?im)^\s*assistant\s*:",
)

# Imperative / override probes. These target the *system persona* rather than
# just containing a banned word — matched in combination with a persona noun
# for precision.
_INJECTION_PATTERNS: Tuple[str, ...] = (
    r"\bdisregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\b",
    r"\boverride\s+(?:the\s+)?(?:system|instructions?|prompt|guidelines?)\b",
    r"\bignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|rules?)\b",
    r"\bnew\s+instructions?\s*:",
    r"\bupdated?\s+instructions?\s*:",
    r"\bforget\s+(?:everything|all|what)\b",
    r"\bfrom\s+now\s+on\s+you\s+(?:will|must|are)\b",
    r"\byour\s+new\s+(?:role|persona|identity|task)\s+is\b",
    r"\bstop\s+being\s+[a-z]+\s+and\b",
    r"\bthe\s+system\s+prompt\s+(?:is|was|says)\b",
    r"\brepeat\s+(?:the\s+)?(?:system|your)\s+prompt\b",
    r"\bprint\s+(?:the\s+)?(?:system|your|previous)\s+(?:prompt|instructions?)\b",
    r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:system|your)\s+(?:prompt|instructions?)\b",
    r"\bpretend\s+(?:to\s+be|you(?:'re|\s+are))\b",
    r"\byou\s+are\s+now\s+(?:going\s+to\s+)?(?:act|roleplay|pretend|behave)\b",
)

# Pre-compile. IGNORECASE + MULTILINE where it helps. The role-confusion list
# already carries per-pattern flags where needed.
_KEYWORD_RE = [re.compile(p, re.IGNORECASE) for p in _KEYWORD_PATTERNS]
_ROLE_CONFUSION_RE = [re.compile(p) for p in _ROLE_CONFUSION_PATTERNS]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# Tokeniser for repetition-burst analysis. Splits on whitespace after
# collapsing runs; we then build n-gram windows (default n=3) and count.
_WS_RE = re.compile(r"\s+", re.UNICODE)

# Control-character stripping regex. Keeps \n \r \t, strips all other C0/C1
# controls and the Unicode BOM.
_CTRL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f\ufeff]"
)

# Weights for the final weighted sum. Must sum to 1.0.
_W_KEYWORD = 0.40
_W_INJECTION = 0.35
_W_ROLE = 0.15
_W_REPETITION = 0.07
_W_CUSTOM = 0.03

# Repetition-burst parameters. A 3-gram repeated >= REP_MIN times saturates
# the repetition sub-score at 1.0; fewer repetitions scale linearly.
_REP_NGRAM = 3
_REP_MIN = 8       # at this count, sub-score = 1.0
_REP_START = 3     # below this count, sub-score = 0.0

# Hard cap on the input length we consider for regex scanning. Anything beyond
# this is truncated (after normalisation) so adversarial inputs cannot make
# the regex engine run unbounded work. 512 KiB is generous for a chat turn.
_MAX_SCAN_BYTES = 512 * 1024


@dataclass(frozen=True)
class JailbreakScore:
    """Result of a single jailbreak classification.

    Attributes:
        score: Aggregate risk score in ``[0.0, 1.0]``. Higher = more likely
            to be a jailbreak / prompt-injection attempt.
        triggered_signals: Sorted, de-duplicated list of signal-family labels
            (``"keyword"``, ``"role_confusion"``, ``"injection"``,
            ``"repetition"``, ``"custom_keyword"``) that contributed a
            non-zero amount to the score.
        is_jailbreak: ``True`` iff ``score`` meets or exceeds the detector
            threshold at the time of scoring.
        details: Per-signal sub-scores (before weighting) for observability.
            Keys are the same as the signal-family labels above.
    """

    score: float
    triggered_signals: List[str]
    is_jailbreak: bool
    details: dict = field(default_factory=dict)


class JailbreakDetector:
    """Heuristic jailbreak / prompt-injection classifier.

    The detector is stateless after construction — :meth:`score` is pure
    with respect to its input and the configuration captured at init time,
    which makes it safe to share a single instance across threads and
    requests.

    Args:
        threshold: Score at or above which :attr:`JailbreakScore.is_jailbreak`
            is ``True``. Defaults to ``0.5``. Must be in ``[0.0, 1.0]``.
        custom_keywords: Optional iterable of extra keyword strings to match
            case-insensitively against the normalised input. These feed the
            low-weight ``custom_keyword`` signal so callers can extend the
            catalog without destabilising the weighting of the built-in
            signals.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        custom_keywords: Optional[List[str]] = None,
    ) -> None:
        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a real number")
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError("threshold must lie in [0.0, 1.0]")
        self.threshold = float(threshold)

        # Compile user-supplied keywords defensively. Non-string / empty
        # entries are skipped rather than raising, so a noisy config does
        # not take the safety filter offline.
        compiled_custom: List[re.Pattern] = []
        raw_custom: List[str] = []
        if custom_keywords is not None:
            for kw in custom_keywords:
                if not isinstance(kw, str):
                    continue
                kw_norm = unicodedata.normalize("NFKC", kw).strip()
                if not kw_norm:
                    continue
                # Treat the keyword as a literal phrase; match case-insensitively
                # and require word-ish boundaries when the keyword is alpha-
                # numeric at its edges.
                pat = re.escape(kw_norm)
                if kw_norm[:1].isalnum():
                    pat = r"(?:\b|_)" + pat
                if kw_norm[-1:].isalnum():
                    pat = pat + r"(?:\b|_)"
                compiled_custom.append(re.compile(pat, re.IGNORECASE))
                raw_custom.append(kw_norm)
        self._custom_re: Tuple[re.Pattern, ...] = tuple(compiled_custom)
        self._custom_raw: Tuple[str, ...] = tuple(raw_custom)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def score(self, text: object) -> JailbreakScore:
        """Score a single text turn.

        Args:
            text: User-controlled input. ``str`` is the expected type, but
                ``bytes``/``bytearray``/``memoryview`` and ``None`` are also
                accepted and coerced rather than raising — this classifier
                is positioned on the untrusted boundary and must not become
                a crash vector itself.

        Returns:
            :class:`JailbreakScore` describing the aggregate risk and the
            set of contributing signals.
        """
        normalised = self._normalise(text)
        if not normalised:
            return JailbreakScore(
                score=0.0,
                triggered_signals=[],
                is_jailbreak=(0.0 >= self.threshold and self.threshold == 0.0),
                details={
                    "keyword": 0.0,
                    "role_confusion": 0.0,
                    "injection": 0.0,
                    "repetition": 0.0,
                    "custom_keyword": 0.0,
                },
            )

        kw_sub = self._keyword_subscore(normalised)
        role_sub = self._role_confusion_subscore(normalised)
        inj_sub = self._injection_subscore(normalised)
        rep_sub = self._repetition_subscore(normalised)
        custom_sub = self._custom_keyword_subscore(normalised)

        total = (
            _W_KEYWORD * kw_sub
            + _W_ROLE * role_sub
            + _W_INJECTION * inj_sub
            + _W_REPETITION * rep_sub
            + _W_CUSTOM * custom_sub
        )
        # Clamp defensively; individual sub-scores are already in [0,1], but
        # float round-off can drift a hair past 1.0.
        if total < 0.0:
            total = 0.0
        elif total > 1.0:
            total = 1.0

        signals: List[str] = []
        if kw_sub > 0.0:
            signals.append("keyword")
        if role_sub > 0.0:
            signals.append("role_confusion")
        if inj_sub > 0.0:
            signals.append("injection")
        if rep_sub > 0.0:
            signals.append("repetition")
        if custom_sub > 0.0:
            signals.append("custom_keyword")
        signals.sort()

        return JailbreakScore(
            score=total,
            triggered_signals=signals,
            is_jailbreak=total >= self.threshold,
            details={
                "keyword": kw_sub,
                "role_confusion": role_sub,
                "injection": inj_sub,
                "repetition": rep_sub,
                "custom_keyword": custom_sub,
            },
        )

    def is_jailbreak(self, text: object) -> bool:
        """Convenience wrapper returning only the boolean classification."""
        return self.score(text).is_jailbreak

    # --------------------------------------------------------------------- #
    # Normalisation
    # --------------------------------------------------------------------- #

    @staticmethod
    def _normalise(text: object) -> str:
        """Coerce arbitrary input to a safe, normalised ``str``.

        * ``None`` → empty string.
        * ``bytes`` / ``bytearray`` / ``memoryview`` → decoded as UTF-8 with
          ``errors="replace"``.
        * Anything else → ``str(text)``; ``repr`` fallback on exotic objects
          whose ``__str__`` raises.
        * NFKC-normalised (folds fullwidth → ASCII, compatibility forms).
        * Control characters and BOMs are stripped.
        * Truncated to :data:`_MAX_SCAN_BYTES` characters as a last line of
          defence against regex DoS on adversarial inputs.
        """
        if text is None:
            return ""
        if isinstance(text, (bytes, bytearray, memoryview)):
            try:
                text = bytes(text).decode("utf-8", errors="replace")
            except Exception:
                return ""
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                try:
                    text = repr(text)
                except Exception:
                    return ""
        try:
            text = unicodedata.normalize("NFKC", text)
        except Exception:
            # Extremely malformed unicode; fall back to what we have.
            pass
        text = _CTRL_RE.sub("", text)
        if len(text) > _MAX_SCAN_BYTES:
            text = text[:_MAX_SCAN_BYTES]
        return text

    # --------------------------------------------------------------------- #
    # Sub-scores — each returns a float in [0.0, 1.0]
    # --------------------------------------------------------------------- #

    @staticmethod
    def _keyword_subscore(text: str) -> float:
        """Fraction of distinct catalog patterns that matched.

        Dividing by the catalog size keeps one bare keyword well below the
        default threshold while still rewarding multi-keyword inputs.
        """
        hits = 0
        for rx in _KEYWORD_RE:
            if rx.search(text) is not None:
                hits += 1
        if hits == 0:
            return 0.0
        # Scale so that 3+ distinct keyword families saturate.
        return min(1.0, hits / 3.0)

    @staticmethod
    def _role_confusion_subscore(text: str) -> float:
        """Any literal chat special token in the user turn is a strong signal."""
        hits = 0
        for rx in _ROLE_CONFUSION_RE:
            if rx.search(text) is not None:
                hits += 1
                if hits >= 2:
                    return 1.0
        if hits == 0:
            return 0.0
        # A single role-confusion token is almost always adversarial.
        return 0.8

    @staticmethod
    def _injection_subscore(text: str) -> float:
        """Imperative overrides targeting the system persona."""
        hits = 0
        for rx in _INJECTION_RE:
            if rx.search(text) is not None:
                hits += 1
        if hits == 0:
            return 0.0
        return min(1.0, hits / 2.0)

    @staticmethod
    def _repetition_subscore(text: str) -> float:
        """Detects high-count repetition of a short phrase.

        Builds 3-gram windows over whitespace-delimited tokens and looks at
        the most common n-gram. Scales linearly from ``_REP_START`` (0.0) to
        ``_REP_MIN`` (1.0).
        """
        if not text:
            return 0.0
        tokens = _WS_RE.split(text.strip().lower())
        tokens = [t for t in tokens if t]
        if len(tokens) < _REP_NGRAM:
            # Degenerate case: single-token repetition (e.g. "HACK " * 100).
            if len(tokens) >= _REP_START:
                top = Counter(tokens).most_common(1)[0][1]
                return JailbreakDetector._rep_scale(top)
            return 0.0
        ngrams = [
            tuple(tokens[i : i + _REP_NGRAM])
            for i in range(len(tokens) - _REP_NGRAM + 1)
        ]
        if not ngrams:
            return 0.0
        top = Counter(ngrams).most_common(1)[0][1]
        # Also consider single-token floods ("HACK HACK HACK ...").
        top1 = Counter(tokens).most_common(1)[0][1]
        return max(
            JailbreakDetector._rep_scale(top),
            JailbreakDetector._rep_scale(top1),
        )

    @staticmethod
    def _rep_scale(count: int) -> float:
        if count < _REP_START:
            return 0.0
        if count >= _REP_MIN:
            return 1.0
        span = _REP_MIN - _REP_START
        return (count - _REP_START) / span

    def _custom_keyword_subscore(self, text: str) -> float:
        """Fraction of user-supplied custom keywords that matched."""
        if not self._custom_re:
            return 0.0
        hits = 0
        for rx in self._custom_re:
            if rx.search(text) is not None:
                hits += 1
        if hits == 0:
            return 0.0
        return min(1.0, hits / max(1, len(self._custom_re)))
