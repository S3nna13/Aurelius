"""Indirect prompt-injection scanner.

This module is a companion to :mod:`src.safety.jailbreak_detector`. Where the
jailbreak detector is positioned on the *direct* user-turn boundary (a user
who is actively trying to coax the model out of its persona), this scanner is
positioned on the *indirect* channel boundary — untrusted content the agent
retrieves, pastes, or receives as a tool result. The threat model is the one
described in Greshake et al. 2023, "Not what you've signed up for: Compromising
Real-World LLM-Integrated Applications with Indirect Prompt Injection"
(arXiv:2302.12173), and catalogued across Simon Willison's injection
write-ups: an attacker plants instructions inside a document, web page, or
API response so that when an LLM-powered agent ingests the content it reads
attacker-authored directives as if they came from its operator.

Design mirrors the sibling jailbreak detector:

* Pure stdlib — only :mod:`re` and :mod:`unicodedata`.
* Deterministic and O(n) over the input.
* Never raises on malformed input; bytes and control characters are coerced
  safely.
* Returns a weighted :class:`InjectionScore` with per-signal attribution.

Signals (each returns a normalised sub-score in ``[0, 1]``):

``embedded_instruction``
    Imperatives targeting the assistant persona embedded in retrieved text:
    "ignore the above", "new directive:", "you are now", "your new task is",
    "print the system prompt".
``html_injection``
    HTML comments, ``<script>`` / ``<iframe>`` tags, and suspicious markup
    that indicate someone is trying to smuggle instructions past a renderer.
``suspicious_url``
    ``data:`` URIs, ``javascript:`` URIs, and bare callback webhook patterns
    (``.ngrok.io``, raw IP literals, webhook catchers like RequestBin).
``role_break``
    Literal chat special tokens appearing inside untrusted content
    (``<|im_start|>``, ``[INST]``, ``<|system|>`` …). Same catalog as the
    jailbreak detector but scored in this context — an attacker-planted
    ``<|system|>`` in a retrieved document is nearly always malicious.
``homoglyph``
    Mixed-script abuse (Cyrillic letters substituted for Latin lookalikes,
    e.g. Cyrillic ``о`` inside the otherwise-Latin word ``Ignоre``) and
    zero-width characters (``U+200B``, ``U+200C``, ``U+200D``, ``U+FEFF``).

The final score is a clamped weighted sum. Two signal families typically
need to fire to exceed the default 0.5 threshold, which keeps the
false-positive rate low on benign content such as a user pasting an HTML
snippet while asking for help with it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Catalogs
# --------------------------------------------------------------------------- #

# Imperative / re-targeting patterns. These are the canonical Greshake et al.
# indirect-injection forms — instructions written as if the document were the
# operator. Matched case-insensitively against the NFKC-normalised input.
_EMBEDDED_INSTRUCTION_PATTERNS: tuple[str, ...] = (
    r"\bignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|preceding)\b",
    r"\bdisregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|preceding)\b",
    r"\bforget\s+(?:everything|all|what|the\s+above|previous)\b",
    r"\bnew\s+(?:instructions?|directive|task|objective|mission)\s*[:\-]",
    r"\bupdated?\s+(?:instructions?|directive|task|objective)\s*[:\-]",
    r"\bsystem\s*(?:prompt|message|instruction)\s*[:\-]",
    r"\byou\s+are\s+now\b",
    r"\byour\s+new\s+(?:role|persona|identity|task|mission|objective|instructions?)\s+(?:is|are)\b",
    r"\bfrom\s+now\s+on\s+you\s+(?:will|must|are|should)\b",
    r"\boverride\s+(?:the\s+)?(?:system|instructions?|prompt|guidelines?|rules?)\b",
    r"\brepeat\s+(?:the\s+)?(?:system|your|previous)\s+(?:prompt|instructions?)\b",
    r"\bprint\s+(?:the\s+)?(?:system|your|previous|above)\s+(?:prompt|instructions?|message)\b",
    r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:system|your)\s+(?:prompt|instructions?|message)\b",
    r"\bstop\s+(?:following|obeying)\s+(?:the\s+)?(?:previous|prior|above|original)\b",
    r"\b(?:send|post|upload|exfiltrate|leak|reveal|disclose)\s+(?:the\s+|your\s+|any\s+|all\s+|confidential\s+)*(?:conversation|chat|history|secrets?|credentials?|api\s*keys?|training\s+data|system\s+prompts?|instructions?)\b",
    r"\bwhen\s+summari[sz]ing,?\s+(?:also|please|first|instead)\b",
    r"\bbefore\s+(?:you\s+)?(?:answer|respond|reply),?\s+(?:also|please|first)\b",
)

# HTML / Markdown injection surfaces. Comments, script/iframe tags, and on*
# event-handler attributes are the classic vectors for hiding instructions
# in a rendered document.
_HTML_INJECTION_PATTERNS: tuple[str, ...] = (
    r"<!--[\s\S]*?-->",
    r"<\s*script\b[^>]*>",
    r"<\s*/\s*script\s*>",
    r"<\s*iframe\b[^>]*>",
    r"<\s*object\b[^>]*>",
    r"<\s*embed\b[^>]*>",
    r"<\s*meta\b[^>]*http-equiv\b[^>]*>",
    r"\son[a-z]+\s*=\s*['\"]",  # onclick=, onerror=, onload=, ...
)

# Suspicious URL / callback patterns. ``data:`` and ``javascript:`` URIs are
# classic exfil/execution vectors. ngrok/RequestBin/webhook.site are the
# canonical attacker-callback hosts seen in indirect-injection proofs of
# concept.
_SUSPICIOUS_URL_PATTERNS: tuple[str, ...] = (
    r"\bdata:[a-z0-9.+-]+/[a-z0-9.+-]+[;,]",
    r"\bjavascript\s*:",
    r"\bvbscript\s*:",
    r"\bfile\s*:/{2,}",
    r"\bhttps?://[^\s/\"']*\.ngrok(?:-free)?\.(?:io|app)\b",
    r"\bhttps?://[^\s/\"']*\brequestbin\.[a-z]+\b",
    r"\bhttps?://[^\s/\"']*\bwebhook\.site\b",
    r"\bhttps?://[^\s/\"']*\bpipedream\.net\b",
    r"\bhttps?://[^\s/\"']*\binteractsh\.com\b",
    r"\bhttps?://[^\s/\"']*\bburpcollaborator\.net\b",
    # Bare-IP HTTP callbacks are rarely legitimate inside retrieved docs.
    r"\bhttps?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/[^\s\"']*)?",
)

# Role-break tokens — same catalog as the jailbreak detector but we score
# them in the *indirect* context. An attacker-planted role token inside a
# web page is essentially always adversarial.
_ROLE_BREAK_PATTERNS: tuple[str, ...] = (
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|system\|>",
    r"<\|user\|>",
    r"<\|assistant\|>",
    r"<\|endoftext\|>",
    r"<\|begin_of_text\|>",
    r"<\|start_header_id\|>",
    r"<\|end_header_id\|>",
    r"<\|eot_id\|>",
    r"\[INST\]",
    r"\[/INST\]",
    r"<<SYS>>",
    r"<</SYS>>",
)

# Pre-compile.
_EMBEDDED_RE = [re.compile(p, re.IGNORECASE) for p in _EMBEDDED_INSTRUCTION_PATTERNS]
_HTML_RE = [re.compile(p, re.IGNORECASE) for p in _HTML_INJECTION_PATTERNS]
_URL_RE = [re.compile(p, re.IGNORECASE) for p in _SUSPICIOUS_URL_PATTERNS]
_ROLE_BREAK_RE = [re.compile(p) for p in _ROLE_BREAK_PATTERNS]

# Zero-width / BOM characters used for invisible payload smuggling. See
# Boucher et al. 2021 "Trojan Source" for the canonical reference on abusing
# invisible characters to hide content from human reviewers.
_ZERO_WIDTH_CHARS = frozenset(
    {
        "\u200b",  # ZERO WIDTH SPACE
        "\u200c",  # ZERO WIDTH NON-JOINER
        "\u200d",  # ZERO WIDTH JOINER
        "\u2060",  # WORD JOINER
        "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / BOM
    }
)

# Control-character stripping regex. Keeps \n \r \t and keeps zero-width
# characters so the homoglyph sub-score can see them; everything else C0/C1
# is removed before regex matching to avoid pathological patterns.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]")

# Word-boundary regex for homoglyph scanning. We look at alphabetic runs and
# check whether the run mixes Latin and Cyrillic / Greek scripts.
_WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)

# Weights for the final weighted sum. Sum to 1.0. The two highest-signal
# families (embedded instructions and role-break tokens) each need another
# contributor to clear the default 0.5 threshold, matching the multi-signal
# policy in the sibling jailbreak detector.
_W_EMBEDDED = 0.50
_W_ROLE_BREAK = 0.20
_W_HTML = 0.12
_W_URL = 0.10
_W_HOMOGLYPH = 0.08

# Hard cap on regex-scan length. 1 MiB retrieved chunk is the realistic upper
# bound for a single tool output; anything beyond is truncated so regex work
# stays bounded under adversarial input.
_MAX_SCAN_BYTES = 1024 * 1024

# Scripts that shadow the basic Latin alphabet with visually-identical
# codepoints. If an alphabetic run contains Latin plus any of these, it is a
# homoglyph candidate.
_SHADOW_SCRIPTS = frozenset({"CYRILLIC", "GREEK"})

# Confusable fold table: the Cyrillic and Greek codepoints that are visually
# indistinguishable from specific ASCII letters. Used to build a second copy
# of the input for regex scanning so that attacker-homoglyphed keywords
# ("Ign\u043ere previous instructions") still light up the
# embedded-instruction and role-break sub-scores. This is a *hand-curated*
# slice of the Unicode confusables table — the 40-odd codepoints below
# cover the overwhelming majority of real-world injection payloads, and we
# deliberately keep the table small so scanning stays O(n) with a tight
# constant factor.
_CONFUSABLE_FOLD = {
    # Cyrillic lowercase lookalikes.
    "\u0430": "a",
    "\u0435": "e",
    "\u043e": "o",
    "\u0440": "p",
    "\u0441": "c",
    "\u0443": "y",
    "\u0445": "x",
    "\u0456": "i",
    "\u0458": "j",
    "\u04cf": "l",
    "\u0501": "d",
    "\u051b": "q",
    "\u051d": "w",
    "\u0491": "g",
    "\u04bb": "h",
    "\u0433": "r",
    # Cyrillic uppercase lookalikes.
    "\u0410": "A",
    "\u0412": "B",
    "\u0415": "E",
    "\u041a": "K",
    "\u041c": "M",
    "\u041d": "H",
    "\u041e": "O",
    "\u0420": "P",
    "\u0421": "C",
    "\u0422": "T",
    "\u0425": "X",
    "\u04c0": "I",
    # Greek lowercase lookalikes.
    "\u03bf": "o",
    "\u03b1": "a",
    "\u03c1": "p",
    "\u03bd": "v",
    "\u03c5": "u",
    "\u03b9": "i",
    # Greek uppercase lookalikes.
    "\u0391": "A",
    "\u0392": "B",
    "\u0395": "E",
    "\u0396": "Z",
    "\u0397": "H",
    "\u0399": "I",
    "\u039a": "K",
    "\u039c": "M",
    "\u039d": "N",
    "\u039f": "O",
    "\u03a1": "P",
    "\u03a4": "T",
    "\u03a5": "Y",
    "\u03a7": "X",
}


@dataclass(frozen=True)
class InjectionScore:
    """Result of a single indirect-injection scan.

    Attributes:
        score: Aggregate risk score in ``[0.0, 1.0]``.
        signals: Sorted, de-duplicated list of signal-family labels that
            contributed a non-zero amount to the score. Labels are drawn
            from ``{"embedded_instruction", "html_injection",
            "suspicious_url", "role_break", "homoglyph"}``.
        is_injection: ``True`` iff ``score`` meets or exceeds the scanner
            threshold at the time of scoring.
        details: Per-signal sub-scores (before weighting) plus the
            ``source`` string the caller passed to :meth:`PromptInjectionScanner.scan`
            and the effective ``threshold`` used for classification.
    """

    score: float
    signals: list[str]
    is_injection: bool
    details: dict[str, object] = field(default_factory=dict)


class PromptInjectionScanner:
    """Heuristic scanner for indirect prompt injection.

    The scanner is stateless after construction; :meth:`scan` is pure with
    respect to its input and the configuration captured at init time, so a
    single instance can be shared across threads and requests.

    Args:
        threshold: Score at or above which :attr:`InjectionScore.is_injection`
            is ``True``. Defaults to ``0.5``. Must lie in ``[0.0, 1.0]``.
        strict_mode: When ``True`` the effective classification threshold is
            lowered by 0.2 (floored at 0.0), making the scanner more eager
            to flag borderline content. Use on high-privilege tool channels
            (e.g. outputs feeding a shell executor) where the cost of a
            missed injection far exceeds a false positive. The raw
            :attr:`InjectionScore.score` is unchanged — only the boolean
            classification and the ``threshold`` entry in ``details`` reflect
            strict mode, which preserves downstream score calibration.
    """

    def __init__(self, threshold: float = 0.5, strict_mode: bool = False) -> None:
        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a real number")
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError("threshold must lie in [0.0, 1.0]")
        if not isinstance(strict_mode, bool):
            raise TypeError("strict_mode must be a bool")
        self.threshold = float(threshold)
        self.strict_mode = strict_mode

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def scan(self, content: object, source: str = "unknown") -> InjectionScore:
        """Scan a single piece of (possibly untrusted) content.

        Args:
            content: The text to scan. ``str`` is expected, but ``bytes`` /
                ``bytearray`` / ``memoryview`` / ``None`` are accepted and
                coerced rather than raising — this scanner sits on the
                untrusted boundary and must not become a crash vector.
            source: Free-form label describing where the content came from
                (``"tool:web_fetch"``, ``"retrieval:kb"``, …). Echoed back in
                ``details["source"]`` for observability; does not affect the
                score.

        Returns:
            :class:`InjectionScore` with the aggregate risk and per-signal
            breakdown.
        """
        effective_threshold = self._effective_threshold()

        normalised = self._normalise(content)
        zero_width_count = self._count_zero_width(normalised)
        # For regex scans we strip zero-width characters so attackers cannot
        # split keywords with an invisible codepoint (``Ign\u200bore``). The
        # raw count has already been captured for the homoglyph sub-score.
        stripped = self._strip_zero_width(normalised)

        if not stripped and zero_width_count == 0:
            return InjectionScore(
                score=0.0,
                signals=[],
                is_injection=(0.0 >= effective_threshold),
                details={
                    "embedded_instruction": 0.0,
                    "html_injection": 0.0,
                    "suspicious_url": 0.0,
                    "role_break": 0.0,
                    "homoglyph": 0.0,
                    "source": source,
                    "threshold": effective_threshold,
                    "strict_mode": self.strict_mode,
                },
            )

        folded = self._fold_confusables(stripped)
        embedded_sub = self._embedded_instruction_subscore(stripped, folded)
        html_sub = self._html_injection_subscore(stripped)
        url_sub = self._suspicious_url_subscore(stripped)
        role_sub = self._role_break_subscore(stripped, folded)
        homo_sub = self._homoglyph_subscore(stripped, zero_width_count)

        total = (
            _W_EMBEDDED * embedded_sub
            + _W_ROLE_BREAK * role_sub
            + _W_HTML * html_sub
            + _W_URL * url_sub
            + _W_HOMOGLYPH * homo_sub
        )
        if total < 0.0:
            total = 0.0
        elif total > 1.0:
            total = 1.0

        signals: list[str] = []
        if embedded_sub > 0.0:
            signals.append("embedded_instruction")
        if html_sub > 0.0:
            signals.append("html_injection")
        if url_sub > 0.0:
            signals.append("suspicious_url")
        if role_sub > 0.0:
            signals.append("role_break")
        if homo_sub > 0.0:
            signals.append("homoglyph")
        signals.sort()

        return InjectionScore(
            score=total,
            signals=signals,
            is_injection=total >= effective_threshold,
            details={
                "embedded_instruction": embedded_sub,
                "html_injection": html_sub,
                "suspicious_url": url_sub,
                "role_break": role_sub,
                "homoglyph": homo_sub,
                "source": source,
                "threshold": effective_threshold,
                "strict_mode": self.strict_mode,
                "zero_width_count": zero_width_count,
            },
        )

    def is_injection(self, content: object, source: str = "unknown") -> bool:
        """Convenience wrapper returning only the boolean classification."""
        return self.scan(content, source=source).is_injection

    # --------------------------------------------------------------------- #
    # Threshold / normalisation
    # --------------------------------------------------------------------- #

    def _effective_threshold(self) -> float:
        if self.strict_mode:
            return max(0.0, self.threshold - 0.2)
        return self.threshold

    @staticmethod
    def _normalise(content: object) -> str:
        """Coerce arbitrary input to a safe, NFKC-normalised ``str``.

        * ``None`` → empty string.
        * ``bytes`` / ``bytearray`` / ``memoryview`` → decoded UTF-8 with
          ``errors="replace"``.
        * Anything else → ``str(content)`` with a ``repr`` fallback.
        * NFKC-normalised. We deliberately do *not* fold script here — the
          homoglyph sub-score must see the original script.
        * C0/C1 control characters stripped (zero-width characters are
          preserved so they can be counted).
        * Truncated to :data:`_MAX_SCAN_BYTES` characters.
        """
        if content is None:
            return ""
        if isinstance(content, (bytes, bytearray, memoryview)):
            try:
                content = bytes(content).decode("utf-8", errors="replace")
            except Exception:
                return ""
        if not isinstance(content, str):
            try:
                content = str(content)
            except Exception:
                try:
                    content = repr(content)
                except Exception:
                    return ""
        try:
            content = unicodedata.normalize("NFKC", content)
        except Exception:  # noqa: S110
            pass
        content = _CTRL_RE.sub("", content)
        if len(content) > _MAX_SCAN_BYTES:
            content = content[:_MAX_SCAN_BYTES]
        return content

    @staticmethod
    def _count_zero_width(text: str) -> int:
        if not text:
            return 0
        # Fast path: no suspect characters.
        count = 0
        for ch in text:
            if ch in _ZERO_WIDTH_CHARS:
                count += 1
        return count

    @staticmethod
    def _fold_confusables(text: str) -> str:
        """Return a copy of ``text`` with Cyrillic/Greek confusables folded.

        Only substitutes codepoints listed in :data:`_CONFUSABLE_FOLD`. The
        output is the same length as the input, so positional signals (line
        numbers, word boundaries) are preserved.
        """
        if not text:
            return text
        # Fast path: skip the rebuild if no shadow codepoint is present.
        for ch in text:
            if ch in _CONFUSABLE_FOLD:
                break
        else:
            return text
        return "".join(_CONFUSABLE_FOLD.get(c, c) for c in text)

    @staticmethod
    def _strip_zero_width(text: str) -> str:
        if not text:
            return text
        # Only rebuild the string if there is at least one hit; avoids a
        # per-character rebuild for the common clean case.
        for ch in text:
            if ch in _ZERO_WIDTH_CHARS:
                return "".join(c for c in text if c not in _ZERO_WIDTH_CHARS)
        return text

    # --------------------------------------------------------------------- #
    # Sub-scores — each returns a float in [0.0, 1.0]
    # --------------------------------------------------------------------- #

    @staticmethod
    def _embedded_instruction_subscore(text: str, folded: str = "") -> float:
        """Fraction of distinct embedded-instruction patterns matched.

        A single strong imperative match scores 0.7 and two or more saturate
        at 1.0. Patterns are tested against both the raw input and the
        confusable-folded copy, so an attacker-homoglyphed keyword such as
        ``Ign\u043ere previous instructions`` still fires this signal.
        """
        if not text:
            return 0.0
        hits = 0
        for rx in _EMBEDDED_RE:
            if rx.search(text) is not None or (
                folded and folded != text and rx.search(folded) is not None
            ):
                hits += 1
        if hits == 0:
            return 0.0
        if hits == 1:
            return 0.9
        return 1.0

    @staticmethod
    def _html_injection_subscore(text: str) -> float:
        """HTML/script surface detection.

        A single ``<!-- -->`` comment is only a weak signal — users legitimately
        paste HTML for help all the time — so we cap a lone comment match at
        0.5. ``<script>`` / ``<iframe>`` / ``on*=`` handlers are much
        stronger; two hits across the catalog saturate.
        """
        if not text:
            return 0.0
        # Index 0 is the plain HTML-comment pattern; everything else in the
        # catalog is the "strong" surface (script/iframe/event handlers).
        comment_hit = _HTML_RE[0].search(text) is not None
        strong_hits = 0
        for rx in _HTML_RE[1:]:
            if rx.search(text) is not None:
                strong_hits += 1
        if strong_hits == 0 and not comment_hit:
            return 0.0
        if strong_hits == 0:
            return 0.5
        return min(1.0, (strong_hits / 2.0) + (0.1 if comment_hit else 0.0))

    @staticmethod
    def _suspicious_url_subscore(text: str) -> float:
        if not text:
            return 0.0
        hits = 0
        for rx in _URL_RE:
            if rx.search(text) is not None:
                hits += 1
        if hits == 0:
            return 0.0
        if hits == 1:
            return 0.7
        return 1.0

    @staticmethod
    def _role_break_subscore(text: str, folded: str = "") -> float:
        if not text:
            return 0.0
        hits = 0
        seen = set()
        for idx, rx in enumerate(_ROLE_BREAK_RE):
            if rx.search(text) is not None:
                seen.add(idx)
            elif folded and folded != text and rx.search(folded) is not None:
                seen.add(idx)
        hits = len(seen)
        if hits == 0:
            return 0.0
        if hits >= 2:
            return 1.0
        # A single planted role token in retrieved content is nearly always
        # malicious — score it high but not saturating.
        return 0.85

    @staticmethod
    def _homoglyph_subscore(text: str, zero_width_count: int) -> float:
        """Detect mixed-script abuse and zero-width-character smuggling.

        For each alphabetic word (Unicode word-character run of length >= 2)
        we determine the Unicode scripts of its codepoints. If Latin is
        mixed with any shadow script (Cyrillic / Greek) the word is flagged.

        Zero-width characters are counted separately and folded in: even a
        single invisible codepoint in retrieved content is suspicious, and
        more than a handful saturates the sub-score.
        """
        if not text and zero_width_count == 0:
            return 0.0

        mixed_words = 0
        total_words = 0
        if text:
            for match in _WORD_RE.finditer(text):
                word = match.group(0)
                total_words += 1
                scripts = set()
                for ch in word:
                    script = _script_of(ch)
                    if script is not None:
                        scripts.add(script)
                    if len(scripts) >= 3:
                        break
                if "LATIN" in scripts and scripts & _SHADOW_SCRIPTS:
                    mixed_words += 1

        # Base score from homoglyph words: even one mixed word is alarming.
        if mixed_words == 0:
            word_component = 0.0
        elif mixed_words == 1:
            word_component = 0.75
        else:
            word_component = 1.0

        # Zero-width contribution: 1 char -> 0.5, >= 3 chars -> saturate.
        if zero_width_count == 0:
            zw_component = 0.0
        elif zero_width_count == 1:
            zw_component = 0.5
        elif zero_width_count == 2:
            zw_component = 0.75
        else:
            zw_component = 1.0

        # Take the max so either vector alone can drive the signal, and
        # escalate slightly when both fire.
        combined = max(word_component, zw_component)
        if word_component > 0.0 and zw_component > 0.0:
            combined = min(1.0, combined + 0.1)
        return combined


# --------------------------------------------------------------------------- #
# Unicode script helper
# --------------------------------------------------------------------------- #

# ``unicodedata`` does not expose script directly, but the first token of
# ``unicodedata.name`` for an alphabetic codepoint is the script name
# ("LATIN SMALL LETTER A", "CYRILLIC SMALL LETTER O", …). We cache the
# resolution per codepoint to keep hot-path scanning fast.
_SCRIPT_CACHE: dict[str, str] = {}


def _script_of(ch: str) -> str | None:
    """Return the Unicode script name for a single character, or ``None``.

    ASCII letters short-circuit to ``"LATIN"``. Non-alphabetic characters
    return ``None`` — callers already restrict scanning to alphabetic runs
    via :data:`_WORD_RE`, but the guard keeps the helper safe to reuse.
    """
    if not ch:
        return None
    cached = _SCRIPT_CACHE.get(ch)
    if cached is not None:
        return cached if cached != "" else None
    # ASCII fast path.
    o = ord(ch)
    if (0x41 <= o <= 0x5A) or (0x61 <= o <= 0x7A):
        _SCRIPT_CACHE[ch] = "LATIN"
        return "LATIN"
    try:
        name = unicodedata.name(ch)
    except ValueError:
        _SCRIPT_CACHE[ch] = ""
        return None
    # Script name is the first whitespace-delimited token of the canonical
    # Unicode name. This matches the approach used by several stdlib-only
    # homoglyph scanners; it is not a replacement for ICU script data but is
    # sufficient for the Latin-vs-{Cyrillic,Greek} confusable axis that
    # dominates real-world injection payloads.
    head = name.split(" ", 1)[0]
    _SCRIPT_CACHE[ch] = head
    return head
