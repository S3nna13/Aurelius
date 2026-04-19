"""Heuristic harm-taxonomy classifier.

A stdlib-only, deterministic harm classifier covering the Llama-Guard
taxonomy (arXiv:2312.06674), the MLCommons AILuminate categories, and the
public OpenAI Moderation categories. It is intentionally a *heuristic*
port of the taxonomy — it produces per-category scores and flags based on
keyword catalogs and compiled regex patterns, so it can run in hot paths
(request admission, streaming filters) with no model weights loaded.

This module is the v1 scaffold: the scoring surface (``HarmClassification``
and the nine category keys in ``HARM_CATEGORIES``) is stable and intended
to be preserved when a trained classifier head is wired in during a later
cycle. The neural head will slot in behind ``HarmTaxonomyClassifier.classify``
and produce logits for the exact same nine categories; upstream consumers
(tool gating, output filtering) should need no code changes.

Design goals
------------

* Pure Python stdlib (``re``, ``unicodedata``). NO transformers, sklearn,
  spacy, or other heavy dependencies.
* Deterministic: same input maps to the same categories and scores.
* Unicode-aware: input is NFKC-normalised, case-folded, and stripped of
  control characters so fullwidth / smart-quoted variants still match.
* Stopword-tolerant: keywords are matched as whole words (word-boundary
  anchors on both sides), so unrelated surrounding punctuation, filler
  words, and quoting do not defeat detection.
* All input is treated as untrusted.

Categories (exactly nine, Llama-Guard + OpenAI Moderation + coding-LLM
specific extensions):

    violence_and_hate
    sexual_content
    criminal_planning
    guns_and_illegal_weapons
    regulated_substances
    self_harm
    child_abuse              (CRITICAL — threshold=0.0, any hit flags)
    privacy_pii
    malicious_code           (CRITICAL for an agentic coding LLM)

child_abuse policy
------------------

This category uses an extremely narrow keyword/pattern set. The threshold
is 0.0 so *any* hit flags. We explicitly accept a higher false-positive
rate here (for example, defensive phrases such as "protecting children
from CSAM is important" will be flagged) because the cost of a false
negative in this category is catastrophic. Downstream review should
distinguish generation requests from meta-discussion; this classifier is
intentionally conservative.

malicious_code policy
---------------------

Because Aurelius is an agentic coding LLM, the ``malicious_code`` category
is considered CRITICAL. Offensive-security primitives (keylogger, RAT,
ransomware, LSASS dumping, reverse shell, C2, SQLi payloads, stealer,
etc.) trigger aggressively. Defensive-phrasing detection is a known-hard
problem and is documented as such by the test suite; a phrase like
"how do I detect a keylogger?" will in general still fire on the bare
keyword. Callers that need a defensive-context allowance should run a
secondary intent classifier on top of this output rather than relying on
heuristic de-escalation here.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Taxonomy definition
# ---------------------------------------------------------------------------

# Each category is a dict of:
#   threshold : float in [0,1], inclusive lower bound for flagging.
#   keywords  : list of lowercase substrings, matched with word boundaries.
#   patterns  : list of regex source strings, compiled case-insensitively.
#
# Scoring per category:
#
#   raw = keyword_hits + 2 * pattern_hits
#   score = min(1.0, raw / saturation)
#
# where saturation is 3.0 for normal categories and 1.0 for ``child_abuse``
# (so a single hit is already saturating — matches the zero-tolerance
# threshold).
HARM_CATEGORIES: Dict[str, Dict[str, object]] = {
    "violence_and_hate": {
        "threshold": 0.5,
        "keywords": [
            "kill", "murder", "lynch", "behead", "massacre", "genocide",
            "ethnic cleansing", "torture", "assault", "stab", "strangle",
            "shoot them", "shoot him", "shoot her", "bomb them",
            "hate group", "white power", "racial slur",
        ],
        "patterns": [
            r"\bi\s+(?:want|am\s+going)\s+to\s+(?:kill|murder|hurt)\s+\w+",
            r"\bhow\s+to\s+(?:kill|murder|beat\s+up|attack)\s+\w+",
            r"\bincite\s+violence\s+against\b",
        ],
    },
    "sexual_content": {
        "threshold": 0.5,
        "keywords": [
            "explicit sex", "pornographic", "porn scene", "erotic story",
            "sexual fantasy", "nsfw story", "xxx scene",
            "hardcore porn", "sex scene", "nude photo",
        ],
        "patterns": [
            r"\bwrite\s+(?:me\s+)?(?:a\s+)?(?:an?\s+)?(?:explicit|erotic|pornographic)\s+\w+",
            r"\bdescribe\s+(?:in\s+)?(?:graphic|explicit)\s+sexual\b",
        ],
    },
    "criminal_planning": {
        "threshold": 0.5,
        "keywords": [
            "launder money", "money laundering", "tax evasion",
            "forge passport", "fake passport", "counterfeit cash",
            "counterfeit currency", "rob a bank", "bank heist",
            "kidnap", "human trafficking", "smuggling route",
            "extort", "blackmail",
        ],
        "patterns": [
            r"\bhow\s+to\s+(?:rob|steal\s+from|break\s+into|burgle)\s+\w+",
            r"\bplan\s+(?:a|the)\s+(?:robbery|heist|burglary|kidnapping)\b",
            r"\bevade\s+(?:the\s+)?(?:police|law\s+enforcement|authorities)\b",
        ],
    },
    "guns_and_illegal_weapons": {
        "threshold": 0.5,
        "keywords": [
            "ghost gun", "untraceable firearm", "3d printed gun",
            "3d-printed gun", "auto sear", "full auto conversion",
            "pipe bomb", "pressure cooker bomb", "ied",
            "improvised explosive", "nerve agent", "sarin",
            "chemical weapon", "bioweapon", "dirty bomb",
            "silencer blueprint", "suppressor blueprint",
        ],
        "patterns": [
            r"\bhow\s+to\s+(?:build|make|manufacture|3d\s*print)\s+(?:a\s+)?(?:gun|rifle|pistol|bomb|explosive|ied)\b",
            r"\bconvert\s+(?:a\s+)?(?:semi|semi-?auto)\w*\s+to\s+full\s*auto\b",
        ],
    },
    "regulated_substances": {
        "threshold": 0.5,
        "keywords": [
            "synthesize meth", "meth synthesis", "cook meth",
            "manufacture cocaine", "cocaine synthesis", "heroin synthesis",
            "fentanyl synthesis", "mdma synthesis", "lsd synthesis",
            "grow psilocybin",
        ],
        "patterns": [
            r"\bhow\s+to\s+(?:make|synthesize|cook|manufacture)\s+(?:meth(?:amphetamine)?|cocaine|heroin|fentanyl|mdma|lsd)\b",
            r"\b(?:methamphetamine|fentanyl)\s+(?:synthesis|recipe|precursors?)\b",
        ],
    },
    "self_harm": {
        "threshold": 0.4,
        "keywords": [
            "kill myself", "suicide method", "suicide methods",
            "end my life", "commit suicide", "hang myself",
            "cut myself", "self-harm", "self harm",
            "overdose to die", "painless suicide",
        ],
        "patterns": [
            r"\bhow\s+(?:do\s+i|can\s+i|to)\s+(?:kill\s+myself|commit\s+suicide|end\s+my\s+life)\b",
            r"\bbest\s+way\s+to\s+(?:kill\s+myself|die|commit\s+suicide)\b",
        ],
    },
    # CRITICAL. Threshold=0.0: any match flags. Extremely narrow set. We
    # deliberately prefer false positives to false negatives here.
    "child_abuse": {
        "threshold": 0.0,
        "keywords": [
            "csam", "child porn", "child pornography", "cp material",
            "underage porn", "underage nude", "lolicon", "shotacon",
            "pedophile how to", "minor nude",
        ],
        "patterns": [
            r"\bsexual(?:ly)?\s+(?:abuse|assault|exploit)\s+(?:a\s+)?(?:child|minor|kid)\b",
            r"\bsexual\s+(?:images?|content|material)\s+of\s+(?:a\s+)?(?:child|minor|kids?)\b",
            r"\bgroom(?:ing)?\s+(?:a\s+)?(?:child|minor|kid)\b",
        ],
    },
    "privacy_pii": {
        "threshold": 0.5,
        "keywords": [
            "social security number", "credit card number",
            "home address of", "phone number of",
            "dox", "doxx", "doxxing",
        ],
        "patterns": [
            # US SSN: NNN-NN-NNNN (rejects all-zero groups via alternation).
            r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
            # 16-digit card number with optional separators; matches common
            # Visa/MC/Discover/Amex(15)-adjacent lengths.
            r"\b(?:\d[ -]?){13,18}\d\b",
            # Explicit request for someone's PII.
            r"\bwhat\s+is\s+\w+(?:\s+\w+)?'?s?\s+(?:home\s+)?address\b",
        ],
    },
    # CRITICAL for an agentic coding LLM.
    "malicious_code": {
        "threshold": 0.4,
        "keywords": [
            "keylogger", "ransomware", "rootkit", "reverse shell",
            "bind shell", "c2 server", "c&c server", "command and control server",
            "dump lsass", "lsass dump", "mimikatz", "hashdump",
            "/etc/shadow", "pass-the-hash", "golden ticket attack",
            "sqli payload", "sql injection payload", "xss payload",
            "xxe payload", "rce payload", "stealer", "infostealer",
            "credential stealer", "password stealer",
            "remote access trojan", "rat malware",
            "ddos tool", "ddos script", "botnet builder", "botnet c2",
            "exploit kit", "malware dropper", "cryptor stub",
            "process hollowing", "dll injection payload",
            "uac bypass", "privilege escalation exploit",
            "buffer overflow exploit",
        ],
        "patterns": [
            r"\bwrite\s+(?:me\s+)?(?:a\s+)?(?:python|c\+\+|rust|go|bash|powershell)?\s*"
            r"(?:script\s+)?(?:that\s+)?(?:is\s+)?(?:a\s+)?"
            r"(?:keylogger|ransomware|rootkit|stealer|rat|trojan|worm|virus|backdoor)\b",
            r"\bexfiltrate\s+(?:credentials|passwords|cookies|tokens|keystrokes|files)\s+to\b",
            r"\bencrypt\s+(?:all\s+)?(?:the\s+)?(?:user'?s?\s+)?files?\s+(?:and\s+)?(?:demand|ransom|for\s+ransom)\b",
            r"\bbypass\s+(?:the\s+)?(?:antivirus|edr|defender|amsi)\b",
            r"\bread\s+/etc/shadow\b",
            r"\bdump\s+(?:the\s+)?(?:sam|ntds\.dit|lsass(?:\.exe)?)\b",
        ],
    },
}

# Per-category saturation denominators. child_abuse saturates at 1 to
# guarantee any hit produces a score >= threshold (threshold=0.0).
_SATURATION: Dict[str, float] = {
    "violence_and_hate": 3.0,
    "sexual_content": 3.0,
    "criminal_planning": 3.0,
    "guns_and_illegal_weapons": 3.0,
    "regulated_substances": 3.0,
    "self_harm": 2.0,
    "child_abuse": 1.0,
    "privacy_pii": 2.0,
    "malicious_code": 3.0,
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarmClassification:
    """Result of classifying a single text turn against the harm taxonomy.

    Attributes
    ----------
    text:
        The raw input text (not normalised — preserved verbatim so callers
        can log / audit the exact input).
    categories:
        List of ``(category_name, score)`` tuples, one per category in the
        module-level ``HARM_CATEGORIES`` ordering. Scores are in ``[0, 1]``.
    top_category:
        The category with the highest score, or ``None`` if all scores are
        zero.
    max_score:
        The maximum score across all categories, in ``[0, 1]``.
    flagged:
        True iff at least one category score meets or exceeds that
        category's threshold (from ``HARM_CATEGORIES[category]['threshold']``
        or the caller's override).
    """

    text: str
    categories: List[Tuple[str, float]]
    top_category: Optional[str]
    max_score: float
    flagged: bool
    triggered: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _normalise(text: str) -> str:
    """NFKC-normalise, strip control characters, and casefold."""
    if not isinstance(text, str):
        # All input is untrusted. Be permissive on type but never silently
        # coerce bytes via implicit decode — force an explicit conversion.
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_RE.sub(" ", text)
    return text.casefold()


def _compile_keyword(keyword: str) -> re.Pattern[str]:
    """Compile a keyword into a whole-word / whole-phrase regex.

    We use ``(?<!\\w)`` and ``(?!\\w)`` rather than ``\\b`` so that keywords
    containing non-word characters (``/etc/shadow``, ``c&c server``,
    ``3d printed gun``) still match at phrase boundaries. The keyword body
    is ``re.escape``'d so literal punctuation is matched verbatim.
    """
    escaped = re.escape(keyword.casefold())
    # Collapse escaped whitespace so multi-space input still matches.
    escaped = re.sub(r"\\\s+", r"\\s+", escaped)
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


class HarmTaxonomyClassifier:
    """Heuristic harm-taxonomy classifier.

    Parameters
    ----------
    thresholds:
        Optional mapping of ``category -> threshold`` that overrides the
        default per-category thresholds from ``HARM_CATEGORIES``. Unknown
        categories raise ``KeyError``.
    custom_patterns:
        Optional mapping of ``category -> list[str]`` of additional regex
        sources to add to that category (they will be compiled with
        ``re.IGNORECASE``). Unknown categories raise ``KeyError``.

    Notes
    -----
    Construction is O(sum of pattern/keyword sizes) — it pre-compiles all
    keyword and pattern regexes once. Classification is O(n) in input
    length per regex; for the default catalog that is ~60 regex passes
    over the (normalised) input.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        custom_patterns: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        # Resolve thresholds.
        self._thresholds: Dict[str, float] = {
            cat: float(spec["threshold"]) for cat, spec in HARM_CATEGORIES.items()  # type: ignore[arg-type]
        }
        if thresholds:
            for cat, val in thresholds.items():
                if cat not in self._thresholds:
                    raise KeyError(
                        f"Unknown harm category for threshold override: {cat!r}"
                    )
                if not (0.0 <= float(val) <= 1.0):
                    raise ValueError(
                        f"Threshold for {cat!r} must be in [0, 1], got {val!r}"
                    )
                self._thresholds[cat] = float(val)

        # Compile keyword and pattern catalogs per category. We keep the
        # individual compiled forms for introspection / debugging, but for
        # hot-path classification we build a *combined* alternation regex
        # per category: one for keywords (with word-boundary anchors) and
        # one for patterns. ``findall`` over the combined regex is O(n) in
        # the input length and ~20x faster than N separate searches for
        # the default catalog of ~60 expressions.
        self._kw: Dict[str, List[re.Pattern[str]]] = {}
        self._pat: Dict[str, List[re.Pattern[str]]] = {}
        self._kw_combined: Dict[str, Optional[re.Pattern[str]]] = {}
        self._pat_combined: Dict[str, Optional[re.Pattern[str]]] = {}
        _pat_sources: Dict[str, List[str]] = {}
        for cat, spec in HARM_CATEGORIES.items():
            kws = list(spec["keywords"])  # type: ignore[arg-type]
            pats = list(spec["patterns"])  # type: ignore[arg-type]
            self._kw[cat] = [_compile_keyword(k) for k in kws]
            self._pat[cat] = [re.compile(p, re.IGNORECASE) for p in pats]
            _pat_sources[cat] = list(pats)

        # Add custom patterns (additive, compiled with IGNORECASE).
        if custom_patterns:
            for cat, pats in custom_patterns.items():
                if cat not in self._pat:
                    raise KeyError(
                        f"Unknown harm category for custom_patterns: {cat!r}"
                    )
                for src in pats:
                    self._pat[cat].append(re.compile(src, re.IGNORECASE))
                    _pat_sources[cat].append(src)

        # Build combined alternation regexes.
        for cat, spec in HARM_CATEGORIES.items():
            kws = list(spec["keywords"])  # type: ignore[arg-type]
            if kws:
                parts = []
                for k in kws:
                    esc = re.escape(k.casefold())
                    esc = re.sub(r"\\\s+", r"\\s+", esc)
                    parts.append(esc)
                combined_src = rf"(?<!\w)(?:{'|'.join(parts)})(?!\w)"
                self._kw_combined[cat] = re.compile(combined_src)
            else:
                self._kw_combined[cat] = None

            pats = _pat_sources[cat]
            if pats:
                combined_src = "|".join(f"(?:{p})" for p in pats)
                self._pat_combined[cat] = re.compile(combined_src, re.IGNORECASE)
            else:
                self._pat_combined[cat] = None

    # -- public API --------------------------------------------------------

    def classify(self, text: str) -> HarmClassification:
        """Classify ``text`` against the full harm taxonomy."""
        if text is None:
            text = ""
        normalised = _normalise(text)

        scores: List[Tuple[str, float]] = []
        triggered: List[str] = []
        top_cat: Optional[str] = None
        top_score: float = 0.0

        for cat in HARM_CATEGORIES:
            if not normalised:
                score = 0.0
            else:
                # Fast gate: run the combined alternation first. If nothing
                # matches, we can skip the per-regex enumeration entirely
                # (the common case on benign input).
                kw_hits = 0
                pat_hits = 0
                kwc = self._kw_combined[cat]
                if kwc is not None and kwc.search(normalised) is not None:
                    kw_hits = sum(
                        1 for rx in self._kw[cat]
                        if rx.search(normalised) is not None
                    )
                patc = self._pat_combined[cat]
                if patc is not None and patc.search(normalised) is not None:
                    pat_hits = sum(
                        1 for rx in self._pat[cat]
                        if rx.search(normalised) is not None
                    )
                raw = float(kw_hits) + 2.0 * float(pat_hits)
                sat = _SATURATION[cat]
                score = min(1.0, raw / sat) if raw > 0 else 0.0

            scores.append((cat, score))
            thr = self._thresholds[cat]
            # For threshold=0.0 (child_abuse) we require a strictly positive
            # score to flag — score == 0 means no hits at all.
            if (thr == 0.0 and score > 0.0) or (thr > 0.0 and score >= thr):
                triggered.append(cat)
            if score > top_score:
                top_score = score
                top_cat = cat

        flagged = bool(triggered)
        return HarmClassification(
            text=text,
            categories=scores,
            top_category=top_cat,
            max_score=top_score,
            flagged=flagged,
            triggered=triggered,
        )

    def is_flagged(self, text: str) -> bool:
        """Return True iff ``classify(text).flagged`` is True."""
        return self.classify(text).flagged


__all__ = [
    "HARM_CATEGORIES",
    "HarmClassification",
    "HarmTaxonomyClassifier",
]
