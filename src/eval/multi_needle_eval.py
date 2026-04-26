"""Multi-needle long-context evaluation harness.

Multi-needle variant of Needle-in-a-Haystack (NIAH). Places ``K`` independent
``KEY=VALUE`` needles at controlled fractional depths within a token-level
filler haystack and scores a retriever's ability to recover all ``K``.

This module is a **pure data builder + scorer**: it does not call any model.
Callers run their own retriever / model against ``sample.prompt`` and pass the
raw text output to :func:`score`. There are **zero foreign imports** — only
the Python standard library is used (``random``, ``dataclasses``, ``re``).

Prompt layout
-------------
The prompt is ``haystack + "\\n\\n" + question``. The haystack is a
whitespace-joined sequence of filler tokens drawn from
:attr:`MultiNeedleConfig.token_vocab`; each needle is inserted as a full
sentence (built from :attr:`MultiNeedleConfig.needle_template`) at a target
token index determined by the ``depth_profile``. Insertion happens between
filler tokens so no filler token is mangled.

Depth profiles
--------------
* ``"uniform"``         — fractions ``(i + 0.5) / K`` for ``i = 0..K-1``.
* ``"clustered_early"`` — all needles land in the first 15%% of the haystack.
* ``"clustered_late"``  — all needles land in the last 15%% of the haystack.
* ``"boundary"``        — half in the first 5%%, half in the last 5%%.

Extensibility
-------------
Custom depth generators may be registered via :func:`register_depth_profile`.
The generator signature is ``(num_needles: int, rng: random.Random) -> tuple[float, ...]``
and must return exactly ``num_needles`` values in ``[0.0, 1.0]``.

Determinism
-----------
All randomness flows through a single ``random.Random(seed)`` instance
constructed from :attr:`MultiNeedleConfig.seed`. Two calls with the same
config produce byte-identical samples.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class MultiNeedleError(Exception):
    """Raised for any validation failure in the multi-needle harness."""


# ---------------------------------------------------------------------------
# Config & result dataclasses
# ---------------------------------------------------------------------------
_DEFAULT_TOKEN_VOCAB: tuple[str, ...] = (
    "the",
    "a",
    "of",
    "to",
    "and",
    "in",
    "that",
    "it",
    "is",
    "as",
    "for",
    "on",
    "with",
    "by",
    "an",
    "but",
    "from",
    "this",
    "be",
    "at",
)


@dataclass(frozen=True)
class MultiNeedleConfig:
    """Configuration for a single multi-needle sample.

    Attributes
    ----------
    num_needles : int
        Number of needles ``K``. Must be in ``[2, 16]``.
    haystack_tokens : int
        Target haystack length in whitespace tokens. Must be in
        ``[64, 65536]``.
    depth_profile : str
        One of ``"uniform"``, ``"clustered_early"``, ``"clustered_late"``,
        ``"boundary"``, or any custom profile registered via
        :func:`register_depth_profile`.
    seed : int
        PRNG seed; all randomness is deterministic in this seed.
    token_vocab : tuple[str, ...]
        Filler token pool. Must be non-empty.
    needle_template : str
        ``str.format``-style template accepting ``{key}`` and ``{value}``.
    key_alphabet : str
        Characters used to sample 2-letter keys. Must be non-empty and
        contain only A–Z characters (upper-case ASCII letters).
    value_digits : int
        Digits per random value string. Must be in ``[2, 12]``.
    """

    num_needles: int
    haystack_tokens: int
    depth_profile: str
    seed: int = 0
    token_vocab: tuple[str, ...] = _DEFAULT_TOKEN_VOCAB
    needle_template: str = "remember the code {key}={value}."
    key_alphabet: str = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    value_digits: int = 6


@dataclass(frozen=True)
class MultiNeedleSample:
    """A fully-materialised multi-needle evaluation sample."""

    needles: tuple[tuple[str, str], ...]
    prompt: str
    question: str
    gold: tuple[tuple[str, str], ...]
    depth_fracs: tuple[float, ...]


@dataclass(frozen=True)
class MultiNeedleVerdict:
    """Score report produced by :func:`score`."""

    sample: MultiNeedleSample
    recovered: tuple[tuple[str, str], ...]
    recall_exact: float
    recall_key: float
    precision: float
    all_or_nothing: bool


# ---------------------------------------------------------------------------
# Depth profile registry
# ---------------------------------------------------------------------------
# A depth generator is: (num_needles, rng) -> tuple[float, ...] of length K.
DepthGenerator = Callable[[int, random.Random], tuple[float, ...]]

DEPTH_PROFILE_REGISTRY: dict[str, DepthGenerator] = {}


def _uniform_profile(k: int, rng: random.Random) -> tuple[float, ...]:
    del rng  # uniform is deterministic in k alone
    return tuple((i + 0.5) / k for i in range(k))


def _clustered_early_profile(k: int, rng: random.Random) -> tuple[float, ...]:
    del rng
    # Evenly spaced inside [0.0, 0.15). All values strictly < 0.15 < 0.2.
    if k == 1:
        return (0.075,)
    step = 0.15 / k
    return tuple((i + 0.5) * step for i in range(k))


def _clustered_late_profile(k: int, rng: random.Random) -> tuple[float, ...]:
    del rng
    # Evenly spaced inside (0.85, 1.0). All values strictly > 0.85 > 0.8.
    if k == 1:
        return (0.925,)
    step = 0.15 / k
    return tuple(0.85 + (i + 0.5) * step for i in range(k))


def _boundary_profile(k: int, rng: random.Random) -> tuple[float, ...]:
    del rng
    # First half strictly in [0.0, 0.05); second half strictly in (0.95, 1.0).
    # Ensures halves satisfy <0.1 and >0.9 respectively.
    n_low = k // 2
    n_high = k - n_low  # when K odd, the extra goes to the high side
    low_step = 0.05 / max(n_low, 1)
    high_step = 0.05 / max(n_high, 1)
    lows = tuple((i + 0.5) * low_step for i in range(n_low))
    highs = tuple(0.95 + (i + 0.5) * high_step for i in range(n_high))
    return lows + highs


DEPTH_PROFILE_REGISTRY["uniform"] = _uniform_profile
DEPTH_PROFILE_REGISTRY["clustered_early"] = _clustered_early_profile
DEPTH_PROFILE_REGISTRY["clustered_late"] = _clustered_late_profile
DEPTH_PROFILE_REGISTRY["boundary"] = _boundary_profile


def register_depth_profile(name: str, generator: DepthGenerator) -> None:
    """Register a custom depth profile generator.

    Parameters
    ----------
    name : str
        Non-empty profile name. Overwrites any existing entry with this name.
    generator : Callable[[int, random.Random], tuple[float, ...]]
        Must return exactly ``num_needles`` fractional depths in
        ``[0.0, 1.0]``. Called once per ``build_sample``.
    """
    if not isinstance(name, str) or not name:
        raise MultiNeedleError("depth profile name must be a non-empty string")
    if not callable(generator):
        raise MultiNeedleError("depth profile generator must be callable")
    DEPTH_PROFILE_REGISTRY[name] = generator


# ---------------------------------------------------------------------------
# Reference benchmark registry
# ---------------------------------------------------------------------------
MULTI_NEEDLE_REGISTRY: dict[str, MultiNeedleConfig] = {
    "niah-mk-small": MultiNeedleConfig(
        num_needles=4,
        haystack_tokens=512,
        depth_profile="uniform",
        seed=0,
    ),
    "niah-mk-medium": MultiNeedleConfig(
        num_needles=8,
        haystack_tokens=4096,
        depth_profile="uniform",
        seed=0,
    ),
    "niah-mk-boundary": MultiNeedleConfig(
        num_needles=8,
        haystack_tokens=4096,
        depth_profile="boundary",
        seed=0,
    ),
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_config(cfg: MultiNeedleConfig) -> None:
    if not isinstance(cfg, MultiNeedleConfig):
        raise MultiNeedleError(f"expected MultiNeedleConfig, got {type(cfg).__name__}")
    if not isinstance(cfg.num_needles, int) or isinstance(cfg.num_needles, bool):
        raise MultiNeedleError(f"num_needles must be int, got {type(cfg.num_needles).__name__}")
    if cfg.num_needles < 2 or cfg.num_needles > 16:
        raise MultiNeedleError(f"num_needles must be in [2, 16], got {cfg.num_needles}")
    if not isinstance(cfg.haystack_tokens, int) or isinstance(cfg.haystack_tokens, bool):
        raise MultiNeedleError(
            f"haystack_tokens must be int, got {type(cfg.haystack_tokens).__name__}"
        )
    if cfg.haystack_tokens < 64 or cfg.haystack_tokens > 65536:
        raise MultiNeedleError(f"haystack_tokens must be in [64, 65536], got {cfg.haystack_tokens}")
    if cfg.depth_profile not in DEPTH_PROFILE_REGISTRY:
        allowed = sorted(DEPTH_PROFILE_REGISTRY.keys())
        raise MultiNeedleError(f"depth_profile must be one of {allowed}, got {cfg.depth_profile!r}")
    if not isinstance(cfg.value_digits, int) or isinstance(cfg.value_digits, bool):
        raise MultiNeedleError(f"value_digits must be int, got {type(cfg.value_digits).__name__}")
    if cfg.value_digits < 2 or cfg.value_digits > 12:
        raise MultiNeedleError(f"value_digits must be in [2, 12], got {cfg.value_digits}")
    if not isinstance(cfg.token_vocab, tuple) or not cfg.token_vocab:
        raise MultiNeedleError("token_vocab must be a non-empty tuple of strings")
    for tok in cfg.token_vocab:
        if not isinstance(tok, str) or not tok:
            raise MultiNeedleError("token_vocab entries must be non-empty strings")
    if not isinstance(cfg.key_alphabet, str) or not cfg.key_alphabet:
        raise MultiNeedleError("key_alphabet must be a non-empty string")
    for ch in cfg.key_alphabet:
        if not ("A" <= ch <= "Z"):
            raise MultiNeedleError(f"key_alphabet must contain only upper-case A-Z, got {ch!r}")
    # Sanity: need enough 2-letter keys in the alphabet to produce K unique ones.
    if len(cfg.key_alphabet) * len(cfg.key_alphabet) < cfg.num_needles:
        raise MultiNeedleError("key_alphabet too small to produce enough unique 2-letter keys")
    if not isinstance(cfg.needle_template, str) or not cfg.needle_template:
        raise MultiNeedleError("needle_template must be a non-empty string")
    if "{key}" not in cfg.needle_template or "{value}" not in cfg.needle_template:
        raise MultiNeedleError("needle_template must contain both {key} and {value} placeholders")


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------
def _gen_unique_keys(k: int, alphabet: str, rng: random.Random) -> tuple[str, ...]:
    # 2-letter keys drawn without replacement.
    seen: dict[str, None] = {}
    attempts = 0
    max_attempts = k * 64 + 1024
    while len(seen) < k and attempts < max_attempts:
        c1 = rng.choice(alphabet)
        c2 = rng.choice(alphabet)
        key = c1 + c2
        if key not in seen:
            seen[key] = None
        attempts += 1
    if len(seen) < k:
        # Deterministic fallback: enumerate pairs in alphabet order and pick
        # the remaining ones, preserving insertion order of `seen`.
        for c1 in alphabet:
            for c2 in alphabet:
                if len(seen) >= k:
                    break
                key = c1 + c2
                if key not in seen:
                    seen[key] = None
            if len(seen) >= k:
                break
    return tuple(seen.keys())


def _gen_value(digits: int, rng: random.Random) -> str:
    return "".join(str(rng.randrange(10)) for _ in range(digits))


QUESTION_TEXT: str = (
    "List every code value you were told. Output one pair per line in the "
    "exact form KEY=VALUE, alphabetical by KEY, no extra commentary."
)


def build_sample(cfg: MultiNeedleConfig) -> MultiNeedleSample:
    """Construct a deterministic multi-needle sample from ``cfg``."""
    _validate_config(cfg)

    rng = random.Random(cfg.seed)

    # 1) Generate K unique keys and K values. Done *first* so the RNG state
    # after this step is independent of the haystack length (useful for
    # testing).
    keys = _gen_unique_keys(cfg.num_needles, cfg.key_alphabet, rng)
    values = tuple(_gen_value(cfg.value_digits, rng) for _ in range(cfg.num_needles))
    needles: tuple[tuple[str, str], ...] = tuple(zip(keys, values))

    # 2) Generate filler tokens.
    vocab = cfg.token_vocab
    filler_tokens = [vocab[rng.randrange(len(vocab))] for _ in range(cfg.haystack_tokens)]

    # 3) Compute per-needle fractional depths via the profile generator.
    gen = DEPTH_PROFILE_REGISTRY[cfg.depth_profile]
    depth_fracs = gen(cfg.num_needles, rng)
    if not isinstance(depth_fracs, tuple) or len(depth_fracs) != cfg.num_needles:
        raise MultiNeedleError(
            f"depth profile {cfg.depth_profile!r} returned "
            f"{type(depth_fracs).__name__} of wrong length"
        )
    for d in depth_fracs:
        if not isinstance(d, (int, float)) or isinstance(d, bool):
            raise MultiNeedleError(
                f"depth profile {cfg.depth_profile!r} returned non-numeric depth"
            )
        if d < 0.0 or d > 1.0:
            raise MultiNeedleError(
                f"depth profile {cfg.depth_profile!r} returned out-of-range depth {d}"
            )

    # 4) Materialise each needle sentence.
    needle_texts = tuple(cfg.needle_template.format(key=k, value=v) for (k, v) in needles)

    # 5) Compute target insertion indices (between filler tokens).
    n = len(filler_tokens)
    raw_indices = [int(round(d * n)) for d in depth_fracs]
    # Pair each needle with its depth and original index, sort by position so
    # we can splice stably; then remember the original order for .needles /
    # .gold (which preserves insertion order == construction order).
    order = list(range(cfg.num_needles))
    # Sort by position ascending. Stable sort preserves relative order for
    # ties, which keeps the result deterministic.
    order.sort(key=lambda i: raw_indices[i])

    # Splice needles into the token list. Because we insert in ascending
    # position order, each insertion shifts subsequent positions by +1; we
    # compensate with an `offset` counter.
    tokens: list = list(filler_tokens)
    offset = 0
    for i in order:
        pos = raw_indices[i] + offset
        if pos < 0:
            pos = 0
        if pos > len(tokens):
            pos = len(tokens)
        tokens.insert(pos, needle_texts[i])
        offset += 1

    haystack = " ".join(tokens)
    question = QUESTION_TEXT
    prompt = haystack + "\n\n" + question

    return MultiNeedleSample(
        needles=needles,
        prompt=prompt,
        question=question,
        gold=needles,
        depth_fracs=tuple(depth_fracs),
    )


# ---------------------------------------------------------------------------
# Recovery parsing
# ---------------------------------------------------------------------------
# A line is considered a candidate recovery if, after stripping leading
# list-bullets / quotes / whitespace and trailing punctuation / whitespace,
# it matches ``KEY=VALUE`` where:
#   * KEY is 1+ upper-case ASCII letters
#   * VALUE is 1+ non-whitespace characters that does not contain '=' or
#     stripped punctuation (we strip trailing .,;:!?)"'` from VALUE)
#   * there is exactly one '=' sign in the stripped token
_BULLET_PREFIX = re.compile(r"^[\s\-\*\u2022>\+]+")
_QUOTE_WRAP = re.compile(r'^[\'"`]+|[\'"`]+$')


def _strip_line(line: str) -> str:
    s = line.strip()
    s = _BULLET_PREFIX.sub("", s)
    s = s.strip()
    # Strip common quote wraps once.
    s = _QUOTE_WRAP.sub("", s)
    s = s.strip()
    return s


def parse_recovery(model_output: str) -> tuple[tuple[str, str], ...]:
    """Parse ``KEY=VALUE`` lines from a model's raw text output.

    Tolerates leading bullet characters (``-``, ``*``, ``•``, ``>``, ``+``),
    surrounding whitespace, and surrounding quote/backtick wrappers. Trailing
    punctuation (``.``, ``,``, ``;``, ``:``, ``!``, ``?``, and the
    quote/backtick characters) is stripped from ``VALUE`` only.

    Rejected lines:
        * any line whose stripped form does not match ``KEY=VALUE`` with
          exactly one ``=``
        * empty ``KEY`` or empty ``VALUE``
        * ``KEY`` containing non-upper-case or non-letter characters
        * ``VALUE`` containing internal whitespace

    Duplicates: the first ``KEY`` wins; subsequent lines with the same key
    are dropped.
    """
    if not isinstance(model_output, str):
        raise MultiNeedleError(f"model_output must be str, got {type(model_output).__name__}")
    if not model_output:
        return tuple()
    out: list = []
    seen_keys: dict[str, None] = {}
    for raw in model_output.splitlines():
        token = _strip_line(raw)
        if not token:
            continue
        # Exactly one '=' required.
        if token.count("=") != 1:
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        # Strip trailing punctuation and quote chars from VALUE (but not KEY).
        value = value.rstrip(".,;:!?\"'`")
        value = value.strip()
        if not key or not value:
            continue
        # KEY must be entirely upper-case ASCII letters.
        if not key.isalpha() or not key.isupper() or not key.isascii():
            continue
        # VALUE must not contain whitespace (needles never do).
        if any(ch.isspace() for ch in value):
            continue
        # VALUE must not contain '=' (defensive; count("=")==1 above already
        # enforces this on the original, but stripping could theoretically
        # re-expose one — keep the check for clarity).
        if "=" in value:
            continue
        if key in seen_keys:
            continue
        seen_keys[key] = None
        out.append((key, value))
    return tuple(out)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(sample: MultiNeedleSample, model_output: str) -> MultiNeedleVerdict:
    """Score ``model_output`` against ``sample.gold``.

    Returns a :class:`MultiNeedleVerdict` with:

    * ``recall_exact``   — fraction of gold pairs where both key and value match
    * ``recall_key``     — fraction of gold keys present in recovery (any value)
    * ``precision``      — fraction of recovered pairs present in gold; if
      recovery is empty, precision is 1.0 (no false positives).
    * ``all_or_nothing`` — ``True`` iff ``recall_exact == 1.0``.
    """
    if not isinstance(sample, MultiNeedleSample):
        raise MultiNeedleError(f"sample must be MultiNeedleSample, got {type(sample).__name__}")
    recovered = parse_recovery(model_output)

    gold_pairs = {(k, v) for (k, v) in sample.gold}
    gold_keys = {k for (k, _) in sample.gold}
    recovered_keys = {k for (k, _) in recovered}
    n_gold = len(sample.gold)

    if n_gold == 0:
        recall_exact = 0.0
        recall_key = 0.0
    else:
        exact_hits = sum(1 for pair in recovered if pair in gold_pairs and pair[0] in gold_keys)
        # De-dup is already enforced by parse_recovery, but clamp to n_gold
        # for safety.
        recall_exact = min(exact_hits, n_gold) / n_gold
        key_hits = len(recovered_keys & gold_keys)
        recall_key = key_hits / n_gold

    if not recovered:
        precision = 1.0
    else:
        true_pos = sum(1 for pair in recovered if pair in gold_pairs)
        precision = true_pos / len(recovered)

    all_or_nothing = recall_exact == 1.0

    return MultiNeedleVerdict(
        sample=sample,
        recovered=recovered,
        recall_exact=recall_exact,
        recall_key=recall_key,
        precision=precision,
        all_or_nothing=all_or_nothing,
    )


__all__ = [
    "MultiNeedleError",
    "MultiNeedleConfig",
    "MultiNeedleSample",
    "MultiNeedleVerdict",
    "build_sample",
    "parse_recovery",
    "score",
    "register_depth_profile",
    "DEPTH_PROFILE_REGISTRY",
    "MULTI_NEEDLE_REGISTRY",
    "QUESTION_TEXT",
]

# Keep `field` referenced for forward-compat with frozen dataclass mutation
# helpers; not used directly today.
_ = field
