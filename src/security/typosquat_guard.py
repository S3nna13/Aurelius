"""Typosquat guard for dependency lockfiles.

Parses ``uv.lock`` and flags packages whose normalized name is within a small
Damerau-Levenshtein distance of a curated list of popular PyPI packages —
without being an exact match. ruff-S and bandit do not detect these supply-
chain risks, so this is an additive defensive layer.

Finding AUR-SEC-2026-0023 (typosquat), AUR-SEC-2026-0024 (bind address gate);
CWE-494 (download of code without integrity check),
CWE-1327 (binding to unrestricted IP).

stdlib-only. No runtime deps beyond the standard library.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# Curated set of popular/legit names we defend. All stored in normalized (PEP 503) form.
POPULAR_PACKAGES: Final[frozenset[str]] = frozenset(
    {
        "torch",
        "transformers",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "requests",
        "urllib3",
        "pyyaml",
        "pillow",
        "matplotlib",
        "beautifulsoup4",
        "click",
        "flask",
        "django",
        "fastapi",
        "pydantic",
        "starlette",
        "uvicorn",
        "aiohttp",
        "httpx",
        "langchain",
        "openai",
        "anthropic",
        "huggingface-hub",
        "tokenizers",
        "safetensors",
        "einops",
        "accelerate",
        "datasets",
        "peft",
        "trl",
        "bitsandbytes",
        "rich",
        "typer",
    }
)


# ---------------------------------------------------------------------------
# Public dataclasses


@dataclass(frozen=True, slots=True)
class TyposquatHit:
    """A single suspicious package near a popular name."""

    package: str
    version: str
    nearest_popular: str
    distance: int
    severity: str  # "high" (dist=1) or "medium" (dist=2)


@dataclass(frozen=True, slots=True)
class TyposquatResult:
    """Result of scanning a lockfile."""

    suspicious: list[TyposquatHit] = field(default_factory=list)
    total_checked: int = 0


# ---------------------------------------------------------------------------
# Helpers


_NORMALIZE_RE = re.compile(r"[-_.]+")


def normalize_package_name(name: str) -> str:
    """PEP 503 normalized form + strip confusable unicode (cyrillic etc.).

    We first map homoglyphs to their ASCII look-alikes (so an attacker can't
    slip ``torсh`` with cyrillic ``с`` past the check), then apply PEP 503
    lowercase/dash normalization.
    """
    # NFKC strips compatibility forms but won't convert cyrillic letters.
    folded = unicodedata.normalize("NFKC", name)
    folded = _fold_confusables(folded)
    return _NORMALIZE_RE.sub("-", folded).lower()


# Minimal homoglyph folding table (cyrillic/greek → latin) covering letters
# that appear in popular package names. Additive-only; never loses info that
# matters for distance comparisons.
_CONFUSABLES: Final[dict[str, str]] = {
    # cyrillic lowercase
    "а": "a", "в": "b", "с": "c", "е": "e", "н": "h", "к": "k",
    "м": "m", "о": "o", "р": "p", "т": "t", "у": "y", "х": "x",
    "і": "i", "ѕ": "s", "ј": "j", "ԁ": "d", "ӏ": "l", "ԛ": "q",
    # greek lowercase
    "α": "a", "ο": "o", "ρ": "p", "τ": "t", "ν": "v", "κ": "k",
    "ε": "e", "ι": "i",
}


def _fold_confusables(s: str) -> str:
    return "".join(_CONFUSABLES.get(ch, ch) for ch in s)


def damerau_levenshtein(a: str, b: str) -> int:
    """Optimal string alignment / Damerau-Levenshtein distance.

    Counts insertions, deletions, substitutions, and adjacent transpositions
    each as cost 1. Implementation uses the classic O(len(a)*len(b)) DP.
    """
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    # (la+1) x (lb+1) matrix
    d: list[list[int]] = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j

    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,         # deletion
                d[i][j - 1] + 1,         # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )
            if (
                i > 1
                and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)  # transposition

    return d[la][lb]


# ---------------------------------------------------------------------------
# Lockfile parsing


_PKG_BLOCK_RE = re.compile(r"\[\[\s*package\s*\]\]")
_NAME_RE = re.compile(r'^\s*name\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
_VERSION_RE = re.compile(r'^\s*version\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
_BAD_HEADER_RE = re.compile(r"\[\[\s*package\s*\](?!\])")  # e.g. "[[package]" missing closing bracket


def _parse_uv_lock(text: str) -> list[tuple[str, str]]:
    """Extract (name, version) tuples from a uv.lock TOML-ish file.

    We do lightweight parsing to avoid a TOML dep; uv.lock has a stable block
    layout. If the file is clearly malformed (e.g. unterminated headers) we
    raise ValueError.
    """
    if _BAD_HEADER_RE.search(text):
        raise ValueError("Malformed uv.lock: unterminated [[package]] header")

    # Split on [[package]] markers
    blocks = _PKG_BLOCK_RE.split(text)
    out: list[tuple[str, str]] = []
    for block in blocks[1:]:  # first chunk is preamble
        name_m = _NAME_RE.search(block)
        ver_m = _VERSION_RE.search(block)
        if name_m is None:
            continue  # skip weird blocks rather than blow up
        name = name_m.group(1)
        version = ver_m.group(1) if ver_m else ""
        out.append((name, version))
    return out


# ---------------------------------------------------------------------------
# Main API


def check_typosquats(
    lockfile_path: Path = Path("uv.lock"),
    allowlist: set[str] | None = None,
) -> TyposquatResult:
    """Scan a uv.lock for names suspiciously close to popular PyPI packages.

    Args:
        lockfile_path: Path to ``uv.lock`` (default: cwd).
        allowlist: Optional set of package names (raw or normalized) that
            should never be flagged — useful when an intentional near-match
            is necessary (e.g. a fork).

    Returns:
        TyposquatResult with ``suspicious`` hits and ``total_checked``.
    """
    path = Path(lockfile_path)
    if not path.exists():
        raise FileNotFoundError(f"Lockfile not found: {path}")

    text = path.read_text(encoding="utf-8")
    packages = _parse_uv_lock(text)

    allow_norm: set[str] = set()
    if allowlist:
        allow_norm = {normalize_package_name(x) for x in allowlist}

    suspicious: list[TyposquatHit] = []
    for raw_name, version in packages:
        norm = normalize_package_name(raw_name)
        # PEP 503 normalization only (no confusable folding) — used to detect
        # homoglyph attacks where the folded name collides with a popular pkg.
        pep503_only = _NORMALIZE_RE.sub("-", raw_name).lower()

        if norm in allow_norm or raw_name in (allowlist or set()):
            continue

        if norm in POPULAR_PACKAGES:
            # Only truly safe if the PEP 503 form *also* matches (i.e. no
            # homoglyph substitution occurred).
            if pep503_only == norm:
                continue
            # Homoglyph attack: name folds to a popular package but isn't
            # actually that package.
            suspicious.append(
                TyposquatHit(
                    package=raw_name,
                    version=version,
                    nearest_popular=norm,
                    distance=1,
                    severity="high",
                )
            )
            continue

        best_name: str | None = None
        best_dist = 99
        for popular in POPULAR_PACKAGES:
            # cheap length prefilter — skip clearly-distant names
            if abs(len(norm) - len(popular)) > 2:
                continue
            dist = damerau_levenshtein(norm, popular)
            if dist < best_dist:
                best_dist = dist
                best_name = popular
                if dist == 1:
                    break

        if best_name is not None and 1 <= best_dist <= 2:
            suspicious.append(
                TyposquatHit(
                    package=raw_name,
                    version=version,
                    nearest_popular=best_name,
                    distance=best_dist,
                    severity="high" if best_dist == 1 else "medium",
                )
            )

    return TyposquatResult(suspicious=suspicious, total_checked=len(packages))


# ---------------------------------------------------------------------------
# Registry (additive)

TYPOSQUAT_REGISTRY: dict[str, object] = {
    "default_popular": POPULAR_PACKAGES,
    "check": check_typosquats,
}


__all__ = [
    "POPULAR_PACKAGES",
    "TYPOSQUAT_REGISTRY",
    "TyposquatHit",
    "TyposquatResult",
    "check_typosquats",
    "damerau_levenshtein",
    "normalize_package_name",
]
