"""Harm filter facet — 9-category harm detection integrated with persona system.

Delegates classification logic to src.safety.harm_taxonomy_classifier
but exposes a facet interface so personas can configure harm filtering
individually (security personas get relaxed thresholds, general personas
get strict thresholds).
"""

from __future__ import annotations

from typing import Any

from ..unified_persona import PersonaFacet

HARM_CATEGORIES = (
    "violence_and_hate",
    "sexual_content",
    "criminal_planning",
    "guns_and_illegal_weapons",
    "regulated_substances",
    "self_harm",
    "child_abuse",
    "privacy_pii",
    "malicious_code",
)


def create_harm_filter_facet(
    categories: tuple[str, ...] | str = "all",
    action: str = "block",
    threshold: float = 0.5,
) -> PersonaFacet:
    if isinstance(categories, str):
        cats = categories
    else:
        cats = list(categories)
    return PersonaFacet(
        facet_type="harm_filter",
        config={"categories": cats, "action": action, "threshold": threshold},
    )


def get_categories(facet: PersonaFacet) -> list[str]:
    cats = facet.config.get("categories", "all")
    if cats == "all":
        return list(HARM_CATEGORIES)
    if isinstance(cats, str):
        return [cats]
    return list(cats)


def get_action(facet: PersonaFacet) -> str:
    return facet.config.get("action", "block")


def get_threshold(facet: PersonaFacet) -> float:
    cat_thresholds = facet.config.get("category_thresholds", {})
    if isinstance(cat_thresholds, dict):
        return cat_thresholds.get("_default", facet.config.get("threshold", 0.5))
    return facet.config.get("threshold", 0.5)


def classify_harm(
    facet: PersonaFacet,
    text: str,
) -> dict[str, Any]:
    categories = get_categories(facet)
    threshold = get_threshold(facet)
    action = get_action(facet)

    text_lower = text.lower()

    category_keywords: dict[str, list[str]] = {
        "violence_and_hate": ["kill", "murder", "attack", "hate", "violent", "assault"],
        "sexual_content": ["sexual", "explicit", "nude", "nsfw"],
        "criminal_planning": ["steal", "robbery", "fraud", "criminal", "illegal activity"],
        "guns_and_illegal_weapons": ["gun", "weapon", "bomb", "explosive", "firearm"],
        "regulated_substances": ["drug", "cocaine", "heroin", "fentanyl", "methamphetamine"],
        "self_harm": ["suicide", "self-harm", "self harm", "kill myself", "end my life"],
        "child_abuse": ["child abuse", "csam"],
        "privacy_pii": ["social security", "ssn", "credit card number", "passport number"],
        "malicious_code": ["keylogger", "ransomware", "reverse shell", "c2", "stealer", "backdoor"],
    }

    scores: dict[str, float] = {}
    flags: dict[str, str] = {}

    for cat in categories:
        keywords = category_keywords.get(cat, [])
        cat_threshold = threshold
        if cat == "child_abuse":
            cat_threshold = 0.0

        keyword_hits = sum(1 for kw in keywords if kw in text_lower)
        if keyword_hits > 0:
            raw_score = min(1.0, keyword_hits / len(keywords)) if keywords else 0.0
        else:
            raw_score = 0.0

        scores[cat] = raw_score
        if raw_score >= cat_threshold:
            flags[cat] = action
        else:
            flags[cat] = "pass"

    any_flagged = any(v != "pass" for v in flags.values())

    return {
        "scores": scores,
        "flags": flags,
        "any_flagged": any_flagged,
        "action": action if any_flagged else "pass",
    }