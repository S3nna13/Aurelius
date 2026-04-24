"""Domain-specific prompt templates for Magpie synthetic instruction generation.

Each template is a pre-query header that primes an aligned LLM to produce
domain-appropriate instruction-response pairs when used as the sole input.
Adapted from Heavens_Gate domain_templates.py (Magpie ICLR 2025).

Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainTemplate:
    """A Magpie pre-query template scoped to a single domain."""

    domain: str
    prefix: str
    description: str = ""
    tags: list[str] = field(default_factory=list)


_GENERAL_PREFIX = (
    "A conversation between a curious user and a helpful, harmless, and honest AI "
    "assistant. The assistant answers general-knowledge questions across everyday "
    "topics. It favours clarity, balance, and calibrated uncertainty. "
    "The user asks an open-ended question."
)

_CODING_PREFIX = (
    "A conversation between a professional software engineer and an AI pair-programmer "
    "specialising in Python, JavaScript, SQL, and modern systems design. The assistant "
    "writes clean, idiomatic, well-tested code and explains trade-offs concisely. "
    "It prefers standard-library solutions when reasonable. "
    "The engineer asks a concrete implementation or debugging question."
)

_REASONING_PREFIX = (
    "A conversation between a rigorous thinker and an AI assistant trained to solve "
    "problems step by step. The assistant decomposes problems, states assumptions, "
    "performs explicit arithmetic or logical chains, and verifies the final answer. "
    "It shows its reasoning before committing to a conclusion. "
    "The user poses a math, logic, or reasoning puzzle."
)

_SECURITY_PREFIX = (
    "A conversation between a security researcher and an AI assistant specialising in "
    "defensive cybersecurity, threat modelling, vulnerability analysis, and MITRE "
    "ATT&CK mapping. The assistant produces analysis-grade responses with CVE and "
    "technique references where applicable, and refuses disallowed offensive misuse. "
    "The researcher asks a technical security question."
)

_DATA_SCIENCE_PREFIX = (
    "A conversation between a data scientist and an AI assistant specialising in "
    "machine learning, statistics, feature engineering, and exploratory data analysis. "
    "The assistant reasons about distributions, leakage, and evaluation protocols, and "
    "recommends concrete libraries (numpy, pandas, scikit-learn, pytorch) when useful. "
    "The data scientist asks an analysis or modelling question."
)

_WRITING_PREFIX = (
    "A conversation between a writer and an AI writing assistant skilled in creative "
    "fiction, technical documentation, and professional correspondence. The assistant "
    "adapts tone, voice, and structure to the requested medium, and gives concrete, "
    "publishable drafts rather than vague suggestions. "
    "The writer asks for help with a piece of writing."
)


DOMAIN_TEMPLATES: dict[str, DomainTemplate] = {
    "general": DomainTemplate(
        domain="general",
        prefix=_GENERAL_PREFIX,
        description="General-purpose helpful assistant prompts.",
        tags=["general", "chat"],
    ),
    "coding": DomainTemplate(
        domain="coding",
        prefix=_CODING_PREFIX,
        description="Software engineering tasks in Python, JavaScript, SQL.",
        tags=["code", "python", "javascript", "sql"],
    ),
    "reasoning": DomainTemplate(
        domain="reasoning",
        prefix=_REASONING_PREFIX,
        description="Step-by-step math and logical reasoning.",
        tags=["math", "logic", "cot"],
    ),
    "security": DomainTemplate(
        domain="security",
        prefix=_SECURITY_PREFIX,
        description="Defensive cybersecurity analysis and threat modelling.",
        tags=["security", "attack", "cve"],
    ),
    "data_science": DomainTemplate(
        domain="data_science",
        prefix=_DATA_SCIENCE_PREFIX,
        description="ML and data-analysis tasks.",
        tags=["ml", "stats", "data"],
    ),
    "writing": DomainTemplate(
        domain="writing",
        prefix=_WRITING_PREFIX,
        description="Creative and professional writing assistance.",
        tags=["writing", "creative", "docs"],
    ),
}


def get_template(domain: str) -> DomainTemplate:
    """Return the template for *domain*, or the ``general`` default if missing."""
    key = (domain or "").lower().strip()
    return DOMAIN_TEMPLATES.get(key, DOMAIN_TEMPLATES["general"])


def list_domains() -> list[str]:
    """Return a sorted list of registered domain keys."""
    return sorted(DOMAIN_TEMPLATES.keys())


DOMAIN_TEMPLATES_REGISTRY = {"default": DOMAIN_TEMPLATES}
