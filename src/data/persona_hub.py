"""PersonaHub (Chan et al. 2024) synthetic data generation via diverse personas.

Conditions instruction generation on diverse personas to increase data diversity
and reduce distribution collapse. Each example is written from the perspective
of a specific persona (e.g. "a high school math teacher", "a skeptical journalist").

Pure Python + stdlib only (no HuggingFace required).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Persona:
    """A single persona used to condition instruction generation."""

    name: str
    description: str
    domain: str
    traits: list[str]

    def to_prompt_prefix(self) -> str:
        """Return a prompt prefix for this persona."""
        return f"As {self.name}, {self.description}."


@dataclass
class PersonaConfig:
    """Configuration for the PersonaHub framework."""

    n_default_personas: int = 20
    domains: list[str] = field(
        default_factory=lambda: [
            "math",
            "science",
            "programming",
            "writing",
            "history",
            "business",
            "education",
            "art",
        ]
    )
    allow_domain_filter: bool = True


# ---------------------------------------------------------------------------
# Default persona factory
# ---------------------------------------------------------------------------


def create_default_personas() -> list[Persona]:
    """Return a list of 20 diverse Persona objects covering all domains."""
    return [
        # --- math ---
        Persona(
            name="a high school math teacher",
            description="you explain mathematical concepts with clarity and patience, using real-world examples to ground abstract ideas",  # noqa: E501
            domain="math",
            traits=["methodical", "patient", "example-driven", "precise"],
        ),
        Persona(
            name="a competitive mathematics olympiad coach",
            description="you challenge students with elegant proofs and non-routine problems, emphasising rigour and creativity",  # noqa: E501
            domain="math",
            traits=["rigorous", "creative", "challenging", "proof-oriented"],
        ),
        # --- science ---
        Persona(
            name="a curiosity-driven research scientist",
            description="you approach problems with systematic experimental thinking and a deep love of discovery",  # noqa: E501
            domain="science",
            traits=["curious", "systematic", "evidence-based", "open-minded"],
        ),
        Persona(
            name="a science communicator",
            description="you translate cutting-edge research into accessible explanations for a general audience",  # noqa: E501
            domain="science",
            traits=["accessible", "enthusiastic", "analogy-rich", "clear"],
        ),
        # --- programming ---
        Persona(
            name="a senior software engineer who loves open source",
            description="you write clean, idiomatic code and champion best practices such as testing, documentation, and code review",  # noqa: E501
            domain="programming",
            traits=["pragmatic", "quality-focused", "collaborative", "opinionated"],
        ),
        Persona(
            name="a security-conscious backend developer",
            description="you think carefully about threat models, input validation, and the principle of least privilege in every system you build",  # noqa: E501
            domain="programming",
            traits=["security-minded", "detail-oriented", "defensive", "thorough"],
        ),
        Persona(
            name="a machine learning researcher",
            description="you approach programming problems through the lens of data, statistics, and scalable computation",  # noqa: E501
            domain="programming",
            traits=["experimental", "mathematical", "scale-aware", "iterative"],
        ),
        # --- writing ---
        Persona(
            name="a seasoned literary fiction author",
            description="you craft prose with deliberate word choice, character depth, and thematic resonance",  # noqa: E501
            domain="writing",
            traits=["lyrical", "introspective", "character-driven", "precise"],
        ),
        Persona(
            name="a skeptical investigative journalist",
            description="you ask hard questions, follow the evidence, and present findings in concise, factual prose",  # noqa: E501
            domain="writing",
            traits=["skeptical", "fact-driven", "concise", "persistent"],
        ),
        # --- history ---
        Persona(
            name="a professor of world history",
            description="you situate events in their broader social, economic, and political context, drawing connections across eras",  # noqa: E501
            domain="history",
            traits=["contextual", "analytical", "comparative", "narrative"],
        ),
        Persona(
            name="a military history enthusiast",
            description="you analyse battles and campaigns through logistics, leadership, and strategic decision-making",  # noqa: E501
            domain="history",
            traits=["strategic", "detail-oriented", "tactical", "critical"],
        ),
        # --- business ---
        Persona(
            name="a startup founder with two exits",
            description="you think in terms of product-market fit, unit economics, and iterative growth loops",  # noqa: E501
            domain="business",
            traits=["results-oriented", "resourceful", "decisive", "risk-tolerant"],
        ),
        Persona(
            name="a management consultant",
            description="you structure ambiguous problems with frameworks, prioritise ruthlessly, and communicate insights to executives",  # noqa: E501
            domain="business",
            traits=["structured", "concise", "data-driven", "persuasive"],
        ),
        Persona(
            name="a personal finance advisor",
            description="you help people understand budgeting, investing, and long-term wealth building in plain language",  # noqa: E501
            domain="business",
            traits=["practical", "empathetic", "conservative", "educational"],
        ),
        # --- education ---
        Persona(
            name="a kindergarten teacher",
            description="you explain concepts through play, stories, and hands-on activities appropriate for young learners",  # noqa: E501
            domain="education",
            traits=["nurturing", "creative", "simple", "encouraging"],
        ),
        Persona(
            name="an instructional design specialist",
            description="you design learning experiences grounded in cognitive science, clear objectives, and measurable outcomes",  # noqa: E501
            domain="education",
            traits=["structured", "evidence-based", "learner-centred", "systematic"],
        ),
        # --- art ---
        Persona(
            name="a contemporary visual artist",
            description="you explore ideas through colour, form, and composition, drawing on art history and cultural critique",  # noqa: E501
            domain="art",
            traits=["expressive", "conceptual", "culturally-aware", "experimental"],
        ),
        Persona(
            name="a film critic and screenwriter",
            description="you analyse narrative structure, cinematography, and subtext, and write with a strong authorial voice",  # noqa: E501
            domain="art",
            traits=["analytical", "opinionated", "narrative-focused", "cinematic"],
        ),
        Persona(
            name="a jazz musician and composer",
            description="you think about harmony, rhythm, and improvisation, drawing parallels between music and other creative domains",  # noqa: E501
            domain="art",
            traits=["improvisational", "harmonic", "rhythmic", "cross-disciplinary"],
        ),
        # --- extra math (to ensure exactly 20) ---
        Persona(
            name="a data scientist at a tech company",
            description="you apply statistics and machine learning to messy real-world datasets, communicating uncertainty clearly",  # noqa: E501
            domain="math",
            traits=["statistical", "pragmatic", "uncertainty-aware", "communicative"],
        ),
    ]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def persona_weighted_sample(
    personas: list[Persona],
    domain_weights: dict[str, float],
    n: int,
) -> list[Persona]:
    """Sample n personas with domain-specific weights (with replacement allowed when necessary).

    Args:
        personas: Pool of Persona objects to sample from.
        domain_weights: Mapping from domain name to non-negative weight.
            Domains absent from the dict receive a default weight of 1.0.
        n: Number of personas to return.

    Returns:
        A list of n Persona objects sampled proportionally to domain weights.
    """
    if not personas:
        return []

    weights = [domain_weights.get(p.domain, 1.0) for p in personas]
    total = sum(weights)
    if total == 0:
        weights = [1.0] * len(personas)
        total = float(len(personas))
    normalized = [w / total for w in weights]

    rng = random.Random()  # use a fresh RNG so we don't perturb the module-level state
    chosen = rng.choices(personas, weights=normalized, k=n)
    return chosen


def estimate_coverage(
    sampled_personas: list[Persona],
    all_personas: list[Persona],
) -> float:
    """Return the fraction of all unique domains represented in sampled_personas.

    Returns 1.0 if all_personas is empty (vacuously true).
    """
    all_domains = {p.domain for p in all_personas}
    if not all_domains:
        return 1.0
    sampled_domains = {p.domain for p in sampled_personas}
    return len(sampled_domains & all_domains) / len(all_domains)


# ---------------------------------------------------------------------------
# Tokenisation helper (pure Python)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Simple lowercase whitespace tokeniser."""
    return text.lower().split()


# ---------------------------------------------------------------------------
# Main PersonaHub class
# ---------------------------------------------------------------------------

_TASK_INSTRUCTIONS: dict[str, str] = {
    "qa": (
        "Please generate a thoughtful question-and-answer pair on a topic relevant "
        "to your expertise, along with a detailed answer."
    ),
    "creative": (
        "Please produce a short creative piece (poem, story snippet, or metaphor) "
        "that reflects your perspective and domain knowledge."
    ),
    "reasoning": (
        "Please pose a challenging reasoning problem and walk through the solution "
        "step by step, explaining your thought process."
    ),
    "coding": (
        "Please write a self-contained code snippet (with comments) that demonstrates "
        "a concept or solves a problem relevant to your expertise."
    ),
    "analysis": (
        "Please analyse a real-world scenario or dataset from your domain, identifying "
        "key insights and implications."
    ),
}


class PersonaHub:
    """Framework for generating diverse synthetic data conditioned on personas."""

    def __init__(
        self,
        personas: list[Persona] | None = None,
        seed: int = 42,
    ) -> None:
        self._rng = random.Random(seed)
        if personas is None:
            self._personas: list[Persona] = create_default_personas()
        else:
            self._personas = list(personas)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_persona(
        self,
        n: int = 1,
        domain: str | None = None,
    ) -> list[Persona]:
        """Sample n personas without replacement within this call.

        Args:
            n: Number of personas to sample.
            domain: If provided, restrict sampling to this domain.

        Returns:
            List of n Persona objects (sampled without replacement).

        Raises:
            ValueError: If n exceeds the available pool size.
        """
        pool = (
            [p for p in self._personas if p.domain == domain]
            if domain is not None
            else list(self._personas)
        )
        if n > len(pool):
            raise ValueError(
                f"Requested {n} personas but only {len(pool)} available"
                + (f" in domain '{domain}'" if domain else "")
            )
        return self._rng.sample(pool, n)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def generate_instruction_prompt(
        self,
        persona: Persona,
        task_type: str = "qa",
    ) -> str:
        """Return a prompt template that combines the persona prefix with a task instruction.

        Args:
            persona: The persona to condition on.
            task_type: One of 'qa', 'creative', 'reasoning', 'coding', 'analysis'.

        Returns:
            A prompt string ready to be sent to an LLM.

        Raises:
            ValueError: If task_type is not recognised.
        """
        if task_type not in _TASK_INSTRUCTIONS:
            raise ValueError(
                f"Unknown task_type '{task_type}'. Choose from: {sorted(_TASK_INSTRUCTIONS)}"
            )
        prefix = persona.to_prompt_prefix()
        task_instr = _TASK_INSTRUCTIONS[task_type]
        return f"{prefix}\n\n{task_instr}"

    # ------------------------------------------------------------------
    # Dataset diversification
    # ------------------------------------------------------------------

    def diversify_dataset(
        self,
        base_instructions: list[str],
        n_personas_per: int = 3,
    ) -> list[dict]:
        """Create persona-conditioned variants of each base instruction.

        For each instruction in base_instructions, n_personas_per different
        personas are sampled (without replacement across the full set, cycling
        back if necessary) and used to create instruction variants.

        Args:
            base_instructions: Original instruction strings to diversify.
            n_personas_per: Number of persona variants per instruction.

        Returns:
            List of dicts, each containing:
                - instruction: the base instruction text
                - persona_name: persona's name
                - domain: persona's domain
                - persona_prefix: the prompt prefix string
        """
        results: list[dict] = []
        for instr in base_instructions:
            # Sample with replacement across persona pool when needed
            personas_needed = n_personas_per
            sampled: list[Persona] = []
            while personas_needed > 0:
                batch = min(personas_needed, len(self._personas))
                sampled.extend(self._rng.sample(self._personas, batch))
                personas_needed -= batch
            for p in sampled[:n_personas_per]:
                results.append(
                    {
                        "instruction": instr,
                        "persona_name": p.name,
                        "domain": p.domain,
                        "persona_prefix": p.to_prompt_prefix(),
                    }
                )
        return results

    # ------------------------------------------------------------------
    # Diversity metrics
    # ------------------------------------------------------------------

    def compute_diversity_score(self, instructions: list[str]) -> float:
        """Estimate dataset diversity as 1 minus average pairwise token overlap.

        Token overlap between two strings is the Jaccard similarity of their
        token sets.  Diversity = 1 - mean(overlap over all pairs).

        Args:
            instructions: List of instruction strings.

        Returns:
            Float in [0.0, 1.0]; higher means more diverse.
            Returns 1.0 for empty or single-element lists.
        """
        n = len(instructions)
        if n < 2:
            return 1.0

        token_sets = [set(_tokenize(s)) for s in instructions]
        total_overlap = 0.0
        n_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                a, b = token_sets[i], token_sets[j]
                union = a | b
                if not union:
                    # Both empty → identical (overlap = 1)
                    overlap = 1.0
                else:
                    overlap = len(a & b) / len(union)
                total_overlap += overlap
                n_pairs += 1

        mean_overlap = total_overlap / n_pairs
        return float(1.0 - mean_overlap)

    # ------------------------------------------------------------------
    # Distribution inspection
    # ------------------------------------------------------------------

    def get_persona_distribution(self) -> dict[str, int]:
        """Return a {domain: count} mapping for all personas in this hub."""
        dist: dict[str, int] = {}
        for p in self._personas:
            dist[p.domain] = dist.get(p.domain, 0) + 1
        return dist
