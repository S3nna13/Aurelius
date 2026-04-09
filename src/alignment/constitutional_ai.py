"""Constitutional AI: iterative self-critique and revision for harmlessness (Anthropic, 2022)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ConstitutionalPrinciple:
    """A single constitutional principle with paired critique and revision prompts."""

    critique_request: str
    revision_request: str
    name: str = ""


# ---------------------------------------------------------------------------
# Default harmlessness principles
# ---------------------------------------------------------------------------

HARMLESSNESS_PRINCIPLES: list[ConstitutionalPrinciple] = [
    ConstitutionalPrinciple(
        name="harmful_content",
        critique_request=(
            "Identify any harmful, dangerous, or potentially injurious content in the response."
        ),
        revision_request=(
            "Rewrite the response to remove any harmful or dangerous content, ensuring it is safe and helpful."
        ),
    ),
    ConstitutionalPrinciple(
        name="discrimination",
        critique_request=(
            "Identify any discriminatory, biased, or offensive language targeting individuals or groups in the response."
        ),
        revision_request=(
            "Rewrite the response to eliminate discriminatory or biased language, treating all people fairly and respectfully."
        ),
    ),
    ConstitutionalPrinciple(
        name="deception",
        critique_request=(
            "Identify any misleading, false, or deceptive claims made in the response."
        ),
        revision_request=(
            "Rewrite the response to be accurate and honest, correcting any misleading claims or acknowledging uncertainty where appropriate."
        ),
    ),
    ConstitutionalPrinciple(
        name="privacy",
        critique_request=(
            "Identify any content in the response that could violate personal privacy or expose sensitive information."
        ),
        revision_request=(
            "Rewrite the response to protect individual privacy and avoid disclosing or encouraging disclosure of sensitive personal information."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------

def format_critique_prompt(response: str, principle: ConstitutionalPrinciple) -> str:
    """Format a prompt asking the model to critique a response."""
    return f"Here is a response: {response}\n\n{principle.critique_request}\nCritique:"


def format_revision_prompt(
    response: str, critique: str, principle: ConstitutionalPrinciple
) -> str:
    """Format a prompt asking the model to revise a response based on a critique."""
    return (
        f"Here is a response: {response}\n\n"
        f"Critique: {critique}\n\n"
        f"{principle.revision_request}\nRevision:"
    )


# ---------------------------------------------------------------------------
# CAIStep dataclass
# ---------------------------------------------------------------------------

@dataclass
class CAIStep:
    """Record of a single critique-revision step."""

    principle: ConstitutionalPrinciple
    original: str
    critique: str
    revised: str
    step_num: int = 0


# ---------------------------------------------------------------------------
# ConstitutionalAILoop
# ---------------------------------------------------------------------------

class ConstitutionalAILoop:
    """Iterative self-critique and revision loop implementing the CAI pipeline.

    Args:
        generate_fn: A callable that maps a prompt string to a generated text string.
        principles: List of :class:`ConstitutionalPrinciple` to apply. Defaults to
            :data:`HARMLESSNESS_PRINCIPLES`.
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        principles: list[ConstitutionalPrinciple] | None = None,
    ) -> None:
        self.generate_fn = generate_fn
        self.principles = principles if principles is not None else HARMLESSNESS_PRINCIPLES

    def critique(self, response: str, principle: ConstitutionalPrinciple) -> str:
        """Generate a critique of *response* with respect to *principle*."""
        prompt = format_critique_prompt(response, principle)
        return self.generate_fn(prompt)

    def revise(self, response: str, critique: str, principle: ConstitutionalPrinciple) -> str:
        """Generate a revised response given a *critique* and *principle*."""
        prompt = format_revision_prompt(response, critique, principle)
        return self.generate_fn(prompt)

    def run_step(
        self,
        response: str,
        principle: ConstitutionalPrinciple,
        step_num: int = 0,
    ) -> CAIStep:
        """Perform one critique-revision step and return a :class:`CAIStep`."""
        critique_text = self.critique(response, principle)
        revised_text = self.revise(response, critique_text, principle)
        return CAIStep(
            principle=principle,
            original=response,
            critique=critique_text,
            revised=revised_text,
            step_num=step_num,
        )

    def run(self, initial_response: str, n_revisions: int = 2) -> list[CAIStep]:
        """Cycle through principles for *n_revisions* total steps.

        Each step takes the previous step's revised output as the new response.

        Returns:
            List of :class:`CAIStep` objects; the final revised response is in
            ``steps[-1].revised``.
        """
        steps: list[CAIStep] = []
        current_response = initial_response

        for i in range(n_revisions):
            principle = self.principles[i % len(self.principles)]
            step = self.run_step(current_response, principle, step_num=i)
            steps.append(step)
            current_response = step.revised

        return steps

    def final_response(self, steps: list[CAIStep]) -> str:
        """Return the last revised response, or empty string if *steps* is empty."""
        return steps[-1].revised if steps else ""


# ---------------------------------------------------------------------------
# SyntheticCAIDataGenerator
# ---------------------------------------------------------------------------

class SyntheticCAIDataGenerator:
    """Generate (harmful_prompt, revised_response) pairs for SFT training.

    Args:
        cai_loop: A :class:`ConstitutionalAILoop` instance to use for revision.
    """

    def __init__(self, cai_loop: ConstitutionalAILoop) -> None:
        self.cai_loop = cai_loop

    def generate_pair(self, harmful_prompt: str, initial_response: str) -> dict:
        """Run the CAI loop and return a training pair dict.

        Returns:
            ``{"prompt": str, "original": str, "revised": str, "n_steps": int}``
        """
        steps = self.cai_loop.run(initial_response)
        revised = self.cai_loop.final_response(steps)
        return {
            "prompt": harmful_prompt,
            "original": initial_response,
            "revised": revised,
            "n_steps": len(steps),
        }

    def generate_dataset(self, prompts_and_responses: list[tuple[str, str]]) -> list[dict]:
        """Map :meth:`generate_pair` over all (prompt, response) tuples."""
        return [self.generate_pair(prompt, response) for prompt, response in prompts_and_responses]


# ---------------------------------------------------------------------------
# Reward heuristic
# ---------------------------------------------------------------------------

def cai_reward_score(original: str, revised: str, principle: ConstitutionalPrinciple) -> float:
    """Simple heuristic reward score based on keyword change between original and revised.

    Returns:
        A float in [0, 1]. Returns 0.0 if the strings are identical, 1.0 if
        completely different (no shared words), and an intermediate value otherwise.
    """
    if original == revised:
        return 0.0

    original_words = set(original.lower().split())
    revised_words = set(revised.lower().split())

    if not original_words:
        return 1.0

    # Fraction of original words removed in the revision
    removed = original_words - revised_words
    score = len(removed) / len(original_words)
    return float(min(max(score, 0.0), 1.0))
