"""Constitutional AI v2: multi-principle critique-revision with convergence detection."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import torch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConstitutionalPrinciple:
    """A single constitutional principle with critique and revision templates."""

    name: str
    critique_prompt: str   # template — must contain {response}
    revision_prompt: str   # template — must contain {response} and {critique}
    weight: float = 1.0


# ---------------------------------------------------------------------------
# Built-in principle library
# ---------------------------------------------------------------------------

HARMLESSNESS_PRINCIPLES: list[ConstitutionalPrinciple] = [
    ConstitutionalPrinciple(
        name="no_harm",
        critique_prompt=(
            "Review the following response and identify any content that could "
            "cause harm to individuals or groups.\n\nResponse: {response}\n\n"
            "Critique:"
        ),
        revision_prompt=(
            "Revise the following response to remove any harmful content identified "
            "in the critique.\n\nResponse: {response}\n\nCritique: {critique}\n\n"
            "Revised response:"
        ),
        weight=1.0,
    ),
    ConstitutionalPrinciple(
        name="no_deception",
        critique_prompt=(
            "Review the following response and identify any misleading, false, or "
            "deceptive statements.\n\nResponse: {response}\n\nCritique:"
        ),
        revision_prompt=(
            "Revise the following response to correct any misleading or deceptive "
            "parts identified in the critique.\n\nResponse: {response}\n\n"
            "Critique: {critique}\n\nRevised response:"
        ),
        weight=1.0,
    ),
    ConstitutionalPrinciple(
        name="helpful",
        critique_prompt=(
            "Review the following response and identify ways it could be more "
            "helpful and useful to the user.\n\nResponse: {response}\n\nCritique:"
        ),
        revision_prompt=(
            "Revise the following response to improve its helpfulness based on "
            "the critique.\n\nResponse: {response}\n\nCritique: {critique}\n\n"
            "Revised response:"
        ),
        weight=1.0,
    ),
]


# ---------------------------------------------------------------------------
# Principle sampling
# ---------------------------------------------------------------------------

def sample_principles(
    principles: list[ConstitutionalPrinciple],
    n: int,
    seed: int | None = None,
) -> list[ConstitutionalPrinciple]:
    """Sample n principles weighted by their weight field.

    Sampling is done with replacement so n may exceed len(principles).

    Args:
        principles: Pool of principles to sample from.
        n: Number of principles to sample.
        seed: Optional random seed for reproducibility.

    Returns:
        List of sampled ConstitutionalPrinciple objects (length n).
    """
    rng = random.Random(seed)
    weights = [p.weight for p in principles]
    return rng.choices(principles, weights=weights, k=n)


# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------

def format_critique_prompt(principle: ConstitutionalPrinciple, response: str) -> str:
    """Fill the {response} placeholder in principle.critique_prompt."""
    return principle.critique_prompt.format(response=response)


def format_revision_prompt(
    principle: ConstitutionalPrinciple,
    response: str,
    critique: str,
) -> str:
    """Fill the {response} and {critique} placeholders in principle.revision_prompt."""
    return principle.revision_prompt.format(response=response, critique=critique)


# ---------------------------------------------------------------------------
# Greedy generation
# ---------------------------------------------------------------------------

def greedy_generate(
    model,
    tokenizer_encode: Callable[[str], list[int]],
    tokenizer_decode: Callable[[list[int]], str],
    prompt: str,
    max_new_tokens: int = 32,
) -> str:
    """Greedily generate text from a prompt string.

    Encodes the prompt, runs the model forward pass for each new token
    using argmax decoding, then decodes and returns only the newly
    generated tokens (not the prompt).

    Args:
        model: AureliusTransformer. Forward signature: (input_ids) -> (_, logits, _).
        tokenizer_encode: Callable mapping str -> list[int].
        tokenizer_decode: Callable mapping list[int] -> str.
        prompt: Input prompt string.
        max_new_tokens: Number of tokens to generate.

    Returns:
        Decoded string of the generated tokens only.
    """
    ids = tokenizer_encode(prompt)
    if not ids:
        return ""

    input_ids = torch.tensor([ids], dtype=torch.long)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    generated: list[int] = []
    current_ids = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            _, logits, _ = model(current_ids)
            next_token = int(logits[:, -1, :].argmax(dim=-1).item())
            generated.append(next_token)
            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_token]], dtype=torch.long, device=device)],
                dim=1,
            )

    return tokenizer_decode(generated)


# ---------------------------------------------------------------------------
# ConstitutionalReviser
# ---------------------------------------------------------------------------

class ConstitutionalReviser:
    """Apply a multi-principle critique-revision pipeline to model responses.

    Args:
        model: AureliusTransformer.
        encode_fn: Callable mapping str -> list[int].
        decode_fn: Callable mapping list[int] -> str.
        principles: Pool of ConstitutionalPrinciple objects to draw from.
        max_new_tokens: Token budget for each generation call.
    """

    def __init__(
        self,
        model,
        encode_fn: Callable[[str], list[int]],
        decode_fn: Callable[[list[int]], str],
        principles: list[ConstitutionalPrinciple],
        max_new_tokens: int = 32,
    ) -> None:
        self.model = model
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.principles = principles
        self.max_new_tokens = max_new_tokens

    def _generate(self, prompt: str) -> str:
        return greedy_generate(
            self.model,
            self.encode_fn,
            self.decode_fn,
            prompt,
            self.max_new_tokens,
        )

    def critique_and_revise(
        self,
        response: str,
        principle: ConstitutionalPrinciple,
    ) -> tuple[str, str]:
        """Generate a critique and a revised response for one principle.

        Args:
            response: Current response text.
            principle: Constitutional principle to apply.

        Returns:
            (critique_text, revised_response) — both decoded strings.
        """
        critique_prompt = format_critique_prompt(principle, response)
        critique_text = self._generate(critique_prompt)

        revision_prompt = format_revision_prompt(principle, response, critique_text)
        revised_response = self._generate(revision_prompt)

        return critique_text, revised_response

    def run_pipeline(
        self,
        initial_response: str,
        n_rounds: int = 2,
    ) -> dict:
        """Run multi-round critique-revision pipeline.

        Each round samples one principle (uniform random from self.principles),
        applies critique_and_revise, and updates the current response.

        Args:
            initial_response: The starting response text.
            n_rounds: Number of critique-revision rounds to run.

        Returns:
            Dict with keys:
                "initial"   : original response string
                "final"     : response after all rounds
                "revisions" : list of revised response strings (one per round)
                "critiques" : list of critique strings (one per round)
                "n_rounds"  : number of rounds performed
        """
        current = initial_response
        revisions: list[str] = []
        critiques: list[str] = []

        for _ in range(n_rounds):
            [principle] = sample_principles(self.principles, n=1)
            critique_text, revised = self.critique_and_revise(current, principle)
            critiques.append(critique_text)
            revisions.append(revised)
            current = revised

        return {
            "initial": initial_response,
            "final": current,
            "revisions": revisions,
            "critiques": critiques,
            "n_rounds": n_rounds,
        }


# ---------------------------------------------------------------------------
# Similarity metric
# ---------------------------------------------------------------------------

def compute_revision_similarity(response1: str, response2: str) -> float:
    """Character-level Jaccard similarity of bigram sets.

    Returns 1.0 for identical strings, 0.0 when the strings share no bigrams.

    Args:
        response1: First string.
        response2: Second string.

    Returns:
        Float in [0, 1].
    """
    if response1 == response2:
        return 1.0

    def bigrams(s: str) -> set[str]:
        return {s[i:i + 2] for i in range(len(s) - 1)}

    bg1 = bigrams(response1)
    bg2 = bigrams(response2)

    if not bg1 and not bg2:
        return 1.0

    union = bg1 | bg2
    if not union:
        return 1.0

    intersection = bg1 & bg2
    return len(intersection) / len(union)
