"""Long-context evaluation: passkey retrieval, needle-in-haystack, multi-hop document QA."""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LongContextConfig:
    """Configuration for long-context evaluations."""

    max_context_len: int = 8192
    n_distractors: int = 10  # filler sentences around needle
    passkey_length: int = 5  # digit passkey length
    eval_positions: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    seed: int = 42


# ---------------------------------------------------------------------------
# Distractor / haystack text generation
# ---------------------------------------------------------------------------

_TEMPLATES = [
    "The researchers conducted an extensive study on the behavioral patterns of various species found in temperate forests.",  # noqa: E501
    "A recent analysis of economic trends suggests that global markets are becoming increasingly interconnected.",  # noqa: E501
    "The ancient city was discovered by archaeologists who spent several decades excavating the site.",  # noqa: E501
    "Scientists have developed a new method for synthesizing complex organic compounds under mild conditions.",  # noqa: E501
    "The committee reviewed the proposed legislation and recommended several amendments before final passage.",  # noqa: E501
    "Observations from the satellite array provided unprecedented detail about atmospheric circulation patterns.",  # noqa: E501
    "The engineering team designed a new bridge structure capable of withstanding extreme weather events.",  # noqa: E501
    "Local authorities implemented a series of measures to improve water quality in the surrounding rivers.",  # noqa: E501
    "The manuscript, recovered from a monastery library, contained previously unknown historical accounts.",  # noqa: E501
    "Advances in materials science have enabled the production of lighter and stronger composites for aerospace use.",  # noqa: E501
    "The survey of public opinion revealed significant variation across different demographic groups.",  # noqa: E501
    "A comprehensive review of the literature was conducted to identify gaps in current knowledge.",
    "The pilot program demonstrated measurable improvements in student outcomes across all grade levels.",  # noqa: E501
    "Nutritional guidelines were updated following the publication of large-scale longitudinal studies.",  # noqa: E501
    "The diplomatic summit concluded with a joint communique outlining shared objectives for the region.",  # noqa: E501
    "Geological surveys confirmed the presence of mineral deposits along the northern ridge.",
    "The software platform was redesigned to improve accessibility for users with visual impairments.",  # noqa: E501
    "Urban planners proposed a mixed-use development strategy to address housing shortages.",
    "The expedition team documented over three hundred plant species in the highland ecosystem.",
    "Financial regulators issued new guidance on risk disclosure requirements for investment products.",  # noqa: E501
]


def generate_distractor_text(rng: random.Random, n_sentences: int) -> str:
    """Generate n_sentences of realistic filler text.

    Each sentence is roughly 50 characters, so the total is approximately
    n_sentences * 50 characters.

    Args:
        rng: Seeded random instance for reproducibility.
        n_sentences: Number of filler sentences to generate.

    Returns:
        String of concatenated filler sentences separated by spaces.
    """
    sentences = [rng.choice(_TEMPLATES) for _ in range(n_sentences)]
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Passkey prompt
# ---------------------------------------------------------------------------


def create_passkey_prompt(
    rng: random.Random,
    config: LongContextConfig,
    position: float,
) -> tuple[str, str]:
    """Create a long context with a hidden numeric passkey.

    Args:
        rng: Seeded random instance.
        config: LongContextConfig controlling lengths and passkey format.
        position: Relative position [0, 1] at which the passkey is inserted.

    Returns:
        (prompt, passkey) where prompt ends with "What is the passkey?"
    """
    # Generate passkey
    passkey = "".join(str(rng.randint(0, 9)) for _ in range(config.passkey_length))

    total_distractors = config.n_distractors
    # Clamp position to a reasonable range so there is text on both sides
    position = max(0.0, min(1.0, position))
    before_count = max(1, int(total_distractors * position))
    after_count = max(1, total_distractors - before_count)

    before_text = generate_distractor_text(rng, before_count)
    after_text = generate_distractor_text(rng, after_count)

    needle_sentence = f"The passkey is: {passkey}."
    prompt = f"{before_text} {needle_sentence} {after_text} What is the passkey?"
    return prompt, passkey


# ---------------------------------------------------------------------------
# Needle-in-haystack prompt
# ---------------------------------------------------------------------------


def create_needle_prompt(
    rng: random.Random,
    needle: str,
    position: float,
    haystack_len: int,
) -> tuple[str, str]:
    """Hide a needle string at a relative position in a haystack of filler text.

    Args:
        rng: Seeded random instance.
        needle: The string to hide in the haystack.
        position: Relative position [0, 1] for insertion.
        haystack_len: Approximate number of distractor sentences.

    Returns:
        (prompt, needle) where prompt asks to retrieve the needle.
    """
    position = max(0.0, min(1.0, position))
    before_count = max(1, int(haystack_len * position))
    after_count = max(1, haystack_len - before_count)

    before_text = generate_distractor_text(rng, before_count)
    after_text = generate_distractor_text(rng, after_count)

    prompt = (
        f"{before_text} {needle} {after_text} "
        f"What was the special phrase or sentence hidden in the text above?"
    )
    return prompt, needle


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def extract_passkey_from_output(output: str, passkey_length: int) -> str | None:
    """Extract a numeric passkey from model output.

    Searches for sequences of exactly `passkey_length` consecutive digits.

    Args:
        output: Raw text generated by the model.
        passkey_length: Expected number of digits.

    Returns:
        First matching digit sequence of the correct length, or None.
    """
    pattern = rf"\b(\d{{{passkey_length}}})\b"
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    # Fallback: find any run of exactly passkey_length digits not part of a longer number
    for m in re.finditer(r"\d+", output):
        if len(m.group()) == passkey_length:
            return m.group()
    return None


# ---------------------------------------------------------------------------
# Greedy generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def greedy_generate_text(
    model: nn.Module,
    tokenizer_encode: Callable[[str], list[int]],
    tokenizer_decode: Callable[[list[int]], str],
    prompt: str,
    max_new: int = 20,
) -> str:
    """Greedy-decode max_new tokens from the model given a text prompt.

    Args:
        model: AureliusTransformer or compatible module.
        tokenizer_encode: Function mapping str -> list[int].
        tokenizer_decode: Function mapping list[int] -> str.
        prompt: Input text.
        max_new: Maximum number of new tokens to generate.

    Returns:
        Generated text string (not including the prompt).
    """
    model.eval()
    device = next(model.parameters()).device
    max_seq_len: int = getattr(getattr(model, "config", None), "max_seq_len", 512)

    input_ids = tokenizer_encode(prompt)
    # Truncate if necessary
    if len(input_ids) > max_seq_len - max_new:
        input_ids = input_ids[-(max_seq_len - max_new) :]

    generated: list[int] = []
    current_ids = input_ids[:]

    for _ in range(max_new):
        x = torch.tensor([current_ids], dtype=torch.long, device=device)
        _loss, logits, _pkv = model(x)
        next_token = int(logits[0, -1].argmax(dim=-1).item())
        generated.append(next_token)
        current_ids.append(next_token)
        if len(current_ids) >= max_seq_len:
            break

    return tokenizer_decode(generated)


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------


class LongContextEvaluator:
    """Runs long-context evaluation tasks against an Aurelius model."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
        config: LongContextConfig,
    ) -> None:
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.config = config

    def evaluate_passkey_retrieval(self, n_eval: int = 10) -> dict[str, float]:
        """Evaluate passkey retrieval accuracy at each eval position.

        Args:
            n_eval: Number of random passkey prompts to generate per position.

        Returns:
            Dict with "mean_accuracy" and "acc_at_{pos}" for each position.
        """
        rng = random.Random(self.config.seed)  # noqa: S311
        results: dict[str, float] = {}
        all_correct: list[bool] = []

        for pos in self.config.eval_positions:
            correct = 0
            for _ in range(n_eval):
                prompt, passkey = create_passkey_prompt(rng, self.config, pos)
                output = greedy_generate_text(
                    self.model,
                    self.tokenizer_encode,
                    self.tokenizer_decode,
                    prompt,
                    max_new=20,
                )
                predicted = extract_passkey_from_output(output, self.config.passkey_length)
                hit = predicted == passkey
                correct += int(hit)
                all_correct.append(hit)

            acc = correct / n_eval
            key = f"acc_at_{pos}"
            results[key] = acc

        results["mean_accuracy"] = sum(all_correct) / len(all_correct) if all_correct else 0.0
        return results

    def evaluate_needle_in_haystack(
        self,
        needles: list[str],
        n_eval: int = 5,
    ) -> dict[str, float]:
        """Evaluate needle retrieval across positions and needles.

        Args:
            needles: List of needle strings to hide and retrieve.
            n_eval: Number of positions tested per needle.

        Returns:
            Dict with "mean_accuracy".
        """
        rng = random.Random(self.config.seed)  # noqa: S311
        positions = [i / max(n_eval - 1, 1) for i in range(n_eval)]
        haystack_len = max(self.config.n_distractors, 4)

        all_correct: list[bool] = []
        for needle in needles:
            for pos in positions:
                prompt, target = create_needle_prompt(rng, needle, pos, haystack_len)
                output = greedy_generate_text(
                    self.model,
                    self.tokenizer_encode,
                    self.tokenizer_decode,
                    prompt,
                    max_new=30,
                )
                hit = target.lower() in output.lower() or target in output
                all_correct.append(hit)

        return {"mean_accuracy": sum(all_correct) / len(all_correct) if all_correct else 0.0}

    @torch.no_grad()
    def compute_perplexity_at_length(
        self,
        text: str,
        chunk_size: int = 512,
    ) -> list[float]:
        """Compute perplexity per chunk of text.

        Args:
            text: Input text to examine.
            chunk_size: Number of tokens per chunk.

        Returns:
            List of perplexity values, one per chunk.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        max_seq_len: int = getattr(getattr(self.model, "config", None), "max_seq_len", 512)
        chunk_size = min(chunk_size, max_seq_len)

        token_ids = self.tokenizer_encode(text)
        if len(token_ids) < 2:
            return []

        perplexities: list[float] = []
        # Slide through token_ids in chunks
        for start in range(0, len(token_ids) - 1, chunk_size):
            chunk = token_ids[start : start + chunk_size + 1]  # +1 for targets
            if len(chunk) < 2:
                break
            # Truncate to max_seq_len
            chunk = chunk[:max_seq_len]
            input_seq = chunk[:-1]
            target_seq = chunk[1:]

            x = torch.tensor([input_seq], dtype=torch.long, device=device)
            targets = torch.tensor([target_seq], dtype=torch.long, device=device)

            _loss, logits, _pkv = self.model(x)
            # logits: (1, S, V)
            log_probs = F.log_softmax(logits, dim=-1)  # (1, S, V)
            # Gather log-probs for target tokens
            target_log_probs = (
                log_probs[0, :, :].gather(1, targets[0, :].unsqueeze(1)).squeeze(1)
            )  # (S,)

            nll = -target_log_probs.mean().item()
            ppl = math.exp(nll) if math.isfinite(nll) else float("inf")
            perplexities.append(ppl)

        return perplexities
