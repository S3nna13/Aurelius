"""Chain-of-thought prompting: format, sample, extract, and vote."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import torch


@dataclass
class CoTConfig:
    """Configuration for chain-of-thought prompting."""

    system_prompt: str = (
        "Think step by step. Show your reasoning, then state your "
        "final answer after 'Therefore:'."
    )
    answer_delimiter: str = "Therefore:"
    n_samples: int = 5  # for self-consistency
    temperature: float = 0.8
    top_p: float = 0.9
    max_new_tokens: int = 128


def format_cot_prompt(question: str, cfg: CoTConfig | None = None) -> str:
    """Format a question with CoT instructions.

    Returns a string like:
    "Think step by step. Show your reasoning, then state your final
    answer after 'Therefore:'.

    Question: {question}
    Answer: Let me think through this step by step."
    """
    cfg = cfg or CoTConfig()
    return (
        f"{cfg.system_prompt}\n\n"
        f"Question: {question}\n"
        f"Answer: Let me think through this step by step."
    )


def extract_answer(text: str, delimiter: str = "Therefore:") -> str | None:
    """Extract the final answer from a CoT response.

    Returns the text after the LAST occurrence of *delimiter*, stripped.
    Returns ``None`` if the delimiter is not found.
    """
    parts = text.split(delimiter)
    if len(parts) < 2:
        return None
    return parts[-1].strip()


def majority_vote(answers: list[str | None]) -> str | None:
    """Return the most common non-None answer.

    If there is a tie the first-occurring winner is returned (preserved by
    :class:`collections.Counter`).  Returns ``None`` when every element is
    ``None``.
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    return counts.most_common(1)[0][0]


@dataclass
class CoTResult:
    """Container for a chain-of-thought sampling run."""

    question: str
    responses: list[str]  # raw generated strings
    extracted_answers: list[str | None]
    final_answer: str | None  # majority vote
    answer_counts: dict[str, int]  # counts per answer


class CoTSampler:
    """Sample N chain-of-thought responses and apply majority voting.

    Works with token IDs -- requires *encode_fn* and *decode_fn* callables.

    Usage::

        sampler = CoTSampler(model, encode_fn=..., decode_fn=..., cfg=cfg)
        result = sampler.sample(question="What is 7 * 8?")
    """

    def __init__(
        self,
        model: torch.nn.Module,
        encode_fn,  # str -> list[int]
        decode_fn,  # list[int] -> str
        cfg: CoTConfig | None = None,
    ) -> None:
        self.model = model
        self.encode = encode_fn
        self.decode = decode_fn
        self.cfg = cfg or CoTConfig()

    def sample(self, question: str) -> CoTResult:
        """Generate *n_samples* responses and apply majority voting."""
        prompt = format_cot_prompt(question, self.cfg)
        prompt_ids = self.encode(prompt)
        input_ids = torch.tensor(prompt_ids).unsqueeze(0)  # (1, S)

        responses: list[str] = []
        extracted: list[str | None] = []

        with torch.no_grad():
            for _ in range(self.cfg.n_samples):
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                )
                new_tokens = output[0, len(prompt_ids) :].tolist()
                response = self.decode(new_tokens)
                responses.append(response)
                extracted.append(
                    extract_answer(response, self.cfg.answer_delimiter)
                )

        final = majority_vote(extracted)
        counts = dict(Counter(a for a in extracted if a is not None))

        return CoTResult(
            question=question,
            responses=responses,
            extracted_answers=extracted,
            final_answer=final,
            answer_counts=counts,
        )
