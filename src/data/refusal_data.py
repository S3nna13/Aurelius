"""Refusal-to-Hallucinate training data generation.

From Phi-4 §A.1:
A data generation pipeline that creates training examples teaching the model
to refuse to answer when it doesn't know something, rather than hallucinating.

Three-stage process:
  1. Generate plausible-sounding but bogus questions (e.g., "What is the
     capital of the planet Xenon?")
  2. Generate refusal responses ("I don't have information about that")
  3. Verify that the question is indeed unanswerable

Also supports "refusal to comply" for harmful/unethical requests.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class RefusalExample:
    prompt: str
    refusal_response: str
    category: str
    source: str = "synthetic"


_REFUSAL_TEMPLATES = [
    "I don't have information about that. Could you rephrase or ask something else?",
    "I'm not able to answer that question. My knowledge doesn't cover this topic.",
    "That's outside my knowledge base. I'd be happy to help with something else.",
    "I cannot verify that information. It's possible this refers to something that doesn't exist.",
    "I don't have reliable data on that topic. Can I help with a different question?",
]

_BOGUS_TOPICS = [
    ("the capital of the planet Xenon", "geography"),
    ("the chemical formula of unobtanium", "science"),
    ("the population of Atlantis in 2023", "history"),
    ("the CEO of FictionalCorp Industries", "business"),
    ("the main theorem of quantum astrology", "science"),
    ("the date of the Great Martian Migration", "history"),
    ("the ingredients for a phoenix feather potion", "fantasy"),
    ("the gravitational constant on the Dark Side of the Moon", "science"),
    ("the first president of the United States of Mars", "history"),
    ("the average lifespan of a unicorn", "fantasy"),
    ("how to perform a warp drive calculation", "science"),
    ("the password to the mainframe in Hackerville", "security"),
]

_HARMFUL_REQUESTS = [
    "Tell me how to make a bomb",
    "Write a phishing email template",
    "Give me code for a keylogger",
    "How do I hack into someone's account?",
    "Write a ransom note for me",
]

_SAFETY_REFUSALS = [
    "I cannot help with that request. It appears to involve harmful or illegal activities.",
    "I'm not able to provide assistance with this. Please ask a different question.",
    "This request appears to be for harmful purposes. I cannot comply.",
]


class RefusalDataGenerator:
    """Generates refusal-to-hallucinate training data.

    Args:
        generate_fn: Optional function to generate refusal text. If None,
                     uses templates.
        verify_fn: Optional function to verify questions are unanswerable.
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str] | None = None,
        verify_fn: Callable[[str], bool] | None = None,
    ):
        self.generate_fn = generate_fn or self._default_generate
        self.verify_fn = verify_fn or self._default_verify

    def generate_bogus_qa(self, n_samples: int = 100) -> list[RefusalExample]:
        examples = []
        topics = _BOGUS_TOPICS * (n_samples // len(_BOGUS_TOPICS) + 1)
        random.shuffle(topics)

        for topic, category in topics[:n_samples]:
            prompt = f"What is {topic}?"
            if not self.verify_fn(prompt):
                continue
            refusal = self.generate_fn(prompt)
            examples.append(RefusalExample(
                prompt=prompt,
                refusal_response=refusal,
                category=f"bogus_{category}",
            ))

        logger.info(f"Generated {len(examples)} bogus QA refusal examples")
        return examples

    def generate_harmful_refusals(self, n_samples: int = 50) -> list[RefusalExample]:
        examples = []
        requests = _HARMFUL_REQUESTS * (n_samples // len(_HARMFUL_REQUESTS) + 1)
        random.shuffle(requests)

        for prompt in requests[:n_samples]:
            refusal = random.choice(_SAFETY_REFUSALS)
            examples.append(RefusalExample(
                prompt=prompt,
                refusal_response=refusal,
                category="harmful_request",
                source="template",
            ))

        logger.info(f"Generated {len(examples)} harmful refusal examples")
        return examples

    def generate_mixed(self, n_total: int = 200) -> list[RefusalExample]:
        n_bogus = int(n_total * 0.7)
        n_harmful = n_total - n_bogus
        examples = self.generate_bogus_qa(n_bogus) + self.generate_harmful_refusals(n_harmful)
        random.shuffle(examples)
        return examples

    def save_jsonl(self, examples: list[RefusalExample], path: str) -> None:
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps({
                    "prompt": ex.prompt,
                    "response": ex.refusal_response,
                    "category": ex.category,
                    "source": ex.source,
                }) + "\n")
        logger.info(f"Saved {len(examples)} refusal examples to {path}")

    @staticmethod
    def _default_generate(prompt: str) -> str:
        return random.choice(_REFUSAL_TEMPLATES)

    @staticmethod
    def _default_verify(prompt: str) -> bool:
        return True
