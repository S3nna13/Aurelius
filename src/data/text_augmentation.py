"""Text-level data augmentation: paraphrase templates, synonym substitution, and instruction variants."""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class TextAugConfig:
    """Configuration for text-level augmentation."""

    p_paraphrase: float = 0.3
    p_synonym: float = 0.2
    p_instruction_variant: float = 0.5
    max_synonyms: int = 3
    seed: int = 42


SYNONYM_DICT: dict[str, list[str]] = {
    "fast": ["quick", "rapid", "swift"],
    "large": ["big", "huge", "enormous"],
    "small": ["tiny", "little", "compact"],
    "good": ["great", "excellent", "fine"],
    "bad": ["poor", "terrible", "awful"],
    "important": ["significant", "crucial", "essential"],
    "difficult": ["hard", "challenging", "tough"],
    "simple": ["easy", "straightforward", "basic"],
    "new": ["recent", "fresh", "novel"],
    "old": ["ancient", "aged", "dated"],
    "make": ["create", "build", "produce"],
    "find": ["locate", "discover", "identify"],
    "help": ["assist", "support", "aid"],
    "show": ["display", "demonstrate", "reveal"],
    "say": ["state", "express", "mention"],
    "know": ["understand", "realize", "recognize"],
    "get": ["obtain", "acquire", "receive"],
    "give": ["provide", "offer", "supply"],
    "think": ["believe", "consider", "suppose"],
    "use": ["employ", "utilize", "apply"],
}

_PARAPHRASE_TEMPLATES = [
    "In other words, {text}",
    "To put it differently, {text}",
    "That is to say, {text}",
    "Put simply, {text}",
    "{text} In essence, this means the same thing.",
]


def apply_synonym_substitution(
    text: str,
    p: float,
    synonym_dict: dict[str, list[str]],
    rng: random.Random,
) -> str:
    """Substitute words with synonyms with probability p.

    Preserves capitalization of the first word. Returns the augmented text.
    """
    words = text.split()
    if not words:
        return text

    result = []
    for i, word in enumerate(words):
        # Strip trailing punctuation for lookup
        stripped = word.rstrip(".,!?;:")
        punct = word[len(stripped):]
        lookup = stripped.lower()

        if rng.random() < p and lookup in synonym_dict:
            synonyms = synonym_dict[lookup]
            chosen = rng.choice(synonyms)
            # Preserve capitalization of the first word in the text
            if i == 0 and stripped and stripped[0].isupper():
                chosen = chosen.capitalize()
            result.append(chosen + punct)
        else:
            result.append(word)

    return " ".join(result)


def paraphrase_with_template(text: str, rng: random.Random) -> str:
    """Wrap text with one of 5 paraphrase templates."""
    template = rng.choice(_PARAPHRASE_TEMPLATES)
    return template.format(text=text)


def generate_instruction_variants(instruction: str, rng: random.Random) -> list[str]:
    """Generate 3 variants of an instruction.

    Variants:
      1. Prefix with "Please " or "Could you "
      2. Append " Thank you." or " Please be thorough."
      3. Rephrase using paraphrase_with_template
    """
    prefix = rng.choice(["Please ", "Could you "])
    variant1 = prefix + instruction

    suffix = rng.choice([" Thank you.", " Please be thorough."])
    variant2 = instruction + suffix

    variant3 = paraphrase_with_template(instruction, rng)

    return [variant1, variant2, variant3]


def augment_text_sample(text: str, config: TextAugConfig, rng: random.Random) -> str:
    """Apply text-level augmentations based on probabilities in config."""
    if rng.random() < config.p_paraphrase:
        text = paraphrase_with_template(text, rng)

    if rng.random() < config.p_synonym:
        text = apply_synonym_substitution(text, config.p_synonym, SYNONYM_DICT, rng)

    return text


def augment_dataset(texts: list[str], config: TextAugConfig) -> list[str]:
    """Augment each text in the dataset using a seeded RNG for reproducibility.

    Returns a list of augmented texts of the same length as the input.
    """
    rng = random.Random(config.seed)
    return [augment_text_sample(text, config, rng) for text in texts]


class InstructionAugmenter:
    """Augments instruction+response pairs with instruction variants."""

    def __init__(self, config: TextAugConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)

    def augment(self, instruction: str, response: str) -> list[tuple[str, str]]:
        """Generate variants of the instruction paired with the original response.

        Returns a list of (instruction_variant, response) tuples.
        """
        variants = generate_instruction_variants(instruction, self._rng)
        return [(variant, response) for variant in variants]

    def augment_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Augment each (instruction, response) pair in a batch.

        Returns a flat list of all (instruction_variant, response) pairs.
        """
        result: list[tuple[str, str]] = []
        for instruction, response in pairs:
            result.extend(self.augment(instruction, response))
        return result
