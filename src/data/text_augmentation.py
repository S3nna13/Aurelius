"""Text-level data augmentation: word deletion, swapping, char noise, and more."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AugmentationConfig:
    """Configuration for text augmentation operations."""

    p_word_delete: float = 0.1
    p_word_swap: float = 0.1
    p_char_noise: float = 0.05
    max_aug_ratio: float = 0.3
    seed: Optional[int] = None


def random_word_deletion(text: str, p: float, rng: random.Random) -> str:
    """Delete each word independently with probability p; never delete all words.

    Args:
        text: Input text string.
        p: Probability of deleting each word.
        rng: Random number generator.

    Returns:
        Augmented text with some words deleted.
    """
    words = text.split()
    if not words:
        return text
    if len(words) == 1:
        return text

    kept = [w for w in words if rng.random() >= p]

    # Never delete all words — keep at least one
    if not kept:
        kept = [rng.choice(words)]

    return " ".join(kept)


def random_word_swap(text: str, p: float, rng: random.Random) -> str:
    """For each word, with probability p swap it with a random other word.

    Applies up to max_aug_ratio * len(words) swaps total (max_aug_ratio is
    taken from the outer scope; here we accept p and apply up to
    int(p * len(words)) + 1 swaps for simplicity, since max_aug_ratio is on
    AugmentationConfig and not passed here — callers using TextAugmentor have
    access to it).

    Args:
        text: Input text string.
        p: Probability of swapping each word.
        rng: Random number generator.

    Returns:
        Augmented text with some words swapped.
    """
    words = text.split()
    if len(words) < 2:
        return text

    words = list(words)
    n = len(words)
    swapped = 0
    max_swaps = max(1, int(p * n))

    for i in range(n):
        if swapped >= max_swaps:
            break
        if rng.random() < p:
            j = rng.randint(0, n - 1)
            words[i], words[j] = words[j], words[i]
            swapped += 1

    return " ".join(words)


def random_char_noise(text: str, p: float, rng: random.Random) -> str:
    """For each non-space character, with probability p replace with a random letter (a-z).

    Preserves spaces.

    Args:
        text: Input text string.
        p: Probability of replacing each non-space character.
        rng: Random number generator.

    Returns:
        Augmented text with some characters replaced.
    """
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch != " " and rng.random() < p:
            chars[i] = chr(ord("a") + rng.randint(0, 25))
    return "".join(chars)


def random_word_insertion(words: List[str], p: float, rng: random.Random) -> List[str]:
    """With probability p per word, insert a random word from the vocabulary at a random position.

    The vocabulary is the words list itself.

    Args:
        words: List of words (also serves as the vocabulary to insert from).
        p: Probability of inserting a new word for each existing word.
        rng: Random number generator.

    Returns:
        New list of words with possible insertions.
    """
    if not words:
        return words

    result = list(words)
    # We iterate over original length to avoid runaway insertion
    for _ in range(len(words)):
        if rng.random() < p:
            insert_word = rng.choice(words)
            pos = rng.randint(0, len(result))
            result.insert(pos, insert_word)

    return result


def synonym_swap_simple(text: str, swap_map: Dict[str, str]) -> str:
    """Replace words in text according to swap_map.

    Case-insensitive match. Preserves original case if the original word
    starts with an uppercase letter.

    Args:
        text: Input text string.
        swap_map: Mapping from word (lowercase key) to replacement word.

    Returns:
        Text with words replaced according to swap_map.
    """
    words = text.split()
    result = []
    for word in words:
        # Strip trailing punctuation for lookup
        stripped = word.rstrip(".,!?;:")
        punct = word[len(stripped):]
        lookup = stripped.lower()

        if lookup in swap_map:
            replacement = swap_map[lookup]
            # Preserve case: if original starts with uppercase, capitalize replacement
            if stripped and stripped[0].isupper():
                replacement = replacement[0].upper() + replacement[1:] if replacement else replacement
            result.append(replacement + punct)
        else:
            result.append(word)

    return " ".join(result)


def compute_edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance at the word level (not character level).

    Uses dynamic programming.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Word-level edit distance between s1 and s2.
    """
    words1 = s1.split()
    words2 = s2.split()
    m, n = len(words1), len(words2)

    # DP table of size (m+1) x (n+1)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if words1[i - 1] == words2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n]


class TextAugmentor:
    """Applies a configurable set of text augmentation operations."""

    _METHOD_ORDER = ["delete", "swap", "char_noise"]

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)

    def augment(self, text: str, methods: Optional[List[str]] = None) -> str:
        """Apply specified augmentation methods in canonical order.

        The canonical order is: ["delete", "swap", "char_noise"].
        If methods is None, all three are applied.
        If methods is an empty list, the original text is returned unchanged.

        Args:
            text: Input text string.
            methods: List of method names to apply. Subset of
                     ["delete", "swap", "char_noise"].

        Returns:
            Augmented text.
        """
        if methods is not None and len(methods) == 0:
            return text

        active = set(methods) if methods is not None else set(self._METHOD_ORDER)

        for method in self._METHOD_ORDER:
            if method not in active:
                continue
            if method == "delete":
                text = random_word_deletion(text, self.config.p_word_delete, self._rng)
            elif method == "swap":
                text = random_word_swap(text, self.config.p_word_swap, self._rng)
            elif method == "char_noise":
                text = random_char_noise(text, self.config.p_char_noise, self._rng)

        return text

    def augment_batch(
        self, texts: List[str], n_augments_per_text: int = 1
    ) -> List[str]:
        """For each text, generate n_augments_per_text augmented versions.

        Returns a flattened list of all augmented texts.

        Args:
            texts: List of input strings.
            n_augments_per_text: Number of augmented versions per input text.

        Returns:
            Flat list of augmented strings (length = len(texts) * n_augments_per_text).
        """
        result: List[str] = []
        for text in texts:
            for _ in range(n_augments_per_text):
                result.append(self.augment(text))
        return result

    def get_augment_stats(self, original: str, augmented: str) -> Dict[str, float]:
        """Compute statistics comparing original and augmented text.

        Returns a dict with keys:
            - word_count_original: word count of original
            - word_count_augmented: word count of augmented
            - edit_distance_ratio: word-level edit distance / max(word counts)
            - words_changed_ratio: fraction of original words not in augmented

        Args:
            original: Original text.
            augmented: Augmented text.

        Returns:
            Dictionary of float statistics.
        """
        orig_words = original.split()
        aug_words = augmented.split()

        wc_orig = float(len(orig_words))
        wc_aug = float(len(aug_words))

        max_wc = max(wc_orig, wc_aug, 1.0)
        edit_dist = float(compute_edit_distance(original, augmented))
        edit_distance_ratio = edit_dist / max_wc

        # words_changed_ratio: fraction of original words that changed
        # Use multiset difference: words in original but not matched in augmented
        aug_counter: Dict[str, int] = {}
        for w in aug_words:
            aug_counter[w] = aug_counter.get(w, 0) + 1

        changed = 0
        for w in orig_words:
            if aug_counter.get(w, 0) > 0:
                aug_counter[w] -= 1
            else:
                changed += 1

        words_changed_ratio = changed / max(wc_orig, 1.0)

        return {
            "word_count_original": wc_orig,
            "word_count_augmented": wc_aug,
            "edit_distance_ratio": edit_distance_ratio,
            "words_changed_ratio": words_changed_ratio,
        }
