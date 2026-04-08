"""Self-consistency decoding (Wang et al. 2022).

Generate K independent completions with temperature > 0, extract the final
answer from each, and return the most common answer via majority vote.
Improves accuracy on reasoning tasks (math, logic, code) without fine-tuning.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from src.model.transformer import AureliusTransformer

EOS_TOKEN_ID: int = 2


@dataclass
class SelfConsistencyConfig:
    """Configuration for self-consistency decoding."""

    n_samples: int = 10
    temperature: float = 0.7
    max_new_tokens: int = 256
    answer_pattern: str = r"(?:answer is|=|Answer:)\s*([^\n,\.]+)"
    aggregation: str = "majority_vote"  # "majority_vote" | "weighted_vote"


@dataclass
class ConsistencyResult:
    """Result from self-consistency decoding."""

    final_answer: str
    vote_counts: dict[str, int]
    confidence: float
    all_completions: list[str]
    all_extracted: list[str]


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def majority_vote(answers: list[str]) -> tuple[str, dict[str, int]]:
    """Return (most_common_answer, vote_counts_dict).

    Normalizes answers by strip+lower for comparison but returns the original
    (non-lowercased) form of the winner.  Ties are broken alphabetically
    (first answer alphabetically among tied winners).  Empty list returns
    ("", {}).
    """
    if not answers:
        return ("", {})

    # Build mapping: normalized -> list of original forms seen
    norm_to_originals: dict[str, list[str]] = {}
    for ans in answers:
        norm = ans.strip().lower()
        norm_to_originals.setdefault(norm, []).append(ans)

    # Count by normalized form
    norm_counts: dict[str, int] = {k: len(v) for k, v in norm_to_originals.items()}

    # Find maximum count
    max_count = max(norm_counts.values())
    tied_norms = [k for k, v in norm_counts.items() if v == max_count]

    # Break tie alphabetically among normalized forms
    winner_norm = sorted(tied_norms)[0]
    # Return the first original form encountered for the winner norm
    winner_original = norm_to_originals[winner_norm][0]

    # Build vote_counts with original forms (first seen per norm key)
    vote_counts: dict[str, int] = {}
    for norm, originals in norm_to_originals.items():
        representative = originals[0]
        vote_counts[representative] = norm_counts[norm]

    return (winner_original, vote_counts)


def weighted_vote(answers: list[str], weights: list[float]) -> tuple[str, dict[str, float]]:
    """Weighted majority vote.

    Returns (most_weighted_answer, weighted_counts).
    weights[i] corresponds to answers[i].
    """
    if not answers:
        return ("", {})

    weighted_counts: dict[str, float] = {}
    for ans, w in zip(answers, weights):
        weighted_counts[ans] = weighted_counts.get(ans, 0.0) + w

    max_weight = max(weighted_counts.values())
    tied = [k for k, v in weighted_counts.items() if v == max_weight]
    winner = sorted(tied)[0]

    return (winner, weighted_counts)


def answer_consistency_score(answers: list[str]) -> float:
    """Return fraction of answers that match the majority answer.

    Perfect consistency = 1.0 (all same), worst = 1/n.
    """
    if not answers:
        return 0.0

    winner, _ = majority_vote(answers)
    winner_norm = winner.strip().lower()
    matching = sum(1 for a in answers if a.strip().lower() == winner_norm)
    return matching / len(answers)


# ---------------------------------------------------------------------------
# Core decoder
# ---------------------------------------------------------------------------

class SelfConsistencyDecoder:
    """Self-consistency decoding via majority vote over sampled completions.

    Args:
        model: AureliusTransformer
        tokenizer_encode: callable str -> list[int]
        tokenizer_decode: callable list[int] -> str
        config: SelfConsistencyConfig
    """

    def __init__(
        self,
        model: AureliusTransformer,
        tokenizer_encode,
        tokenizer_decode,
        config: SelfConsistencyConfig | None = None,
    ) -> None:
        self.model = model
        self._encode = tokenizer_encode
        self._decode = tokenizer_decode
        self.config = config or SelfConsistencyConfig()
        self.model.train(False)  # set to inference mode

    @torch.no_grad()
    def _sample_completion(self, prompt_ids: list[int], temperature: float) -> list[int]:
        """Sample one completion from the model with the given temperature.

        Generates until max_new_tokens or EOS (token id=2).
        Returns generated token ids (not including prompt).
        """
        generated: list[int] = []
        input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)  # (1, S)
        past_key_values = None
        cur_ids = input_ids

        for _ in range(self.config.max_new_tokens):
            _, logits, past_key_values = self.model(cur_ids, past_key_values=past_key_values)
            next_logits = logits[:, -1, :]  # (1, vocab)

            if temperature == 0:
                next_token_id = int(next_logits.argmax(dim=-1).item())
            else:
                scaled_logits = next_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                next_token_id = int(torch.multinomial(probs, num_samples=1).item())

            generated.append(next_token_id)

            if next_token_id == EOS_TOKEN_ID:
                break

            # Prepare next step: only the newly generated token
            cur_ids = torch.tensor([[next_token_id]], dtype=torch.long)

        return generated

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from generated text using config.answer_pattern.

        Returns stripped match or "" if not found.
        """
        match = re.search(self.config.answer_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def decode(self, prompt: str) -> ConsistencyResult:
        """Generate n_samples completions and aggregate via majority vote.

        Confidence = votes_for_winner / n_samples.
        """
        prompt_ids = self._encode(prompt)
        all_completions: list[str] = []
        all_extracted: list[str] = []

        for _ in range(self.config.n_samples):
            token_ids = self._sample_completion(prompt_ids, self.config.temperature)
            completion_text = self._decode(token_ids)
            all_completions.append(completion_text)
            all_extracted.append(self._extract_answer(completion_text))

        if self.config.aggregation == "weighted_vote":
            weights = [1.0] * len(all_extracted)
            winner, wc = weighted_vote(all_extracted, weights)
            vote_counts = {k: int(v) for k, v in wc.items()}
        else:
            winner, vote_counts = majority_vote(all_extracted)

        winner_norm = winner.strip().lower()
        votes_for_winner = sum(
            1 for a in all_extracted if a.strip().lower() == winner_norm
        )
        confidence = votes_for_winner / self.config.n_samples if self.config.n_samples > 0 else 0.0

        return ConsistencyResult(
            final_answer=winner,
            vote_counts=vote_counts,
            confidence=confidence,
            all_completions=all_completions,
            all_extracted=all_extracted,
        )

    def decode_batch(self, prompts: list[str]) -> list[ConsistencyResult]:
        """Decode each prompt independently. Returns list of ConsistencyResult."""
        return [self.decode(prompt) for prompt in prompts]


# ---------------------------------------------------------------------------
# Chain-of-thought sampler
# ---------------------------------------------------------------------------

class ChainOfThoughtSampler:
    """Generate chain-of-thought reasoning paths before extracting answers.

    Prompts the model with a CoT trigger prefix and collects reasoning +
    answer pairs.

    Args:
        model: AureliusTransformer
        tokenizer_encode: callable str -> list[int]
        tokenizer_decode: callable list[int] -> str
        cot_trigger: str prepended to question to elicit reasoning
        answer_trigger: str that introduces the final answer
        config: optional SelfConsistencyConfig for generation parameters
    """

    def __init__(
        self,
        model: AureliusTransformer,
        tokenizer_encode,
        tokenizer_decode,
        cot_trigger: str = "Let's think step by step:",
        answer_trigger: str = "Therefore, the answer is",
        config: SelfConsistencyConfig | None = None,
    ) -> None:
        self.model = model
        self._encode = tokenizer_encode
        self._decode = tokenizer_decode
        self.cot_trigger = cot_trigger
        self.answer_trigger = answer_trigger
        self.config = config or SelfConsistencyConfig()
        self._decoder = SelfConsistencyDecoder(model, tokenizer_encode, tokenizer_decode, self.config)
        self.model.train(False)

    def sample_with_cot(
        self,
        question: str,
        n_samples: int = 5,
        temperature: float = 0.7,
    ) -> list[dict]:
        """Generate n_samples CoT completions.

        Returns list of {'reasoning': str, 'answer': str, 'full_text': str}
        """
        cot_prompt = f"{question}\n{self.cot_trigger}"
        prompt_ids = self._encode(cot_prompt)

        results: list[dict] = []
        for _ in range(n_samples):
            token_ids = self._decoder._sample_completion(prompt_ids, temperature)
            full_text = self._decode(token_ids)

            if self.answer_trigger in full_text:
                parts = full_text.split(self.answer_trigger, 1)
                reasoning = parts[0].strip()
                answer = parts[1].strip() if len(parts) > 1 else ""
            else:
                reasoning = full_text.strip()
                answer = ""

            results.append({
                "reasoning": reasoning,
                "answer": answer,
                "full_text": full_text,
            })

        return results

    def aggregate(self, samples: list[dict]) -> dict:
        """Aggregate CoT samples.

        Returns {'final_answer': str, 'confidence': float,
                 'consistent_reasoning': bool,
                 'vote_counts': dict}
        """
        answers = [s["answer"] for s in samples]
        final_answer, vote_counts = majority_vote(answers)

        n = len(samples)
        if n == 0:
            return {
                "final_answer": "",
                "confidence": 0.0,
                "consistent_reasoning": False,
                "vote_counts": {},
            }

        winner_norm = final_answer.strip().lower()
        votes_for_winner = sum(
            1 for a in answers if a.strip().lower() == winner_norm
        )
        confidence = votes_for_winner / n
        consistent_reasoning = confidence > 0.5

        return {
            "final_answer": final_answer,
            "confidence": confidence,
            "consistent_reasoning": consistent_reasoning,
            "vote_counts": vote_counts,
        }
