"""Chain-of-thought prompting: structured multi-step reasoning with evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class CoTConfig:
    max_reasoning_tokens: int = 128
    max_answer_tokens: int = 32
    temperature: float = 0.7
    n_samples: int = 1  # for self-consistency (majority vote)
    cot_trigger: str = "Let's think step by step."
    answer_trigger: str = "Therefore, the answer is:"


@dataclass
class ReasoningStep:
    """One step in a chain of thought."""

    step_idx: int
    content: str  # text of this step
    confidence: float  # estimated confidence (0-1)


@dataclass
class CoTOutput:
    """Full chain-of-thought output."""

    reasoning_steps: list[ReasoningStep]
    final_answer: str
    raw_tokens: list[int]
    n_reasoning_tokens: int


def parse_reasoning_steps(text: str, step_delimiter: str = "\n") -> list[str]:
    """Split reasoning text into individual steps by delimiter.

    Returns list of non-empty step strings.
    """
    parts = text.split(step_delimiter)
    return [p for p in parts if p.strip()]


def extract_final_answer(text: str, answer_trigger: str) -> str:
    """Extract text after answer_trigger.

    Returns empty string if not found.
    """
    idx = text.find(answer_trigger)
    if idx == -1:
        return ""
    return text[idx + len(answer_trigger) :].strip()


def compute_step_confidence(
    model: nn.Module,
    step_tokens: list[int],
    context_tokens: list[int],
) -> float:
    """Estimate confidence for a reasoning step as mean token log probability.

    Returns float in [0, 1] (sigmoid of mean log prob).
    """
    if not step_tokens:
        return torch.sigmoid(torch.tensor(0.0)).item()

    all_tokens = context_tokens + step_tokens
    input_ids = torch.tensor(all_tokens, dtype=torch.long).unsqueeze(0)  # (1, S)

    with torch.no_grad():
        _loss, logits, _pkv = model(input_ids)

    # logits: (1, S, vocab_size)
    log_probs = torch.log_softmax(logits[0], dim=-1)  # (S, vocab_size)

    # Evaluate probabilities of step tokens only.
    # step_tokens[i] is at position context_len + i; its logit comes from
    # context_len + i - 1 (the previous token predicts it).
    context_len = len(context_tokens)
    total_log_prob = 0.0
    count = 0
    for i, tok in enumerate(step_tokens):
        pred_pos = context_len + i - 1
        if pred_pos < 0:
            continue
        total_log_prob += log_probs[pred_pos, tok].item()
        count += 1

    if count == 0:
        mean_log_prob = 0.0
    else:
        mean_log_prob = total_log_prob / count

    return torch.sigmoid(torch.tensor(mean_log_prob)).item()


def greedy_decode_n_tokens(
    model: nn.Module,
    input_ids: Tensor,
    n_tokens: int,
    temperature: float = 1.0,
) -> tuple[list[int], list[float]]:
    """Greedy decode n_tokens from model given input_ids.

    Returns (token_ids, log_probs).
    """
    generated_ids: list[int] = []
    generated_log_probs: list[float] = []

    current_ids = input_ids.clone()
    past_key_values = None

    with torch.no_grad():
        for _ in range(n_tokens):
            if past_key_values is not None:
                # Feed only the last token when using KV cache
                feed_ids = current_ids[:, -1:]
            else:
                feed_ids = current_ids

            _loss, logits, past_key_values = model(feed_ids, past_key_values=past_key_values)

            # logits: (1, S, vocab_size) — take last position
            last_logits = logits[0, -1, :]  # (vocab_size,)

            if temperature != 1.0 and temperature > 0.0:
                last_logits = last_logits / temperature

            log_probs = torch.log_softmax(last_logits, dim=-1)
            next_token = torch.argmax(log_probs).item()
            next_log_prob = log_probs[next_token].item()

            generated_ids.append(int(next_token))
            generated_log_probs.append(float(next_log_prob))

            # Append next token to sequence for next step
            next_tensor = torch.tensor([[next_token]], dtype=torch.long)
            current_ids = torch.cat([current_ids, next_tensor], dim=1)

    return generated_ids, generated_log_probs


class ChainOfThoughtGenerator:
    """Generates chain-of-thought reasoning from a model."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
        config: CoTConfig,
    ) -> None:
        self.model = model
        self.encode = tokenizer_encode
        self.decode = tokenizer_decode
        self.config = config

    def generate(self, prompt_ids: list[int]) -> CoTOutput:
        """Generate reasoning + answer given prompt token ids.

        Returns CoTOutput with parsed steps and final answer.
        """
        input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)  # (1, S)

        # Generate reasoning tokens
        reasoning_token_ids, _ = greedy_decode_n_tokens(
            self.model,
            input_ids,
            self.config.max_reasoning_tokens,
            temperature=self.config.temperature,
        )

        reasoning_text = self.decode(reasoning_token_ids)

        # Generate answer tokens from prompt + reasoning
        all_so_far = prompt_ids + reasoning_token_ids
        full_ids = torch.tensor(all_so_far, dtype=torch.long).unsqueeze(0)

        answer_token_ids, _ = greedy_decode_n_tokens(
            self.model,
            full_ids,
            self.config.max_answer_tokens,
            temperature=self.config.temperature,
        )

        answer_text = self.decode(answer_token_ids)
        final_answer = extract_final_answer(answer_text, self.config.answer_trigger)
        if not final_answer:
            # Fall back to the raw answer text
            final_answer = answer_text.strip()

        # Parse reasoning into steps
        raw_step_texts = parse_reasoning_steps(reasoning_text)
        steps: list[ReasoningStep] = []
        for i, step_text in enumerate(raw_step_texts):
            step_toks = list(step_text.encode("utf-8", errors="replace"))
            conf = compute_step_confidence(self.model, step_toks, prompt_ids)
            steps.append(ReasoningStep(step_idx=i, content=step_text, confidence=conf))

        raw_tokens = reasoning_token_ids + answer_token_ids

        return CoTOutput(
            reasoning_steps=steps,
            final_answer=final_answer,
            raw_tokens=raw_tokens,
            n_reasoning_tokens=len(reasoning_token_ids),
        )

    def generate_with_trigger(self, prompt_ids: list[int]) -> CoTOutput:
        """Prepend cot_trigger tokens, then generate."""
        trigger_ids = self.encode(self.config.cot_trigger)
        combined_ids = prompt_ids + trigger_ids
        return self.generate(combined_ids)


class SelfConsistencyDecoder:
    """Sample multiple CoT paths, take majority vote on final answer."""

    def __init__(self, generator: ChainOfThoughtGenerator) -> None:
        self.generator = generator

    def decode(self, prompt_ids: list[int]) -> tuple[str, dict]:
        """Sample n_samples reasoning paths, majority vote on answer.

        Returns (best_answer, stats) where stats has:
            'n_samples': int
            'answer_counts': dict[str, int]
            'confidence': float — fraction agreeing with majority
        """
        n = self.generator.config.n_samples
        answer_counts: dict[str, int] = {}

        for _ in range(n):
            output = self.generator.generate(prompt_ids)
            ans = output.final_answer.strip()
            answer_counts[ans] = answer_counts.get(ans, 0) + 1

        if not answer_counts:
            best_answer = ""
            confidence = 0.0
        else:
            best_answer = max(answer_counts, key=lambda k: answer_counts[k])
            confidence = answer_counts[best_answer] / n

        stats = {
            "n_samples": n,
            "answer_counts": answer_counts,
            "confidence": confidence,
        }
        return best_answer, stats


class CoTEvaluator:
    """Evaluates chain-of-thought quality."""

    def __init__(self, generator: ChainOfThoughtGenerator) -> None:
        self.generator = generator

    def evaluate_step_coherence(self, output: CoTOutput) -> float:
        """Mean confidence across all reasoning steps. Returns float in [0, 1]."""
        if not output.reasoning_steps:
            return 0.0
        return sum(s.confidence for s in output.reasoning_steps) / len(output.reasoning_steps)

    def evaluate_answer_presence(self, output: CoTOutput) -> bool:
        """True if final_answer is non-empty."""
        return bool(output.final_answer.strip())

    def batch_evaluate(self, outputs: list[CoTOutput]) -> dict:
        """Evaluate a batch of CoT outputs.

        Returns dict with: 'mean_coherence', 'answer_presence_rate', 'mean_steps'
        """
        if not outputs:
            return {
                "mean_coherence": 0.0,
                "answer_presence_rate": 0.0,
                "mean_steps": 0.0,
            }

        coherences = [self.evaluate_step_coherence(o) for o in outputs]
        presences = [self.evaluate_answer_presence(o) for o in outputs]
        step_counts = [len(o.reasoning_steps) for o in outputs]

        return {
            "mean_coherence": sum(coherences) / len(coherences),
            "answer_presence_rate": sum(presences) / len(presences),
            "mean_steps": sum(step_counts) / len(step_counts),
        }
