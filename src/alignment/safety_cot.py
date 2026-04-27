"""Safety reasoning via chain-of-thought (inspired by gpt-oss-safeguard).

Instead of outputting a binary safe/unsafe label, the model generates reasoning
about WHY something is safe or unsafe, then gives a verdict.

Protocol:
1. Prompt = "[SAFETY_ANALYSIS]\nContent: {content}\nStep {i}: "
2. Generate reasoning step
3. Repeat for cfg.effort steps
4. Prompt for verdict: "Based on the above analysis, verdict: "
5. Parse SAFE/UNSAFE/UNCERTAIN from generated text
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Enums and configuration dataclasses
# ---------------------------------------------------------------------------


class SafetyVerdict(IntEnum):
    SAFE = 0
    UNSAFE = 1
    UNCERTAIN = 2


class ReasoningEffort(IntEnum):
    LOW = 1  # 1 reasoning step
    MEDIUM = 3  # 3 reasoning steps
    HIGH = 5  # full CoT chain


@dataclass
class SafetyCoTConfig:
    effort: ReasoningEffort = ReasoningEffort.MEDIUM
    max_reasoning_tokens: int = 128  # tokens per reasoning step
    max_verdict_tokens: int = 10  # tokens for final verdict
    verdict_token_safe: str = "SAFE"  # noqa: S105
    verdict_token_unsafe: str = "UNSAFE"  # noqa: S105
    verdict_token_uncertain: str = "UNCERTAIN"  # noqa: S105
    temperature: float = 0.0  # greedy for safety decisions
    safety_threshold: float = 0.5  # probability threshold for UNSAFE verdict


@dataclass
class SafetyCoTResult:
    verdict: SafetyVerdict
    reasoning_steps: list[str]  # list of generated reasoning strings
    unsafe_probability: float  # model's confidence in UNSAFE
    safe_probability: float
    n_reasoning_tokens: int  # total tokens used for reasoning


@dataclass
class SafetyPolicyConfig:
    """Bring-your-own-policy configuration."""

    policy_text: str  # written safety policy to follow
    examples: list[tuple[str, SafetyVerdict]] = None  # few-shot examples


# ---------------------------------------------------------------------------
# Core generation helper
# ---------------------------------------------------------------------------


def _generate(
    model: torch.nn.Module,
    input_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    vocab_size: int,
) -> list[int]:
    """Greedy / temperature-sampled autoregressive generation.

    Returns a list of newly generated token ids (not including the prompt).
    Clips input to model's max_seq_len.
    """
    model.training = False
    model.eval()
    max_seq_len = model.config.max_seq_len
    # Clip prompt to leave room for generated tokens
    max_prompt_len = max(1, max_seq_len - max_new_tokens)
    if len(input_ids) > max_prompt_len:
        input_ids = input_ids[-max_prompt_len:]

    generated: list[int] = []
    current_ids = list(input_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            ids_tensor = torch.tensor([current_ids], dtype=torch.long)
            _, logits, _ = model(ids_tensor)
            # logits: (1, S, V) — take last position
            last_logits = logits[0, -1, :]  # (V,)

            if temperature == 0.0 or temperature < 1e-8:
                next_token = int(last_logits.argmax().item())
            else:
                probs = F.softmax(last_logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())

            generated.append(next_token)
            current_ids.append(next_token)
            # Stay within max_seq_len
            if len(current_ids) >= max_seq_len:
                break

    return generated


def _get_verdict_token_probs(
    model: torch.nn.Module,
    input_ids: list[int],
    safe_token_ids: list[int],
    unsafe_token_ids: list[int],
    uncertain_token_ids: list[int],
    vocab_size: int,
) -> tuple[float, float, float]:
    """Compute normalized probabilities for SAFE/UNSAFE/UNCERTAIN at the next token position."""
    model.eval()
    max_seq_len = model.config.max_seq_len
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[-max_seq_len:]

    with torch.no_grad():
        ids_tensor = torch.tensor([input_ids], dtype=torch.long)
        _, logits, _ = model(ids_tensor)
        last_logits = logits[0, -1, :]  # (V,)
        probs = F.softmax(last_logits, dim=-1)  # (V,)

    def _token_prob(token_ids: list[int]) -> float:
        if not token_ids:
            return 0.0
        # Use first subtoken as representative
        tid = token_ids[0] % vocab_size
        return float(probs[tid].item())

    safe_p = _token_prob(safe_token_ids)
    unsafe_p = _token_prob(unsafe_token_ids)
    uncertain_p = _token_prob(uncertain_token_ids)

    total = safe_p + unsafe_p + uncertain_p
    if total < 1e-12:
        return 1.0 / 3, 1.0 / 3, 1.0 / 3
    return safe_p / total, unsafe_p / total, uncertain_p / total


# ---------------------------------------------------------------------------
# SafetyCoTReasoner
# ---------------------------------------------------------------------------


class SafetyCoTReasoner:
    """Generates chain-of-thought safety reasoning then gives a verdict.

    Protocol:
    1. Prompt = "[SAFETY_ANALYSIS]\nContent: {content}\nStep {i}: "
    2. Generate reasoning step
    3. Repeat for cfg.effort steps
    4. Prompt for verdict: "Based on the above analysis, verdict: "
    5. Parse SAFE/UNSAFE/UNCERTAIN from generated text
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
        cfg: SafetyCoTConfig,
    ) -> None:
        self.model = model
        self.encode = tokenizer_encode
        self.decode = tokenizer_decode
        self.cfg = cfg
        self._vocab_size = model.config.vocab_size

    def build_reasoning_prompt(self, content: str, step: int, prior_steps: list[str]) -> str:
        """Build the prompt for reasoning step `step` given prior reasoning."""
        lines = ["[SAFETY_ANALYSIS]", f"Content: {content}"]
        for i, prev in enumerate(prior_steps, start=1):
            lines.append(f"Step {i}: {prev}")
        lines.append(f"Step {step}: ")
        return "\n".join(lines)

    def build_verdict_prompt(self, content: str, reasoning_steps: list[str]) -> str:
        """Build the prompt asking for SAFE/UNSAFE/UNCERTAIN verdict."""
        lines = ["[SAFETY_ANALYSIS]", f"Content: {content}"]
        for i, step_text in enumerate(reasoning_steps, start=1):
            lines.append(f"Step {i}: {step_text}")
        lines.append("Based on the above analysis, verdict: ")
        return "\n".join(lines)

    def extract_verdict(self, generated_text: str) -> tuple[SafetyVerdict, float, float]:
        """Parse verdict and probabilities from generated text.

        Returns (verdict, unsafe_prob, safe_prob).
        """
        text_upper = generated_text.upper()

        # Check for each verdict keyword — UNSAFE must be checked before SAFE
        # since the string "UNSAFE" contains "SAFE"
        has_unsafe = self.cfg.verdict_token_unsafe.upper() in text_upper
        has_safe = (not has_unsafe) and (self.cfg.verdict_token_safe.upper() in text_upper)
        has_uncertain = self.cfg.verdict_token_uncertain.upper() in text_upper

        if has_unsafe:
            verdict = SafetyVerdict.UNSAFE
            unsafe_prob = 0.9
            safe_prob = 0.05
        elif has_safe:
            verdict = SafetyVerdict.SAFE
            unsafe_prob = 0.05
            safe_prob = 0.9
        elif has_uncertain:
            verdict = SafetyVerdict.UNCERTAIN
            unsafe_prob = 0.4
            safe_prob = 0.4
        else:
            verdict = SafetyVerdict.UNCERTAIN
            unsafe_prob = 1.0 / 3
            safe_prob = 1.0 / 3

        return verdict, unsafe_prob, safe_prob

    def _generate_text(self, prompt: str, max_new_tokens: int) -> tuple[str, int]:
        """Tokenize prompt, generate, decode. Returns (generated_text, n_tokens)."""
        input_ids = self.encode(prompt)
        generated_ids = _generate(
            self.model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=self.cfg.temperature,
            vocab_size=self._vocab_size,
        )
        return self.decode(generated_ids), len(generated_ids)

    def analyze(self, content: str) -> SafetyCoTResult:
        """Full analysis: reasoning chain + verdict."""
        n_steps = int(self.cfg.effort)
        reasoning_steps: list[str] = []
        total_reasoning_tokens = 0

        for step_i in range(1, n_steps + 1):
            prompt = self.build_reasoning_prompt(content, step_i, reasoning_steps)
            step_text, n_tokens = self._generate_text(
                prompt, max_new_tokens=self.cfg.max_reasoning_tokens
            )
            reasoning_steps.append(step_text)
            total_reasoning_tokens += n_tokens

        # Generate verdict
        verdict_prompt = self.build_verdict_prompt(content, reasoning_steps)
        verdict_text, _ = self._generate_text(
            verdict_prompt, max_new_tokens=self.cfg.max_verdict_tokens
        )

        # Get token-level probabilities for the verdict tokens
        verdict_input_ids = self.encode(verdict_prompt)
        safe_ids = self.encode(self.cfg.verdict_token_safe)
        unsafe_ids = self.encode(self.cfg.verdict_token_unsafe)
        uncertain_ids = self.encode(self.cfg.verdict_token_uncertain)

        safe_p, unsafe_p, uncertain_p = _get_verdict_token_probs(
            self.model,
            verdict_input_ids,
            safe_ids,
            unsafe_ids,
            uncertain_ids,
            self._vocab_size,
        )

        # Parse text verdict
        text_verdict, _text_unsafe_prob, _text_safe_prob = self.extract_verdict(verdict_text)

        # Apply threshold to determine final verdict
        if unsafe_p >= self.cfg.safety_threshold:
            verdict = SafetyVerdict.UNSAFE
        elif text_verdict != SafetyVerdict.UNCERTAIN:
            verdict = text_verdict
        else:
            if unsafe_p > safe_p:
                verdict = SafetyVerdict.UNSAFE
            elif safe_p > unsafe_p:
                verdict = SafetyVerdict.SAFE
            else:
                verdict = SafetyVerdict.UNCERTAIN

        return SafetyCoTResult(
            verdict=verdict,
            reasoning_steps=reasoning_steps,
            unsafe_probability=float(unsafe_p),
            safe_probability=float(safe_p),
            n_reasoning_tokens=total_reasoning_tokens,
        )

    def analyze_batch(self, contents: list[str]) -> list[SafetyCoTResult]:
        """Analyze multiple pieces of content."""
        return [self.analyze(c) for c in contents]


# ---------------------------------------------------------------------------
# PolicyGuidedReasoner
# ---------------------------------------------------------------------------


class PolicyGuidedReasoner(SafetyCoTReasoner):
    """Extends SafetyCoTReasoner with explicit policy context."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
        cot_cfg: SafetyCoTConfig,
        policy_cfg: SafetyPolicyConfig,
    ) -> None:
        super().__init__(model, tokenizer_encode, tokenizer_decode, cot_cfg)
        self.policy_cfg = policy_cfg

    def build_reasoning_prompt(self, content: str, step: int, prior_steps: list[str]) -> str:
        """Prepend policy_text to the reasoning prompt."""
        base_prompt = super().build_reasoning_prompt(content, step, prior_steps)
        # Prepend policy section
        policy_section = f"[POLICY]\n{self.policy_cfg.policy_text}\n"

        # Add few-shot examples if provided
        if self.policy_cfg.examples:
            examples_lines = ["[EXAMPLES]"]
            for ex_content, ex_verdict in self.policy_cfg.examples:
                examples_lines.append(f"Content: {ex_content} -> {ex_verdict.name}")
            policy_section += "\n".join(examples_lines) + "\n"

        return policy_section + base_prompt


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------


def calibrate_safety_threshold(
    reasoner: SafetyCoTReasoner,
    labeled_examples: list[tuple[str, SafetyVerdict]],
    target_recall: float = 0.95,
) -> float:
    """Run reasoner on examples, find threshold that achieves target_recall.

    Returns optimal safety_threshold (float in [0, 1]).
    """
    if not labeled_examples:
        return 0.5

    # Run all examples through reasoner and collect unsafe_probabilities
    results = []
    for content, true_verdict in labeled_examples:
        result = reasoner.analyze(content)
        results.append((result.unsafe_probability, true_verdict))

    # Get all positive (UNSAFE) examples
    positive_probs = [p for p, v in results if v == SafetyVerdict.UNSAFE]
    if not positive_probs:
        return 0.5

    # Sort candidate thresholds (ascending)
    all_probs = sorted({p for p, _ in results})

    best_threshold = 0.0
    best_recall = 0.0

    for threshold in all_probs:
        # Recall = TP / (TP + FN)
        tp = sum(1 for p in positive_probs if p >= threshold)
        recall = tp / len(positive_probs)

        if recall >= target_recall:
            # This threshold still achieves target recall
            best_threshold = threshold
            best_recall = recall

    # If no threshold achieves target_recall, use 0.0 (catch everything)
    if best_recall < target_recall:
        best_threshold = 0.0

    # Clamp to [0, 1]
    return float(max(0.0, min(1.0, best_threshold)))
