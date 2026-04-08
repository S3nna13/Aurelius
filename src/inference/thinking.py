"""Think-before-answer: extended chain-of-thought with budget forcing.

Generates a scratchpad "thinking" sequence, then the final answer.
Optionally uses a process reward model to score each thinking step.

This enables reasoning at inference time (like DeepSeek-R1/o1 style).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn as nn


@dataclass
class ThinkingConfig:
    think_token_budget: int = 512     # max tokens for scratchpad
    answer_token_budget: int = 256    # max tokens for final answer
    think_start_token: int | None = None  # token to start thinking (None = no special token)
    think_end_token: int | None = None    # token to end thinking
    answer_start_token: int | None = None  # token to start final answer
    temperature: float = 1.0
    top_p: float = 0.9
    eos_token_id: int | None = None
    stop_thinking_on_eos: bool = True  # if eos generated during thinking, stop thinking early


@dataclass
class ThinkingResult:
    """Result of think-before-answer generation."""
    thinking_ids: torch.Tensor    # (S_think,) scratchpad tokens
    answer_ids: torch.Tensor      # (S_ans,) final answer tokens
    full_ids: torch.Tensor        # (S_think + S_ans,) full sequence
    thinking_tokens_used: int
    answer_tokens_used: int

    @property
    def total_tokens(self) -> int:
        return self.thinking_tokens_used + self.answer_tokens_used


def generate_thinking(
    model: nn.Module,
    prompt_ids: torch.Tensor,    # (1, S_prompt)
    cfg: ThinkingConfig,
) -> torch.Tensor:
    """Generate the thinking (scratchpad) tokens.

    Uses model.generate() with think_token_budget as max_new_tokens.
    Returns (S_think,) thinking token ids.
    """
    # Normalise to (1, S_prompt)
    if prompt_ids.dim() == 1:
        input_ids = prompt_ids.unsqueeze(0)
    else:
        input_ids = prompt_ids

    prompt_len = input_ids.shape[1]

    # Optionally prepend think_start_token to the generation context
    gen_input = input_ids
    if cfg.think_start_token is not None:
        start = torch.tensor([[cfg.think_start_token]], dtype=input_ids.dtype)
        gen_input = torch.cat([input_ids, start], dim=1)

    with torch.no_grad():
        output = model.generate(
            gen_input,
            max_new_tokens=cfg.think_token_budget,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            eos_token_id=cfg.eos_token_id if cfg.stop_thinking_on_eos else None,
        )  # (1, gen_input_len + generated_len)

    thinking_ids = output[0, gen_input.shape[1]:]  # (S_think,)

    # Trim at think_end_token if configured
    if cfg.think_end_token is not None:
        matches = (thinking_ids == cfg.think_end_token).nonzero(as_tuple=False)
        if matches.numel() > 0:
            end_pos = matches[0].item()
            thinking_ids = thinking_ids[:end_pos]

    return thinking_ids


def generate_answer(
    model: nn.Module,
    prompt_ids: torch.Tensor,    # (1, S_prompt)
    thinking_ids: torch.Tensor,  # (S_think,)
    cfg: ThinkingConfig,
) -> torch.Tensor:
    """Generate the final answer given prompt + thinking tokens.

    Concatenates prompt + thinking as context, then generates the answer.
    Returns (S_ans,) answer token ids.
    """
    # Normalise prompt to (1, S_prompt)
    if prompt_ids.dim() == 1:
        prompt_2d = prompt_ids.unsqueeze(0)
    else:
        prompt_2d = prompt_ids

    # Build context: prompt + thinking
    thinking_2d = thinking_ids.view(1, -1)  # (1, S_think)

    # Optionally add think_end_token and answer_start_token as separators
    parts = [prompt_2d, thinking_2d]
    if cfg.think_end_token is not None:
        end_tok = torch.tensor([[cfg.think_end_token]], dtype=prompt_2d.dtype)
        parts.append(end_tok)
    if cfg.answer_start_token is not None:
        ans_tok = torch.tensor([[cfg.answer_start_token]], dtype=prompt_2d.dtype)
        parts.append(ans_tok)

    context = torch.cat(parts, dim=1)  # (1, context_len)
    context_len = context.shape[1]

    with torch.no_grad():
        output = model.generate(
            context,
            max_new_tokens=cfg.answer_token_budget,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            eos_token_id=cfg.eos_token_id,
        )  # (1, context_len + generated_len)

    answer_ids = output[0, context_len:]  # (S_ans,)
    return answer_ids


def think_and_answer(
    model: nn.Module,
    prompt_ids: torch.Tensor,   # (1, S_prompt) or (S_prompt,)
    cfg: ThinkingConfig | None = None,
) -> ThinkingResult:
    """Full think-then-answer pipeline.

    1. Generate thinking tokens
    2. Generate answer conditioned on prompt + thinking
    3. Return ThinkingResult
    """
    cfg = cfg or ThinkingConfig()

    # Normalise to (1, S_prompt)
    if prompt_ids.dim() == 1:
        prompt_2d = prompt_ids.unsqueeze(0)
    else:
        prompt_2d = prompt_ids

    # Step 1: generate thinking
    thinking_ids = generate_thinking(model, prompt_2d, cfg)

    # Step 2: generate answer conditioned on prompt + thinking
    answer_ids = generate_answer(model, prompt_2d, thinking_ids, cfg)

    # Step 3: assemble full sequence
    full_ids = torch.cat([thinking_ids.view(-1), answer_ids.view(-1)], dim=0)

    return ThinkingResult(
        thinking_ids=thinking_ids.view(-1),
        answer_ids=answer_ids.view(-1),
        full_ids=full_ids,
        thinking_tokens_used=thinking_ids.numel(),
        answer_tokens_used=answer_ids.numel(),
    )


class ThinkingReranker:
    """Generate multiple think-chains and pick the best answer.

    For each of n_candidates:
    1. Generate thinking -> answer
    2. Score answer with reward_fn
    3. Return best (thinking, answer) pair
    """

    def __init__(
        self,
        model: nn.Module,
        n_candidates: int = 4,
        reward_fn=None,   # callable(answer_ids: Tensor) -> float, or None (use log-prob)
        cfg: ThinkingConfig | None = None,
    ) -> None:
        self.model = model
        self.n_candidates = n_candidates
        self.reward_fn = reward_fn
        self.cfg = cfg or ThinkingConfig()

    def _score_answer(
        self,
        prompt_ids: torch.Tensor,  # (1, S_prompt)
        thinking_ids: torch.Tensor,  # (S_think,)
        answer_ids: torch.Tensor,    # (S_ans,)
    ) -> float:
        """Score the answer using reward_fn or model log-prob."""
        if answer_ids.numel() == 0:
            return float("-inf")

        if self.reward_fn is not None:
            return float(self.reward_fn(answer_ids))

        # Default: use model log-prob of answer given prompt + thinking as context
        import torch.nn.functional as F

        prompt_1d = prompt_ids.view(-1)
        thinking_1d = thinking_ids.view(-1)
        answer_1d = answer_ids.view(-1)

        context = torch.cat([prompt_1d, thinking_1d], dim=0)
        full_ids = torch.cat([context, answer_1d], dim=0).unsqueeze(0)  # (1, S)
        context_len = context.shape[0]
        ans_len = answer_1d.shape[0]

        with torch.no_grad():
            _, logits, _ = self.model(full_ids)  # (1, S, vocab)

        # Logits at [context_len-1 : context_len + ans_len - 1] predict answer tokens
        gen_logits = logits[0, context_len - 1: context_len + ans_len - 1, :]
        log_probs = F.log_softmax(gen_logits, dim=-1)
        token_log_probs = log_probs[torch.arange(ans_len), answer_1d]
        return token_log_probs.mean().item()

    def generate_best(
        self,
        prompt_ids: torch.Tensor,
    ) -> ThinkingResult:
        """Generate n_candidates think-answer pairs, return best by reward."""
        # Normalise to (1, S_prompt)
        if prompt_ids.dim() == 1:
            prompt_2d = prompt_ids.unsqueeze(0)
        else:
            prompt_2d = prompt_ids

        candidates: list[ThinkingResult] = []
        scores: list[float] = []

        for _ in range(self.n_candidates):
            result = think_and_answer(self.model, prompt_2d, self.cfg)
            score = self._score_answer(prompt_2d, result.thinking_ids, result.answer_ids)
            candidates.append(result)
            scores.append(score)

        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        return candidates[best_idx]
