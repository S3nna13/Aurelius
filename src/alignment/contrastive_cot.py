"""Aurelius — Contrastive Chain-of-Thought (CCoT) alignment.

Train models to prefer reasoning chains that lead to correct answers over chains
that lead to incorrect answers. Uses DPO-style contrastive loss on
(question, correct_cot, wrong_cot) triples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration and data dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CCoTConfig:
    """Configuration for Contrastive Chain-of-Thought training."""
    beta: float = 0.1           # KL regularization temperature
    margin: float = 0.5         # minimum preference margin
    max_seq_len: int = 512      # maximum sequence length
    cot_weight: float = 1.0     # weight for CoT loss vs answer loss
    use_sft_loss: bool = True   # also train on correct CoT with SFT


@dataclass
class CCoTExample:
    """A single contrastive CoT training example."""
    question: str
    correct_cot: str
    correct_answer: str
    wrong_cot: str
    wrong_answer: str


# ---------------------------------------------------------------------------
# Core utility: sequence log-prob
# ---------------------------------------------------------------------------

def compute_sequence_log_prob(
    model: nn.Module,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute mean log-prob of response tokens.

    Args:
        model: Forward signature: (loss, logits, present_key_values) = model(input_ids).
        input_ids: Shape (B, T).
        response_mask: Shape (B, T) boolean or 0/1 int tensor; True/1 for response tokens.

    Returns:
        Shape (B,) — mean log-prob per sequence over masked positions.
        Returns 0 for sequences where no token is masked.
    """
    _, logits, _ = model(input_ids)  # (B, T, vocab_size)

    # Shift: position t predicts token t+1
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, vocab_size)

    # Gather log-prob of actual next token
    targets = input_ids[:, 1:].unsqueeze(-1)               # (B, T-1, 1)
    token_lp = log_probs.gather(2, targets).squeeze(-1)    # (B, T-1)

    # Shift mask to align with next-token predictions
    mask = response_mask[:, 1:].float()                    # (B, T-1)

    # Mean over masked positions; guard against all-zero mask
    masked_sum = (token_lp * mask).sum(dim=-1)             # (B,)
    mask_count = mask.sum(dim=-1).clamp(min=1.0)           # (B,)
    return masked_sum / mask_count                         # (B,)


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _build_sequence_with_mask(
    question: str,
    cot: str,
    answer: str,
    tokenizer_encode: Callable[[str], list[int]],
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize question+cot+answer and build a response mask.

    The question portion is masked out (False); cot+answer tokens are the response (True).

    Returns:
        input_ids: (1, L)
        response_mask: (1, L) bool — True for cot+answer tokens
    """
    q_ids = tokenizer_encode(question)
    r_ids = tokenizer_encode(cot + answer)

    full_ids = q_ids + r_ids
    mask_vals = [False] * len(q_ids) + [True] * len(r_ids)

    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
        mask_vals = mask_vals[:max_seq_len]

    input_ids = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0)       # (1, L)
    response_mask = torch.tensor(mask_vals, dtype=torch.bool).unsqueeze(0)  # (1, L)
    return input_ids, response_mask


# ---------------------------------------------------------------------------
# Main loss function
# ---------------------------------------------------------------------------

def compute_ccot_loss(
    model: nn.Module,
    ref_model: nn.Module,
    examples: list[CCoTExample],
    tokenizer_encode: Callable[[str], list[int]],
    config: CCoTConfig,
) -> tuple[torch.Tensor, dict]:
    """Compute the CCoT loss over a list of examples.

    For each example:
      - Tokenize question + correct_cot + correct_answer -> correct sequence
      - Tokenize question + wrong_cot + wrong_answer -> wrong sequence
      - Compute log-probs under policy and reference
      - DPO-style contrastive loss with margin
      - Optionally add SFT loss on correct CoT tokens

    Returns:
        (total_loss, metrics_dict) where metrics_dict has keys:
        contrastive_loss, sft_loss, margin_achieved, accuracy
    """
    contrastive_losses: list[torch.Tensor] = []
    sft_losses: list[torch.Tensor] = []
    n_correct = 0

    for ex in examples:
        # Build sequences
        correct_ids, correct_mask = _build_sequence_with_mask(
            ex.question, ex.correct_cot, ex.correct_answer,
            tokenizer_encode, config.max_seq_len,
        )
        wrong_ids, wrong_mask = _build_sequence_with_mask(
            ex.question, ex.wrong_cot, ex.wrong_answer,
            tokenizer_encode, config.max_seq_len,
        )

        # Policy log-probs
        log_pi_correct = compute_sequence_log_prob(model, correct_ids, correct_mask)  # (1,)
        log_pi_wrong = compute_sequence_log_prob(model, wrong_ids, wrong_mask)         # (1,)

        # Reference log-probs (no gradient)
        with torch.no_grad():
            log_ref_correct = compute_sequence_log_prob(ref_model, correct_ids, correct_mask)
            log_ref_wrong = compute_sequence_log_prob(ref_model, wrong_ids, wrong_mask)

        # DPO-style contrastive loss with margin
        reward_margin = config.beta * (
            (log_pi_correct - log_ref_correct) - (log_pi_wrong - log_ref_wrong)
        ) - config.margin

        c_loss = -F.logsigmoid(reward_margin).mean()
        contrastive_losses.append(c_loss)

        # Track accuracy: does policy prefer correct over wrong?
        with torch.no_grad():
            if (log_pi_correct - log_pi_wrong).item() > 0:
                n_correct += 1

        # SFT loss on correct CoT tokens
        if config.use_sft_loss:
            _, logits, _ = model(correct_ids)             # (1, L, vocab_size)
            shift_logits = logits[:, :-1, :].contiguous()  # (1, L-1, vocab_size)
            shift_labels = correct_ids[:, 1:].contiguous() # (1, L-1)
            shift_mask = correct_mask[:, 1:].float()       # (1, L-1)

            token_losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            )  # (L-1,)

            mask_flat = shift_mask.view(-1)
            denom = mask_flat.sum().clamp(min=1.0)
            sft_loss = (token_losses * mask_flat).sum() / denom
            sft_losses.append(sft_loss)

    # Aggregate
    contrastive_loss = torch.stack(contrastive_losses).mean()

    if config.use_sft_loss and sft_losses:
        sft_loss_val = torch.stack(sft_losses).mean()
        total_loss = contrastive_loss + config.cot_weight * sft_loss_val
        sft_loss_item = sft_loss_val.item()
    else:
        total_loss = contrastive_loss
        sft_loss_item = 0.0

    accuracy = n_correct / len(examples) if examples else 0.0

    # Compute margin_achieved: mean of (log_pi_correct - log_pi_wrong)
    with torch.no_grad():
        margin_vals: list[float] = []
        for ex in examples:
            c_ids, c_mask = _build_sequence_with_mask(
                ex.question, ex.correct_cot, ex.correct_answer,
                tokenizer_encode, config.max_seq_len,
            )
            w_ids, w_mask = _build_sequence_with_mask(
                ex.question, ex.wrong_cot, ex.wrong_answer,
                tokenizer_encode, config.max_seq_len,
            )
            lpc = compute_sequence_log_prob(model, c_ids, c_mask)
            lpw = compute_sequence_log_prob(model, w_ids, w_mask)
            margin_vals.append((lpc - lpw).item())
        margin_achieved = sum(margin_vals) / len(margin_vals) if margin_vals else 0.0

    metrics = {
        "contrastive_loss": contrastive_loss.item(),
        "sft_loss": sft_loss_item,
        "margin_achieved": margin_achieved,
        "accuracy": accuracy,
    }

    return total_loss, metrics


# ---------------------------------------------------------------------------
# CoT Quality Scorer
# ---------------------------------------------------------------------------

def _generate_tokens(
    model: nn.Module,
    input_ids: list[int],
    max_new_tokens: int,
    vocab_size: int,
) -> list[int]:
    """Greedy autoregressive generation. Returns newly generated token ids."""
    generated: list[int] = []
    current_ids = list(input_ids)

    for _ in range(max_new_tokens):
        t = torch.tensor([current_ids], dtype=torch.long)
        with torch.no_grad():
            _, logits, _ = model(t)
        next_token = int(logits[0, -1, :].argmax(dim=-1).item())
        generated.append(next_token)
        current_ids.append(next_token)

    return generated


class CoTQualityScorer:
    """Score and rank chain-of-thought reasoning quality using model perplexity."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
    ) -> None:
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode

    def score_cot(self, question: str, cot: str, answer: str) -> float:
        """Score CoT quality by (negative) normalized perplexity.

        Lower perplexity -> higher score. Returns value in [0, 1].
        Score = 1 / (1 + perplexity), where perplexity = exp(-mean_log_prob).
        """
        text = question + cot + answer
        ids = self.tokenizer_encode(text)
        if len(ids) < 2:
            return 0.0

        input_ids = torch.tensor([ids], dtype=torch.long)
        mask = torch.ones(1, len(ids), dtype=torch.bool)

        with torch.no_grad():
            mean_lp = compute_sequence_log_prob(self.model, input_ids, mask)  # (1,)

        perplexity = float(torch.exp(-mean_lp).item())
        score = 1.0 / (1.0 + perplexity)
        return float(max(0.0, min(1.0, score)))

    def rank_cots(
        self,
        question: str,
        cots: list[tuple[str, str]],
    ) -> list[tuple[float, str, str]]:
        """Rank (cot, answer) pairs by score. Returns sorted list (highest score first).

        Returns:
            List of (score, cot, answer) sorted descending by score.
        """
        scored = [
            (self.score_cot(question, cot, answer), cot, answer)
            for cot, answer in cots
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def generate_contrastive_pairs(
        self,
        question: str,
        n_pairs: int = 4,
    ) -> list[CCoTExample]:
        """Generate n_pairs CoTs by sampling, then split into better/worse pairs by score.

        Returns list of CCoTExample with higher-scoring CoT as correct, lower as wrong.
        """
        vocab_size = 256  # works with small test models
        max_new_tokens = 8

        q_ids = self.tokenizer_encode(question)

        # Generate 2 * n_pairs candidates
        n_candidates = max(2, n_pairs * 2)
        candidates: list[tuple[str, str]] = []
        for i in range(n_candidates):
            varied_ids = q_ids + [i % vocab_size]
            gen_ids = _generate_tokens(self.model, varied_ids, max_new_tokens, vocab_size)
            half = max(1, len(gen_ids) // 2)
            cot_text = self.tokenizer_decode(gen_ids[:half])
            answer_text = self.tokenizer_decode(gen_ids[half:])
            candidates.append((cot_text, answer_text))

        # Score and sort
        ranked = self.rank_cots(question, candidates)

        # Pair top half (better) with bottom half (worse)
        n = len(ranked)
        half = n // 2
        better = ranked[:half]
        worse = ranked[half:]

        examples: list[CCoTExample] = []
        for i in range(min(n_pairs, half, len(worse))):
            _, good_cot, good_ans = better[i]
            _, bad_cot, bad_ans = worse[i]
            examples.append(
                CCoTExample(
                    question=question,
                    correct_cot=good_cot,
                    correct_answer=good_ans,
                    wrong_cot=bad_cot,
                    wrong_answer=bad_ans,
                )
            )

        return examples


# ---------------------------------------------------------------------------
# CCoT Trainer
# ---------------------------------------------------------------------------

class CCoTTrainer:
    """Trainer for Contrastive Chain-of-Thought alignment."""

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        config: CCoTConfig,
        optimizer: torch.optim.Optimizer,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.optimizer = optimizer
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode

        # Freeze reference model
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    def train_step(self, examples: list[CCoTExample]) -> dict:
        """Compute CCoT loss, backward pass, optimizer step. Returns metrics dict."""
        self.model.train()
        self.optimizer.zero_grad()

        loss, metrics = compute_ccot_loss(
            self.model,
            self.ref_model,
            examples,
            self.tokenizer_encode,
            self.config,
        )

        loss.backward()
        self.optimizer.step()

        metrics["loss"] = loss.item()
        return metrics

    def evaluate(self, examples: list[CCoTExample]) -> dict:
        """Compute CCoT metrics without updating model weights."""
        self.model.eval()
        with torch.no_grad():
            loss, metrics = compute_ccot_loss(
                self.model,
                self.ref_model,
                examples,
                self.tokenizer_encode,
                self.config,
            )
        metrics["loss"] = loss.item()
        return metrics
