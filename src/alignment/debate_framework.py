"""Self-Play Debate framework: multiple debater agents argue positions, judge decides."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DebateConfig:
    """Configuration for the debate framework."""

    d_model: int = 32
    vocab_size: int = 64
    n_layers: int = 2
    n_turns: int = 2
    max_new_tokens: int = 8
    lr_debaters: float = 1e-4
    lr_judge: float = 1e-3


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------


class _CausalSelfAttention(nn.Module):
    """Minimal multi-head causal self-attention."""

    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # [B, T, 3*C]
        q, k, v = qkv.split(C, dim=-1)  # each [B, T, C]

        # reshape to [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale  # [B, heads, T, T]

        # causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # [B, heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        # Use 2 heads when d_model < 8, else 4
        n_heads = 2 if d_model < 8 else 4
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_heads=n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# DebaterModel
# ---------------------------------------------------------------------------


class DebaterModel(nn.Module):
    """Simple autoregressive transformer debater."""

    def __init__(self, d_model: int, vocab_size: int, n_layers: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([_TransformerBlock(d_model) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: [B, T] integer token ids

        Returns:
            logits: [B, T, vocab_size]
        """
        x = self.embed(input_ids)  # [B, T, d_model]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)  # [B, T, vocab_size]

    @torch.no_grad()
    def generate_argument(
        self, context_ids: torch.Tensor, max_new: int
    ) -> torch.Tensor:
        """Greedy autoregressive decoding.

        Args:
            context_ids: [B, T] prompt token ids
            max_new: number of new tokens to generate

        Returns:
            [B, T + max_new] token ids
        """
        ids = context_ids
        for _ in range(max_new):
            logits = self.forward(ids)  # [B, T_cur, vocab]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
            ids = torch.cat([ids, next_token], dim=1)
        return ids


# ---------------------------------------------------------------------------
# JudgeModel
# ---------------------------------------------------------------------------


class JudgeModel(nn.Module):
    """Transformer encoder that assigns support/oppose verdicts."""

    def __init__(self, d_model: int, vocab_size: int, n_layers: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([_TransformerBlock(d_model) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        # Pool over sequence then project to 2 classes (support / oppose)
        self.verdict_head = nn.Linear(d_model, 2, bias=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: [B, T]

        Returns:
            verdict_logits: [B, 2]  (support=0, oppose=1)
        """
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        pooled = x.mean(dim=1)  # [B, d_model]
        return self.verdict_head(pooled)  # [B, 2]

    def score_argument(self, argument_ids: torch.Tensor) -> torch.Tensor:
        """Return softmax probability of 'support' class.

        Args:
            argument_ids: [B, T]

        Returns:
            scores: [B] in (0, 1)
        """
        verdict_logits = self.forward(argument_ids)  # [B, 2]
        probs = F.softmax(verdict_logits, dim=-1)  # [B, 2]
        return probs[:, 0]  # probability of "support"


# ---------------------------------------------------------------------------
# DebateResult
# ---------------------------------------------------------------------------


@dataclass
class DebateResult:
    """Result of a single debate round."""

    transcript_ids: torch.Tensor  # [B, T_full]
    a_score: float
    b_score: float
    winner: str  # "a", "b", or "tie"


# ---------------------------------------------------------------------------
# DebateRound
# ---------------------------------------------------------------------------


class DebateRound:
    """Orchestrates a debate between debater_a, debater_b, judged by judge."""

    def __init__(
        self,
        debater_a: DebaterModel,
        debater_b: DebaterModel,
        judge: JudgeModel,
    ) -> None:
        self.debater_a = debater_a
        self.debater_b = debater_b
        self.judge = judge

    def run_round(
        self, question_ids: torch.Tensor, n_turns: int = 2, max_new: int = 8
    ) -> DebateResult:
        """Run a full debate round.

        Debaters alternate generating arguments appended to a shared transcript.
        The judge scores the final transcript.

        Args:
            question_ids: [B, T] initial context
            n_turns: total number of turns (debater_a and debater_b alternate)
            max_new: tokens each debater generates per turn

        Returns:
            DebateResult
        """
        transcript = question_ids
        for turn_idx in range(n_turns):
            debater = self.debater_a if turn_idx % 2 == 0 else self.debater_b
            transcript = debater.generate_argument(transcript, max_new=max_new)

        # Judge scores the full transcript
        a_score_tensor = self.judge.score_argument(transcript)  # [B]
        a_score = float(a_score_tensor.mean().item())
        b_score = 1.0 - a_score

        if a_score > b_score:
            winner = "a"
        elif b_score > a_score:
            winner = "b"
        else:
            winner = "tie"

        return DebateResult(
            transcript_ids=transcript,
            a_score=a_score,
            b_score=b_score,
            winner=winner,
        )


# ---------------------------------------------------------------------------
# SelfPlayDebateTrainer
# ---------------------------------------------------------------------------


class SelfPlayDebateTrainer:
    """Trains debaters via REINFORCE and judge via BCE."""

    def __init__(
        self,
        debater_a: DebaterModel,
        debater_b: DebaterModel,
        judge: JudgeModel,
        lr_debaters: float = 1e-4,
        lr_judge: float = 1e-3,
    ) -> None:
        self.debater_a = debater_a
        self.debater_b = debater_b
        self.judge = judge
        self.opt_a = torch.optim.Adam(debater_a.parameters(), lr=lr_debaters)
        self.opt_b = torch.optim.Adam(debater_b.parameters(), lr=lr_debaters)
        self.opt_judge = torch.optim.Adam(judge.parameters(), lr=lr_judge)
        self._debate_round = DebateRound(debater_a, debater_b, judge)

    def judge_step(
        self, transcript_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Train the judge to classify which debater is correct.

        Args:
            transcript_ids: [B, T]
            labels: [B] binary float (1 = debater_a correct, 0 = debater_b correct)

        Returns:
            scalar BCE loss
        """
        self.opt_judge.zero_grad()
        scores = self.judge.score_argument(transcript_ids)  # [B] in (0,1)
        loss = F.binary_cross_entropy(scores, labels.float())
        loss.backward()
        self.opt_judge.step()
        return loss.detach()

    def debater_step(
        self, question_ids: torch.Tensor, reward_signal: torch.Tensor, max_new: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """REINFORCE update for both debaters.

        debater_a maximises expected judge score (reward_signal).
        debater_b minimises it (adversarial: reward = 1 - signal).

        The log-probabilities are computed from a forward pass over the
        generated transcript tokens.

        Args:
            question_ids: [B, T] prompt
            reward_signal: [B] float reward in [0,1]
            max_new: tokens each debater generates

        Returns:
            (loss_a, loss_b) scalar tensors
        """
        # ---- debater_a ----
        self.opt_a.zero_grad()
        # Generate from debater_a with gradients
        gen_a = question_ids
        for _ in range(max_new):
            logits = self.debater_a(gen_a)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_a = torch.cat([gen_a, next_tok.detach()], dim=1)

        # Log-prob of generated tokens under debater_a
        q_len = question_ids.shape[1]
        logits_a = self.debater_a(gen_a[:, :-1])  # [B, T+max_new-1, vocab]
        gen_tokens_a = gen_a[:, q_len:]  # [B, max_new]
        log_probs_a = F.log_softmax(logits_a[:, q_len - 1:, :], dim=-1)  # [B, max_new, vocab]
        selected_log_probs_a = log_probs_a.gather(
            2, gen_tokens_a.unsqueeze(-1)
        ).squeeze(-1)  # [B, max_new]
        sum_log_prob_a = selected_log_probs_a.sum(dim=1)  # [B]
        # Maximise: loss = -reward * log_prob
        loss_a = -(reward_signal * sum_log_prob_a).mean()
        loss_a.backward()
        self.opt_a.step()

        # ---- debater_b (adversarial) ----
        self.opt_b.zero_grad()
        gen_b = question_ids
        for _ in range(max_new):
            logits_b = self.debater_b(gen_b)
            next_tok_b = logits_b[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_b = torch.cat([gen_b, next_tok_b.detach()], dim=1)

        logits_b_full = self.debater_b(gen_b[:, :-1])
        gen_tokens_b = gen_b[:, q_len:]
        log_probs_b = F.log_softmax(logits_b_full[:, q_len - 1:, :], dim=-1)
        selected_log_probs_b = log_probs_b.gather(
            2, gen_tokens_b.unsqueeze(-1)
        ).squeeze(-1)
        sum_log_prob_b = selected_log_probs_b.sum(dim=1)
        # Adversarial: b tries to win *against* a, so reward for b is inverted
        reward_b = 1.0 - reward_signal
        loss_b = -(reward_b * sum_log_prob_b).mean()
        loss_b.backward()
        self.opt_b.step()

        return loss_a.detach(), loss_b.detach()

    def self_play_step(
        self,
        question_ids: torch.Tensor,
        n_turns: int = 2,
        max_new: int = 8,
    ) -> dict:
        """Run one full self-play step: debate → judge → update all models.

        Args:
            question_ids: [B, T]
            n_turns: debate turns
            max_new: tokens per turn

        Returns:
            dict with keys: "judge_loss", "loss_a", "loss_b", "winner"
        """
        # Run debate (no grad, just for judge training data)
        with torch.no_grad():
            result = self._debate_round.run_round(
                question_ids, n_turns=n_turns, max_new=max_new
            )

        transcript = result.transcript_ids

        # Judge training: label = 1 if a won, 0 if b won
        B = question_ids.shape[0]
        a_wins = float(result.a_score > result.b_score)
        labels = torch.full((B,), a_wins, dtype=torch.float32)
        judge_loss = self.judge_step(transcript, labels)

        # Debater training via REINFORCE
        reward_signal = self.judge.score_argument(transcript).detach()  # [B]
        loss_a, loss_b = self.debater_step(question_ids, reward_signal, max_new=max_new)

        return {
            "judge_loss": judge_loss,
            "loss_a": loss_a,
            "loss_b": loss_b,
            "winner": result.winner,
        }


# ---------------------------------------------------------------------------
# DebateEvaluator
# ---------------------------------------------------------------------------


class DebateEvaluator:
    """Computes aggregate metrics over a list of DebateResult objects."""

    def win_rate(self, results: List[DebateResult]) -> dict:
        """Fraction of wins for a, b, and ties.

        Args:
            results: list of DebateResult

        Returns:
            {"a_wins": float, "b_wins": float, "ties": float}
        """
        if not results:
            return {"a_wins": 0.0, "b_wins": 0.0, "ties": 0.0}

        n = len(results)
        a_wins = sum(1 for r in results if r.winner == "a") / n
        b_wins = sum(1 for r in results if r.winner == "b") / n
        ties = sum(1 for r in results if r.winner == "tie") / n
        return {"a_wins": a_wins, "b_wins": b_wins, "ties": ties}

    def argument_diversity(self, results: List[DebateResult]) -> float:
        """Average pairwise token-level edit distance between transcripts.

        Args:
            results: list of DebateResult

        Returns:
            mean edit distance >= 0
        """
        if len(results) < 2:
            return 0.0

        transcripts = [r.transcript_ids[0].tolist() for r in results]  # use batch index 0

        total_dist = 0.0
        count = 0
        for i in range(len(transcripts)):
            for j in range(i + 1, len(transcripts)):
                total_dist += _token_edit_distance(transcripts[i], transcripts[j])
                count += 1

        return total_dist / count if count > 0 else 0.0

    def judge_confidence(self, results: List[DebateResult]) -> float:
        """Mean |a_score - 0.5| * 2.  0 = random, 1 = certain.

        Args:
            results: list of DebateResult

        Returns:
            float in [0, 1]
        """
        if not results:
            return 0.0
        return float(
            sum(abs(r.a_score - 0.5) * 2.0 for r in results) / len(results)
        )


def _token_edit_distance(seq_a: list, seq_b: list) -> int:
    """Standard Levenshtein edit distance between two token sequences."""
    m, n = len(seq_a), len(seq_b)
    # dp[i][j] = edit distance between seq_a[:i] and seq_b[:j]
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]
