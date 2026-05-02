"""Latent reasoning: invisible thinking tokens that improve answer quality without appearing in output."""  # noqa: E501

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LatentReasoningConfig:
    """Configuration for latent reasoning (Quiet-STaR / Thinking Tokens style)."""

    n_thought_tokens: int = 8
    thought_token_id: int = 1
    max_answer_tokens: int = 64
    temperature: float = 1.0
    use_thought_supervision: bool = True
    thought_start_id: int = 2
    thought_end_id: int = 3


class ThoughtTokenEmbedding(nn.Module):
    """Special embedding for latent thought tokens — learnable per-position embeddings."""

    def __init__(self, d_model: int, n_thought_tokens: int) -> None:
        super().__init__()
        self.thought_embeddings = nn.Embedding(n_thought_tokens, d_model)

    def forward(self, thought_ids: Tensor) -> Tensor:
        return self.thought_embeddings(thought_ids)


def prepend_thought_tokens(hidden_states: Tensor, thought_embeds: Tensor) -> Tensor:
    return torch.cat([thought_embeds, hidden_states], dim=1)


def extract_answer_from_thoughts(full_output: Tensor, n_thought_tokens: int) -> Tensor:
    return full_output[:, n_thought_tokens:, ...]


class LatentReasoningLayer(nn.Module):
    """Transformer layer augmented with latent thought tokens."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        config: LatentReasoningConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.thought_embed = ThoughtTokenEmbedding(d_model, config.n_thought_tokens)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        n_thoughts = self.config.n_thought_tokens

        thought_ids = torch.arange(n_thoughts, device=x.device).unsqueeze(0).expand(B, -1)
        thought_embeds = self.thought_embed(thought_ids)

        full_seq = prepend_thought_tokens(x, thought_embeds)
        attended, _ = self.attn(full_seq, full_seq, full_seq)

        x_attended = extract_answer_from_thoughts(attended, n_thoughts)
        x = self.norm1(x + x_attended)
        x = self.norm2(x + self.ffn(x))
        return x


def compute_thought_supervision_loss(
    thought_logits: Tensor,
    target_ids: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    if (target_ids == ignore_index).all():
        return torch.tensor(0.0, device=thought_logits.device, requires_grad=False)

    B, n_thoughts, V = thought_logits.shape
    logits_flat = thought_logits.reshape(B * n_thoughts, V)
    targets_flat = target_ids.reshape(B * n_thoughts)
    return F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)


class LatentReasoningDecoder:
    """Full inference loop with latent reasoning (Quiet-STaR style)."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
        config: LatentReasoningConfig,
    ) -> None:
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.config = config

    def _forward_logits(self, input_ids: Tensor) -> Tensor:
        out = self.model(input_ids)
        if isinstance(out, tuple):
            return out[0]
        return out

    def generate_with_thoughts(self, prompt: str) -> tuple[str, list[int]]:
        cfg = self.config
        input_ids = self.tokenizer_encode(prompt)
        device = next(self.model.parameters()).device

        thought_ids: list[int] = []
        current_ids = list(input_ids)
        self.model.eval()
        with torch.no_grad():
            for _ in range(cfg.n_thought_tokens):
                inp = torch.tensor([current_ids], dtype=torch.long, device=device)
                logits = self._forward_logits(inp)
                next_logits = logits[0, -1, :] / max(cfg.temperature, 1e-8)
                next_id = int(torch.multinomial(torch.softmax(next_logits, dim=-1), 1).item())
                thought_ids.append(next_id)
                current_ids.append(next_id)

            answer_ids: list[int] = []
            for _ in range(cfg.max_answer_tokens):
                inp = torch.tensor([current_ids], dtype=torch.long, device=device)
                logits = self._forward_logits(inp)
                next_logits = logits[0, -1, :] / max(cfg.temperature, 1e-8)
                next_id = int(torch.multinomial(torch.softmax(next_logits, dim=-1), 1).item())
                answer_ids.append(next_id)
                current_ids.append(next_id)

        answer_text = self.tokenizer_decode(answer_ids)
        return answer_text, thought_ids

    def generate_direct(self, prompt: str) -> str:
        cfg = self.config
        input_ids = self.tokenizer_encode(prompt)
        device = next(self.model.parameters()).device

        current_ids = list(input_ids)
        answer_ids: list[int] = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(cfg.max_answer_tokens):
                inp = torch.tensor([current_ids], dtype=torch.long, device=device)
                logits = self._forward_logits(inp)
                next_logits = logits[0, -1, :] / max(cfg.temperature, 1e-8)
                next_id = int(torch.multinomial(torch.softmax(next_logits, dim=-1), 1).item())
                answer_ids.append(next_id)
                current_ids.append(next_id)

        return self.tokenizer_decode(answer_ids)


def measure_thought_benefit(
    model: nn.Module,
    prompts: list[str],
    tokenizer_encode: Callable[[str], list[int]],
    tokenizer_decode: Callable[[list[int]], str],
    config: LatentReasoningConfig,
) -> dict[str, float]:
    decoder = LatentReasoningDecoder(model, tokenizer_encode, tokenizer_decode, config)
    device = next(model.parameters()).device
    model.eval()

    thought_losses: list[float] = []
    direct_losses: list[float] = []

    with torch.no_grad():
        for prompt in prompts:
            prompt_ids = tokenizer_encode(prompt)
            steps = min(config.max_answer_tokens, 8)

            # Thought path: first generate thought tokens, then measure answer NLL
            thought_context = list(prompt_ids)
            for _ in range(config.n_thought_tokens):
                inp = torch.tensor([thought_context], dtype=torch.long, device=device)
                logits = decoder._forward_logits(inp)
                probs = torch.softmax(logits[0, -1, :] / max(config.temperature, 1e-8), dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
                thought_context.append(next_id)

            loss_sum_thought = 0.0
            current = list(thought_context)
            for _ in range(steps):
                inp = torch.tensor([current], dtype=torch.long, device=device)
                logits = decoder._forward_logits(inp)
                probs = torch.softmax(logits[0, -1, :], dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
                loss_sum_thought += -float(torch.log(probs[next_id] + 1e-10))
                current.append(next_id)
            thought_ppl = float(torch.exp(torch.tensor(loss_sum_thought / max(steps, 1))))

            # Direct path
            loss_sum_direct = 0.0
            current = list(prompt_ids)
            for _ in range(steps):
                inp = torch.tensor([current], dtype=torch.long, device=device)
                logits = decoder._forward_logits(inp)
                probs = torch.softmax(logits[0, -1, :], dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
                loss_sum_direct += -float(torch.log(probs[next_id] + 1e-10))
                current.append(next_id)
            direct_ppl = float(torch.exp(torch.tensor(loss_sum_direct / max(steps, 1))))

            thought_losses.append(thought_ppl)
            direct_losses.append(direct_ppl)

    thought_perplexity = sum(thought_losses) / len(thought_losses)
    direct_perplexity = sum(direct_losses) / len(direct_losses)
    relative_improvement = (direct_perplexity - thought_perplexity) / max(direct_perplexity, 1e-10)

    return {
        "thought_perplexity": thought_perplexity,
        "direct_perplexity": direct_perplexity,
        "relative_improvement": relative_improvement,
    }
