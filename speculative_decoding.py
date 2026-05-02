import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


@dataclass
class SpeculativeConfig:
    gamma: int = 5
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9


class MemoryContextProjector(nn.Module):
    def __init__(self, d_mem: int, d_draft: int):
        super().__init__()
        self.proj = nn.Linear(d_mem, d_draft)
        self.norm = nn.LayerNorm(d_draft)

    def forward(self, memory_context: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(memory_context))


class MemoryAwareDraftModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, n_layers: int = 4, d_mem: int = 768):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.memory_projector = MemoryContextProjector(d_mem, d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4, activation=F.gelu,
                                       batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, memory_context: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)

        mem = self.memory_projector(memory_context)
        mem = mem.unsqueeze(1)
        x = torch.cat([mem, x], dim=1)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x[:, 1:, :])
        return logits

    def generate_draft(self, input_ids: torch.Tensor, memory_context: torch.Tensor,
                       gamma: int, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
        drafts = []
        cur_ids = input_ids
        for _ in range(gamma):
            logits = self.forward(cur_ids, memory_context)
            logits = logits[:, -1, :]
            logits = self._apply_top_k_top_p(logits, top_k, top_p)
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            drafts.append(next_token)
            cur_ids = torch.cat([cur_ids, next_token], dim=1)
        return torch.cat(drafts, dim=1)

    def _apply_top_k_top_p(self, logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            threshold = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < threshold, float('-inf'), logits)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[mask] = float('-inf')
            logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)
        return logits


class SpeculativeDecoder(nn.Module):
    def __init__(self, config: SpeculativeConfig):
        super().__init__()
        self.config = config

    @staticmethod
    def _rejection_sample(acceptance_probs: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
        uniform = torch.rand(acceptance_probs.shape, generator=rng, device=acceptance_probs.device)
        return (uniform < acceptance_probs).long()

    @staticmethod
    def _residual_distribution(target_probs: torch.Tensor, draft_probs: torch.Tensor,
                               accepted_until: int) -> torch.Tensor:
        residual = target_probs - draft_probs
        residual = residual.clamp(min=0)
        residual_sum = residual.sum(dim=-1, keepdim=True)
        residual = torch.where(residual_sum > 0, residual / residual_sum,
                               torch.ones_like(residual) / residual.size(-1))
        return residual

    def generate_with_speculation(
        self,
        input_ids: torch.Tensor,
        target_model: nn.Module,
        draft_model: MemoryAwareDraftModel,
        memory_context: torch.Tensor,
        max_new_tokens: int,
        gamma: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: int | None = None,
    ) -> torch.Tensor:
        gamma = gamma if gamma is not None else self.config.gamma
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p

        rng = torch.Generator(device=input_ids.device)
        rng.seed()

        output = input_ids.clone()
        device = input_ids.device
        b = input_ids.shape[0]
        total_generated = 0

        while total_generated < max_new_tokens:
            cur_len = output.shape[1]
            cur_gamma = min(gamma, max_new_tokens - total_generated)

            with torch.no_grad():
                draft_tokens = draft_model.generate_draft(
                    output, memory_context, cur_gamma, temperature, top_k, top_p
                )

            draft_input = torch.cat([output, draft_tokens], dim=1)

            with torch.no_grad():
                draft_logits_all = draft_model(draft_input, memory_context)
                target_logits_all = target_model(draft_input)

            target_probs = F.softmax(target_logits_all / temperature, dim=-1)
            draft_probs = F.softmax(draft_logits_all / temperature, dim=-1)

            n_drafts = cur_gamma
            acceptance_probs = torch.zeros(b, n_drafts, device=device)
            accepted_tokens = []

            for i in range(n_drafts):
                pos = cur_len + i
                p_target = target_probs[:, pos - 1, :]
                p_draft = draft_probs[:, pos - 1, :]
                token = draft_tokens[:, i]

                token_probs_target = p_target.gather(-1, token.unsqueeze(-1)).squeeze(-1)
                token_probs_draft = p_draft.gather(-1, token.unsqueeze(-1)).squeeze(-1)
                ratio = torch.where(
                    token_probs_draft > 0,
                    token_probs_target / token_probs_draft,
                    torch.zeros_like(token_probs_target)
                )
                acceptance_probs[:, i] = ratio

            mask = self._rejection_sample(acceptance_probs, rng)

            accepted_count = 0
            for i in range(n_drafts):
                if mask[0, i]:
                    accepted_tokens.append(draft_tokens[:, i:i+1])
                    accepted_count += 1
                else:
                    p_target = target_probs[:, cur_len + i - 1, :]
                    p_draft = draft_probs[:, cur_len + i - 1, :]
                    residual = self._residual_distribution(p_target, p_draft, i)
                    resampled = torch.multinomial(residual, num_samples=1)
                    accepted_tokens.append(resampled)
                    break
            else:
                p_target_last = target_probs[:, cur_len + n_drafts - 1, :]
                p_draft_last = draft_probs[:, cur_len + n_drafts - 1, :]
                residual = self._residual_distribution(p_target_last, p_draft_last, n_drafts)
                extra_token = torch.multinomial(residual, num_samples=1)
                accepted_tokens.append(extra_token)
                accepted_count += 1

            new_tokens = torch.cat(accepted_tokens, dim=1)
            output = torch.cat([output, new_tokens], dim=1)
            total_generated += new_tokens.shape[1]

        return output
