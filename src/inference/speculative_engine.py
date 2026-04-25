from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SpecConfig:
    n_draft_tokens: int = 4
    temperature: float = 1.0
    top_p: float = 1.0
    vocab_size: int = 32000


@dataclass
class SpecResult:
    accepted_tokens: list[int]
    n_accepted: int
    bonus_token: int | None
    acceptance_rate: float


class SpeculativeEngine:
    """Draft-then-verify speculative decoding (Leviathan et al. 2023)."""

    def __init__(self, config: SpecConfig) -> None:
        self.config = config

    def _sample(self, logits: torch.Tensor) -> int:
        if self.config.temperature == 0.0:
            return int(logits.argmax().item())
        probs = F.softmax(logits / self.config.temperature, dim=-1)
        if self.config.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = (cumulative - sorted_probs) >= self.config.top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()
            chosen = torch.multinomial(sorted_probs, 1).item()
            return int(sorted_idx[int(chosen)].item())
        return int(torch.multinomial(probs, 1).item())

    def draft(self, draft_logits: torch.Tensor) -> list[int]:
        tokens: list[int] = []
        for i in range(draft_logits.size(0)):
            tokens.append(self._sample(draft_logits[i]))
        return tokens

    def verify(
        self,
        draft_tokens: list[int],
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> SpecResult:
        n_draft = len(draft_tokens)
        draft_probs = F.softmax(draft_logits.float(), dim=-1)
        target_probs_all = F.softmax(target_logits.float(), dim=-1)

        accepted: list[int] = []
        n_accepted = 0

        for i, tok in enumerate(draft_tokens):
            p_draft = float(draft_probs[i, tok].clamp(min=1e-9))
            p_target = float(target_probs_all[i, tok].clamp(min=0.0))
            accept_prob = min(1.0, p_target / p_draft)
            u = float(torch.rand(1).item())
            if u <= accept_prob:
                accepted.append(tok)
                n_accepted += 1
            else:
                corrected = (target_probs_all[i] - draft_probs[i]).clamp(min=0.0)
                mass = corrected.sum()
                if mass > 0:
                    corrected = corrected / mass
                else:
                    corrected = torch.ones_like(corrected) / corrected.size(0)
                bonus = int(torch.multinomial(corrected, 1).item())
                acceptance_rate = n_accepted / n_draft
                return SpecResult(
                    accepted_tokens=accepted,
                    n_accepted=n_accepted,
                    bonus_token=bonus,
                    acceptance_rate=acceptance_rate,
                )

        # All draft tokens accepted; sample bonus from target at position n_draft.
        if target_logits.size(0) > n_draft:
            bonus_logits = target_probs_all[n_draft]
        else:
            bonus_logits = target_probs_all[-1]
        bonus = int(torch.multinomial(bonus_logits, 1).item())
        return SpecResult(
            accepted_tokens=accepted,
            n_accepted=n_accepted,
            bonus_token=bonus,
            acceptance_rate=float(n_accepted / n_draft) if n_draft > 0 else 0.0,
        )

    def step(self, draft_logits: torch.Tensor, target_logits: torch.Tensor) -> SpecResult:
        draft_tokens = self.draft(draft_logits)
        return self.verify(draft_tokens, draft_logits, target_logits)

    def expected_speedup(self, acceptance_rate: float) -> float:
        n = self.config.n_draft_tokens
        cost_ratio = 0.1
        numerator = 1.0 - acceptance_rate ** (n + 1)
        denominator = (1.0 - acceptance_rate) * (1.0 + n * cost_ratio)
        if denominator == 0.0:
            return 1.0
        return numerator / denominator
