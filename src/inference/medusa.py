"""Medusa: Simple Framework for Accelerating LLM Inference (arXiv:2401.10774).

Adds K parallel draft heads to the base LM. Each head predicts K tokens ahead
from the same final hidden state. During generation, heads propose K candidates
simultaneously; greedy verification accepts a consistent prefix.

Speedup depends on acceptance rate: with M heads, best case is M+1 tokens per
base model forward pass (all heads correct) vs 1 token without Medusa.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MedusaConfig:
    num_heads: int = 3  # number of draft heads (predicts up to 3 tokens ahead)
    max_new_tokens: int = 256
    temperature: float = 1.0
    eos_token_id: int | None = None


class MedusaModel(nn.Module):
    """AureliusTransformer with K Medusa draft heads for faster generation.

    The base model backbone is frozen by default; only Medusa heads are trained.

    Args:
        base_model: Trained AureliusTransformer.
        cfg: Medusa configuration.
        freeze_base: If True, freeze base model weights (only train heads).
    """

    def __init__(
        self,
        base_model: nn.Module,
        cfg: MedusaConfig | None = None,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.cfg = cfg or MedusaConfig()

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)

        d_model = base_model.config.d_model
        vocab_size = base_model.config.vocab_size

        # K independent linear heads, each predicting vocab distribution
        self.medusa_heads = nn.ModuleList(
            [nn.Linear(d_model, vocab_size, bias=False) for _ in range(self.cfg.num_heads)]
        )
        for head in self.medusa_heads:
            nn.init.normal_(head.weight, std=0.02)

    def _get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract all hidden states from base model backbone.

        Returns: (B, seq_len, d_model)
        """
        x = self.base.embed(input_ids)
        freqs_cis = self.base.freqs_cis[: input_ids.shape[1]]
        for layer in self.base.layers:
            x, _, _ = layer(x, freqs_cis, mask=None, past_kv=None)
        return self.base.norm(x)  # (B, seq_len, d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, list[torch.Tensor]]:
        """Forward pass returning base loss + per-head logits.

        Args:
            input_ids: (B, seq_len)
            labels: (B, seq_len) — if provided, compute cross-entropy losses

        Returns:
            (total_loss, base_logits, head_logits_list)
            - total_loss: None if labels not provided; else sum of base + head losses
            - base_logits: (B, seq_len, vocab_size) from base model lm_head
            - head_logits_list: list of (B, seq_len, vocab_size) per Medusa head
        """
        hidden = self._get_hidden_states(input_ids)  # (B, S, d_model)
        base_logits = self.base.lm_head(hidden)  # (B, S, vocab)

        head_logits = [head(hidden) for head in self.medusa_heads]  # list of (B, S, vocab)

        total_loss = None
        if labels is not None:
            vocab_size = self.base.config.vocab_size

            # Base model loss: predict token t+1 from position t
            base_loss = F.cross_entropy(
                base_logits[:, :-1].contiguous().view(-1, vocab_size),
                labels[:, 1:].contiguous().view(-1),
            )

            # Head i loss: head i predicts token t+(i+2) from position t
            # (head 0 -> offset 2, head 1 -> offset 3, ..., head K-1 -> offset K+1)
            head_losses = []
            for i, h_logits in enumerate(head_logits):
                offset = i + 2  # predict i+2 positions ahead
                if h_logits.shape[1] > offset:
                    loss_i = F.cross_entropy(
                        h_logits[:, :-offset].contiguous().view(-1, vocab_size),
                        labels[:, offset:].contiguous().view(-1),
                    )
                    head_losses.append(loss_i)

            total_loss = base_loss + sum(head_losses)

        return total_loss, base_logits, head_logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate tokens using Medusa (simple greedy: accept all K+1 candidates per step).

        At each step:
        1. Run one forward pass to get base + head logits
        2. Draft candidates: base argmax + each head's argmax
        3. Append all accepted tokens (base + heads) up to max_new_tokens

        Args:
            input_ids: (1, prompt_len)

        Returns:
            (1, prompt_len + generated_len)
        """
        cfg = self.cfg
        generated = input_ids.clone()
        tokens_generated = 0
        max_seq_len = self.base.config.max_seq_len

        while tokens_generated < cfg.max_new_tokens:
            remaining = cfg.max_new_tokens - tokens_generated

            # Truncate context to max_seq_len to avoid exceeding positional encoding bounds
            context = generated[:, -max_seq_len:] if generated.shape[1] > max_seq_len else generated

            _, base_logits, head_logits = self.forward(context)

            # Get base token (step +1)
            if cfg.temperature != 1.0:
                base_probs = F.softmax(base_logits[0, -1] / cfg.temperature, dim=-1)
                base_tok = torch.multinomial(base_probs, 1).squeeze(-1)
            else:
                base_tok = base_logits[0, -1].argmax()

            # Get head tokens (steps +2, +3, ...)
            head_toks = []
            for h_logits in head_logits[: remaining - 1]:  # don't overshoot
                if cfg.temperature != 1.0:
                    h_probs = F.softmax(h_logits[0, -1] / cfg.temperature, dim=-1)
                    head_toks.append(torch.multinomial(h_probs, 1).squeeze(-1))
                else:
                    head_toks.append(h_logits[0, -1].argmax())

            # Collect all candidates: [base, head_0, head_1, ...]
            new_toks = [base_tok] + head_toks

            # Find EOS cutoff
            eos_cut = len(new_toks)
            if cfg.eos_token_id is not None:
                for j, t in enumerate(new_toks):
                    if t.item() == cfg.eos_token_id:
                        eos_cut = j + 1
                        break

            new_toks = new_toks[:eos_cut]
            new_tensor = torch.stack(new_toks).view(1, -1)
            generated = torch.cat([generated, new_tensor], dim=1)
            tokens_generated += len(new_toks)

            if eos_cut < len([base_tok] + head_toks):
                break

        return generated
