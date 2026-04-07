"""Contrastive search decoding (Su et al., 2022 - arXiv:2202.06417).

Selects tokens that are both high-probability and contextually distinct,
preventing repetition without sacrificing fluency.

Score: (1-alpha) * model_prob - alpha * max_cos_sim(candidate_hidden, context_hidden)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ContrastiveConfig:
    k: int = 5                    # top-k candidates to score
    alpha: float = 0.6            # penalty weight (0=greedy, 1=pure contrastive)
    max_new_tokens: int = 256
    eos_token_id: int | None = None


def _get_last_hidden(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Extract the last-token hidden state from the model backbone.

    Runs embed -> layers -> norm without computing the lm_head.

    Args:
        input_ids: (1, seq_len)

    Returns:
        (d_model,) — last-token normalized hidden state.
    """
    x = model.embed(input_ids)
    freqs_cis = model.freqs_cis[:input_ids.shape[1]]
    for layer in model.layers:
        x, _ = layer(x, freqs_cis, mask=None, past_kv=None)
    x = model.norm(x)
    return x[0, -1]  # (d_model,)


@torch.no_grad()
def contrastive_search(
    model: nn.Module,
    input_ids: torch.Tensor,
    cfg: ContrastiveConfig | None = None,
) -> torch.Tensor:
    """Generate tokens using contrastive search.

    Args:
        model: AureliusTransformer (must have embed, layers, norm, lm_head attributes).
        input_ids: (1, prompt_len) — input token ids.
        cfg: Contrastive search configuration.

    Returns:
        (1, prompt_len + generated_len) — input + generated tokens.
    """
    if cfg is None:
        cfg = ContrastiveConfig()

    generated = input_ids.clone()

    # Collect hidden states of all past tokens for contrastive penalty.
    # Initialize with hidden states of the prompt tokens.
    context_hidden = []
    for pos in range(generated.shape[1]):
        h = _get_last_hidden(model, generated[:, :pos + 1])
        context_hidden.append(h)

    for _ in range(cfg.max_new_tokens):
        # Step 1: Get top-k candidate tokens from current sequence.
        _, logits, _ = model(generated)
        probs = F.softmax(logits[0, -1], dim=-1)  # (vocab_size,)
        top_k_probs, top_k_ids = torch.topk(probs, cfg.k)

        # Step 2: Score each candidate via the contrastive objective.
        context_h = torch.stack(context_hidden)          # (t, d_model)
        context_h_norm = F.normalize(context_h, dim=-1)  # (t, d_model)

        best_score = float("-inf")
        best_token = top_k_ids[0]
        best_hidden = None

        for i in range(cfg.k):
            cand_id = top_k_ids[i:i + 1]                              # (1,)
            cand_input = torch.cat([generated, cand_id.unsqueeze(0)], dim=1)  # (1, seq+1)
            cand_h = _get_last_hidden(model, cand_input)              # (d_model,)
            cand_h_norm = F.normalize(cand_h.unsqueeze(0), dim=-1)   # (1, d_model)

            # Max cosine similarity with any past hidden state.
            cos_sims = (cand_h_norm @ context_h_norm.T).squeeze(0)   # (t,)
            max_cos = cos_sims.max().item()

            # Contrastive score.
            score = (1 - cfg.alpha) * top_k_probs[i].item() - cfg.alpha * max_cos

            if score > best_score:
                best_score = score
                best_token = cand_id
                best_hidden = cand_h

        # Append the winning token's hidden state to context.
        # best_hidden is always set here because we iterate cfg.k >= 1 candidates.
        context_hidden.append(best_hidden)
        next_token = best_token.unsqueeze(0)  # (1, 1)

        generated = torch.cat([generated, next_token], dim=1)

        if cfg.eos_token_id is not None and next_token.item() == cfg.eos_token_id:
            break

    return generated
