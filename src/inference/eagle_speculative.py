"""EAGLE: Extrapolation Algorithm for Greater Language-model Efficiency.

EAGLE (arXiv:2401.15077) is a speculative decoding method that uses the *target
model's own penultimate hidden states* to drive a lightweight autoregressive
draft head, eliminating the need for a separate smaller draft model.

Core algorithm
--------------
1. Run target model forward pass → last hidden state h_t and token embedding e_t.
2. Draft head takes (h_{t-1} concat e_t) → predicts next K tokens greedily.
3. Verify all K draft tokens against the target model in one batched forward pass.
4. Accept tokens up to the first mismatch via standard speculative rejection sampling.

Public API
----------
EAGLEConfig       — configuration dataclass
EAGLEDraftHead    — lightweight 1-layer autoregressive draft module
EAGLEDecoder      — wraps target model + draft head for speculative generation
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EAGLEConfig:
    """Configuration for EAGLE speculative decoding."""

    d_model: int = 512
    vocab_size: int = 32000
    n_draft_steps: int = 4       # how many tokens to draft per round
    draft_hidden_dim: int = 256  # hidden size of the draft head (small!)
    temperature: float = 1.0


# ---------------------------------------------------------------------------
# Draft head
# ---------------------------------------------------------------------------

class EAGLEDraftHead(nn.Module):
    """Lightweight 1-layer autoregressive draft predictor.

    The draft head accepts the concatenation of:
    - the previous target hidden state  h_{t-1}  (d_model dims)
    - the current token embedding        e_t      (d_model dims)

    and projects through a single hidden layer to predict vocabulary logits.

    Input:  (B, T, d_model + d_model)   # concat of prev hidden & current embed
    Output: (B, T, vocab_size)           # draft logits
    """

    def __init__(self, config: EAGLEConfig) -> None:
        super().__init__()
        self.config = config
        in_dim = config.d_model * 2  # concat of hidden state + token embed

        # 1-layer transformer-style MLP with layer norm
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, config.draft_hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.draft_hidden_dim, config.vocab_size, bias=False)

        # Initialise weights
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)

    def forward(self, hidden: Tensor, token_embed: Tensor) -> Tensor:
        """Compute draft logits.

        Parameters
        ----------
        hidden:      (B, T, d_model) — target model's penultimate hidden states.
        token_embed: (B, T, d_model) — token embedding for the current token ids.

        Returns
        -------
        logits: (B, T, vocab_size)
        """
        x = torch.cat([hidden, token_embed], dim=-1)  # (B, T, 2*d_model)
        x = self.norm(x)
        x = self.fc1(x)        # (B, T, draft_hidden_dim)
        x = self.act(x)
        logits = self.fc2(x)   # (B, T, vocab_size)
        return logits


# ---------------------------------------------------------------------------
# EAGLE Decoder
# ---------------------------------------------------------------------------

class EAGLEDecoder:
    """Wraps a target model and a draft head for EAGLE speculative generation.

    The target model must expose an ``embedding`` attribute (nn.Embedding) or a
    callable ``embed(token_ids)`` method, and its ``forward`` must return a tuple
    ``(logits, hidden_state)`` where ``hidden_state`` is the last hidden state
    tensor of shape ``(B, T, d_model)``.

    Parameters
    ----------
    target_model:
        The frozen base language model.  ``forward(input_ids)`` must return
        ``(logits, hidden_state)``.
    draft_head:
        An :class:`EAGLEDraftHead` instance.
    config:
        An :class:`EAGLEConfig` instance.
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_head: EAGLEDraftHead,
        config: EAGLEConfig,
    ) -> None:
        self.target_model = target_model
        self.draft_head = draft_head
        self.config = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_token_embed(self, token_ids: Tensor) -> Tensor:
        """Retrieve token embeddings from the target model.

        Tries, in order:
        1. ``target_model.embedding(token_ids)``
        2. ``target_model.embed(token_ids)``
        3. Falls back to a zero tensor of shape (B, T, d_model).
        """
        if hasattr(self.target_model, "embedding") and isinstance(
            self.target_model.embedding, nn.Embedding
        ):
            return self.target_model.embedding(token_ids)
        if hasattr(self.target_model, "embed") and callable(self.target_model.embed):
            return self.target_model.embed(token_ids)
        # Fallback: zero embedding (acceptable for testing / mock models)
        B, T = token_ids.shape
        return torch.zeros(B, T, self.config.d_model, device=token_ids.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draft(self, input_ids: Tensor, features: Tensor) -> Tensor:
        """Draft n_draft_steps tokens autoregressively using the draft head.

        Parameters
        ----------
        input_ids: (B, T)       — current context token ids.
        features:  (B, T, d_model) — target model hidden states for ``input_ids``.

        Returns
        -------
        draft_ids: (B, n_draft_steps)  — greedy draft token ids.
        """
        cfg = self.config
        B = input_ids.shape[0]

        current_ids = input_ids          # (B, T)
        current_hidden = features        # (B, T, d_model)
        all_draft: list[Tensor] = []

        with torch.no_grad():
            for _ in range(cfg.n_draft_steps):
                # Token embedding for the current context
                token_embed = self._get_token_embed(current_ids)  # (B, T', d_model)

                # Align sequence lengths: take the last min(T_h, T_e) positions
                T_h = current_hidden.shape[1]
                T_e = token_embed.shape[1]
                T_min = min(T_h, T_e)
                h = current_hidden[:, -T_min:, :]      # (B, T_min, d_model)
                e = token_embed[:, -T_min:, :]          # (B, T_min, d_model)

                draft_logits = self.draft_head(h, e)    # (B, T_min, vocab_size)
                next_logits = draft_logits[:, -1, :]    # (B, vocab_size) — last position

                # Greedy decode
                next_token = next_logits.argmax(dim=-1)  # (B,)
                all_draft.append(next_token)

                # Extend context: append the new draft token
                current_ids = torch.cat(
                    [current_ids, next_token.unsqueeze(1)], dim=1
                )  # (B, T+1)

                # Approximate the new hidden state by repeating the last hidden
                # vector (the draft head has no target-model forward pass here).
                new_h = current_hidden[:, -1:, :]  # (B, 1, d_model)
                current_hidden = torch.cat([current_hidden, new_h], dim=1)

        draft_ids = torch.stack(all_draft, dim=1)  # (B, n_draft_steps)
        return draft_ids

    def verify_and_accept(
        self,
        input_ids: Tensor,
        draft_ids: Tensor,
        target_logits: Tensor,
        draft_logits: Tensor,
    ) -> tuple[Tensor, int]:
        """Standard speculative rejection-sampling acceptance.

        For each draft position i with draft token t_i:
          - p_target = softmax(target_logits)[t_i]
          - p_draft  = softmax(draft_logits)[t_i]
          - accept_prob = min(1, p_target / p_draft)
          - draw u ~ Uniform(0,1); accept if u < accept_prob, else resample and stop.
        If all K tokens are accepted, sample a bonus token from the target.

        Parameters
        ----------
        input_ids:     (B, T)           — original prompt tokens.
        draft_ids:     (B, n_draft)     — proposed draft token ids.
        target_logits: (B, T+n_draft, V) — target model logits over [prompt | draft].
        draft_logits:  (B, n_draft, V)  — draft head logits for each draft step.

        Returns
        -------
        accepted_ids: (B, k)  — newly accepted token ids (k between 0 and n_draft+1).
        n_accepted:   int     — number of draft tokens accepted (before bonus).
        """
        cfg = self.config
        B = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        n_draft = draft_ids.shape[1]
        temp = max(float(cfg.temperature), 1e-8)
        device = input_ids.device

        accepted: list[Tensor] = []
        n_accepted = 0

        for i in range(n_draft):
            # Target logit at position (prompt_len - 1 + i) predicts draft_ids[:, i]
            target_pos = prompt_len - 1 + i
            t_logits_i = target_logits[:, target_pos, :]      # (B, V)
            t_probs = F.softmax(t_logits_i / temp, dim=-1)    # (B, V)

            d_logits_i = draft_logits[:, i, :]                # (B, V)
            d_probs = F.softmax(d_logits_i / temp, dim=-1)    # (B, V)

            draft_tok = draft_ids[:, i]                        # (B,)
            batch_idx = torch.arange(B, device=device)

            p_target = t_probs[batch_idx, draft_tok]           # (B,)
            p_draft = d_probs[batch_idx, draft_tok]            # (B,)

            accept_prob = torch.clamp(p_target / (p_draft + 1e-10), max=1.0)  # (B,)
            u = torch.rand(B, device=device)

            # Decision based on batch index 0 (standard single-batch inference)
            if u[0].item() <= accept_prob[0].item():
                accepted.append(draft_tok)
                n_accepted += 1
            else:
                # Resample from corrected distribution max(0, p_target - p_draft)
                adj = (t_probs - p_draft.unsqueeze(-1)).clamp(min=0.0)  # (B, V)
                adj_sum = adj.sum(dim=-1, keepdim=True)
                adj = torch.where(adj_sum < 1e-10, t_probs, adj / (adj_sum + 1e-10))
                fallback = torch.multinomial(adj, num_samples=1).squeeze(-1)  # (B,)
                accepted.append(fallback)
                break

        # Bonus token when all drafts accepted
        if n_accepted == n_draft:
            bonus_pos = prompt_len + n_draft - 1
            bonus_logits = target_logits[:, bonus_pos, :]
            bonus_probs = F.softmax(bonus_logits / temp, dim=-1)
            bonus_tok = torch.multinomial(bonus_probs, num_samples=1).squeeze(-1)
            accepted.append(bonus_tok)

        if accepted:
            accepted_ids = torch.stack(accepted, dim=1)  # (B, k)
        else:
            accepted_ids = torch.zeros(B, 0, dtype=torch.long, device=device)

        return accepted_ids, n_accepted

    def generate_step(self, input_ids: Tensor) -> tuple[Tensor, dict]:
        """One full EAGLE speculative decoding step.

        1. Run target model on ``input_ids`` → (target_logits, hidden_states).
        2. Run draft head on (hidden_states, token_embeds) → K draft tokens.
        3. Run target model on ``[input_ids | draft_ids]`` → verification logits.
        4. Accept tokens via rejection sampling.

        Parameters
        ----------
        input_ids: (B, T)  — current context.

        Returns
        -------
        new_ids:   (B, k)   — newly generated token ids.
        stats:     dict     — includes 'n_accepted' and 'n_draft'.
        """
        cfg = self.config
        B, T = input_ids.shape
        device = input_ids.device

        with torch.no_grad():
            # Step 1: target model forward on prompt → get features
            target_out = self.target_model(input_ids)
            if isinstance(target_out, (tuple, list)):
                target_logits_prompt, hidden_states = target_out[0], target_out[1]
            else:
                # Fallback: if model only returns logits, create dummy hidden states
                target_logits_prompt = target_out
                hidden_states = torch.zeros(B, T, cfg.d_model, device=device)

        # Step 2: draft K tokens using the draft head
        draft_ids = self.draft(input_ids, hidden_states)  # (B, n_draft_steps)

        # Collect draft head logits for verify_and_accept
        with torch.no_grad():
            token_embed = self._get_token_embed(input_ids)  # (B, T, d_model)

            T_h = hidden_states.shape[1]
            T_e = token_embed.shape[1]
            T_min = min(T_h, T_e)
            h = hidden_states[:, -T_min:, :]
            e = token_embed[:, -T_min:, :]
            draft_logits_all = self.draft_head(h, e)  # (B, T_min, vocab_size)

            # We only need the last T_min positions; draft steps extend from last
            # position, so we replicate last draft logit for each step position.
            # Build (B, n_draft_steps, vocab_size) by querying per-step.
            n_draft = cfg.n_draft_steps
            draft_logit_steps: list[Tensor] = []

            current_ids = input_ids
            current_hidden = hidden_states
            for _ in range(n_draft):
                te_step = self._get_token_embed(current_ids)  # (B, T', d_model)
                T_h2 = current_hidden.shape[1]
                T_e2 = te_step.shape[1]
                T_m = min(T_h2, T_e2)
                h2 = current_hidden[:, -T_m:, :]
                e2 = te_step[:, -T_m:, :]
                step_logits = self.draft_head(h2, e2)   # (B, T_m, vocab_size)
                draft_logit_steps.append(step_logits[:, -1, :])  # (B, vocab_size)

                # Extend with next draft token
                next_tok = draft_logit_steps[-1].argmax(dim=-1)   # (B,)
                current_ids = torch.cat([current_ids, next_tok.unsqueeze(1)], dim=1)
                new_h = current_hidden[:, -1:, :]
                current_hidden = torch.cat([current_hidden, new_h], dim=1)

        draft_logits = torch.stack(draft_logit_steps, dim=1)  # (B, n_draft, vocab_size)

        # Step 3: target model forward on full sequence [prompt | draft]
        full_ids = torch.cat([input_ids, draft_ids], dim=1)  # (B, T + n_draft)
        with torch.no_grad():
            target_out_full = self.target_model(full_ids)
            if isinstance(target_out_full, (tuple, list)):
                target_logits_full = target_out_full[0]
            else:
                target_logits_full = target_out_full

        # Step 4: verify and accept
        new_ids, n_accepted = self.verify_and_accept(
            input_ids, draft_ids, target_logits_full, draft_logits
        )

        stats = {
            "n_accepted": n_accepted,
            "n_draft": n_draft,
            "acceptance_rate": n_accepted / n_draft if n_draft > 0 else 0.0,
        }

        return new_ids, stats
