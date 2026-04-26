"""Thought Token / Pause Token Training.

Fine-tunes a model to insert K latent computation tokens before answering.
The thought tokens are trained end-to-end; at inference, they're prepended
to the prompt and the model generates through them before producing output.

References:
    Goyal et al. 2023 (Pause Tokens) — https://arxiv.org/abs/2310.02226
    Zelikman et al. 2024 — related work on latent reasoning tokens
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ThoughtTokenConfig:
    """Configuration for thought-token / pause-token training.

    Args:
        n_thought_tokens: Number of pause tokens to insert between prompt and response.
        thought_token_id: Token ID used to fill thought positions in input_ids.
        loss_on_thoughts: If True, include thought token positions in the training loss
            (predict the next thought token); if False, mask them out with -100.
    """

    def __init__(
        self,
        n_thought_tokens: int = 4,
        thought_token_id: int = 1,
        loss_on_thoughts: bool = False,
    ) -> None:
        self.n_thought_tokens = n_thought_tokens
        self.thought_token_id = thought_token_id
        self.loss_on_thoughts = loss_on_thoughts


# ---------------------------------------------------------------------------
# ThoughtTokenInserter
# ---------------------------------------------------------------------------


class ThoughtTokenInserter:
    """Manages insertion of thought tokens into sequences.

    Args:
        config: ThoughtTokenConfig.
    """

    def __init__(self, config: ThoughtTokenConfig) -> None:
        self.config = config

    def insert_thoughts(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Append thought tokens to the end of each prompt in the batch.

        Args:
            input_ids: (B, T) long tensor of prompt token IDs.

        Returns:
            augmented_ids: (B, T + n_thought_tokens) — prompt followed by thought tokens.
            thought_mask:  (B, T + n_thought_tokens) bool — True at thought positions.
        """
        B, T = input_ids.shape
        K = self.config.n_thought_tokens

        thought_tokens = torch.full(
            (B, K),
            fill_value=self.config.thought_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        augmented_ids = torch.cat([input_ids, thought_tokens], dim=1)  # (B, T+K)

        thought_mask = torch.zeros(B, T + K, dtype=torch.bool, device=input_ids.device)
        thought_mask[:, T:] = True  # last K positions are thought tokens

        return augmented_ids, thought_mask

    def create_labels(
        self,
        augmented_ids: Tensor,
        thought_mask: Tensor,
        response_ids: Tensor,
    ) -> Tensor:
        """Build labels for the full sequence [prompt | thoughts | response].

        Prompt positions are always masked with -100 (no loss).
        Thought positions are masked with -100 unless config.loss_on_thoughts=True,
        in which case the label at thought position t is the next token
        (thought[t+1] or response[0] for the last thought position).
        Response positions carry their actual token IDs as targets.

        Args:
            augmented_ids: (B, T + K) token IDs (prompt + thought tokens).
            thought_mask:  (B, T + K) bool mask — True at thought positions.
            response_ids:  (B, T_r) response token IDs.

        Returns:
            labels: (B, T + K + T_r) long tensor.
        """
        B, TK = augmented_ids.shape
        T_r = response_ids.shape[1]
        device = augmented_ids.device

        # Full sequence: [prompt | thoughts | response]
        full_ids = torch.cat([augmented_ids, response_ids], dim=1)  # (B, TK + T_r)

        # Start with everything masked
        labels = torch.full((B, TK + T_r), fill_value=-100, dtype=torch.long, device=device)

        # Response positions get their token IDs
        labels[:, TK:] = response_ids

        if self.config.loss_on_thoughts:
            # At each thought position t, the label is the next token in the full sequence
            # i.e., labels[:, t] = full_ids[:, t + 1]
            # For thought positions in [T_start, T_start+K), that's [T_start+1, T_start+K+1)
            # We only unmask the thought positions (not the prompt prefix)
            thought_start = TK - self.config.n_thought_tokens  # = T (prompt length)
            thought_end = TK  # exclusive

            for t in range(thought_start, thought_end):
                # label at position t is the token at position t+1
                labels[:, t] = full_ids[:, t + 1]

        return labels


# ---------------------------------------------------------------------------
# PauseTokenEmbedding
# ---------------------------------------------------------------------------


class PauseTokenEmbedding(nn.Module):
    """Learnable positional embeddings for thought / pause tokens.

    Each of the K thought positions has its own embedding vector so that
    the model can distinguish different reasoning steps.

    Args:
        n_thought_tokens: Number of thought tokens (K).
        d_model: Embedding dimension.
    """

    def __init__(self, n_thought_tokens: int, d_model: int) -> None:
        super().__init__()
        self.n_thought_tokens = n_thought_tokens
        self.d_model = d_model
        self.embeddings = nn.Embedding(n_thought_tokens, d_model)
        self.register_buffer("position_ids", torch.arange(n_thought_tokens))

    def forward(self) -> Tensor:
        """Return all thought token embeddings.

        Returns:
            (n_thought_tokens, d_model) float tensor.
        """
        return self.embeddings(self.position_ids)  # type: ignore[arg-type]

    def get_token_embedding(self, position: int) -> Tensor:
        """Return the embedding for a single thought position.

        Args:
            position: Index in [0, n_thought_tokens).

        Returns:
            (d_model,) float tensor.
        """
        idx = torch.tensor(position, dtype=torch.long, device=self.position_ids.device)
        return self.embeddings(idx)


# ---------------------------------------------------------------------------
# ThoughtTokenTrainer
# ---------------------------------------------------------------------------


class ThoughtTokenTrainer:
    """Trainer that fine-tunes a model to reason through thought tokens.

    The model sees [prompt | K thought tokens | response] at train time.
    Loss is computed only on response positions (and optionally thought positions).

    Args:
        model_fn: Callable ``(input_ids: LongTensor(B, T)) -> logits: (B, T, V)``.
        optimizer: PyTorch optimizer for gradient updates.
        config: ThoughtTokenConfig.
        inserter: ThoughtTokenInserter instance (must share config).
    """

    def __init__(
        self,
        model_fn: Callable[[Tensor], Tensor],
        optimizer: torch.optim.Optimizer,
        config: ThoughtTokenConfig,
        inserter: ThoughtTokenInserter,
    ) -> None:
        self.model_fn = model_fn
        self.optimizer = optimizer
        self.config = config
        self.inserter = inserter

    def train_step(self, input_ids: Tensor, response_ids: Tensor) -> dict[str, float]:
        """Perform one gradient update step.

        Args:
            input_ids:    (B, T) prompt token IDs.
            response_ids: (B, T_r) target response token IDs.

        Returns:
            Dict with keys:
                "loss"            — total scalar loss (float)
                "n_thought_tokens" — number of thought tokens used (int as float)
                "response_loss"   — loss on response positions only (float)
        """
        # Build augmented prompt and labels
        augmented_ids, thought_mask = self.inserter.insert_thoughts(input_ids)
        labels = self.inserter.create_labels(augmented_ids, thought_mask, response_ids)

        # Full input sequence: [prompt | thoughts | response]
        full_input = torch.cat([augmented_ids, response_ids], dim=1)  # (B, TK + T_r)

        # Forward pass
        self.optimizer.zero_grad()
        logits = self.model_fn(full_input)  # (B, TK + T_r, V)

        B, S, V = logits.shape

        # Shift for next-token prediction: logit[t] predicts token[t+1]
        shift_logits = logits[:, :-1, :].contiguous()  # (B, S-1, V)
        shift_labels = labels[:, 1:].contiguous()  # (B, S-1)

        total_loss = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Compute response-only loss for reporting
        TK = augmented_ids.shape[1]
        # Response positions in shifted labels start at TK-1 (since shift removes one)
        response_shift_labels = labels[:, TK:].contiguous()  # (B, T_r)
        response_shift_logits = logits[
            :, TK - 1 : TK - 1 + response_shift_labels.shape[1], :
        ].contiguous()

        response_mask = response_shift_labels != -100
        if response_mask.any():
            response_loss = F.cross_entropy(
                response_shift_logits.reshape(-1, V),
                response_shift_labels.reshape(-1),
                ignore_index=-100,
            )
        else:
            response_loss = total_loss

        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "n_thought_tokens": float(self.config.n_thought_tokens),
            "response_loss": response_loss.item(),
        }

    def generate_with_thoughts(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        """Generate tokens after running the model through thought tokens.

        The prompt is augmented with thought tokens, then the model greedily
        decodes ``max_new_tokens`` response tokens.

        Args:
            input_ids:      (B, T) prompt token IDs (B=1 typical at inference).
            max_new_tokens: Number of response tokens to generate.

        Returns:
            (max_new_tokens,) long tensor of generated token IDs (first batch item).
        """
        augmented_ids, _ = self.inserter.insert_thoughts(input_ids)  # (B, T+K)

        context = augmented_ids  # (B, T+K)
        generated: list[Tensor] = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model_fn(context)  # (B, S, V)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
                generated.append(next_token)
                context = torch.cat([context, next_token], dim=1)

        # Stack and return first batch item, squeezed to 1D
        all_generated = torch.cat(generated, dim=1)  # (B, max_new_tokens)
        return all_generated[0]  # (max_new_tokens,)
