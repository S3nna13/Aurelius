"""
Chain-of-Thought Verifier — verifies step-by-step reasoning chains.

Classes
-------
StepEncoder         : Encodes a single reasoning step into a fixed-size vector.
ChainEncoder        : Models dependencies across a sequence of step encodings.
VerifierHead        : Produces per-step and chain-level validity scores.
CoTVerifierModel    : Full verifier (step → chain → head).
VerifierTrainer     : Training utilities with combined BCE loss.
ProcessRewardModel  : Process-supervision reward model over individual steps.
VerifierConfig      : Dataclass of default hyper-parameters.
"""

import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transformer_encoder(d_model: int, n_layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=max(1, d_model // 8),
        dim_feedforward=d_model * 4,
        dropout=0.0,
        batch_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


# ---------------------------------------------------------------------------
# StepEncoder
# ---------------------------------------------------------------------------

class StepEncoder(nn.Module):
    """Encodes a single reasoning step (token sequence) into a fixed-size vector.

    Uses a learnable CLS token prepended to the embedded token sequence.
    The CLS representation after the transformer is returned as the step vector.

    Parameters
    ----------
    d_model    : Hidden dimension.
    vocab_size : Vocabulary size for the token embedding table.
    n_layers   : Number of transformer encoder layers.
    """

    def __init__(self, d_model: int, vocab_size: int, n_layers: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.pos_embedding = nn.Embedding(512, d_model)
        self.encoder = _make_transformer_encoder(d_model, n_layers)

    def forward(self, step_ids: torch.Tensor) -> torch.Tensor:
        """Encode a batch of token sequences.

        Parameters
        ----------
        step_ids : LongTensor of shape [B, T_step]

        Returns
        -------
        Tensor of shape [B, d_model] — CLS pooled representation.
        """
        B, T = step_ids.shape
        tok_emb = self.embedding(step_ids)                          # [B, T, d]
        positions = torch.arange(T, device=step_ids.device)
        tok_emb = tok_emb + self.pos_embedding(positions).unsqueeze(0)

        cls = self.cls_token.expand(B, -1, -1)                     # [B, 1, d]
        x = torch.cat([cls, tok_emb], dim=1)                       # [B, T+1, d]

        out = self.encoder(x)                                       # [B, T+1, d]
        return out[:, 0, :]                                         # [B, d]


# ---------------------------------------------------------------------------
# ChainEncoder
# ---------------------------------------------------------------------------

class ChainEncoder(nn.Module):
    """Models dependencies across a sequence of step encodings.

    Takes the per-step vectors produced by StepEncoder and contextualises
    them with a transformer encoder that attends across steps.

    Parameters
    ----------
    d_model  : Hidden dimension (must match StepEncoder output).
    n_layers : Number of transformer encoder layers.
    """

    def __init__(self, d_model: int, n_layers: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.pos_embedding = nn.Embedding(512, d_model)
        self.encoder = _make_transformer_encoder(d_model, n_layers)

    def forward(self, step_encodings: torch.Tensor) -> torch.Tensor:
        """Contextualise step representations across the chain.

        Parameters
        ----------
        step_encodings : FloatTensor of shape [B, n_steps, d_model]

        Returns
        -------
        Tensor of shape [B, n_steps, d_model] — contextual step representations.
        """
        B, S, D = step_encodings.shape
        positions = torch.arange(S, device=step_encodings.device)
        x = step_encodings + self.pos_embedding(positions).unsqueeze(0)
        return self.encoder(x)                                      # [B, S, d]


# ---------------------------------------------------------------------------
# VerifierHead
# ---------------------------------------------------------------------------

class VerifierHead(nn.Module):
    """Produces per-step validity scores and a single chain-level score.

    Parameters
    ----------
    d_model : Hidden dimension (must match ChainEncoder output).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.step_scorer = nn.Linear(d_model, 1)
        self.chain_scorer = nn.Linear(d_model, 1)

    def forward(self, chain_enc: torch.Tensor):
        """Score every step and the overall chain.

        Parameters
        ----------
        chain_enc : FloatTensor of shape [B, n_steps, d_model]

        Returns
        -------
        step_scores  : FloatTensor [B, n_steps] — per-step validity in [0, 1].
        chain_score  : FloatTensor [B]           — overall validity in [0, 1].
        """
        # Per-step scores
        step_scores = torch.sigmoid(self.step_scorer(chain_enc)).squeeze(-1)  # [B, S]

        # Chain score from mean-pooled representation
        pooled = chain_enc.mean(dim=1)                              # [B, d]
        chain_score = torch.sigmoid(self.chain_scorer(pooled)).squeeze(-1)    # [B]

        return step_scores, chain_score


# ---------------------------------------------------------------------------
# CoTVerifierModel
# ---------------------------------------------------------------------------

class CoTVerifierModel(nn.Module):
    """Full Chain-of-Thought Verifier.

    Encodes each step, stacks them, contextualises with ChainEncoder,
    then scores with VerifierHead.

    Parameters
    ----------
    d_model    : Hidden dimension shared across all sub-modules.
    vocab_size : Vocabulary size.
    n_layers   : Number of transformer layers for both encoders.
    """

    def __init__(self, d_model: int, vocab_size: int, n_layers: int = 2) -> None:
        super().__init__()
        self.step_encoder = StepEncoder(d_model, vocab_size, n_layers)
        self.chain_encoder = ChainEncoder(d_model, n_layers)
        self.head = VerifierHead(d_model)

    def forward(self, step_ids_list: List[torch.Tensor]):
        """Verify a chain of reasoning steps.

        Parameters
        ----------
        step_ids_list : list of LongTensors each with shape [B, T_step].
                        Each element is one step across the batch.

        Returns
        -------
        step_scores : FloatTensor [B, n_steps]
        chain_score : FloatTensor [B]
        """
        # Encode each step individually: list of [B, d_model]
        step_vecs = [self.step_encoder(s) for s in step_ids_list]

        # Stack to [B, n_steps, d_model]
        step_encodings = torch.stack(step_vecs, dim=1)

        # Contextualise across chain
        chain_enc = self.chain_encoder(step_encodings)              # [B, n_steps, d]

        # Score
        return self.head(chain_enc)


# ---------------------------------------------------------------------------
# VerifierTrainer
# ---------------------------------------------------------------------------

class VerifierTrainer:
    """Training utilities for CoTVerifierModel.

    Parameters
    ----------
    model : CoTVerifierModel instance.
    lr    : Learning rate for Adam optimiser.
    """

    def __init__(self, model: CoTVerifierModel, lr: float = 1e-4) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def step_loss(
        self,
        step_scores: torch.Tensor,
        step_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy loss on per-step validity labels.

        Parameters
        ----------
        step_scores : FloatTensor [B, n_steps] — predicted probabilities.
        step_labels : FloatTensor [B, n_steps] — ground truth (0 or 1).

        Returns
        -------
        Scalar loss tensor.
        """
        return F.binary_cross_entropy(step_scores, step_labels.float())

    def chain_loss(
        self,
        chain_score: torch.Tensor,
        chain_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy loss on overall chain validity.

        Parameters
        ----------
        chain_score  : FloatTensor [B] — predicted probability.
        chain_labels : FloatTensor [B] — ground truth (0 or 1).

        Returns
        -------
        Scalar loss tensor.
        """
        return F.binary_cross_entropy(chain_score, chain_labels.float())

    def combined_loss(
        self,
        step_scores: torch.Tensor,
        chain_score: torch.Tensor,
        step_labels: torch.Tensor,
        chain_labels: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Weighted combination of step and chain losses.

        Parameters
        ----------
        alpha : Weight for step loss; (1 - alpha) for chain loss.

        Returns
        -------
        Scalar loss tensor.
        """
        sl = self.step_loss(step_scores, step_labels)
        cl = self.chain_loss(chain_score, chain_labels)
        return alpha * sl + (1.0 - alpha) * cl

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        step_ids_list: List[torch.Tensor],
        step_labels: torch.Tensor,
        chain_labels: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Single forward + backward + optimiser step.

        Parameters
        ----------
        step_ids_list : list of LongTensors [B, T_step], one per step.
        step_labels   : FloatTensor [B, n_steps].
        chain_labels  : FloatTensor [B].
        alpha         : Step-loss weight.

        Returns
        -------
        Scalar loss tensor (detached).
        """
        self.model.train()
        self.optimizer.zero_grad()

        step_scores, chain_score = self.model(step_ids_list)
        loss = self.combined_loss(step_scores, chain_score, step_labels, chain_labels, alpha)

        loss.backward()
        self.optimizer.step()

        return loss.detach()


# ---------------------------------------------------------------------------
# ProcessRewardModel
# ---------------------------------------------------------------------------

class ProcessRewardModel(nn.Module):
    """Process-supervision reward model that scores individual steps.

    Architecture mirrors StepEncoder but ends with a scalar reward output.

    Parameters
    ----------
    d_model    : Hidden dimension.
    vocab_size : Vocabulary size.
    n_layers   : Number of transformer encoder layers.
    """

    def __init__(self, d_model: int, vocab_size: int, n_layers: int = 2) -> None:
        super().__init__()
        self.step_encoder = StepEncoder(d_model, vocab_size, n_layers)
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, step_ids: torch.Tensor) -> torch.Tensor:
        """Compute a scalar reward for each item in the batch.

        Parameters
        ----------
        step_ids : LongTensor [B, T_step]

        Returns
        -------
        FloatTensor [B] — reward score in [0, 1].
        """
        enc = self.step_encoder(step_ids)                           # [B, d]
        return torch.sigmoid(self.reward_head(enc)).squeeze(-1)     # [B]

    def outcome_from_steps(self, step_scores: torch.Tensor) -> torch.Tensor:
        """Estimate overall chain correctness as the product of per-step scores.

        Joint probability that every step in the chain is correct.

        Parameters
        ----------
        step_scores : FloatTensor [B, n_steps] — per-step correctness probs.

        Returns
        -------
        FloatTensor [B] — joint probability in [0, 1].
        """
        return step_scores.prod(dim=-1)                             # [B]


# ---------------------------------------------------------------------------
# VerifierConfig
# ---------------------------------------------------------------------------

@dataclass
class VerifierConfig:
    """Default hyper-parameters for the CoT verifier pipeline."""

    d_model: int = 32
    vocab_size: int = 64
    n_layers: int = 2
    lr: float = 1e-4
    alpha: float = 0.5
    max_steps: int = 8
    max_step_len: int = 16
