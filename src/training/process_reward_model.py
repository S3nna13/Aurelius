"""Process Reward Model (PRM) for Aurelius.

Implements step-level reward scoring for reasoning chains, following:
  - "Let's Verify Step by Step" (Lightman et al., arXiv:2305.20050)
  - "Math-Shepherd" (Wang et al., arXiv:2312.08935)

PRM assigns scalar rewards to each step in a reasoning chain, enabling:
  - Step-level verification during training (alignment-adjacent)
  - Best-of-N step selection at inference / test-time compute scaling

Architecture:
  - Backbone: AureliusTransformer hidden states (no lm_head logits needed)
  - PRMHead: Linear(d_model -> 1) applied only at [STEP] token positions
  - Training: Binary CE loss over step positions (correct=1, incorrect=0, pad=-1)
  - Inference: sigmoid(logit) gives per-step probability of correctness;
               best chain selected by min-score or mean-score aggregation
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# PRMHead
# ---------------------------------------------------------------------------

class PRMHead(nn.Module):
    """Linear head: d_model -> 1 scalar reward per step position.

    Applied only at positions where input_ids == step_token_id.
    Output shape before squeeze: [B, num_steps, 1]
    Output shape after squeeze:  [B, num_steps]
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1, bias=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Score step-position hidden states.

        Args:
            hidden: [num_valid_steps, d_model]  -- hidden states at step positions

        Returns:
            [num_valid_steps]  -- raw logits (pre-sigmoid)
        """
        return self.linear(hidden).squeeze(-1)  # [N]


# ---------------------------------------------------------------------------
# PRMLoss
# ---------------------------------------------------------------------------

class PRMLoss(nn.Module):
    """Binary cross-entropy loss over step positions only.

    Label convention:
        1  -> step is correct
        0  -> step is incorrect
       -1  -> padding (ignored)
    """

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BCE loss ignoring padded positions.

        Args:
            logits: [B, S]  -- raw logits at each step slot
            labels: [B, S]  -- binary labels (0/1), -1 for padding

        Returns:
            Scalar BCE loss over non-padded positions.
        """
        mask = labels != -1                  # [B, S]
        if mask.sum() == 0:
            return logits.sum() * 0.0       # differentiable zero

        valid_logits = logits[mask]          # [N]
        valid_labels = labels[mask].float()  # [N]
        return F.binary_cross_entropy_with_logits(valid_logits, valid_labels)


# ---------------------------------------------------------------------------
# ProcessRewardModel
# ---------------------------------------------------------------------------

class ProcessRewardModel(nn.Module):
    """PRM: scores each step in a reasoning chain.

    Architecture:
    - Shares backbone with the base LM (AureliusTransformer)
    - Special [STEP] token (step_token_id) inserted at each step boundary
    - At each [STEP] position, PRMHead maps hidden states -> scalar reward
    - Training: BCE loss at each step (correct=1 / incorrect=0, pad=-1)

    Forward signature:
        input_ids       : [B, T]    (contains step_token_id at step boundaries)
        labels          : [B, S]    (binary labels, one per step, -1 for padding)
                                    S = max steps in batch
        attention_mask  : [B, T]    (optional, 1 for real tokens, 0 for pad)

    Returns:
        (loss_or_None, step_rewards: [B, S])
        where S = max number of step tokens found across the batch.
    """

    def __init__(self, config: AureliusConfig) -> None:
        super().__init__()
        self.config = config
        # Use step_token_id from config if available, else default to 50256
        self.step_token_id: int = getattr(config, "prm_step_token_id", 50256)
        # Clamp to valid vocab range
        if self.step_token_id >= config.vocab_size:
            self.step_token_id = config.vocab_size - 1

        self.backbone = AureliusTransformer(config)
        self.head = PRMHead(config.d_model)
        self.loss_fn = PRMLoss()

    # ------------------------------------------------------------------
    # Internal: extract hidden states via forward hook
    # ------------------------------------------------------------------

    def _get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run backbone and return final hidden states [B, T, d_model].

        We hook onto the final RMSNorm layer of the transformer (backbone.norm)
        to get the hidden states *before* the lm_head projection.
        """
        captured: list[torch.Tensor] = []

        def _hook(_module: nn.Module, _inp: tuple, output: torch.Tensor) -> None:
            captured.append(output)

        handle = self.backbone.norm.register_forward_hook(_hook)
        try:
            # AureliusTransformer.forward returns (loss, logits, kv_cache)
            # We only need it for the side-effect on the hook.
            self.backbone(input_ids, mask=attention_mask)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError(
                "PRMHead hook failed to capture hidden states from backbone.norm"
            )

        return captured[0]  # [B, T, d_model]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """Score each step; optionally compute BCE loss.

        Args:
            input_ids      : [B, T]
            labels         : [B, S] optional; S = max steps in batch
            attention_mask : [B, T] optional

        Returns:
            (loss, step_rewards)
            - loss          : scalar Tensor or None
            - step_rewards  : [B, S]  -- raw logits per step (sigmoid for probs)
        """
        B, T = input_ids.shape

        # --- 1. Get backbone hidden states ----------------------------------
        hidden = self._get_hidden_states(input_ids, attention_mask)  # [B, T, d_model]

        # --- 2. Find step token positions -----------------------------------
        step_mask = input_ids == self.step_token_id  # [B, T] bool
        step_counts = step_mask.sum(dim=1)           # [B]
        S = int(step_counts.max().item())
        if S == 0:
            # No step tokens -- return zeros [B, 1]
            step_rewards = torch.zeros(B, 1, device=input_ids.device, dtype=hidden.dtype)
            loss = None if labels is None else step_rewards.sum() * 0.0
            return loss, step_rewards

        # --- 3. Gather hidden states at step positions ----------------------
        # Build padded step_hidden: [B, S, d_model]
        step_hidden_list: list[torch.Tensor] = []
        for b in range(B):
            positions = step_mask[b].nonzero(as_tuple=False).squeeze(1)  # [n_b]
            n_b = positions.shape[0]
            h_b = hidden[b, positions, :]   # [n_b, d_model]
            if n_b < S:
                pad = torch.zeros(
                    S - n_b, self.config.d_model,
                    device=hidden.device, dtype=hidden.dtype,
                )
                h_b = torch.cat([h_b, pad], dim=0)   # [S, d_model]
            step_hidden_list.append(h_b)

        step_hidden = torch.stack(step_hidden_list, dim=0)  # [B, S, d_model]

        # --- 4. Score via PRMHead -------------------------------------------
        # Reshape for linear: [B*S, d_model] -> [B*S] -> [B, S]
        flat_hidden = step_hidden.view(B * S, self.config.d_model)
        flat_logits = self.head.linear(flat_hidden).squeeze(-1)  # [B*S]
        step_rewards = flat_logits.view(B, S)                    # [B, S]

        # Zero out padded positions (where there are no real steps)
        for b in range(B):
            n_b = int(step_counts[b].item())
            if n_b < S:
                step_rewards[b, n_b:] = 0.0

        # --- 5. Loss --------------------------------------------------------
        loss: Optional[torch.Tensor] = None
        if labels is not None:
            # labels: [B, S_label]; S_label may differ from S
            S_label = labels.shape[1]
            if S_label < S:
                pad_labels = torch.full(
                    (B, S - S_label), -1,
                    dtype=labels.dtype, device=labels.device,
                )
                labels = torch.cat([labels, pad_labels], dim=1)  # [B, S]
            elif S_label > S:
                labels = labels[:, :S]

            loss = self.loss_fn(step_rewards, labels)

        return loss, step_rewards


# ---------------------------------------------------------------------------
# StepDataCollator
# ---------------------------------------------------------------------------

class StepDataCollator:
    """Collates step-annotated examples for PRM training.

    Each example is a dict with:
        "input_ids"      : list[int] or Tensor[T]
        "step_positions" : list[int]   -- indices where step tokens appear
        "step_labels"    : list[int]   -- 0/1 label per step

    Output batch dict:
        "input_ids"      : LongTensor[B, T_max]  -- right-padded with pad_id
        "attention_mask" : LongTensor[B, T_max]  -- 1 real, 0 pad
        "step_positions" : LongTensor[B, S_max]  -- right-padded with -1
        "labels"         : LongTensor[B, S_max]  -- right-padded with -1
    """

    def __init__(self, pad_id: int = 0) -> None:
        self.pad_id = pad_id

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        B = len(examples)

        # Determine max sequence length
        T_max = max(
            (len(ex["input_ids"]) for ex in examples),
            default=1,
        )
        # Determine max step count
        S_max = max(
            (len(ex["step_positions"]) for ex in examples),
            default=1,
        )

        input_ids_out = torch.full((B, T_max), self.pad_id, dtype=torch.long)
        attention_mask_out = torch.zeros(B, T_max, dtype=torch.long)
        step_positions_out = torch.full((B, S_max), -1, dtype=torch.long)
        labels_out = torch.full((B, S_max), -1, dtype=torch.long)

        for i, ex in enumerate(examples):
            ids = ex["input_ids"]
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            T = len(ids)
            input_ids_out[i, :T] = torch.tensor(ids, dtype=torch.long)
            attention_mask_out[i, :T] = 1

            positions = ex["step_positions"]
            step_lbls = ex["step_labels"]
            n = len(positions)
            if n > 0:
                step_positions_out[i, :n] = torch.tensor(positions, dtype=torch.long)
                labels_out[i, :n] = torch.tensor(step_lbls, dtype=torch.long)

        return {
            "input_ids": input_ids_out,
            "attention_mask": attention_mask_out,
            "step_positions": step_positions_out,
            "labels": labels_out,
        }


# ---------------------------------------------------------------------------
# PRMInference
# ---------------------------------------------------------------------------

class PRMInference:
    """Best-of-N step selection using PRM scores.

    Given N candidate chains, each a string with steps separated by
    `step_token`, score each step and select the chain with the
    highest aggregate score (min-score or mean-score aggregation).
    """

    def __init__(
        self,
        prm: ProcessRewardModel,
        aggregation: str = "min",
        tokenizer_encode=None,
    ) -> None:
        """
        Args:
            prm          : Trained ProcessRewardModel.
            aggregation  : "min" (default, conservative) or "mean".
            tokenizer_encode : Optional callable(str) -> list[int].
                               If None, a simple character-level encoding is used
                               (for testing without a real tokenizer).
        """
        if aggregation not in ("min", "mean"):
            raise ValueError(f"aggregation must be 'min' or 'mean', got {aggregation!r}")
        self.prm = prm
        self.aggregation = aggregation
        self._encode = tokenizer_encode if tokenizer_encode is not None else self._default_encode

    # ------------------------------------------------------------------
    # Encoding fallback (testing / no-tokenizer path)
    # ------------------------------------------------------------------

    def _default_encode(self, text: str) -> list[int]:
        """Simple character-level encoding into vocab range for testing."""
        vocab = self.prm.config.vocab_size
        return [ord(c) % vocab for c in text]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_chain(
        self,
        chain: str,
        step_token: str = "\n\n",
    ) -> list[float]:
        """Score each step in a single chain string.

        Args:
            chain      : Full reasoning chain as a string.
            step_token : Delimiter that separates steps.

        Returns:
            List of per-step correctness probabilities (via sigmoid).
        """
        step_token_id = self.prm.step_token_id
        vocab = self.prm.config.vocab_size

        # Tokenize the chain, inserting step_token_id at each delimiter
        parts = chain.split(step_token)
        input_ids_list: list[int] = []
        for k, part in enumerate(parts):
            input_ids_list.extend(self._encode(part))
            if k < len(parts) - 1:
                input_ids_list.append(step_token_id)

        if not input_ids_list:
            return []

        # Clamp to vocab range
        input_ids_list = [min(t, vocab - 1) for t in input_ids_list]

        device = next(self.prm.parameters()).device
        input_ids = torch.tensor(
            [input_ids_list], dtype=torch.long, device=device,
        )  # [1, T]

        self.prm.eval()
        _, step_rewards = self.prm(input_ids, labels=None)  # [1, S]
        probs = torch.sigmoid(step_rewards[0])               # [S]
        return probs.tolist()

    @torch.no_grad()
    def select_best(
        self,
        candidates: list[list[str]],
        step_token: str = "\n\n",
    ) -> int:
        """Select the best candidate chain by aggregate step score.

        Args:
            candidates : list of N candidate chains.
                         Each chain is a list of step strings.
                         Pass ["step1", "step2", ...] for each candidate.
            step_token : Delimiter inserted between steps.

        Returns:
            Index of the best candidate (0-based).
        """
        if not candidates:
            raise ValueError("candidates must be non-empty")

        scores: list[float] = []
        for chain_steps in candidates:
            chain_str = step_token.join(chain_steps)
            step_scores = self.score_chain(chain_str, step_token=step_token)
            if not step_scores:
                agg = 0.0
            elif self.aggregation == "min":
                agg = min(step_scores)
            else:  # mean
                agg = sum(step_scores) / len(step_scores)
            scores.append(agg)

        return int(max(range(len(scores)), key=lambda i: scores[i]))


# ---------------------------------------------------------------------------
# Math-Shepherd: synthetic step label generation via MC rollouts
# ---------------------------------------------------------------------------

def generate_synthetic_step_labels(
    model: ProcessRewardModel,
    input_ids: torch.Tensor,
    step_positions: list[int],
    correct_threshold: float = 0.5,
) -> list[int]:
    """Generate synthetic step labels via Monte Carlo rollouts (Math-Shepherd).

    For each step boundary, run a forward pass on the prefix up to that step
    and score the last step. If the score > threshold, label the step as 1.

    This is a lightweight approximation of Math-Shepherd's MC rollout approach:
    in production you would sample from the LM and verify the final answer.
    Here we use the PRM's own scoring as a proxy verifier (self-consistency).

    Args:
        model           : Trained or partially-trained ProcessRewardModel.
        input_ids       : [1, T] input token sequence.
        step_positions  : Positions of step tokens in input_ids.
        correct_threshold: Score threshold for labeling a step as correct.

    Returns:
        list[int] -- binary labels (0/1), one per step.
    """
    labels: list[int] = []
    with torch.no_grad():
        for pos in step_positions:
            # Score the prefix up to and including this step token
            prefix = input_ids[:, : pos + 1]  # [1, pos+1]
            _, rewards = model(prefix)          # [1, S]
            # Use the last step's score as the rollout proxy
            if rewards.shape[1] > 0:
                score = torch.sigmoid(rewards[0, -1]).item()
            else:
                score = 0.0
            labels.append(1 if score >= correct_threshold else 0)
    return labels
