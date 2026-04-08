"""Aurelius — Online Direct Preference Optimization (Online DPO).

Generates chosen/rejected preference pairs on-the-fly from the policy model
itself, scoring candidates with a reward model and computing DPO loss.

Algorithm:
  For each prompt:
    1. Sample n_candidates completions from the current policy.
    2. Score each full sequence (prompt + response) with the reward model.
    3. Select chosen = argmax reward, rejected = argmin reward.
    4. Compute DPO loss against the frozen reference model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OnlineDPOConfig:
    """Configuration for online DPO training."""

    n_candidates: int = 4          # completions to generate per prompt
    max_new_tokens: int = 64       # maximum response length in tokens
    temperature: float = 1.0       # sampling temperature (1.0 = categorical sampling)
    dpo_beta: float = 0.1          # KL penalty coefficient (same role as DPO beta)
    top_p: float = 1.0             # nucleus sampling threshold (1.0 = disabled)
    reward_batch_size: int = 8     # batch size for reward model scoring


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_response_log_probs(
    model: nn.Module,
    full_ids: torch.Tensor,   # (1, seq_len) — prompt + response
    prompt_len: int,
) -> torch.Tensor:
    """Return the summed log-prob of the response tokens under *model*.

    Args:
        model: Policy or reference model.
               Forward signature: (loss, logits, present_key_values) = model(input_ids)
        full_ids: Shape (1, seq_len) — prompt concatenated with response.
        prompt_len: Number of prompt tokens; response starts at index prompt_len.

    Returns:
        Scalar tensor — sum of log-probs over response positions.
    """
    _, logits, _ = model(full_ids)                    # (1, seq_len, vocab)
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1, seq_len-1, vocab)

    # Build response mask: 1 for positions prompt_len..seq_len-1, shifted by 1
    seq_len = full_ids.shape[1]
    # After the shift, the response tokens to predict start at index (prompt_len - 1)
    # because position i predicts token i+1.
    mask = torch.zeros(seq_len - 1, dtype=torch.float, device=full_ids.device)
    resp_start = max(prompt_len - 1, 0)
    mask[resp_start:] = 1.0

    token_lp = log_probs[0].gather(1, full_ids[0, 1:].unsqueeze(-1)).squeeze(-1)  # (seq_len-1,)
    return (token_lp * mask).sum()


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus (top-p) filtering to a 1-D logits vector."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[sorted_indices_to_remove] = float("-inf")
    # Scatter back to original indexing
    filtered = torch.full_like(logits, float("-inf"))
    filtered[sorted_indices] = sorted_logits
    return filtered


# ---------------------------------------------------------------------------
# OnlineDPOTrainer
# ---------------------------------------------------------------------------

class OnlineDPOTrainer:
    """Online DPO trainer that generates preference pairs from the live policy.

    Args:
        model: Trainable policy (AureliusTransformer).
        ref_model: Frozen reference model (deep copy of the initial policy).
        reward_model: Reward model used to score candidates.
        optimizer: PyTorch optimizer bound to *model*'s parameters.
        cfg: OnlineDPOConfig.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: OnlineDPOConfig,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def generate_candidates(self, prompt_ids: torch.Tensor) -> list[torch.Tensor]:
        """Generate *n_candidates* response tensors for a single prompt.

        Args:
            prompt_ids: 1-D tensor of shape (prompt_len,).

        Returns:
            List of n_candidates 1-D tensors.  Each tensor is the *response*
            tokens only (not the prompt prefix).
        """
        candidates: list[torch.Tensor] = []
        device = prompt_ids.device

        self.model.eval()
        with torch.no_grad():
            for _ in range(self.cfg.n_candidates):
                current = prompt_ids.clone()  # (seq,)

                for _ in range(self.cfg.max_new_tokens):
                    input_ids = current.unsqueeze(0)  # (1, seq)
                    _, logits, _ = self.model(input_ids)
                    next_logits = logits[0, -1, :]  # (vocab,)

                    if self.cfg.temperature != 1.0:
                        next_logits = next_logits / self.cfg.temperature

                    # Top-p filtering
                    if self.cfg.top_p < 1.0:
                        next_logits = _top_p_filter(next_logits, self.cfg.top_p)

                    # Sample or greedy
                    if self.cfg.temperature == 0.0:
                        next_token = next_logits.argmax(dim=-1, keepdim=True)
                    else:
                        probs = F.softmax(next_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)

                    current = torch.cat([current, next_token], dim=0)

                # Return only the response part (strip prompt)
                response = current[prompt_ids.shape[0]:]
                candidates.append(response)

        return candidates

    # ------------------------------------------------------------------
    # Pair selection
    # ------------------------------------------------------------------

    def select_pair(
        self,
        prompt_ids: torch.Tensor,
        candidates: list[torch.Tensor],
        reward_model: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score candidates and return (chosen, rejected) full sequences.

        Args:
            prompt_ids: 1-D prompt tensor, shape (prompt_len,).
            candidates: List of 1-D response tensors.
            reward_model: Callable that accepts (B, seq_len) and returns (B,) scores.

        Returns:
            Tuple of (chosen_ids, rejected_ids) — full sequences (prompt + response),
            each a 1-D tensor.
        """
        prompt_len = prompt_ids.shape[0]
        device = prompt_ids.device

        # Build full sequences and find the maximum length
        full_seqs = [torch.cat([prompt_ids, resp], dim=0) for resp in candidates]
        max_len = max(s.shape[0] for s in full_seqs)

        # Pad to same length for batched scoring
        padded = torch.zeros(len(full_seqs), max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(full_seqs):
            padded[i, : seq.shape[0]] = seq

        # Score in reward_batch_size chunks
        scores: list[float] = []
        reward_model.eval()
        with torch.no_grad():
            for start in range(0, len(full_seqs), self.cfg.reward_batch_size):
                chunk = padded[start : start + self.cfg.reward_batch_size]
                chunk_scores = reward_model(chunk)  # (chunk_size,)
                scores.extend(chunk_scores.tolist())

        scores_tensor = torch.tensor(scores, dtype=torch.float, device=device)
        chosen_idx = int(scores_tensor.argmax().item())
        rejected_idx = int(scores_tensor.argmin().item())

        return full_seqs[chosen_idx], full_seqs[rejected_idx]

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, prompt_ids_batch: torch.Tensor) -> dict[str, float]:
        """Run one Online DPO training step over a batch of prompts.

        For each prompt in the batch:
          1. Generate n_candidates completions.
          2. Select the best (chosen) and worst (rejected) by reward.
          3. Compute DPO loss against the frozen reference model.

        The loss is averaged across the batch.

        Args:
            prompt_ids_batch: Shape (B, prompt_len) — batch of prompts.

        Returns:
            Dict with keys: loss, chosen_reward, rejected_reward, reward_margin.
            All values are Python floats.
        """
        self.model.train()
        self.ref_model.eval()

        batch_size = prompt_ids_batch.shape[0]
        device = prompt_ids_batch.device

        total_loss = torch.tensor(0.0, device=device)
        total_chosen_reward = 0.0
        total_rejected_reward = 0.0

        for i in range(batch_size):
            prompt_ids = prompt_ids_batch[i]  # (prompt_len,)
            prompt_len = prompt_ids.shape[0]

            # 1. Generate candidates
            candidates = self.generate_candidates(prompt_ids)

            # 2. Select pair
            chosen_ids, rejected_ids = self.select_pair(
                prompt_ids, candidates, self.reward_model
            )

            # Record rewards for metrics
            with torch.no_grad():
                r_chosen = self.reward_model(chosen_ids.unsqueeze(0)).item()
                r_rejected = self.reward_model(rejected_ids.unsqueeze(0)).item()
            total_chosen_reward += r_chosen
            total_rejected_reward += r_rejected

            # 3. Compute DPO loss
            # Policy log-probs (with gradient)
            self.model.train()
            pi_chosen = _compute_response_log_probs(
                self.model, chosen_ids.unsqueeze(0), prompt_len
            )
            pi_rejected = _compute_response_log_probs(
                self.model, rejected_ids.unsqueeze(0), prompt_len
            )

            # Reference log-probs (frozen)
            with torch.no_grad():
                ref_chosen = _compute_response_log_probs(
                    self.ref_model, chosen_ids.unsqueeze(0), prompt_len
                )
                ref_rejected = _compute_response_log_probs(
                    self.ref_model, rejected_ids.unsqueeze(0), prompt_len
                )

            # DPO loss: -log σ(β * ((log π_c - log π_ref_c) - (log π_r - log π_ref_r)))
            log_ratio_diff = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
            loss_i = -F.logsigmoid(self.cfg.dpo_beta * log_ratio_diff)
            total_loss = total_loss + loss_i

        # Average across batch
        avg_loss = total_loss / batch_size

        # Backprop and optimizer step
        self.optimizer.zero_grad()
        avg_loss.backward()
        self.optimizer.step()

        avg_chosen_reward = total_chosen_reward / batch_size
        avg_rejected_reward = total_rejected_reward / batch_size

        return {
            "loss": avg_loss.item(),
            "chosen_reward": avg_chosen_reward,
            "rejected_reward": avg_rejected_reward,
            "reward_margin": avg_chosen_reward - avg_rejected_reward,
        }
