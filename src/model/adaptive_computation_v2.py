"""
Adaptive Computation Time (Graves 2016) + Universal Transformer (Dehghani et al. 2018).

Implements per-token halting via ACT and a shared-weight (recurrent) transformer layer.
Pure PyTorch only — no third-party ML libraries.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# HaltingUnit
# ---------------------------------------------------------------------------


class HaltingUnit(nn.Module):
    """Per-token halting probability predictor.

    Maps each token representation to a scalar halt probability in (0, 1).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Return per-token halting probabilities.

        Args:
            x: (B, T, D) token representations.

        Returns:
            halting_probs: (B, T) values in (0, 1).
        """
        return torch.sigmoid(self.proj(x)).squeeze(-1)  # (B, T)

    def should_halt(self, cumulative_prob: Tensor, threshold: float = 0.99) -> Tensor:
        """Return True for tokens whose cumulative halting probability exceeds threshold.

        Args:
            cumulative_prob: (B, T) cumulative probabilities accumulated so far.
            threshold: Scalar float — halt when cumulative_prob > threshold.

        Returns:
            Boolean mask (B, T).
        """
        return cumulative_prob > threshold


# ---------------------------------------------------------------------------
# ACTState
# ---------------------------------------------------------------------------


class ACTState:
    """Mutable state object that accumulates ACT information across recurrent steps."""

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        d_model: int,
        device: str = "cpu",
    ) -> None:
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.device = device

        self.cumulative_prob: Tensor = torch.zeros(batch_size, seq_len, device=device)
        self.remainder: Tensor = torch.zeros(batch_size, seq_len, device=device)
        self.n_updates: Tensor = torch.zeros(batch_size, seq_len, device=device)
        self.accumulated_output: Tensor = torch.zeros(batch_size, seq_len, d_model, device=device)
        # Bool mask: True means the token has already halted.
        self.halted: Tensor = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        # Keep the last step's output for the remainder term.
        self._last_x: Tensor = torch.zeros(batch_size, seq_len, d_model, device=device)

    # ------------------------------------------------------------------

    def update(
        self,
        x: Tensor,
        p: Tensor,
        threshold: float = 0.99,
        eps: float = 1e-6,
    ) -> None:
        """Update ACT state for one recurrent step.

        For non-halted tokens:
          - Add p to cumulative_prob.
          - Add p * x to accumulated_output.
          - If cumulative_prob > threshold (or > 1 - eps): mark halted,
            store remainder.

        Args:
            x:         (B, T, D) output of the shared layer at this step.
            p:         (B, T)    halting probabilities at this step.
            threshold: Cumulative probability above which a token halts.
            eps:       Small value so we never divide by zero in remainder.
        """
        # Store for finalize remainder term.
        self._last_x = x

        active = ~self.halted  # (B, T) tokens still being updated

        # Update cumulative probability for active tokens only.
        new_cum = self.cumulative_prob + p * active.float()

        # Halt when cumulative prob crosses the requested threshold *or*
        # reaches the hard cap of 1 - eps.
        halt_threshold = min(threshold, 1.0 - eps)
        halting_now = active & (new_cum > halt_threshold)  # (B, T)

        # For tokens halting now, the effective probability is the remainder:
        #   r = 1 - (cumulative_prob_before_this_step)
        # For still-active tokens, use p directly.
        effective_p = torch.where(
            halting_now,
            torch.clamp(1.0 - self.cumulative_prob, min=0.0),
            p * active.float(),
        )

        # Accumulate weighted outputs.
        self.accumulated_output = self.accumulated_output + effective_p.unsqueeze(-1) * x

        # Store remainder values for tokens halting now.
        self.remainder = torch.where(
            halting_now,
            torch.clamp(1.0 - self.cumulative_prob, min=0.0),
            self.remainder,
        )

        # Update cumulative prob (cap at 1.0).
        self.cumulative_prob = torch.clamp(new_cum, max=1.0)

        # Update step counters.
        self.n_updates = self.n_updates + active.float()

        # Mark newly halted tokens.
        self.halted = self.halted | halting_now

    # ------------------------------------------------------------------

    def finalize(self) -> Tensor:
        """Return the final ACT output.

        For tokens that halted mid-loop the accumulated_output already contains
        the remainder-weighted last step. For tokens that were still active when
        we ran out of steps we add remainder * last_x explicitly.

        Returns:
            output: (B, T, D)
        """
        # Tokens that never halted by the threshold need the remainder term.
        never_halted = ~self.halted
        if never_halted.any():
            remainder = (1.0 - self.cumulative_prob) * never_halted.float()
            self.accumulated_output = (
                self.accumulated_output + remainder.unsqueeze(-1) * self._last_x
            )
            self.halted = torch.ones_like(self.halted)

        return self.accumulated_output


# ---------------------------------------------------------------------------
# UniversalTransformerLayer
# ---------------------------------------------------------------------------


class UniversalTransformerLayer(nn.Module):
    """A standard transformer block whose weights are shared across recurrent steps.

    Adds a temporal (step-index) positional encoding so the model can distinguish
    which recurrent iteration it is on.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------

    def _temporal_encoding(self, x: Tensor, step: int) -> Tensor:
        """Add a sin/cos temporal encoding derived from the recurrent step index.

        The encoding follows the standard sinusoidal schedule but uses `step`
        in place of the position index, broadcasted to every sequence position.

        Returns:
            x + temporal_enc  (B, T, D)
        """
        device = x.device
        d = self.d_model

        # Frequency denominators: shape (d//2,)
        i = torch.arange(0, d, 2, dtype=torch.float32, device=device)
        denom = torch.pow(10000.0, i / d)  # (d//2,)

        # step is a plain int — divide as a float scalar.
        angles = step / denom  # (d//2,)

        # Interleave sin / cos into a (D,) vector then reshape to (1, 1, D).
        enc = torch.zeros(d, device=device)
        half = d // 2
        enc[0::2] = torch.sin(angles[: half + d % 2])
        enc[1::2] = torch.cos(angles[:half])

        return x + enc.view(1, 1, d).to(x.dtype)

    # ------------------------------------------------------------------

    def forward(self, x: Tensor, step: int) -> Tensor:
        """Apply one recurrent step of the Universal Transformer.

        Args:
            x:    (B, T, D) input.
            step: int — which recurrent iteration we are on (0-indexed).

        Returns:
            out: (B, T, D)
        """
        # Add temporal encoding.
        x = self._temporal_encoding(x, step)

        # Self-attention sub-layer.
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward sub-layer.
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


# ---------------------------------------------------------------------------
# ACTTransformer
# ---------------------------------------------------------------------------


class ACTTransformer(nn.Module):
    """Universal Transformer with Adaptive Computation Time halting.

    Runs a single shared UTLayer recurrently until each token decides to halt
    (or until max_steps is reached).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        vocab_size: int,
        max_steps: int = 8,
        act_threshold: float = 0.99,
        time_penalty: float = 0.01,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_steps = max_steps
        self.act_threshold = act_threshold
        self.time_penalty = time_penalty

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.ut_layer = UniversalTransformerLayer(d_model, n_heads)
        self.halting_unit = HaltingUnit(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    # ------------------------------------------------------------------

    def forward(self, input_ids: Tensor):
        """Run ACT loop and return logits, auxiliary loss, and mean steps.

        Args:
            input_ids: (B, T) integer token ids.

        Returns:
            logits:     (B, T, vocab_size)
            aux_loss:   scalar tensor — time_penalty * mean(n_updates)
            mean_steps: float — average computational steps used per token
        """
        B, T = input_ids.shape
        device = input_ids.device

        x = self.embedding(input_ids)  # (B, T, D)

        state = ACTState(B, T, self.d_model, device=str(device))

        for step in range(self.max_steps):
            h = self.ut_layer(x, step)  # (B, T, D)
            p = self.halting_unit(h)  # (B, T)
            state.update(h, p, threshold=self.act_threshold)

            if state.halted.all():
                break

        output = state.finalize()  # (B, T, D)
        logits = self.lm_head(output)  # (B, T, V)

        # Auxiliary loss: penalise unnecessary computation.
        mean_steps_tensor = state.n_updates.mean()
        if self.time_penalty == 0.0:
            aux_loss = torch.tensor(0.0, device=device)
        else:
            aux_loss = self.time_penalty * mean_steps_tensor

        mean_steps: float = mean_steps_tensor.item()

        return logits, aux_loss, mean_steps


# ---------------------------------------------------------------------------
# FixedDepthUniversalTransformer
# ---------------------------------------------------------------------------


class FixedDepthUniversalTransformer(nn.Module):
    """Universal Transformer with a fixed (non-adaptive) number of recurrent steps.

    Useful as an ablation: the same shared layer is applied exactly n_steps times.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        vocab_size: int,
        n_steps: int = 4,
    ) -> None:
        super().__init__()
        self.n_steps = n_steps

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.ut_layer = UniversalTransformerLayer(d_model, n_heads)
        self.lm_head = nn.Linear(d_model, vocab_size)

    # ------------------------------------------------------------------

    def forward(self, input_ids: Tensor) -> Tensor:
        """Apply the shared layer exactly n_steps times.

        Args:
            input_ids: (B, T) integer token ids.

        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.embedding(input_ids)

        for step in range(self.n_steps):
            x = self.ut_layer(x, step)

        return self.lm_head(x)
