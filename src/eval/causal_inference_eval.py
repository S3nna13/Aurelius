"""Causal Inference Tools for LLM Evaluation.

Provides counterfactual generation, causal effect estimation, and
mediation analysis for understanding model behavior.

Classes:
    CounterfactualGenerator  — Token-level interventions and logit diffs
    CausalEffectEstimator    — Total, direct, and indirect causal effects
    AttentionPatternAnalyzer — Attention rollout and flow
    LogitLensAnalyzer        — Intermediate representation decoding
    CausalEvalConfig         — Default configuration for evaluation experiments
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CausalEvalConfig:
    """Default hyper-parameters for causal evaluation experiments."""

    vocab_size: int = 64
    d_model: int = 32
    n_layers: int = 2
    n_attention_heads: int = 4
    logit_lens_k: int = 5


# ---------------------------------------------------------------------------
# CounterfactualGenerator
# ---------------------------------------------------------------------------


class CounterfactualGenerator:
    """Generate counterfactual sequences and measure their effect on model outputs.

    Works with any nn.Module whose forward signature accepts input_ids of
    shape [B, T] and returns a tensor of shape [B, T, vocab_size] **or** a
    tuple whose first or second element is [B, T, vocab_size].

    The helper ``_logits(input_ids)`` normalises these conventions.
    """

    def __init__(self, model: nn.Module, vocab_size: int) -> None:
        self.model = model
        self.vocab_size = vocab_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run a no-grad forward pass and return logits [B, T, vocab_size]."""
        with torch.no_grad():
            out = self.model(input_ids)
        # Accept: Tensor | tuple(loss_or_none, logits, ...) | tuple(logits, ...)
        if isinstance(out, torch.Tensor):
            return out
        # Tuple: try index-1 first (loss, logits, ...) then index-0
        if isinstance(out, (tuple, list)):
            # If element 0 is None or a scalar it's (loss, logits, ...)
            first = out[0]
            if first is None or (isinstance(first, torch.Tensor) and first.dim() == 0):
                return out[1]
            # Otherwise element 0 might be logits
            if isinstance(first, torch.Tensor) and first.dim() == 3:
                return first
            # Fall back to index 1
            return out[1]
        raise ValueError(f"Unexpected model output type: {type(out)}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def intervene_token(
        self,
        input_ids: torch.Tensor,  # [B, T]
        pos: int,
        new_token_id: int,
    ) -> torch.Tensor:  # [B, T]
        """Return a copy of input_ids with token at *pos* replaced by *new_token_id*."""
        counterfactual = input_ids.clone()
        counterfactual[:, pos] = new_token_id
        return counterfactual

    def intervene_span(
        self,
        input_ids: torch.Tensor,  # [B, T]
        start: int,
        end: int,
        fill_id: int = 0,
    ) -> torch.Tensor:  # [B, T]
        """Return a copy with tokens in [start, end) replaced by *fill_id* (ablation)."""
        counterfactual = input_ids.clone()
        counterfactual[:, start:end] = fill_id
        return counterfactual

    def generate_minimal_pair(
        self,
        input_ids: torch.Tensor,  # [B, T]
        pos: int,
        token_a: int,
        token_b: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two sequences that differ only at *pos* (token_a vs token_b)."""
        seq_a = self.intervene_token(input_ids, pos, token_a)
        seq_b = self.intervene_token(input_ids, pos, token_b)
        return seq_a, seq_b

    def logit_diff(
        self,
        input_ids_a: torch.Tensor,  # [B, T]
        input_ids_b: torch.Tensor,  # [B, T]
        target_pos: int,
    ) -> torch.Tensor:  # [B]
        """Per-example difference of max logits at *target_pos*.

        Returns model(a)[:, target_pos, :].max(-1).values
               - model(b)[:, target_pos, :].max(-1).values
        """
        logits_a = self._logits(input_ids_a)  # [B, T, V]
        logits_b = self._logits(input_ids_b)  # [B, T, V]
        max_a = logits_a[:, target_pos, :].max(dim=-1).values  # [B]
        max_b = logits_b[:, target_pos, :].max(dim=-1).values  # [B]
        return max_a - max_b


# ---------------------------------------------------------------------------
# CausalEffectEstimator
# ---------------------------------------------------------------------------


class CausalEffectEstimator:
    """Estimate total, direct, and indirect causal effects of token interventions.

    All effects are computed via the *do-calculus*: we intervene on the token
    at *intervention_pos* and measure the change in the max-logit at
    *outcome_pos*.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._gen: CounterfactualGenerator | None = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_gen(self, vocab_size: int = 0) -> CounterfactualGenerator:
        """Lazily create a CounterfactualGenerator sharing the same model."""
        if self._gen is None:
            self._gen = CounterfactualGenerator(self.model, vocab_size)
        return self._gen

    def _max_logit_at(self, input_ids: torch.Tensor, pos: int) -> float:
        """Mean over batch of max logit at *pos*."""
        gen = self._get_gen()
        logits = gen._logits(input_ids)  # [B, T, V]
        return logits[:, pos, :].max(dim=-1).values.mean().item()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def total_effect(
        self,
        input_ids: torch.Tensor,  # [B, T]
        intervention_pos: int,
        token_a: int,
        token_b: int,
        outcome_pos: int,
    ) -> float:
        """E[Y | do(X=a)] − E[Y | do(X=b)] via logit difference.

        Uses the max logit at *outcome_pos* as the outcome Y.
        """
        gen = self._get_gen()
        seq_a = gen.intervene_token(input_ids, intervention_pos, token_a)
        seq_b = gen.intervene_token(input_ids, intervention_pos, token_b)
        effect_a = self._max_logit_at(seq_a, outcome_pos)
        effect_b = self._max_logit_at(seq_b, outcome_pos)
        return effect_a - effect_b

    def direct_effect(
        self,
        input_ids: torch.Tensor,  # [B, T]
        intervention_pos: int,
        token_a: int,
        token_b: int,
        outcome_pos: int,
        mediator_pos: int,
    ) -> float:
        """Direct causal effect, holding the mediator token fixed.

        We freeze the mediator to its value under token_a and vary only the
        intervention token.  This approximates the direct effect not mediated
        through *mediator_pos*.

        Concretely:
            1. Build seq_a (intervention=token_a).
            2. Copy mediator token from seq_a into seq_b as a "frozen mediator".
            3. direct_effect = outcome(seq_a) − outcome(seq_b_frozen_mediator).
        """
        gen = self._get_gen()
        seq_a = gen.intervene_token(input_ids, intervention_pos, token_a)
        seq_b = gen.intervene_token(input_ids, intervention_pos, token_b)

        # Freeze mediator in seq_b to its value in seq_a
        mediator_token = seq_a[:, mediator_pos].clone()
        seq_b_frozen = seq_b.clone()
        seq_b_frozen[:, mediator_pos] = mediator_token

        effect_a = self._max_logit_at(seq_a, outcome_pos)
        effect_b = self._max_logit_at(seq_b_frozen, outcome_pos)
        return effect_a - effect_b

    def indirect_effect(self, total: float, direct: float) -> float:
        """Indirect (mediated) causal effect = total − direct."""
        return total - direct


# ---------------------------------------------------------------------------
# AttentionPatternAnalyzer
# ---------------------------------------------------------------------------


class AttentionPatternAnalyzer:
    """Extract and analyse attention patterns from a transformer model.

    Assumes the model has a ``layers`` attribute (nn.ModuleList of
    TransformerBlock-like modules) each containing an ``attn`` sub-module
    that exposes attention weights during its forward pass.

    If the attention module caches its last computed weights in
    ``attn.attn_weights`` (set inside its own forward), those are used
    directly.  Otherwise the analyser hooks into the *output* of the
    attention module and captures whatever weight-like tensor is returned
    alongside the attended output.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_get_attn_module(layer: nn.Module) -> nn.Module | None:
        """Return the attention sub-module of a TransformerBlock, or None."""
        for name in ("attn", "attention", "self_attn", "self_attention"):
            mod = getattr(layer, name, None)
            if mod is not None:
                return mod
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_attention_weights(
        self,
        input_ids: torch.Tensor,  # [B, T]
    ) -> list[torch.Tensor]:  # list[Tensor [B, n_heads, T, T]]
        """Register forward hooks to capture attention weight tensors.

        For each transformer layer the hook intercepts whatever the attention
        sub-module returns and looks for a tensor with 4 dimensions
        (B, n_heads, T, T).  If the attention module exposes an
        ``attn_weights`` attribute it takes priority.

        Returns a list of length n_layers, each tensor [B, n_heads, T, T].
        """
        captured: list[torch.Tensor | None] = []
        hooks = []

        def _make_hook(idx: int):
            def hook(module, inp, output):
                # 1. Check for explicit attribute
                w = getattr(module, "attn_weights", None)
                if w is not None and isinstance(w, torch.Tensor) and w.dim() == 4:
                    captured[idx] = w.detach().clone()
                    return
                # 2. Scan output tuple / tensor for [B, *, T, T] shape
                tensors = output if isinstance(output, (tuple, list)) else (output,)
                B = input_ids.shape[0]
                T = input_ids.shape[1]
                for t in tensors:
                    if (
                        isinstance(t, torch.Tensor)
                        and t.dim() == 4
                        and t.shape[0] == B
                        and t.shape[2] == T
                        and t.shape[3] == T
                    ):
                        captured[idx] = t.detach().clone()
                        return

            return hook

        layers = list(self.model.layers)
        for i, layer in enumerate(layers):
            captured.append(None)
            attn_mod = self._try_get_attn_module(layer)
            target = attn_mod if attn_mod is not None else layer
            hooks.append(target.register_forward_hook(_make_hook(i)))

        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        # Replace None entries with uniform attention as a safe fallback
        B, T = input_ids.shape
        n_heads = 1
        results: list[torch.Tensor] = []
        for w in captured:
            if w is not None:
                n_heads = w.shape[1]
                results.append(w)
            else:
                # Uniform fallback: [B, n_heads, T, T]
                uniform = torch.full((B, n_heads, T, T), 1.0 / T)
                results.append(uniform)
        return results

    def attention_rollout(
        self,
        attn_weights: list[torch.Tensor],  # list[Tensor [B, n_heads, T, T]]
    ) -> torch.Tensor:  # [B, T, T]
        """Compute attention rollout (Abnar & Zuidema, 2020).

        At each layer we:
          1. Average over attention heads → [B, T, T].
          2. Add the identity (residual connection): A_hat = 0.5*A + 0.5*I.
          3. Row-normalise so each row sums to 1.
          4. Matrix-multiply across layers.

        Returns the final rollout matrix; entry [b, i, j] is the importance
        of input token j for position i.
        """
        if not attn_weights:
            raise ValueError("attn_weights must be a non-empty list")

        B = attn_weights[0].shape[0]
        T = attn_weights[0].shape[2]
        eye = torch.eye(T, device=attn_weights[0].device).unsqueeze(0).expand(B, -1, -1)

        rollout: torch.Tensor | None = None

        for layer_w in attn_weights:
            # Average heads: [B, T, T]
            avg = layer_w.mean(dim=1)
            # Add residual and row-normalise
            mixed = 0.5 * avg + 0.5 * eye
            row_sums = mixed.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            mixed = mixed / row_sums

            if rollout is None:
                rollout = mixed
            else:
                rollout = torch.bmm(mixed, rollout)

        return rollout  # type: ignore[return-value]

    def attention_flow(
        self,
        attn_weights: list[torch.Tensor],  # list[Tensor [B, n_heads, T, T]]
        target_pos: int,
    ) -> torch.Tensor:  # [B, T]
        """Approximate max-flow importance of each input token for *target_pos*.

        Uses the attention rollout matrix: the row corresponding to
        *target_pos* gives the flow from each input token.
        """
        rollout = self.attention_rollout(attn_weights)  # [B, T, T]
        return rollout[:, target_pos, :]  # [B, T]


# ---------------------------------------------------------------------------
# LogitLensAnalyzer
# ---------------------------------------------------------------------------


class LogitLensAnalyzer:
    """Decode intermediate transformer hidden states into vocabulary logits.

    The "logit lens" (nostalgebraist, 2020) projects each layer's output
    through the final language-model head to produce a distribution over the
    vocabulary at every layer, revealing how predictions form across depth.

    The analyser assumes:
      - The model has a ``layers`` attribute (nn.ModuleList).
      - The model has a ``lm_head`` (nn.Linear or similar) for token prediction.
      - Optionally, a ``norm`` attribute (final RMSNorm / LayerNorm) that is
        applied before ``lm_head``.  If present it is applied per-layer too,
        following standard logit-lens practice.
    """

    def __init__(self, model: nn.Module, vocab_size: int) -> None:
        self.model = model
        self.vocab_size = vocab_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_intermediate_logits(
        self,
        input_ids: torch.Tensor,  # [B, T]
    ) -> list[torch.Tensor]:  # list[Tensor [B, T, vocab_size]]  (len = n_layers)
        """Project each layer's hidden state through lm_head.

        Hooks on each TransformerBlock output capture the hidden state before
        it is passed to the next layer.
        """
        hidden_states: list[torch.Tensor | None] = [None] * len(self.model.layers)
        hooks = []

        def _make_hook(idx: int):
            def hook(module, inp, output):
                hs = output[0] if isinstance(output, (tuple, list)) else output
                if isinstance(hs, torch.Tensor) and hs.dim() == 3:
                    hidden_states[idx] = hs.detach().clone()

            return hook

        for i, layer in enumerate(self.model.layers):
            hooks.append(layer.register_forward_hook(_make_hook(i)))

        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None:
            raise AttributeError("Model must have a 'lm_head' attribute for LogitLensAnalyzer")

        norm_fn = getattr(self.model, "norm", None)

        intermediate_logits: list[torch.Tensor] = []
        for hs in hidden_states:
            if hs is None:
                # Fallback: zeros
                B, T = input_ids.shape
                intermediate_logits.append(torch.zeros(B, T, self.vocab_size))
                continue
            with torch.no_grad():
                h = norm_fn(hs) if norm_fn is not None else hs
                logits = lm_head(h)  # [B, T, vocab_size]
            intermediate_logits.append(logits)

        return intermediate_logits

    def top_k_predictions(
        self,
        logits: torch.Tensor,  # [B, T, vocab_size]
        k: int = 5,
    ) -> list[list[int]]:
        """Return the top-k token indices for each (batch, position) pair.

        Returns a flat list of length B*T, each element being a list of k
        token indices sorted descending by logit.
        """
        B, T, V = logits.shape
        top_k_indices = logits.topk(min(k, V), dim=-1).indices  # [B, T, k]
        result: list[list[int]] = []
        for b in range(B):
            for t in range(T):
                result.append(top_k_indices[b, t].tolist())
        return result

    def prediction_confidence_by_layer(
        self,
        intermediate_logits: list[torch.Tensor],  # list[Tensor [B, T, vocab]]
        final_tokens: torch.Tensor,  # [B, T]
    ) -> list[float]:
        """Fraction of positions where a layer's top prediction matches final_tokens.

        Returns a list of floats in [0, 1], one per layer.
        """
        confidences: list[float] = []
        for logits in intermediate_logits:
            # argmax over vocabulary → [B, T]
            pred = logits.argmax(dim=-1)  # [B, T]
            correct = (pred == final_tokens).float()  # [B, T]
            confidences.append(correct.mean().item())
        return confidences
