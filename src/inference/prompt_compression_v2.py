"""LLMLingua-inspired prompt compression: score token importance and compress
prompts to a target ratio while preserving key information.

Methods:
- Perplexity-based: tokens with higher surprise (lower predicted probability)
  are more important; unexpected tokens carry more information.
- Attention-based: tokens that receive more attention from other tokens
  are considered more important.
- Gradient-based: tokens whose embeddings have larger gradient magnitude
  under a cross-entropy loss are more important.

References:
    LLMLingua (Jiang et al., 2023): coarse-to-fine prompt compression.
    Selective Context (Li et al., 2023): compression via self-information.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# TokenImportanceScorer
# ---------------------------------------------------------------------------


class TokenImportanceScorer:
    """Score each token's importance in a prompt.

    Args:
        model: Language model (nn.Module).  For perplexity/gradient methods
            the model must accept (B, T) integer ids and return (B, T, V) logits
            (or a tuple whose first element is that tensor).  For attention
            method the model must expose nn.MultiheadAttention sub-modules.
        method: One of "perplexity", "attention", "gradient".
    """

    VALID_METHODS = {"perplexity", "attention", "gradient"}

    def __init__(self, model: nn.Module, method: str = "perplexity") -> None:
        if method not in self.VALID_METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {self.VALID_METHODS}.")
        self.model = model
        self.method = method

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and extract (B, T, V) logits."""
        out = self.model(input_ids)
        if isinstance(out, torch.Tensor):
            logits = out
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            raise TypeError(
                f"Model output type {type(out)} is not supported; expected Tensor or tuple/list."
            )
        if logits.dim() == 2:
            # (B, V) — single-step model; expand to (B, 1, V)
            logits = logits.unsqueeze(1)
        return logits  # (B, T, V)

    # ------------------------------------------------------------------
    # Scoring methods
    # ------------------------------------------------------------------

    def perplexity_score(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute surprise (negative log-prob) at each token position.

        Higher surprise = more important (unexpected tokens carry more info).

        Args:
            input_ids: (1, T) integer token IDs.

        Returns:
            scores (T,): non-negative float32 surprise values.
        """
        T = input_ids.shape[1]

        with torch.no_grad():
            logits = self._get_logits(input_ids)  # (1, T, V)

        # Shift: logits[t] predicts token at position t+1
        # For position 0 we have no preceding context — assign score 0.
        if T == 1:
            return torch.zeros(1)

        # logits[:, :-1, :] predicts tokens input_ids[:, 1:]
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1, T-1, V)
        targets = input_ids[:, 1:]  # (1, T-1)

        # Gather log-prob of the actual next token
        gathered = log_probs[0, torch.arange(T - 1), targets[0]]  # (T-1,)
        surprise = -gathered  # (T-1,)

        # Position 0 gets score equal to the mean of the rest (no context)
        mean_surprise = surprise.mean()
        scores = torch.cat([mean_surprise.unsqueeze(0), surprise], dim=0)  # (T,)
        return scores.clamp(min=0.0)

    def attention_score(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mean received attention per token, averaged over all layers and heads.

        Args:
            input_ids: (1, T) integer token IDs.

        Returns:
            scores (T,): non-negative float32 attention-mass values.
        """
        T = input_ids.shape[1]
        captured: list[torch.Tensor] = []

        def _hook(_module: nn.Module, _inp, out):
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                weights = out[1]
                if weights is not None and isinstance(weights, torch.Tensor):
                    captured.append(weights.detach())

        handles = []
        for module in self.model.modules():
            if isinstance(module, nn.MultiheadAttention):
                handles.append(module.register_forward_hook(_hook))

        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            for h in handles:
                h.remove()

        if not captured:
            # No MHA layers — fall back to uniform
            return torch.ones(T) / T

        pooled: list[torch.Tensor] = []
        for w in captured:
            w = w.float()
            if w.dim() == 4:  # (B, heads, T, T)
                w = w[0].mean(dim=0)  # (T, T)
            elif w.dim() == 3:  # (B, T, T)  [already avg'd heads]
                w = w[0]  # (T, T)
            elif w.dim() == 2:  # (T, T)
                pass
            else:
                continue
            pooled.append(w)

        if not pooled:
            return torch.ones(T) / T

        avg_attn = torch.stack(pooled).mean(dim=0)  # (T, T)
        # Column-sum: how much total attention each token *receives*
        scores = avg_attn.sum(dim=0)  # (T,)
        return scores.clamp(min=0.0)

    def gradient_score(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Gradient magnitude of the embedding w.r.t. cross-entropy loss.

        Args:
            input_ids: (1, T) integer token IDs.

        Returns:
            scores (T,): non-negative float32 gradient norms.
        """
        T = input_ids.shape[1]
        if T == 1:
            return torch.zeros(1)

        # We need gradients — temporarily enable them
        # Find the embedding layer
        embedding_layer: nn.Embedding | None = None
        for module in self.model.modules():
            if isinstance(module, nn.Embedding):
                embedding_layer = module
                break

        if embedding_layer is None:
            # No embedding found — fall back to perplexity score
            return self.perplexity_score(input_ids)

        # Forward pass with gradient tracking on the embeddings
        ids = input_ids.detach()
        embeds = embedding_layer(ids).float()  # (1, T, d)
        embeds.requires_grad_(True)

        # We need to call the model with embedded inputs rather than ids.
        # Since we can't guarantee the model supports that interface, we use
        # a hook to replace the embedding output.
        embed_out: list[torch.Tensor] = []

        def _embed_hook(_module, _inp, _out):
            embed_out.clear()
            embed_out.append(embeds)
            return embeds

        handle = embedding_layer.register_forward_hook(_embed_hook)
        try:
            logits = self._get_logits(ids)  # (1, T, V)
        finally:
            handle.remove()

        if logits.shape[1] < 2:
            return torch.zeros(T)

        # CE loss: predict tokens 1..T from positions 0..T-1
        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_targets = ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_targets)
        loss.backward()

        if embeds.grad is None:
            return torch.zeros(T)

        grad_norms = embeds.grad[0].norm(dim=-1)  # (T,)
        return grad_norms.detach().clamp(min=0.0)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def score(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute importance scores using the configured method.

        Args:
            input_ids: (1, T) integer token IDs.

        Returns:
            (T,) importance scores.
        """
        if self.method == "perplexity":
            return self.perplexity_score(input_ids)
        if self.method == "attention":
            return self.attention_score(input_ids)
        # gradient
        return self.gradient_score(input_ids)


# ---------------------------------------------------------------------------
# TokenSelector
# ---------------------------------------------------------------------------


class TokenSelector:
    """Select which tokens to keep based on importance scores.

    Args:
        compression_ratio: Fraction of tokens to keep (0.0 to 1.0).
    """

    def __init__(self, compression_ratio: float = 0.5) -> None:
        if not (0.0 <= compression_ratio <= 1.0):
            raise ValueError(f"compression_ratio must be in [0, 1], got {compression_ratio}.")
        self.compression_ratio = compression_ratio

    def target_length(self, original_length: int) -> int:
        """Return ceil(original_length * compression_ratio), minimum 2."""
        raw = math.ceil(original_length * self.compression_ratio)
        # Always keep at least first + last (2 tokens) unless seq is length 1
        return max(min(2, original_length), raw)

    def select(
        self,
        scores: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the most important tokens.

        Always keeps the first and last token as context anchors.

        Args:
            scores: (T,) importance scores.
            input_ids: (T,) or (1, T) token IDs.

        Returns:
            kept_ids: (K,) token IDs at kept positions.
            kept_indices: (K,) sorted positions in the original sequence.
        """
        if input_ids.dim() == 2:
            ids_1d = input_ids[0]
        else:
            ids_1d = input_ids

        T = ids_1d.shape[0]
        k = self.target_length(T)

        if T <= 2 or k >= T:
            # Keep everything
            kept_indices = torch.arange(T, dtype=torch.long)
            return ids_1d.clone(), kept_indices

        # Mandatory: first and last
        mandatory = {0, T - 1}
        k = max(k, len(mandatory))

        # Score with mandatory positions locked in
        masked_scores = scores.clone().float()
        masked_scores[0] = float("inf")
        masked_scores[T - 1] = float("inf")

        _, top_indices = torch.topk(masked_scores, k=k, sorted=False)
        kept_indices, _ = torch.sort(top_indices)  # restore order

        kept_ids = ids_1d[kept_indices]
        return kept_ids, kept_indices


# ---------------------------------------------------------------------------
# ChunkCompressor
# ---------------------------------------------------------------------------


class ChunkCompressor:
    """Compress a long prompt in non-overlapping chunks.

    Args:
        scorer: TokenImportanceScorer instance.
        selector: TokenSelector instance.
        chunk_size: Number of tokens per chunk.
    """

    def __init__(
        self,
        scorer: TokenImportanceScorer,
        selector: TokenSelector,
        chunk_size: int = 32,
    ) -> None:
        self.scorer = scorer
        self.selector = selector
        self.chunk_size = chunk_size

    def compress_chunk(self, chunk_ids: torch.Tensor) -> torch.Tensor:
        """Score tokens in this chunk and select important ones.

        Args:
            chunk_ids: (1, C) or (C,) token IDs for one chunk.

        Returns:
            compressed_ids: (K,) kept token IDs.
        """
        if chunk_ids.dim() == 1:
            chunk_ids_2d = chunk_ids.unsqueeze(0)
        else:
            chunk_ids_2d = chunk_ids

        scores = self.scorer.score(chunk_ids_2d)  # (C,)
        kept_ids, _ = self.selector.select(scores, chunk_ids_2d)
        return kept_ids  # (K,)

    def compress(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Split into chunks, compress each, concatenate.

        Args:
            input_ids: (1, T) or (T,) token IDs.

        Returns:
            compressed_ids: (1, T_c) compressed token IDs.
            compression_stats: dict with keys 'original_len', 'compressed_len', 'ratio'.
        """
        if input_ids.dim() == 2:
            ids_1d = input_ids[0]
        else:
            ids_1d = input_ids

        T = ids_1d.shape[0]
        compressed_parts: list[torch.Tensor] = []

        start = 0
        while start < T:
            end = min(start + self.chunk_size, T)
            chunk = ids_1d[start:end]  # (C,)
            kept = self.compress_chunk(chunk)  # (K,)
            compressed_parts.append(kept)
            start = end

        if compressed_parts:
            compressed_ids_1d = torch.cat(compressed_parts, dim=0)
        else:
            compressed_ids_1d = ids_1d.clone()

        compressed_len = compressed_ids_1d.shape[0]
        ratio = compressed_len / T if T > 0 else 1.0

        stats = {
            "original_len": T,
            "compressed_len": compressed_len,
            "ratio": ratio,
        }
        return compressed_ids_1d.unsqueeze(0), stats  # (1, T_c)


# ---------------------------------------------------------------------------
# SemanticPreserver
# ---------------------------------------------------------------------------


class SemanticPreserver:
    """Evaluate how well a compressed prompt preserves semantic content.

    Args:
        model: Language model (nn.Module) that accepts (B, T) token IDs
            and returns (B, T, d_model) hidden states or logits.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def _get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run model and return mean-pooled representation."""
        with torch.no_grad():
            out = self.model(input_ids)

        if isinstance(out, torch.Tensor):
            h = out
        elif isinstance(out, (tuple, list)):
            h = out[0]
        else:
            raise TypeError(f"Unsupported model output type: {type(out)}")

        if h.dim() == 2:
            # (B, d) — already pooled
            return h[0].float()
        if h.dim() == 3:
            # (B, T, d) — mean pool over sequence
            return h[0].float().mean(dim=0)
        raise ValueError(f"Unexpected hidden state shape: {h.shape}")

    def semantic_similarity(
        self,
        original_ids: torch.Tensor,
        compressed_ids: torch.Tensor,
    ) -> float:
        """Cosine similarity between mean-pooled representations.

        Returns a value in [-1, 1].  Identical sequences → 1.0.

        Args:
            original_ids: (1, T) token IDs.
            compressed_ids: (1, T_c) token IDs.

        Returns:
            float cosine similarity.
        """
        h_orig = self._get_hidden_states(original_ids)  # (d,)
        h_comp = self._get_hidden_states(compressed_ids)  # (d,)

        # Cosine similarity
        dot = (h_orig * h_comp).sum()
        norm_o = h_orig.norm()
        norm_c = h_comp.norm()

        if norm_o == 0 or norm_c == 0:
            return 0.0
        sim = dot / (norm_o * norm_c)
        return float(sim.clamp(-1.0, 1.0))

    def _sequence_perplexity(self, input_ids: torch.Tensor) -> float:
        """Compute perplexity of the sequence under the model.

        Args:
            input_ids: (1, T) token IDs.

        Returns:
            perplexity (positive float).
        """
        T = input_ids.shape[1]
        if T < 2:
            return 1.0

        with torch.no_grad():
            out = self.model(input_ids)

        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out

        if logits.dim() == 2:
            logits = logits.unsqueeze(1)

        # NLL over positions 1..T given logits at 0..T-1
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1, T-1, V)
        targets = input_ids[:, 1:]  # (1, T-1)
        gathered = log_probs[0, torch.arange(T - 1), targets[0]]  # (T-1,)
        nll = -gathered.mean().item()
        return math.exp(nll)

    def perplexity_ratio(
        self,
        original_ids: torch.Tensor,
        compressed_ids: torch.Tensor,
    ) -> float:
        """Ratio of perplexities: ppl(compressed) / ppl(original).

        A ratio close to 1.0 indicates good compression quality.

        Args:
            original_ids: (1, T) token IDs.
            compressed_ids: (1, T_c) token IDs.

        Returns:
            positive float.
        """
        ppl_orig = self._sequence_perplexity(original_ids)
        ppl_comp = self._sequence_perplexity(compressed_ids)
        if ppl_orig == 0.0:
            return 1.0
        return ppl_comp / ppl_orig


# ---------------------------------------------------------------------------
# IterativeCompressor
# ---------------------------------------------------------------------------


class IterativeCompressor:
    """Iteratively compress a prompt until the target ratio is reached.

    Each iteration compresses by a fixed aggressive step (0.7 of current
    length) until ``len(compressed) / len(original) ≤ target_ratio`` or
    ``max_iterations`` is exhausted.

    Args:
        scorer: TokenImportanceScorer instance.
        target_ratio: Desired compression ratio (fraction to keep).
        max_iterations: Hard limit on compression iterations.
    """

    STEP_RATIO = 0.7  # aggressive per-iteration keep fraction

    def __init__(
        self,
        scorer: TokenImportanceScorer,
        target_ratio: float = 0.3,
        max_iterations: int = 5,
    ) -> None:
        if not (0.0 <= target_ratio <= 1.0):
            raise ValueError(f"target_ratio must be in [0, 1], got {target_ratio}.")
        self.scorer = scorer
        self.target_ratio = target_ratio
        self.max_iterations = max_iterations

    def compress(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Compress iteratively until target_ratio or max_iterations.

        Args:
            input_ids: (1, T) token IDs.

        Returns:
            compressed_ids: (1, T_c) token IDs.
            n_iterations: number of iterations performed.
        """
        if input_ids.dim() == 1:
            current = input_ids.unsqueeze(0)
        else:
            current = input_ids.clone()

        original_len = current.shape[1]
        n_iterations = 0

        for _ in range(self.max_iterations):
            current_len = current.shape[1]
            # Check if we've already hit the target
            if original_len > 0 and current_len / original_len <= self.target_ratio:
                break

            # One aggressive compression step: keep STEP_RATIO of current tokens
            selector = TokenSelector(compression_ratio=self.STEP_RATIO)
            scores = self.scorer.score(current)  # (T,)
            kept_ids, _ = selector.select(scores, current)  # (K,)
            current = kept_ids.unsqueeze(0)  # (1, K)
            n_iterations += 1

            # Re-check after compression
            if original_len > 0 and current.shape[1] / original_len <= self.target_ratio:
                break

        return current, n_iterations
