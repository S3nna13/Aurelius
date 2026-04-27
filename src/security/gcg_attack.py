"""GCG adversarial suffix search for language models.

Finds a discrete token suffix that, when appended to a prompt, maximally
increases the probability of a target string via greedy coordinate gradient
optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GCGConfig:
    """Configuration for the GCG adversarial suffix search."""

    suffix_len: int = 10
    n_candidates: int = 16
    n_steps: int = 20
    topk: int = 128
    seed: int = 42


class GCGAttack:
    """Greedy Coordinate Gradient adversarial suffix search."""

    def __init__(self, model: object, config: GCGConfig | None = None) -> None:
        self.model = model
        self.config = config or GCGConfig()

        # Resolve embedding weight at runtime
        if hasattr(model, "embed") and hasattr(model.embed, "weight"):
            self._embed_weight = model.embed.weight
        elif hasattr(model, "tok_emb") and hasattr(model.tok_emb, "weight"):
            self._embed_weight = model.tok_emb.weight
        else:
            raise AttributeError(
                "Cannot locate embedding weight. Expected model.embed.weight or "
                "model.tok_emb.weight."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _vocab_size(self) -> int:
        return self._embed_weight.shape[0]

    def _device(self) -> torch.device:
        return self._embed_weight.device

    def _token_gradients(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient of cross-entropy loss w.r.t. one-hot suffix embeddings.

        Uses a straight-through estimator: forward pass through the embedding
        matrix using one-hot vectors so the gradient flows back to token space.

        Args:
            input_ids: (1, prefix_len + suffix_len) full input token ids.
            target_ids: (1, target_len) target token ids.

        Returns:
            Tensor of shape (suffix_len, vocab_size).
        """
        cfg = self.config
        vocab_size = self._vocab_size()
        self._device()

        # Identify suffix slice within input_ids
        seq_len = input_ids.shape[1]
        suffix_start = seq_len - cfg.suffix_len

        # Build one-hot for the suffix positions; keep gradients here
        suffix_ids = input_ids[0, suffix_start:]  # (suffix_len,)
        one_hot = F.one_hot(suffix_ids, num_classes=vocab_size).float()  # (suffix_len, vocab_size)
        one_hot = one_hot.detach().requires_grad_(True)

        # Straight-through: compute embeddings for the full sequence but
        # replace suffix positions with differentiable one-hot @ W_e
        embed_weight = self._embed_weight  # (vocab_size, d_model)

        # Prefix embeddings (no grad needed)
        with torch.no_grad():
            prefix_embeds = embed_weight[input_ids[0, :suffix_start]]  # (prefix_len, d_model)

        # Suffix embeddings via differentiable matmul
        suffix_embeds = one_hot @ embed_weight  # (suffix_len, d_model)

        # Concatenate and add batch dim
        all_embeds = torch.cat([prefix_embeds.detach(), suffix_embeds], dim=0).unsqueeze(0)
        # (1, seq_len, d_model)

        # Append target tokens (no grad)
        with torch.no_grad():
            target_embeds = embed_weight[target_ids[0]]  # (target_len, d_model)
        torch.cat([all_embeds, target_embeds.unsqueeze(0)], dim=1)
        # (1, seq_len + target_len, d_model)

        # Forward pass manually through the model using embeddings
        # We need to call the model in a way that accepts pre-computed embeddings.
        # Since AureliusTransformer.forward expects input_ids, we compute the
        # loss manually from logits by re-running with full_input_ids and
        # extracting the logit positions that predict the target tokens.
        #
        # Strategy: concatenate input_ids + target_ids and run a single
        # forward pass, then compute cross-entropy on the target positions.

        full_ids = torch.cat([input_ids, target_ids], dim=1)  # (1, seq_len + target_len)
        full_len = full_ids.shape[1]
        target_len = target_ids.shape[1]

        # We need differentiable embeddings for suffix positions only.
        # Reconstruct full input via embedding index lookups + one-hot for suffix.
        with torch.no_grad():
            pre_suffix = embed_weight[full_ids[0, :suffix_start]]  # (prefix_len, d_model)
            post_suffix = embed_weight[
                full_ids[0, suffix_start + cfg.suffix_len :]
            ]  # (target_len, d_model)

        suffix_embeds_for_full = one_hot @ embed_weight  # (suffix_len, d_model)
        combined = torch.cat(
            [pre_suffix.detach(), suffix_embeds_for_full, post_suffix.detach()],
            dim=0,
        ).unsqueeze(0)  # (1, full_len, d_model)

        # Run transformer layers manually (mirrors AureliusTransformer.forward)
        model = self.model
        x = combined
        freqs_cis = model.freqs_cis[:full_len]

        for layer in model.layers:
            x, _, _ = layer(x, freqs_cis, None, None)

        x = model.norm(x)
        logits = model.lm_head(x)  # (1, full_len, vocab_size)

        # Loss: predict target tokens from the positions just before them
        # Positions [seq_len-1 : seq_len-1+target_len] in logits predict
        # positions [seq_len : seq_len+target_len] = target tokens.
        pred_logits = logits[
            0, seq_len - 1 : seq_len - 1 + target_len, :
        ]  # (target_len, vocab_size)
        loss = F.cross_entropy(pred_logits, target_ids[0])

        loss.backward()

        grad = one_hot.grad  # (suffix_len, vocab_size)
        return grad.detach()

    def _sample_candidates(
        self,
        current_suffix: torch.Tensor,
        token_grads: torch.Tensor,
    ) -> torch.Tensor:
        """Sample candidate suffix replacements guided by token gradients.

        For each candidate, picks a random suffix position uniformly, then
        samples a replacement token from the top-k tokens at that position
        by negative gradient (ascending probability).

        Args:
            current_suffix: (suffix_len,) current suffix token ids.
            token_grads: (suffix_len, vocab_size) gradient tensor.

        Returns:
            Tensor of shape (n_candidates, suffix_len).
        """
        cfg = self.config
        suffix_len = current_suffix.shape[0]
        n_candidates = cfg.n_candidates
        topk = min(cfg.topk, self._vocab_size())

        # Top-k tokens per position by most negative gradient
        # (lower gradient = better token substitution)
        neg_grad = -token_grads  # (suffix_len, vocab_size)
        topk_tokens = neg_grad.topk(topk, dim=-1).indices  # (suffix_len, topk)

        candidates = current_suffix.unsqueeze(0).expand(n_candidates, -1).clone()
        # (n_candidates, suffix_len)

        # For each candidate, choose a random position and a random top-k token
        positions = torch.randint(0, suffix_len, (n_candidates,), device=current_suffix.device)
        token_indices = torch.randint(0, topk, (n_candidates,), device=current_suffix.device)

        for i in range(n_candidates):
            pos = positions[i].item()
            tok_idx = token_indices[i].item()
            candidates[i, pos] = topk_tokens[pos, tok_idx]

        return candidates

    def _eval_loss(
        self,
        prefix_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> float:
        """Evaluate cross-entropy loss of target tokens given prefix + suffix.

        Args:
            prefix_ids: (1, prefix_len) prefix token ids.
            suffix_ids: (1, suffix_len) or (suffix_len,) suffix token ids.
            target_ids: (1, target_len) target token ids.

        Returns:
            Scalar float cross-entropy loss.
        """
        if suffix_ids.dim() == 1:
            suffix_ids = suffix_ids.unsqueeze(0)

        full_ids = torch.cat([prefix_ids, suffix_ids, target_ids], dim=1)
        seq_len = prefix_ids.shape[1] + suffix_ids.shape[1]
        target_len = target_ids.shape[1]

        with torch.no_grad():
            _, logits, _ = self.model(full_ids)

        # Predict target tokens
        pred_logits = logits[0, seq_len - 1 : seq_len - 1 + target_len, :]
        loss = F.cross_entropy(pred_logits, target_ids[0])
        return loss.item()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        prefix_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, float, list[float]]:
        """Run GCG greedy coordinate search.

        Args:
            prefix_ids: (1, prefix_len) prefix token ids.
            target_ids: (1, target_len) target token ids.

        Returns:
            best_suffix: LongTensor of shape (suffix_len,).
            best_loss: float, minimum loss achieved.
            loss_history: list of per-step best losses (length n_steps).
        """
        cfg = self.config
        torch.manual_seed(cfg.seed)

        device = self._device()
        vocab_size = self._vocab_size()

        # Initialise suffix with random tokens
        current_suffix = torch.randint(0, vocab_size, (cfg.suffix_len,), device=device)

        best_suffix = current_suffix.clone()
        best_loss = float("inf")
        loss_history: list[float] = []

        for step in range(cfg.n_steps):
            # Compute gradients for current suffix
            input_ids = torch.cat([prefix_ids, current_suffix.unsqueeze(0)], dim=1)
            token_grads = self._token_gradients(input_ids, target_ids)

            # Sample candidate replacements
            candidates = self._sample_candidates(current_suffix, token_grads)
            # (n_candidates, suffix_len)

            # Evaluate all candidates
            step_best_loss = float("inf")
            step_best_suffix = current_suffix.clone()

            for i in range(cfg.n_candidates):
                cand_suffix = candidates[i]  # (suffix_len,)
                loss_val = self._eval_loss(prefix_ids, cand_suffix, target_ids)
                if loss_val < step_best_loss:
                    step_best_loss = loss_val
                    step_best_suffix = cand_suffix.clone()

            current_suffix = step_best_suffix
            loss_history.append(step_best_loss)

            if step_best_loss < best_loss:
                best_loss = step_best_loss
                best_suffix = step_best_suffix.clone()

        return best_suffix, best_loss, loss_history
