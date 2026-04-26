"""Fusion-in-Decoder (FiD) style multi-passage reading comprehension.

FiD encodes each passage independently through transformer layers up to a
chosen fusion layer, then mean-pools across passages to form a fused
representation used to condition decoding.

Reference: Izacard & Grave (2020), "Leveraging Passage Retrieval with
Generative Models for Open Domain Question Answering".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FiDConfig:
    """Configuration for Fusion-in-Decoder."""

    n_passages: int = 5
    max_passage_len: int = 32
    max_answer_len: int = 16
    fusion_layer: int = -1  # which layer to fuse at (-1 = last layer)


def encode_passages(model: nn.Module, passage_ids_list: list[torch.Tensor]) -> torch.Tensor:
    """Encode each passage through model layers up to fusion_layer and mean-pool.

    Uses a forward hook on the last transformer block to capture hidden states
    before the final norm and LM head. Each passage is run independently.

    Args:
        model: AureliusTransformer instance.
        passage_ids_list: List of (1, T) or (T,) token ID tensors, one per passage.

    Returns:
        Fused representation (1, T, d_model) -- mean-pooled across passages,
        where T is the length of the first passage.
    """
    captured: list[torch.Tensor] = []

    def _hook(module, inp, out):
        # out may be (hidden, kv) tuple -- always take out[0]
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h.detach())

    last_layer = model.layers[-1]
    handle = last_layer.register_forward_hook(_hook)

    hidden_states: list[torch.Tensor] = []
    try:
        model.eval()
        with torch.no_grad():
            for passage_ids in passage_ids_list:
                # Ensure shape (1, T)
                if passage_ids.dim() == 1:
                    passage_ids = passage_ids.unsqueeze(0)
                captured.clear()
                model(passage_ids)
                # captured[0] is (1, T, d_model) from this passage
                hidden_states.append(captured[0])
    finally:
        handle.remove()

    # Truncate/pad to the length of the first passage for uniform stacking
    T = hidden_states[0].shape[1] if hidden_states else 1
    aligned = []
    for hs in hidden_states:
        if hs.shape[1] >= T:
            aligned.append(hs[:, :T, :])
        else:
            pad = torch.zeros(1, T - hs.shape[1], hs.shape[2], dtype=hs.dtype, device=hs.device)
            aligned.append(torch.cat([hs, pad], dim=1))

    # Stack: (n_passages, 1, T, d_model) -> mean over passages -> (1, T, d_model)
    stacked = torch.stack(aligned, dim=0)  # (n_passages, 1, T, d_model)
    fused = stacked.mean(dim=0)  # (1, T, d_model)
    return fused


def fuse_representations(hidden_states_list: list[torch.Tensor]) -> torch.Tensor:
    """Fuse a list of per-passage hidden states by mean-pooling across passages.

    Args:
        hidden_states_list: List of (1, T, d_model) tensors, one per passage.
            All tensors must have the same T and d_model.

    Returns:
        Fused tensor of shape (1, T, d_model).
    """
    if not hidden_states_list:
        raise ValueError("hidden_states_list must not be empty")

    stacked = torch.stack(hidden_states_list, dim=0)  # (n_passages, 1, T, d_model)
    fused = stacked.mean(dim=0)  # (1, T, d_model)
    return fused


class FusionInDecoder:
    """FiD-style reader that encodes multiple passages and fuses before decoding.

    Usage::
        cfg = FiDConfig(n_passages=3, max_passage_len=32, max_answer_len=16)
        fid = FusionInDecoder(model, cfg)
        fused = fid.encode(passage_ids_list)
        tokens = fid.generate(question_ids, passage_ids_list, max_new_tokens=8)
        scores = fid.score_passages(question_ids, passage_ids_list)
    """

    def __init__(self, model: nn.Module, config: FiDConfig) -> None:
        self.model = model
        self.config = config

    def encode(self, passage_ids_list: list[torch.Tensor]) -> torch.Tensor:
        """Encode all passages through the model and return fused hidden state.

        Args:
            passage_ids_list: List of (1, T) or (T,) passage token ID tensors.

        Returns:
            Fused hidden representation of shape (1, T, d_model).
        """
        return encode_passages(self.model, passage_ids_list)

    def generate(
        self,
        question_ids: torch.Tensor,
        passage_ids_list: list[torch.Tensor],
        max_new_tokens: int = 16,
    ) -> torch.Tensor:
        """Encode passages, prepend question tokens, then decode greedily.

        The question and all passage tokens are concatenated into a single
        prefix; the model then generates autoregressively from that prefix.

        Args:
            question_ids: (1, Q) or (Q,) question token IDs.
            passage_ids_list: List of (1, T) or (T,) passage token ID tensors.
            max_new_tokens: Number of new tokens to generate.

        Returns:
            Generated token IDs of shape (max_new_tokens,).
        """
        # Ensure question_ids is (1, Q)
        if question_ids.dim() == 1:
            question_ids = question_ids.unsqueeze(0)

        # Concatenate question with all passage tokens into a single prefix
        parts = [question_ids]
        for pid in passage_ids_list:
            if pid.dim() == 1:
                pid = pid.unsqueeze(0)
            parts.append(pid)

        prefix = torch.cat(parts, dim=1)  # (1, Q + n_passages*T)

        # Truncate prefix if it would leave no room for generation
        max_prefix = self.model.config.max_seq_len - max_new_tokens
        if prefix.shape[1] > max_prefix:
            prefix = prefix[:, :max_prefix]

        generated = prefix.clone()

        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if generated.shape[1] >= self.model.config.max_seq_len:
                    break
                _, logits, _ = self.model(generated)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
                generated = torch.cat([generated, next_token], dim=1)

        # Return only the newly generated tokens
        new_tokens = generated[0, prefix.shape[1] :]
        return new_tokens

    def score_passages(
        self,
        question_ids: torch.Tensor,
        passage_ids_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """Score each passage by its relevance to the question.

        Relevance is the cosine similarity between the mean-pooled last-layer
        hidden states of the question and each passage.

        Args:
            question_ids: (1, Q) or (Q,) question token IDs.
            passage_ids_list: List of (1, T) or (T,) passage token ID tensors.

        Returns:
            Float tensor of shape (n_passages,) with relevance scores.
        """
        if question_ids.dim() == 1:
            question_ids = question_ids.unsqueeze(0)

        last_layer = self.model.layers[-1]

        def _run_and_capture(ids: torch.Tensor) -> torch.Tensor:
            """Run model on ids (1, S), return mean-pooled last-layer hidden (d_model,)."""
            captured: list[torch.Tensor] = []

            def _hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured.append(h.detach())

            handle = last_layer.register_forward_hook(_hook)
            self.model.eval()
            with torch.no_grad():
                self.model(ids)
            handle.remove()
            # (1, S, d_model) -> (d_model,)
            return captured[0].mean(dim=1).squeeze(0)

        q_repr = _run_and_capture(question_ids)
        q_norm = F.normalize(q_repr.float(), dim=-1)

        scores = []
        for pid in passage_ids_list:
            if pid.dim() == 1:
                pid = pid.unsqueeze(0)
            p_repr = _run_and_capture(pid)
            p_norm = F.normalize(p_repr.float(), dim=-1)
            score = (q_norm * p_norm).sum().item()
            scores.append(score)

        return torch.tensor(scores, dtype=torch.float)
