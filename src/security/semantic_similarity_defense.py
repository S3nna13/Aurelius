"""Semantic similarity defense for the Aurelius LLM research platform.

Detects paraphrase-based evasion of security filters by comparing the
semantic embedding similarity between a reference prompt and a query.
Also provides a query consistency check that runs the same logical query
in multiple surface forms and verifies that model outputs agree.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from src.model.transformer import AureliusTransformer


class SimilarityDefense:
    """Detects paraphrase-based evasion via cosine similarity over model embeddings.

    Attributes:
        model: The AureliusTransformer instance used for embedding extraction.
        similarity_threshold: Cosine similarity above which two prompts are
            considered paraphrases of one another.
        consistency_threshold: Fraction of output-agreeing pairs below which a
            set of query variants is considered inconsistent.
    """

    def __init__(
        self,
        model: AureliusTransformer,
        similarity_threshold: float = 0.85,
        consistency_threshold: float = 0.7,
    ) -> None:
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.consistency_threshold = consistency_threshold

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract a mean-pooled sentence embedding from the last transformer layer.

        Registers a forward hook on the final transformer block to capture its
        output hidden states, runs a no-grad forward pass, then removes the hook.

        Args:
            input_ids: (batch, seq_len) or (seq_len,) token id tensor.

        Returns:
            1-D tensor of shape (d_model,) representing the mean-pooled embedding
            averaged over the batch dimension when batch size > 1.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        captured: list[torch.Tensor] = []

        def hook_fn(module, input, output):
            # TransformerBlock returns (hidden_states, kv_tuple); grab hidden states.
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append(hidden.detach())

        handle = self.model.layers[-1].register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            handle.remove()

        # captured[0]: (batch, seq_len, d_model)
        hidden_states = captured[0]
        # Mean-pool over sequence dimension, then over batch dimension.
        embedding = hidden_states.mean(dim=1).mean(dim=0)  # (d_model,)
        return embedding

    def cosine_similarity(self, embed_a: torch.Tensor, embed_b: torch.Tensor) -> float:
        """Compute cosine similarity between two embedding vectors.

        Args:
            embed_a: 1-D tensor of shape (d_model,).
            embed_b: 1-D tensor of shape (d_model,).

        Returns:
            Scalar float in [-1, 1].
        """
        a = embed_a.float()
        b = embed_b.float()
        sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)
        return float(sim.item())

    def is_paraphrase_evasion(
        self,
        reference_ids: torch.Tensor,
        query_ids: torch.Tensor,
    ) -> bool:
        """Determine whether a query is a paraphrase evasion of a reference prompt.

        A query is flagged as a paraphrase evasion when its cosine similarity to
        the reference exceeds similarity_threshold, indicating the two prompts
        convey the same semantics despite different surface forms.

        Args:
            reference_ids: Token ids for the reference (potentially blocked) prompt.
            query_ids: Token ids for the incoming query to evaluate.

        Returns:
            True if cosine_similarity(embed(reference_ids), embed(query_ids))
            exceeds similarity_threshold.
        """
        sim = self.cosine_similarity(self.embed(reference_ids), self.embed(query_ids))
        return sim > self.similarity_threshold

    def output_consistency(
        self,
        model: AureliusTransformer,
        list_of_input_ids: List[torch.Tensor],
        top_k: int = 1,
    ) -> float:
        """Measure how consistently the model predicts the same token across input variants.

        For each pair of inputs, checks whether the top-1 predicted token at the last
        sequence position is identical. Returns the fraction of pairs that agree.

        Args:
            model: The transformer model to query.
            list_of_input_ids: List of (batch, seq_len) or (seq_len,) input tensors.
            top_k: Number of top tokens to consider (only top-1 is used currently).

        Returns:
            Float in [0, 1]: fraction of pairs that produce the same top-1 token.
            Returns 1.0 when the list contains fewer than two elements.
        """
        n = len(list_of_input_ids)
        if n < 2:
            return 1.0

        top_tokens: list[int] = []
        with torch.no_grad():
            for ids in list_of_input_ids:
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                _, logits, _ = model(ids)
                # logits: (batch, seq_len, vocab_size) — take last position, first batch elem
                token = int(logits[0, -1, :].argmax().item())
                top_tokens.append(token)

        agree = 0
        total_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                if top_tokens[i] == top_tokens[j]:
                    agree += 1

        return agree / total_pairs if total_pairs > 0 else 1.0

    def is_consistent(
        self,
        model: AureliusTransformer,
        list_of_input_ids: List[torch.Tensor],
    ) -> bool:
        """Check whether a set of query variants produces consistent model outputs.

        Args:
            model: The transformer model to query.
            list_of_input_ids: List of input tensors representing the same logical query
                expressed in different surface forms.

        Returns:
            True if output_consistency >= consistency_threshold.
        """
        return self.output_consistency(model, list_of_input_ids) >= self.consistency_threshold

    def scan(
        self,
        reference_ids: torch.Tensor,
        query_ids: torch.Tensor,
    ) -> dict:
        """Run a full similarity scan of query against a reference prompt.

        Args:
            reference_ids: Token ids for the reference prompt.
            query_ids: Token ids for the query to evaluate.

        Returns:
            Dictionary with keys:
                'similarity' (float): cosine similarity between the two embeddings.
                'is_evasion' (bool): whether the query is flagged as paraphrase evasion.
                'similarity_threshold' (float): the threshold used for the decision.
        """
        sim = self.cosine_similarity(self.embed(reference_ids), self.embed(query_ids))
        return {
            "similarity": sim,
            "is_evasion": sim > self.similarity_threshold,
            "similarity_threshold": self.similarity_threshold,
        }
