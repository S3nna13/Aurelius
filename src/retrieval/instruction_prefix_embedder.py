"""Instruction-prefix-aware embedder wrapper.

Wraps a trained :class:`DenseEmbedder` with an instruction prefix that is
prepended to the input text before tokenization, following the recipes of
the Instructor and BGE-M3 model families. Different tasks (``query``,
``passage``, ``code_search_query``, ...) select different prefixes, which
steer the same underlying encoder toward task-appropriate representations
without retraining or parameter duplication.

References:
    - Su et al. (2022). "One Embedder, Any Task: Instruction-Finetuned
      Text Embeddings." arXiv:2212.09741.
    - Chen et al. (2024). "BGE M3-Embedding: Multi-Lingual,
      Multi-Functionality, Multi-Granularity Text Embeddings Through
      Self-Knowledge Distillation." arXiv:2402.03216.

The wrapper is pure PyTorch and only imports from within ``src.retrieval``.

Public surface:
    - INSTRUCTION_PREFIXES
    - InstructionPrefixEmbedder
"""

from __future__ import annotations

from typing import Callable, List

import torch

from .dense_embedding_trainer import DenseEmbedder


INSTRUCTION_PREFIXES: dict[str, str] = {
    "query": "Represent this sentence for searching relevant passages: ",
    "passage": "Represent this passage for retrieval: ",
    "code_search_query": "Represent this query for searching relevant code snippets: ",
    "code_search_passage": "Represent this code snippet for retrieval: ",
    "question_answering": "Represent this question for retrieving supporting answers: ",
    "classification": "Represent this sentence for classification: ",
}


def _set_inference_mode(module: torch.nn.Module) -> None:
    """Switch module to inference (eval) mode without dropout noise."""
    module.train(False)


class InstructionPrefixEmbedder:
    """Task-conditioned wrapper around a :class:`DenseEmbedder`.

    The wrapper holds a frozen reference to an existing embedder and a
    tokenizer callable ``str -> list[int]``. On each call, the configured
    task's prefix is concatenated to the input text before tokenization,
    so the downstream encoder sees ``f"{prefix}{text}"``.

    Unknown tasks raise ``KeyError``; sequences that do not fit within
    ``embedder.config.max_seq_len`` after prefixing raise ``ValueError``
    (we pick raise over silent truncation to avoid corrupting retrieval
    semantics).
    """

    def __init__(
        self,
        dense_embedder: DenseEmbedder,
        tokenizer: Callable[[str], List[int]],
    ) -> None:
        if not isinstance(dense_embedder, DenseEmbedder):
            raise TypeError(
                "dense_embedder must be DenseEmbedder, got "
                f"{type(dense_embedder).__name__}"
            )
        if not callable(tokenizer):
            raise TypeError(
                f"tokenizer must be callable, got {type(tokenizer).__name__}"
            )
        self.dense_embedder = dense_embedder
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_prefix(self, task: str) -> str:
        if task not in INSTRUCTION_PREFIXES:
            raise KeyError(
                f"unknown task {task!r}; known tasks: "
                f"{sorted(INSTRUCTION_PREFIXES)}"
            )
        return INSTRUCTION_PREFIXES[task]

    def _tokenize(self, text: str, task: str) -> torch.Tensor:
        prefix = self._resolve_prefix(task)
        prefixed = prefix + text
        ids = self.tokenizer(prefixed)
        if not isinstance(ids, list):
            raise TypeError(
                "tokenizer must return list[int], got "
                f"{type(ids).__name__}"
            )
        max_len = self.dense_embedder.config.max_seq_len
        if len(ids) == 0:
            # Preserve at least one token so the encoder receives a
            # well-formed [B, T>=1] tensor; use pad id so the attention
            # mask treats it as empty.
            ids = [self.dense_embedder.config.pad_token_id]
        if len(ids) > max_len:
            raise ValueError(
                f"tokenized length {len(ids)} exceeds max_seq_len {max_len} "
                f"after prefixing for task={task!r}"
            )
        return torch.tensor(ids, dtype=torch.long)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode(self, text: str, task: str = "query") -> torch.Tensor:
        """Encode a single text string under ``task`` to ``[embed_dim]``."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        ids = self._tokenize(text, task).unsqueeze(0)  # [1, T]
        _set_inference_mode(self.dense_embedder)
        with torch.no_grad():
            out = self.dense_embedder(ids)  # [1, D]
        return out.squeeze(0)

    def encode_batch(self, texts: List[str], task: str) -> torch.Tensor:
        """Encode a list of texts to ``[B, embed_dim]``.

        Pads each row to the batch max length with ``pad_token_id`` so
        the encoder receives a rectangular tensor; the embedder's
        attention mask already zeroes those positions out.
        """
        if not isinstance(texts, list):
            raise TypeError(
                f"texts must be list[str], got {type(texts).__name__}"
            )
        if len(texts) == 0:
            raise ValueError("texts must be non-empty")
        for t in texts:
            if not isinstance(t, str):
                raise TypeError(
                    f"all texts must be str, got {type(t).__name__}"
                )
        token_lists = [self._tokenize(t, task).tolist() for t in texts]
        max_len = max(len(ids) for ids in token_lists)
        pad_id = self.dense_embedder.config.pad_token_id
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in token_lists]
        batch = torch.tensor(padded, dtype=torch.long)  # [B, T]
        _set_inference_mode(self.dense_embedder)
        with torch.no_grad():
            out = self.dense_embedder(batch)  # [B, D]
        return out

    def similarity(
        self,
        query: str,
        passages: List[str],
        query_task: str = "query",
        passage_task: str = "passage",
    ) -> List[float]:
        """Return cosine similarities between ``query`` and each passage.

        Embeddings are already L2-normalized by :class:`DenseEmbedder`,
        so a single inner product is the cosine similarity.
        """
        if not isinstance(passages, list):
            raise TypeError(
                f"passages must be list[str], got {type(passages).__name__}"
            )
        if len(passages) == 0:
            raise ValueError("passages must be non-empty")
        q = self.encode(query, task=query_task)  # [D]
        p = self.encode_batch(passages, task=passage_task)  # [N, D]
        sims = (p @ q.unsqueeze(-1)).squeeze(-1)  # [N]
        return [float(x) for x in sims.tolist()]


__all__ = [
    "INSTRUCTION_PREFIXES",
    "InstructionPrefixEmbedder",
]
