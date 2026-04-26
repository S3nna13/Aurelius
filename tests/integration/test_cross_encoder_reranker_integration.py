"""Integration tests for the cross-encoder reranker registry wiring.

Verifies:
    1. ``RERANKER_REGISTRY`` exposes ``"cross_encoder"``.
    2. The registered class can be instantiated with a tiny config and
       rerank a 3-document candidate list.
    3. Existing retriever registry entries (``"bm25"``, ``"hybrid_rrf"``)
       are still present and untouched.
    4. Importing :mod:`src.retrieval` does NOT transitively import
       :mod:`src.model` — the reranker must stay decoupled from the
       frozen core transformer stack. Verified in a hermetic subprocess.
"""

from __future__ import annotations

import subprocess
import sys

import torch

from src.retrieval import (
    EMBEDDING_REGISTRY,
    RERANKER_REGISTRY,
    RETRIEVER_REGISTRY,
)
from src.retrieval.cross_encoder_reranker import (
    CrossEncoderConfig,
    CrossEncoderReranker,
)


def test_registry_contains_cross_encoder() -> None:
    assert "cross_encoder" in RERANKER_REGISTRY
    assert RERANKER_REGISTRY["cross_encoder"] is CrossEncoderReranker


def test_existing_retriever_registry_untouched() -> None:
    # bm25 + hybrid_rrf should still be registered; cross_encoder must
    # not have leaked into the retriever registry.
    assert "bm25" in RETRIEVER_REGISTRY
    assert "hybrid_rrf" in RETRIEVER_REGISTRY
    assert "cross_encoder" not in RETRIEVER_REGISTRY
    assert "cross_encoder" not in EMBEDDING_REGISTRY


def test_instantiate_tiny_and_rerank_three_docs() -> None:
    cfg = CrossEncoderConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        max_seq_len=16,
        dropout=0.0,
        sep_token_id=1,
    )
    torch.manual_seed(7)
    cls = RERANKER_REGISTRY["cross_encoder"]
    model = cls(cfg)
    ranked = model.rerank([2, 3], [[4, 5], [6, 7, 8], [9]])
    assert len(ranked) == 3
    assert {idx for idx, _ in ranked} == {0, 1, 2}
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_retrieval_import_does_not_pull_in_src_model() -> None:
    """Hermetic subprocess import check.

    If ``src.retrieval`` (or the reranker module) accidentally imports
    from ``src.model``, that module will appear in ``sys.modules`` after
    the import completes. This guarantees architectural decoupling from
    the frozen core transformer.
    """
    code = (
        "import sys\n"
        "import src.retrieval  # noqa: F401\n"
        "import src.retrieval.cross_encoder_reranker  # noqa: F401\n"
        "leaked = sorted(m for m in sys.modules if m == 'src.model' "
        "or m.startswith('src.model.'))\n"
        "print(repr(leaked))\n"
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "[]", (
        f"src.model modules leaked into retrieval import: {proc.stdout!r}"
    )
