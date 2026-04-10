"""RAG-Fusion: multi-query generation + reciprocal rank fusion (RRF) retrieval.

This module implements RAG-Fusion, which is distinct from the basic RAG in
src/inference/rag.py. Where rag.py uses a VectorStore + EmbeddingExtractor
for dense retrieval, rag_fusion.py:

  1. Generates *multiple query variations* from the model.
  2. Retrieves candidates for each variation independently.
  3. Fuses all ranked lists using Reciprocal Rank Fusion (RRF).
  4. Builds a context string and generates a final answer.

No external index or embedding model is required; similarity is computed via
character-trigram Jaccard distance, making this fully self-contained.
"""

from __future__ import annotations

import re
import torch
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration & data structures
# ---------------------------------------------------------------------------

@dataclass
class RAGFusionConfig:
    """Configuration for the RAG-Fusion pipeline."""
    n_queries: int = 4
    top_k_per_query: int = 5
    final_top_k: int = 10
    rrf_k: int = 60
    use_hyde: bool = False


@dataclass
class Document:
    """A retrieved document with metadata."""
    doc_id: str
    text: str
    score: float = 0.0
    source: str = ""


# ---------------------------------------------------------------------------
# Trigram similarity
# ---------------------------------------------------------------------------

def _trigrams(text: str) -> set:
    t = text.lower()
    if len(t) < 3:
        return {t} if t else set()
    return {t[i:i + 3] for i in range(len(t) - 2)}


def compute_query_doc_similarity(query: str, doc: str) -> float:
    """Character trigram Jaccard similarity between query and doc. Returns [0,1]."""
    q_tri = _trigrams(query)
    d_tri = _trigrams(doc)
    if not q_tri and not d_tri:
        return 1.0
    if not q_tri or not d_tri:
        return 0.0
    intersection = len(q_tri & d_tri)
    union = len(q_tri | d_tri)
    return intersection / union


# ---------------------------------------------------------------------------
# Query variation generation
# ---------------------------------------------------------------------------

def generate_query_variations(
    model,
    original_query: str,
    n_queries: int,
    tokenizer_encode,
    tokenizer_decode,
    max_new_tokens: int = 32,
) -> list:
    """Generate n_queries variations of original_query using model.

    Prompt: "Generate {n_queries} search query variations for: {original_query}\\nQueries:\\n1."

    Parses numbered lines from output. Falls back to original_query if parsing yields fewer
    than n_queries results.

    Returns:
        List of exactly n_queries strings.
    """
    prompt = (
        f"Generate {n_queries} search query variations for: {original_query}\n"
        f"Queries:\n1."
    )

    input_ids_list = tokenizer_encode(prompt)
    if isinstance(input_ids_list, torch.Tensor):
        input_ids = input_ids_list.unsqueeze(0) if input_ids_list.dim() == 1 else input_ids_list
    else:
        input_ids = torch.tensor([input_ids_list], dtype=torch.long)

    generated = input_ids.clone()

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            _, logits, _ = model(generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    new_token_ids = generated[0, input_ids.shape[1]:].tolist()
    generated_text = tokenizer_decode(new_token_ids)

    # Prepend "1." because the prompt already started that line
    full_text = "1." + generated_text
    pattern = re.compile(r"^\s*\d+[.)]\s*(.+)", re.MULTILINE)
    matches = pattern.findall(full_text)

    variations = []
    for m in matches:
        cleaned = m.strip()
        if cleaned:
            variations.append(cleaned)
        if len(variations) >= n_queries:
            break

    while len(variations) < n_queries:
        variations.append(original_query)

    return variations[:n_queries]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(ranked_lists: list, k: int = 60) -> list:
    """Fuse multiple ranked Document lists using RRF.

    RRF(d) = sum_i 1/(k + rank_i(d))

    Returns deduplicated Documents sorted by RRF score descending.
    """
    rrf_scores: dict = {}
    best_doc: dict = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0.0) + 1.0 / (k + rank)
            if doc.doc_id not in best_doc:
                best_doc[doc.doc_id] = doc

    fused = []
    for doc_id, score in rrf_scores.items():
        d = best_doc[doc_id]
        fused.append(Document(doc_id=d.doc_id, text=d.text, score=score, source=d.source))

    fused.sort(key=lambda d: d.score, reverse=True)
    return fused


# ---------------------------------------------------------------------------
# MockRetriever
# ---------------------------------------------------------------------------

class MockRetriever:
    """Retriever backed by a static list of Documents using trigram similarity."""

    def __init__(self, documents: list) -> None:
        self.documents = documents

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Return top_k docs ranked by compute_query_doc_similarity."""
        scored = []
        for doc in self.documents:
            sim = compute_query_doc_similarity(query, doc.text)
            scored.append(Document(doc_id=doc.doc_id, text=doc.text, score=sim, source=doc.source))
        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# RAGFusionPipeline
# ---------------------------------------------------------------------------

class RAGFusionPipeline:
    """End-to-end RAG-Fusion pipeline: multi-query -> retrieve -> RRF -> answer."""

    def __init__(self, model, retriever, config: RAGFusionConfig, tokenizer_encode, tokenizer_decode) -> None:
        self.model = model
        self.retriever = retriever
        self.config = config
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode

    def retrieve_and_fuse(self, query: str) -> list:
        """Generate query variations, retrieve for each, fuse with RRF.

        Returns up to config.final_top_k Documents sorted by RRF score.
        """
        variations = generate_query_variations(
            self.model, query, self.config.n_queries,
            self.tokenizer_encode, self.tokenizer_decode,
        )
        ranked_lists = [
            self.retriever.retrieve(q, top_k=self.config.top_k_per_query)
            for q in variations
        ]
        fused = reciprocal_rank_fusion(ranked_lists, k=self.config.rrf_k)
        return fused[: self.config.final_top_k]

    def build_context(self, docs: list) -> str:
        """Format docs as 'Document 1: {text}\\nDocument 2: {text}\\n...'"""
        lines = [f"Document {i}: {doc.text}" for i, doc in enumerate(docs, start=1)]
        return "\n".join(lines)

    def generate_answer(self, query: str, context: str, max_new_tokens: int = 64) -> str:
        """Generate answer given query + context string."""
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        input_ids_list = self.tokenizer_encode(prompt)
        if isinstance(input_ids_list, torch.Tensor):
            input_ids = input_ids_list.unsqueeze(0) if input_ids_list.dim() == 1 else input_ids_list
        else:
            input_ids = torch.tensor([input_ids_list], dtype=torch.long)

        generated = input_ids.clone()
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                _, logits, _ = self.model(generated)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

        new_ids = generated[0, input_ids.shape[1]:].tolist()
        return self.tokenizer_decode(new_ids)

    def run(self, query: str) -> dict:
        """End-to-end: retrieve_and_fuse -> build_context -> generate_answer.

        Returns dict with keys: query, answer, docs_retrieved, queries_used.
        """
        variations = generate_query_variations(
            self.model, query, self.config.n_queries,
            self.tokenizer_encode, self.tokenizer_decode,
        )
        ranked_lists = [
            self.retriever.retrieve(q, top_k=self.config.top_k_per_query)
            for q in variations
        ]
        fused = reciprocal_rank_fusion(ranked_lists, k=self.config.rrf_k)
        top_docs = fused[: self.config.final_top_k]

        context = self.build_context(top_docs)
        answer = self.generate_answer(query, context)

        return {
            "query": query,
            "answer": answer,
            "docs_retrieved": len(top_docs),
            "queries_used": variations,
        }
