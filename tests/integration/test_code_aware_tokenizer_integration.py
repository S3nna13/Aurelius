"""Integration: CodeAwareTokenizer wired into BM25Retriever for code retrieval."""

from __future__ import annotations

import pytest

import src.retrieval as retrieval
from src.retrieval import BM25Retriever, CodeAwareTokenizer


CODE_CORPUS = [
    # doc 0: python utility
    "def get_user_name(user_id):\n    return db.users.find(user_id)\n",
    # doc 1: python class with camelCase method
    "class HTTPServer:\n    def handleRequest(self, req):\n        return req.path\n",
    # doc 2: javascript function
    "function parseXMLDoc(src) { return new DOMParser().parseFromString(src); }",
    # doc 3: rust function
    "fn compute_checksum(buf: &[u8]) -> u32 { buf.iter().map(|b| *b as u32).sum() }",
    # doc 4: go function
    "package main\nfunc HandleConnection(c net.Conn) { defer c.Close() }",
    # doc 5: unrelated prose-ish doc
    "the quick brown fox jumps over the lazy dog many times each day",
]


def test_public_exports_intact():
    # Pre-existing names must still be exported.
    for name in (
        "BM25Retriever",
        "HybridRetriever",
        "RETRIEVER_REGISTRY",
        "EMBEDDING_REGISTRY",
        "RERANKER_REGISTRY",
        "DenseEmbedder",
        "reciprocal_rank_fusion",
    ):
        assert hasattr(retrieval, name), f"{name} missing from src.retrieval"
    # New name exposed.
    assert hasattr(retrieval, "CodeAwareTokenizer")


def test_tokenizer_constructible_from_package():
    tok = retrieval.CodeAwareTokenizer()
    assert isinstance(tok, CodeAwareTokenizer)


def test_bm25_with_code_aware_tokenizer_camel_query():
    tok = CodeAwareTokenizer(language="auto", keep_keywords=False)
    r = BM25Retriever(tokenizer=tok)
    r.add_documents(CODE_CORPUS)

    # Query by a camelCase fragment; the tokenizer must split so that
    # "handleRequest" matches doc 1 which contains the same identifier.
    hits = r.query("handleRequest", k=3)
    assert hits, "expected at least one hit"
    top_doc = hits[0][0]
    assert top_doc == 1, f"expected doc 1 top-ranked, got {top_doc} (hits={hits})"


def test_bm25_with_code_aware_tokenizer_snake_query():
    tok = CodeAwareTokenizer(language="python", keep_keywords=False)
    r = BM25Retriever(tokenizer=tok)
    r.add_documents(CODE_CORPUS)

    # snake_case query -> should retrieve doc 0
    hits = r.query("get_user_name", k=3)
    assert hits
    assert hits[0][0] == 0


def test_bm25_with_code_aware_tokenizer_dotted_identifier():
    tok = CodeAwareTokenizer(language="python", keep_keywords=False)
    r = BM25Retriever(tokenizer=tok)
    r.add_documents(CODE_CORPUS)

    # A dotted sub-part should retrieve doc 0, which contains "db.users.find".
    hits = r.query("users", k=3)
    assert hits
    assert hits[0][0] == 0


def test_bm25_with_code_aware_tokenizer_cross_language():
    tok = CodeAwareTokenizer(language="auto", keep_keywords=False)
    r = BM25Retriever(tokenizer=tok)
    r.add_documents(CODE_CORPUS)

    # Rust-specific identifier should surface doc 3.
    hits = r.query("compute_checksum", k=3)
    assert hits and hits[0][0] == 3

    # Go-specific identifier should surface doc 4.
    hits = r.query("HandleConnection", k=3)
    assert hits and hits[0][0] == 4


def test_end_to_end_registry_bm25_still_works():
    # Smoke-test the pre-existing BM25 registry entry with the new tokenizer.
    BM25Cls = retrieval.RETRIEVER_REGISTRY["bm25"]
    tok = retrieval.CodeAwareTokenizer(keep_keywords=False)
    r = BM25Cls(tokenizer=tok)
    r.add_documents(CODE_CORPUS)
    hits = r.query("parseXMLDoc", k=3)
    assert hits and hits[0][0] == 2
