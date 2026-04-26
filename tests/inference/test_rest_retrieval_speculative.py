"""Tests for REST: Retrieval-Based Speculative Decoding (arXiv:2311.08252).

Covers RESTDatastore, RESTDecoder, and RESTAccelerator in accordance with the
test-rigor floor specified in the implementation spec (15 tests).
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.inference.rest_retrieval_speculative import (
    RESTAccelerator,
    RESTDatastore,
    RESTDecoder,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_target_probs_fn(vocab_size: int, fixed_next: list[int]):
    """Return a target_probs_fn that always predicts *fixed_next* greedily.

    ``fixed_next[i]`` is the argmax token at position i (0-indexed over the
    full sequence length passed to the callable).  If the sequence is longer
    than fixed_next, we cycle.
    """

    def _fn(token_ids: list[int]) -> Tensor:
        T = len(token_ids)
        probs = torch.zeros(T, vocab_size)
        for i in range(T):
            tok = fixed_next[i % len(fixed_next)]
            probs[i, tok] = 1.0
        return probs

    return _fn


# ---------------------------------------------------------------------------
# RESTDatastore tests
# ---------------------------------------------------------------------------


class TestRESTDatastore:
    def test_add_and_retrieve_correct_next_token(self):
        """Test 1: add_document + retrieve returns correct next token."""
        ds = RESTDatastore(n=2)
        ds.add_document([10, 20, 30, 40])
        result = ds.retrieve([10, 20])
        assert 30 in result, f"Expected 30 in {result}"

    def test_unknown_ngram_returns_empty(self):
        """Test 2: unknown n-gram returns empty list without crashing."""
        ds = RESTDatastore(n=2)
        ds.add_document([1, 2, 3])
        result = ds.retrieve([99, 88])
        assert result == []

    def test_frequency_ordering(self):
        """Test 3: most common next token appears first in results."""
        ds = RESTDatastore(n=2)
        # Token 5 follows [1,2] three times; token 6 follows once
        ds.add_document([1, 2, 5])
        ds.add_document([1, 2, 5])
        ds.add_document([1, 2, 5])
        ds.add_document([1, 2, 6])
        result = ds.retrieve([1, 2], top_k=5)
        assert result[0] == 5, f"Expected 5 first (most common), got {result}"

    def test_unigram_n1_works(self):
        """Test 12: n=1 (unigram) datastore functions correctly."""
        ds = RESTDatastore(n=1)
        ds.add_document([7, 8, 9])
        result = ds.retrieve([7])
        assert 8 in result

    def test_multiple_documents_accumulated(self):
        """Test 13: multiple documents are accumulated in the same datastore."""
        ds = RESTDatastore(n=2)
        ds.add_document([1, 2, 3])
        ds.add_document([4, 5, 6])
        r1 = ds.retrieve([1, 2])
        r2 = ds.retrieve([4, 5])
        assert 3 in r1
        assert 6 in r2

    def test_context_too_short_returns_empty(self):
        """Test 11 (partial): single-token context with n=2 returns empty."""
        ds = RESTDatastore(n=2)
        ds.add_document([1, 2, 3])
        result = ds.retrieve([1])  # only 1 token, need 2 for n=2
        assert result == []

    def test_top_k_limits_results(self):
        """Retrieve honours the top_k limit."""
        ds = RESTDatastore(n=1)
        ds.add_document([0, 1])
        ds.add_document([0, 2])
        ds.add_document([0, 3])
        result = ds.retrieve([0], top_k=2)
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# RESTDecoder tests
# ---------------------------------------------------------------------------


class TestRESTDecoder:
    def _make_decoder_with_docs(self, docs, n=2, gamma=4):
        ds = RESTDatastore(n=n)
        for doc in docs:
            ds.add_document(doc)
        return RESTDecoder(datastore=ds, gamma=gamma)

    def test_draft_length_at_most_gamma(self):
        """Test 4: draft returns at most γ tokens."""
        dec = self._make_decoder_with_docs([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], gamma=4)
        result = dec.draft([1, 2])
        assert len(result) <= 4

    def test_draft_empty_datastore_returns_empty(self):
        """Test 5: empty datastore → empty draft (no crash)."""
        ds = RESTDatastore(n=2)
        dec = RESTDecoder(datastore=ds, gamma=4)
        result = dec.draft([1, 2])
        assert result == []

    def test_verify_accepts_correct_tokens(self):
        """Test 6: verify accepts draft tokens where target agrees."""
        vocab = 100
        ds = RESTDatastore(n=2)
        dec = RESTDecoder(datastore=ds, gamma=4)
        context = [1, 2]
        draft = [10, 11, 12]
        # target always predicts exactly the right token at each position
        # positions: 0-indexed over full_seq = [1,2,10,11,12]
        # pred_pos for draft[0] = ctx_len-1 + 0 = 1 → predicts 10
        # pred_pos for draft[1] = 1+1 = 2 → predicts 11
        # pred_pos for draft[2] = 1+2 = 3 → predicts 12
        # bonus_pos = 1+3 = 4 → predicts 99
        target_fn = make_target_probs_fn(vocab, [1, 10, 11, 12, 99])
        accepted, n_accepted = dec.verify(context, draft, target_fn)
        assert n_accepted == 3
        assert accepted == [10, 11, 12, 99]

    def test_verify_rejects_at_first_mismatch(self):
        """Test 7: verify stops at first mismatch; subsequent tokens discarded."""
        vocab = 100
        ds = RESTDatastore(n=2)
        dec = RESTDecoder(datastore=ds, gamma=4)
        context = [1, 2]
        draft = [10, 11, 12]
        # At pred_pos=1 target predicts 10 (match), at pred_pos=2 target
        # predicts 99 (mismatch with draft[1]=11) → n_accepted=1
        # bonus token at pred_pos=2 → 99
        target_fn = make_target_probs_fn(vocab, [1, 10, 99, 12, 50])
        accepted, n_accepted = dec.verify(context, draft, target_fn)
        assert n_accepted == 1
        assert accepted == [10, 99]

    def test_verify_n_accepted_0_when_all_wrong(self):
        """Test 8: n_accepted = 0 when first token is wrong."""
        vocab = 100
        ds = RESTDatastore(n=2)
        dec = RESTDecoder(datastore=ds, gamma=4)
        context = [1, 2]
        draft = [10, 11]
        # pred_pos=1 target predicts 50 ≠ 10 → immediate rejection
        # bonus token at pred_pos=1 → 50
        target_fn = make_target_probs_fn(vocab, [1, 50, 99])
        accepted, n_accepted = dec.verify(context, draft, target_fn)
        assert n_accepted == 0
        assert accepted == [50]

    def test_verify_n_accepted_equals_gamma_when_all_correct(self):
        """Test 9: n_accepted = gamma when all draft tokens are correct."""
        vocab = 100
        gamma = 3
        ds = RESTDatastore(n=2)
        dec = RESTDecoder(datastore=ds, gamma=gamma)
        context = [1, 2]
        draft = [10, 11, 12]
        # pred_pos for each: 1, 2, 3; bonus at 4 → 99
        target_fn = make_target_probs_fn(vocab, [1, 10, 11, 12, 99])
        accepted, n_accepted = dec.verify(context, draft, target_fn)
        assert n_accepted == gamma
        # accepted = [10, 11, 12, 99] (3 draft + 1 bonus)
        assert len(accepted) == gamma + 1

    def test_draft_determinism(self):
        """Test 10: same context produces same draft across repeated calls."""
        dec = self._make_decoder_with_docs([[1, 2, 3, 4, 5, 6]], gamma=3)
        r1 = dec.draft([1, 2])
        r2 = dec.draft([1, 2])
        assert r1 == r2

    def test_single_token_context_no_crash(self):
        """Test 11: single-token context does not crash (n=2 → empty draft)."""
        dec = self._make_decoder_with_docs([[1, 2, 3, 4]], n=2, gamma=4)
        # context has 1 token, n=2 retrieval needs 2 → empty draft
        result = dec.draft([1])
        assert isinstance(result, list)

    def test_bonus_token_appended_when_all_gamma_accepted(self):
        """Test 15: accepted list is length gamma+1 when all draft accepted."""
        vocab = 100
        gamma = 4
        ds = RESTDatastore(n=2)
        dec = RESTDecoder(datastore=ds, gamma=gamma)
        context = [0, 1]
        draft = [10, 11, 12, 13]
        # Arrange target to accept all 4 draft tokens
        # pred_pos for draft[i] = ctx_len-1 + i = 1+i
        # pos 1→10, 2→11, 3→12, 4→13, bonus at 5→99
        target_fn = make_target_probs_fn(vocab, [0, 10, 11, 12, 13, 99])
        accepted, n_accepted = dec.verify(context, draft, target_fn)
        assert n_accepted == gamma
        assert accepted[-1] == 99  # bonus token present
        assert len(accepted) == gamma + 1


# ---------------------------------------------------------------------------
# RESTAccelerator tests
# ---------------------------------------------------------------------------


class TestRESTAccelerator:
    def test_step_returns_non_empty_token_list(self):
        """Test 14: RESTAccelerator.step always returns at least one new token."""
        ds = RESTDatastore(n=2)
        ds.add_document([1, 2, 3, 4, 5])
        accel = RESTAccelerator(datastore=ds, gamma=4, top_k=5)
        vocab = 100
        context = [1, 2]
        # Simple oracle target that agrees with likely draft [3,4,5,...]
        target_fn = make_target_probs_fn(vocab, [1, 3, 4, 5, 6, 99])
        new_tokens, n_accepted = accel.step(context, target_fn)
        assert len(new_tokens) >= 1
        assert isinstance(n_accepted, int)
        assert n_accepted >= 0

    def test_step_with_empty_datastore_returns_bonus_token(self):
        """Accelerator with empty datastore still produces one bonus token."""
        ds = RESTDatastore(n=2)  # empty, no documents
        accel = RESTAccelerator(datastore=ds, gamma=4, top_k=5)
        vocab = 50
        context = [5, 6]
        target_fn = make_target_probs_fn(vocab, [5, 42])
        new_tokens, n_accepted = accel.step(context, target_fn)
        # No draft tokens → verify called with empty list → 1 bonus token
        assert len(new_tokens) == 1
        assert n_accepted == 0
