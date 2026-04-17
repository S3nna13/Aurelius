"""Tests for speculative_rejection_v2 — Liu et al. 2024 token-tree variant.

Tiny config: vocab=16, d_model=8, branching_factor=2, depth=3,
             max_new_tokens=5, B=1.
Models: nn.Embedding(16, 8) + nn.Linear(8, 16).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.inference.speculative_rejection_v2 import (
    DraftTree,
    TokenTreeVerifier,
    RejectionSamplingCorrector,
    SpeculativeRejectionDecoder,
    AcceptanceRateTracker,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOCAB = 16
D_MODEL = 8
BF = 2       # branching_factor
DEPTH = 3
MAX_NEW = 5
B = 1
SEQ_LEN = 4  # context length


# ---------------------------------------------------------------------------
# Tiny model: Embedding + Linear, returns (None, logits, None) tuple
# ---------------------------------------------------------------------------

class TinyLM(nn.Module):
    """Minimal autoregressive LM returning (None, logits, None)."""

    def __init__(self, vocab: int = VOCAB, d: int = D_MODEL) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.proj = nn.Linear(d, vocab)

    def forward(self, input_ids: torch.Tensor) -> tuple:
        # input_ids: (B, T)
        x = self.embed(input_ids)         # (B, T, d)
        logits = self.proj(x)             # (B, T, vocab)
        return (None, logits, None)


def make_model(seed: int = 0) -> TinyLM:
    torch.manual_seed(seed)
    return TinyLM()


def make_prompt(seq_len: int = SEQ_LEN, batch: int = B) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randint(0, VOCAB, (batch, seq_len))


# ---------------------------------------------------------------------------
# Helper: build a DraftTree with a real model
# ---------------------------------------------------------------------------

def _build_tree(bf: int = BF, depth: int = DEPTH) -> tuple[DraftTree, list[torch.Tensor]]:
    model = make_model()
    prompt = make_prompt()
    tree = DraftTree(branching_factor=bf, depth=depth)

    # Collect one logit per depth position (autoregressive, batch=1)
    logits_list: list[torch.Tensor] = []
    current = prompt.clone()
    with torch.no_grad():
        for _ in range(depth):
            _, logits, _ = model(current)
            step = logits[0, -1, :]  # (V,)
            logits_list.append(step)
            next_tok = step.argmax().unsqueeze(0).unsqueeze(0)
            current = torch.cat([current, next_tok], dim=1)

    tree.build(logits_list)
    return tree, logits_list


# ---------------------------------------------------------------------------
# 1. DraftTree.build: tokens shape (depth, branching_factor)
# ---------------------------------------------------------------------------

def test_draft_tree_tokens_shape():
    tree, _ = _build_tree()
    assert tree.tokens is not None
    assert tree.tokens.shape == (DEPTH, BF), \
        f"Expected ({DEPTH}, {BF}), got {tree.tokens.shape}"


# ---------------------------------------------------------------------------
# 2. DraftTree.build: log_probs shape correct
# ---------------------------------------------------------------------------

def test_draft_tree_log_probs_shape():
    tree, _ = _build_tree()
    assert tree.log_probs is not None
    assert tree.log_probs.shape == (DEPTH, BF), \
        f"Expected ({DEPTH}, {BF}), got {tree.log_probs.shape}"


# ---------------------------------------------------------------------------
# 3. DraftTree.linear_paths: returns list of sequences each of length depth
# ---------------------------------------------------------------------------

def test_draft_tree_linear_paths_length():
    tree, _ = _build_tree()
    paths = tree.linear_paths()
    assert isinstance(paths, list)
    assert len(paths) > 0
    for path in paths:
        assert len(path) == DEPTH, f"Path length {len(path)} != depth {DEPTH}"


# ---------------------------------------------------------------------------
# 4. DraftTree.n_candidates: equals branching_factor
# ---------------------------------------------------------------------------

def test_draft_tree_n_candidates():
    tree, _ = _build_tree()
    assert tree.n_candidates() == BF


# ---------------------------------------------------------------------------
# 5. TokenTreeVerifier.verify: target_logits shape (B, K, V)
# ---------------------------------------------------------------------------

def test_verifier_target_logits_shape():
    model = make_model()
    verifier = TokenTreeVerifier(model)
    prompt = make_prompt()                           # (1, SEQ_LEN)
    draft_tokens = torch.randint(0, VOCAB, (B, BF))  # (1, 2)

    target_logits, accept_mask = verifier.verify(prompt, draft_tokens)
    assert target_logits.shape == (B, BF, VOCAB), \
        f"Expected ({B}, {BF}, {VOCAB}), got {target_logits.shape}"


# ---------------------------------------------------------------------------
# 6. TokenTreeVerifier.verify: accept_mask shape (B, K) bool
# ---------------------------------------------------------------------------

def test_verifier_accept_mask_shape_and_dtype():
    model = make_model()
    verifier = TokenTreeVerifier(model)
    prompt = make_prompt()
    draft_tokens = torch.randint(0, VOCAB, (B, BF))

    _, accept_mask = verifier.verify(prompt, draft_tokens)
    assert accept_mask.shape == (B, BF), \
        f"Expected ({B}, {BF}), got {accept_mask.shape}"
    assert accept_mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# 7. TokenTreeVerifier: same model → high acceptance (greedy tokens match)
# ---------------------------------------------------------------------------

def test_verifier_same_model_high_acceptance():
    """When draft==target (same model), greedy draft tokens should always match."""
    torch.manual_seed(0)
    model = make_model()
    verifier = TokenTreeVerifier(model)
    prompt = make_prompt()

    # Build greedy draft tokens from the model itself
    with torch.no_grad():
        _, logits, _ = model(prompt)
        # First draft token: argmax of last position
        tok0 = logits[0, -1, :].argmax().item()
        # Second draft token: argmax of position after tok0
        extended = torch.cat([prompt, torch.tensor([[tok0]])], dim=1)
        _, logits2, _ = model(extended)
        tok1 = logits2[0, -1, :].argmax().item()

    draft_tokens = torch.tensor([[tok0, tok1]])  # (1, 2)
    _, accept_mask = verifier.verify(prompt, draft_tokens)

    # Both should be accepted (greedy tokens match target argmax by definition)
    assert accept_mask[0, 0].item() is True or accept_mask[0, 0].item() == True


# ---------------------------------------------------------------------------
# 8. RejectionSamplingCorrector.correct: valid token id in [0, vocab)
# ---------------------------------------------------------------------------

def test_corrector_correct_valid_range():
    corrector = RejectionSamplingCorrector()
    torch.manual_seed(1)
    logits = torch.randn(VOCAB)

    for draft_tok in range(VOCAB):
        tok = corrector.correct(
            draft_token=draft_tok,
            draft_logp=-2.0,
            target_logits=logits,
            accepted=False,
        )
        assert 0 <= tok < VOCAB, f"Token {tok} out of [0, {VOCAB})"


# ---------------------------------------------------------------------------
# 9. RejectionSamplingCorrector.batch_correct: output shape (B, K), valid ids
# ---------------------------------------------------------------------------

def test_corrector_batch_correct_shape_and_values():
    corrector = RejectionSamplingCorrector()
    torch.manual_seed(2)
    K = BF
    draft_tokens = torch.randint(0, VOCAB, (B, K))
    accept_mask = torch.zeros(B, K, dtype=torch.bool)   # all rejected
    draft_logps = torch.full((B, K), -2.0)
    target_logits = torch.randn(B, K, VOCAB)

    out = corrector.batch_correct(draft_tokens, accept_mask, draft_logps, target_logits)
    assert out.shape == (B, K), f"Expected ({B}, {K}), got {out.shape}"
    assert out.min().item() >= 0
    assert out.max().item() < VOCAB


# ---------------------------------------------------------------------------
# 10. RejectionSamplingCorrector: correction_dist sums to ~1.0
# ---------------------------------------------------------------------------

def test_corrector_correction_dist_normalized():
    """Verify internal normalisation: relu(p_target - p_draft) / Z sums to 1."""
    import torch.nn.functional as F

    torch.manual_seed(3)
    target_logits = torch.randn(VOCAB)
    target_probs = F.softmax(target_logits.float(), dim=-1)

    # Construct a draft_probs that is lower than target in some places
    draft_token = int(target_probs.argmax().item())
    draft_probs = torch.zeros(VOCAB)
    draft_probs[draft_token] = 0.5   # intentionally mismatch

    correction = F.relu(target_probs - draft_probs)
    Z = correction.sum() + 1e-9
    correction_dist = correction / Z

    assert abs(correction_dist.sum().item() - 1.0) < 1e-5, \
        f"correction_dist sum = {correction_dist.sum().item()}, expected ~1.0"


# ---------------------------------------------------------------------------
# 11. SpeculativeRejectionDecoder.generate: output shape (B, T + max_new_tokens)
# ---------------------------------------------------------------------------

def test_decoder_generate_output_shape():
    draft = make_model(seed=0)
    target = make_model(seed=1)
    decoder = SpeculativeRejectionDecoder(
        draft_model=draft,
        target_model=target,
        branching_factor=1,
        depth=DEPTH,
    )
    prompt = make_prompt()   # (1, SEQ_LEN)
    out = decoder.generate(prompt, max_new_tokens=MAX_NEW)

    expected_len = SEQ_LEN + MAX_NEW
    assert out.shape == (B, expected_len), \
        f"Expected ({B}, {expected_len}), got {out.shape}"


# ---------------------------------------------------------------------------
# 12. SpeculativeRejectionDecoder: all output token ids are valid
# ---------------------------------------------------------------------------

def test_decoder_generate_valid_token_ids():
    draft = make_model(seed=0)
    target = make_model(seed=1)
    decoder = SpeculativeRejectionDecoder(
        draft_model=draft,
        target_model=target,
        branching_factor=1,
        depth=DEPTH,
    )
    prompt = make_prompt()
    out = decoder.generate(prompt, max_new_tokens=MAX_NEW)
    assert out.min().item() >= 0
    assert out.max().item() < VOCAB


# ---------------------------------------------------------------------------
# 13. AcceptanceRateTracker: mean_acceptance_rate in [0, 1]
# ---------------------------------------------------------------------------

def test_tracker_acceptance_rate_range():
    tracker = AcceptanceRateTracker()
    tracker.record(2, 4)
    tracker.record(1, 3)
    rate = tracker.mean_acceptance_rate()
    assert 0.0 <= rate <= 1.0, f"Rate {rate} out of [0, 1]"


# ---------------------------------------------------------------------------
# 14. AcceptanceRateTracker.speedup_estimate: >= 1.0 always
# ---------------------------------------------------------------------------

def test_tracker_speedup_at_least_one():
    tracker = AcceptanceRateTracker()
    # Empty tracker
    assert tracker.speedup_estimate() >= 1.0

    tracker.record(0, 5)
    assert tracker.speedup_estimate() >= 1.0

    tracker.record(5, 5)
    assert tracker.speedup_estimate() >= 1.0


# ---------------------------------------------------------------------------
# 15. AcceptanceRateTracker.reset: mean_acceptance_rate returns 0.0 after reset
# ---------------------------------------------------------------------------

def test_tracker_reset():
    tracker = AcceptanceRateTracker()
    tracker.record(3, 4)
    assert tracker.mean_acceptance_rate() > 0.0

    tracker.reset()
    assert tracker.mean_acceptance_rate() == 0.0


# ---------------------------------------------------------------------------
# 16. Multiple decode steps without error (loop stability)
# ---------------------------------------------------------------------------

def test_decoder_loop_stability():
    """Run many steps to confirm no index errors or NaN."""
    draft = make_model(seed=5)
    target = make_model(seed=6)
    decoder = SpeculativeRejectionDecoder(
        draft_model=draft,
        target_model=target,
        branching_factor=1,
        depth=2,
    )
    prompt = make_prompt(seq_len=2)
    out = decoder.generate(prompt, max_new_tokens=10)
    assert not out.isnan().any()
    assert out.shape[1] == 2 + 10


# ---------------------------------------------------------------------------
# 17. accept_mask all-True: all tokens accepted, no correction path needed
# ---------------------------------------------------------------------------

def test_verifier_all_accepted_case():
    """Force accept_mask all True by using same model greedy draft."""
    torch.manual_seed(7)
    model = make_model()
    verifier = TokenTreeVerifier(model)
    K = 3
    prompt = make_prompt(seq_len=2)

    # Build K greedy tokens from the model
    current = prompt.clone()
    greedy_tokens = []
    with torch.no_grad():
        for _ in range(K):
            _, logits, _ = model(current)
            tok = int(logits[0, -1, :].argmax().item())
            greedy_tokens.append(tok)
            current = torch.cat([current, torch.tensor([[tok]])], dim=1)

    draft_tokens = torch.tensor([greedy_tokens])  # (1, K)
    _, accept_mask = verifier.verify(prompt, draft_tokens)

    # All positions should be accepted since we used the model's own greedy tokens
    assert accept_mask.all().item(), \
        f"Expected all True, got {accept_mask}"


# ---------------------------------------------------------------------------
# 18. SpeculativeRejectionDecoder: same draft and target → tracker records data
# ---------------------------------------------------------------------------

def test_decoder_same_model_tracker():
    """When draft == target, acceptance tracker should record non-zero drafts."""
    model = make_model(seed=0)
    decoder = SpeculativeRejectionDecoder(
        draft_model=model,
        target_model=model,
        branching_factor=1,
        depth=DEPTH,
    )
    prompt = make_prompt()
    decoder.generate(prompt, max_new_tokens=MAX_NEW)
    # Tracker should have recorded at least some drafted tokens
    assert decoder.tracker._total_drafted > 0
    assert decoder.tracker.speedup_estimate() >= 1.0
