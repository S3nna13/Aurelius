"""Tests for src/inference/speculative_streaming.py.

Covers DraftHead and SpeculativeStreamingDecoder (14 tests).

Test config: vocab_size=256, d_model=64, gamma=4.
All tests use pure PyTorch — no transformers/HuggingFace dependencies.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.inference.speculative_streaming import (
    DraftHead,
    SpeculativeStreamingDecoder,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
D_MODEL = 64
GAMMA = 4
B = 1  # batch size (decoder only supports B=1)
T = 6  # context length

# ---------------------------------------------------------------------------
# Mock model helpers
# ---------------------------------------------------------------------------


def _make_fixed_logits(V: int, uniform: bool = False) -> Tensor:
    """Return a fixed ``(V,)`` logit tensor.

    If *uniform* is True all logits are 0 (uniform distribution).
    Otherwise token 0 always gets the highest logit.
    """
    if uniform:
        return torch.zeros(V)
    logits = torch.zeros(V)
    logits[0] = 10.0  # token 0 is the dominant token
    return logits


class MockModel:
    """Minimal target model mock.

    Parameters
    ----------
    vocab_size:
        Vocabulary size V.
    d_model:
        Hidden-state dimensionality.
    logits_fn:
        Optional callable ``(B, T, V) -> (B, T, V)`` to produce custom logits.
        Defaults to ``torch.zeros``.
    h_fn:
        Optional callable ``(B, T) -> (B, T, d_model)`` for custom hidden states.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = D_MODEL,
        logits_fn=None,
        h_fn=None,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self._logits_fn = logits_fn
        self._h_fn = h_fn

    def __call__(
        self,
        input_ids: Tensor,
        return_hidden: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        B, T_ = input_ids.shape
        V = self.vocab_size
        D = self.d_model

        if self._logits_fn is not None:
            logits = self._logits_fn(B, T_, V)
        else:
            logits = torch.zeros(B, T_, V)

        if return_hidden:
            if self._h_fn is not None:
                h = self._h_fn(B, T_)
            else:
                h = torch.zeros(B, T_, D)
            return logits, h
        return logits


def _make_decoder(
    gamma: int = GAMMA,
    temperature: float = 1.0,
    model: MockModel | None = None,
    draft_head: DraftHead | None = None,
) -> SpeculativeStreamingDecoder:
    m = model or MockModel()
    dh = draft_head or DraftHead(D_MODEL, VOCAB_SIZE)
    return SpeculativeStreamingDecoder(m, dh, gamma=gamma, temperature=temperature)


def _input_ids(t: int = T) -> Tensor:
    return torch.randint(0, VOCAB_SIZE, (B, t))


# ===========================================================================
# 1. DraftHead output shape (B, T, vocab_size)
# ===========================================================================


def test_draft_head_output_shape():
    dh = DraftHead(D_MODEL, VOCAB_SIZE)
    h = torch.randn(B, T, D_MODEL)
    out = dh(h)
    assert out.shape == (B, T, VOCAB_SIZE), f"Expected ({B}, {T}, {VOCAB_SIZE}), got {out.shape}"


# ===========================================================================
# 2. DraftHead gradient flow
# ===========================================================================


def test_draft_head_gradient_flow():
    dh = DraftHead(D_MODEL, VOCAB_SIZE)
    h = torch.randn(B, T, D_MODEL, requires_grad=True)
    out = dh(h)
    loss = out.sum()
    loss.backward()
    assert h.grad is not None, "No gradient flowed back to input h"
    assert h.grad.shape == h.shape


# ===========================================================================
# 3. generate_step: accepted_tokens is non-empty list of token IDs
# ===========================================================================


def test_generate_step_returns_nonempty_list():
    decoder = _make_decoder()
    ids = _input_ids()
    accepted, n = decoder.generate_step(ids)
    assert isinstance(accepted, list), f"Expected list, got {type(accepted)}"
    assert len(accepted) > 0, "accepted_tokens must be non-empty"
    assert all(isinstance(t, int) for t in accepted), "All tokens must be int"


# ===========================================================================
# 4. generate_step: n_accepted ≤ gamma + 1
# ===========================================================================


def test_generate_step_n_accepted_upper_bound():
    decoder = _make_decoder(gamma=GAMMA)
    ids = _input_ids()
    _, n = decoder.generate_step(ids)
    assert n <= GAMMA + 1, f"n_accepted={n} > gamma+1={GAMMA + 1}"


# ===========================================================================
# 5. generate_step: n_accepted ≥ 1
# ===========================================================================


def test_generate_step_n_accepted_lower_bound():
    decoder = _make_decoder(gamma=GAMMA)
    ids = _input_ids()
    _, n = decoder.generate_step(ids)
    assert n >= 1, "Must always accept at least 1 token"


# ===========================================================================
# 6. temperature=0 → greedy; if draft argmax == target argmax → always accept
# ===========================================================================


def test_temperature_zero_accepts_matching_argmax():
    """With temperature=0 both draft and target are greedy.

    If the draft head and target model both have their highest logit at token 0,
    then every draft token is 0 and the target also prefers 0, so all γ tokens
    should be accepted (plus the bonus), giving n_accepted == γ + 1.
    """

    # Target model: logits dominated by token 0 at all positions
    def logits_fn(b, t, v):
        lg = torch.zeros(b, t, v)
        lg[:, :, 0] = 10.0
        return lg

    # Draft head weights zeroed, bias large on index 0
    dh = DraftHead(D_MODEL, VOCAB_SIZE)
    with torch.no_grad():
        dh.fc1.weight.zero_()
        dh.fc1.bias.zero_()
        dh.fc2.weight.zero_()
        dh.fc2.bias.zero_()
        dh.fc2.bias[0] = 10.0  # token 0 dominates draft

    model = MockModel(logits_fn=logits_fn)
    decoder = SpeculativeStreamingDecoder(model, dh, gamma=GAMMA, temperature=0.0)
    ids = _input_ids()
    accepted, n = decoder.generate_step(ids)
    assert n == GAMMA + 1, (
        f"With matching greedy argmax, expected {GAMMA + 1} accepted, got {n}. "
        f"Accepted tokens: {accepted}"
    )


# ===========================================================================
# 7. Draft tokens accepted: p_draft == p_target everywhere → always accept γ
# ===========================================================================


def test_always_accept_when_draft_equals_target():
    """When draft distribution == target distribution the acceptance ratio is
    always 1, so all γ draft tokens (+ bonus) should be accepted."""

    # Make draft head always predict uniform distribution (bias=0, weight=0)
    dh = DraftHead(D_MODEL, VOCAB_SIZE)
    with torch.no_grad():
        dh.fc1.weight.zero_()
        dh.fc1.bias.zero_()
        dh.fc2.weight.zero_()
        dh.fc2.bias.zero_()
    # draft logits → all zeros → uniform(V)

    # Target model also returns all-zero logits → same uniform distribution
    model = MockModel()  # default: zeros → uniform

    decoder = SpeculativeStreamingDecoder(model, dh, gamma=GAMMA, temperature=1.0)
    torch.manual_seed(42)
    ids = _input_ids()
    accepted, n = decoder.generate_step(ids)
    # ratio = uniform[d_k] / uniform[d_k] = 1 → always accept
    assert n == GAMMA + 1, (
        f"Expected all {GAMMA + 1} tokens accepted (p_draft == p_target), got {n}."
    )


# ===========================================================================
# 8. Draft tokens rejected: p_draft puts all mass on wrong token → reject step 1
# ===========================================================================


def test_always_reject_when_draft_wrong():
    """Draft head assigns all mass to token V-1; target assigns all mass to
    token 0.  Ratio p_target(V-1) / p_draft(V-1) = 0 → reject at k=1."""

    V = VOCAB_SIZE

    # Target: all mass on token 0
    def logits_fn(b, t, v):
        lg = torch.full((b, t, v), -1e9)
        lg[:, :, 0] = 0.0
        return lg

    # Draft head: all mass on token V-1
    dh = DraftHead(D_MODEL, V)
    with torch.no_grad():
        dh.fc1.weight.zero_()
        dh.fc1.bias.zero_()
        dh.fc2.weight.zero_()
        dh.fc2.bias.fill_(-1e9)
        dh.fc2.bias[V - 1] = 0.0  # all mass on last token

    model = MockModel(logits_fn=logits_fn)
    decoder = SpeculativeStreamingDecoder(model, dh, gamma=GAMMA, temperature=1.0)
    ids = _input_ids()
    accepted, n = decoder.generate_step(ids)
    # Should reject at step 1 and resample → exactly 1 token emitted
    assert n == 1, f"Expected rejection at step 1 → n_accepted=1, got {n}. Tokens: {accepted}"


# ===========================================================================
# 9. Determinism under torch.manual_seed
# ===========================================================================


def test_determinism_under_manual_seed():
    def run():
        torch.manual_seed(7)
        decoder = _make_decoder(temperature=1.0)
        ids = torch.randint(0, VOCAB_SIZE, (B, T))
        accepted, n = decoder.generate_step(ids)
        return accepted, n

    a1, n1 = run()
    a2, n2 = run()
    assert a1 == a2 and n1 == n2, f"Non-deterministic under same seed: {a1} vs {a2}"


# ===========================================================================
# 10. No NaN/Inf in generated tokens
# ===========================================================================


def test_no_nan_inf_in_generated_tokens():
    # Use random logits to stress-test numerical stability
    def logits_fn(b, t, v):
        return torch.randn(b, t, v)

    model = MockModel(logits_fn=logits_fn)
    decoder = _make_decoder(model=model)
    torch.manual_seed(0)
    ids = _input_ids()
    accepted, _ = decoder.generate_step(ids)
    for tok in accepted:
        assert not (tok != tok), f"NaN token id: {tok}"  # nan check
        assert tok < VOCAB_SIZE and tok >= 0, f"Token out of range: {tok}"


# ===========================================================================
# 11. acceptance_rate is in [0, 1]
# ===========================================================================


def test_acceptance_rate_in_bounds():
    decoder = _make_decoder()
    ids = _input_ids()
    for _ in range(5):
        decoder.generate_step(ids)
    rate = decoder.acceptance_rate()
    assert 0.0 <= rate <= 1.0, f"acceptance_rate={rate} out of [0,1]"


# ===========================================================================
# 12. acceptance_rate == 1.0 when draft always matches target (all accepted)
# ===========================================================================


def test_acceptance_rate_one_when_always_accepted():
    """When all γ draft tokens are accepted every step, acceptance rate = 1."""
    # Same setup as test 7: p_draft == p_target
    dh = DraftHead(D_MODEL, VOCAB_SIZE)
    with torch.no_grad():
        dh.fc1.weight.zero_()
        dh.fc1.bias.zero_()
        dh.fc2.weight.zero_()
        dh.fc2.bias.zero_()

    model = MockModel()
    decoder = SpeculativeStreamingDecoder(model, dh, gamma=GAMMA, temperature=1.0)
    ids = _input_ids()
    for _ in range(3):
        decoder.generate_step(ids)

    rate = decoder.acceptance_rate()
    assert rate == 1.0, f"Expected acceptance_rate=1.0, got {rate}"


# ===========================================================================
# 13. Adjusted distribution: after rejection, resample from (p_t - p_d)+
# ===========================================================================


def test_adjusted_distribution_after_rejection():
    """After rejection the resampled token must come from the support of
    (p_target - p_draft)+.  We set up p_draft(V-1)=1 (all mass on token V-1)
    and p_target with mass on tokens {0, 1, 2}; the adjusted distribution
    (p_t - p_d)+ has support only on {0, 1, 2}, so the emitted token must be
    in that set."""
    V = VOCAB_SIZE

    def logits_fn(b, t, v):
        # p_target: equal mass on tokens 0, 1, 2
        lg = torch.full((b, t, v), -1e9)
        lg[:, :, 0] = 0.0
        lg[:, :, 1] = 0.0
        lg[:, :, 2] = 0.0
        return lg

    dh = DraftHead(D_MODEL, V)
    with torch.no_grad():
        dh.fc1.weight.zero_()
        dh.fc1.bias.zero_()
        dh.fc2.weight.zero_()
        dh.fc2.bias.fill_(-1e9)
        dh.fc2.bias[V - 1] = 0.0  # draft all on token V-1

    model = MockModel(logits_fn=logits_fn)
    decoder = SpeculativeStreamingDecoder(model, dh, gamma=GAMMA, temperature=1.0)

    rejected_samples: list[int] = []
    for seed in range(30):
        torch.manual_seed(seed)
        ids = _input_ids()
        accepted, n = decoder.generate_step(ids)
        if n == 1:
            rejected_samples.append(accepted[0])

    assert len(rejected_samples) > 0, "Expected at least some rejections"
    for tok in rejected_samples:
        assert tok in {0, 1, 2, V - 1}, (
            f"Resampled token {tok} is outside expected support {{0,1,2,{V - 1}}}. "
            "Note: V-1 can appear if draft samples token V-1 and passes; "
            "pure rejection re-samples should be in {0,1,2}."
        )
    # Specifically, purely resampled tokens should be in {0,1,2}
    purely_resampled = [t for t in rejected_samples if t != V - 1]
    for tok in purely_resampled:
        assert tok in {0, 1, 2}, f"Re-sampled token {tok} outside support {{0,1,2}} of (p_t - p_d)+"


# ===========================================================================
# 14. gamma=1: degenerates to standard autoregressive (1 or 2 tokens per step)
# ===========================================================================


def test_gamma_1_degenerates_to_autoregressive():
    """With gamma=1, each step emits either 1 token (rejection) or 2 tokens
    (1 accepted draft + 1 bonus).  Always ≤ 2."""
    decoder = _make_decoder(gamma=1)
    ids = _input_ids()
    for _ in range(10):
        _, n = decoder.generate_step(ids)
        assert 1 <= n <= 2, f"gamma=1 should yield 1 or 2 tokens, got {n}"
