"""Tests for Mixture-of-Agents inference (Wang et al. 2024)."""
import pytest
import torch
import torch.nn as nn

from src.inference.mixture_of_agents import MixtureOfAgents, MoAConfig
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_cfg():
    return AureliusConfig(
        d_model=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=128,
    )


@pytest.fixture(scope="module")
def small_model(small_cfg):
    torch.manual_seed(42)
    model = AureliusTransformer(small_cfg)
    model.eval()
    return model


def _identity_encode(x):
    return x


def _identity_decode(x):
    return x


def _make_moa(model, n_proposers=3, n_rounds=1, max_new_tokens=4):
    proposers = [model] * n_proposers
    cfg = MoAConfig(
        n_proposers=n_proposers,
        n_rounds=n_rounds,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        aggregator_temperature=0.0,
    )
    return MixtureOfAgents(
        proposers=proposers,
        aggregator=model,
        tokenizer_encode=_identity_encode,
        tokenizer_decode=_identity_decode,
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_moa_config_defaults():
    cfg = MoAConfig()
    assert cfg.n_proposers == 3
    assert cfg.n_rounds == 2
    assert cfg.max_new_tokens == 128
    assert cfg.temperature == 0.7
    assert cfg.aggregator_temperature == 0.0


def test_generate_proposals_count(small_model):
    moa = _make_moa(small_model, n_proposers=3, max_new_tokens=4)
    input_ids = torch.randint(0, 256, (6,))
    proposals = moa.generate_proposals(input_ids)
    assert len(proposals) == 3


def test_generate_proposals_are_tensors(small_model):
    moa = _make_moa(small_model, n_proposers=2, max_new_tokens=4)
    input_ids = torch.randint(0, 256, (6,))
    proposals = moa.generate_proposals(input_ids)
    for p in proposals:
        assert isinstance(p, torch.Tensor)
        assert p.dim() == 1


def test_build_aggregator_context_contains_prompt(small_model):
    moa = _make_moa(small_model, n_proposers=2, max_new_tokens=4)
    prompt_ids = torch.tensor([10, 20, 30], dtype=torch.long)
    proposals = [torch.tensor([1, 2, 3], dtype=torch.long),
                 torch.tensor([4, 5, 6], dtype=torch.long)]
    context = moa.build_aggregator_context(prompt_ids, proposals)
    assert isinstance(context, torch.Tensor)
    assert context.dim() == 1
    # Prompt tokens must appear at the very start
    assert context[:3].tolist() == [10, 20, 30]


def test_build_aggregator_context_contains_sep(small_model):
    """SEP token (id=2) must appear between proposals."""
    moa = _make_moa(small_model, n_proposers=2, max_new_tokens=4)
    prompt_ids = torch.tensor([10, 20], dtype=torch.long)
    proposals = [torch.tensor([1, 2, 3], dtype=torch.long),
                 torch.tensor([7, 8, 9], dtype=torch.long)]
    context = moa.build_aggregator_context(prompt_ids, proposals)
    # Token 2 (SEP/EOS) must appear somewhere after the prompt
    tokens_after_prompt = context[len(prompt_ids):].tolist()
    assert 2 in tokens_after_prompt


def test_generate_returns_tensor(small_model):
    moa = _make_moa(small_model, n_proposers=2, n_rounds=1, max_new_tokens=4)
    input_ids = torch.randint(0, 256, (5,))
    out = moa.generate(input_ids)
    assert isinstance(out, torch.Tensor)
    assert out.dim() == 1


def test_generate_simple_returns_tensor(small_model):
    moa = _make_moa(small_model, n_proposers=2, n_rounds=1, max_new_tokens=4)
    input_ids = torch.randint(0, 256, (5,))
    out = moa.generate_simple(input_ids)
    assert isinstance(out, torch.Tensor)
    assert out.dim() == 1


def test_moa_single_proposer(small_model):
    """MoA must work with a single proposer."""
    moa = _make_moa(small_model, n_proposers=1, max_new_tokens=4)
    input_ids = torch.randint(0, 256, (5,))
    out = moa.generate(input_ids)
    assert isinstance(out, torch.Tensor)
    assert out.dim() == 1


def test_moa_multi_round(small_model):
    """n_rounds=2 must produce an output tensor."""
    moa = _make_moa(small_model, n_proposers=2, n_rounds=2, max_new_tokens=4)
    input_ids = torch.randint(0, 256, (5,))
    out = moa.generate(input_ids)
    assert isinstance(out, torch.Tensor)
    assert out.dim() == 1
    assert len(out) > 0


def test_moa_same_proposer_reused(small_model):
    """Passing the same model instance for all proposers should work fine."""
    proposers = [small_model, small_model, small_model]
    cfg = MoAConfig(n_proposers=3, n_rounds=1, max_new_tokens=4, temperature=0.5)
    moa = MixtureOfAgents(
        proposers=proposers,
        aggregator=small_model,
        tokenizer_encode=_identity_encode,
        tokenizer_decode=_identity_decode,
        cfg=cfg,
    )
    input_ids = torch.randint(0, 256, (5,))
    out = moa.generate(input_ids)
    assert isinstance(out, torch.Tensor)
    assert out.dim() == 1
