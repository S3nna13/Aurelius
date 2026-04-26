"""Tests for the CAI training loop (critique -> revision -> SFT)."""

import math

import pytest
import torch

from src.alignment.cai_trainer import (
    CAIConfig,
    CAITrainer,
    CritiqueRevisionPair,
    greedy_generate,
    stochastic_generate,
    top_k_sample,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
MAX_SEQ_LEN = 512
MAX_NEW_TOKENS = 4
PROMPT_LEN = 8


@pytest.fixture(scope="module")
def small_model():
    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    )
    model = AureliusTransformer(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def cai_config():
    return CAIConfig(
        n_critique_rounds=1,
        max_new_tokens_critique=MAX_NEW_TOKENS,
        max_new_tokens_revision=MAX_NEW_TOKENS,
    )


@pytest.fixture(scope="module")
def tokenizer_encode():
    """Simple byte-level encode that stays within vocab_size=256."""

    def encode(text: str) -> list[int]:
        return [b % VOCAB_SIZE for b in text.encode("utf-8")][:32]

    return encode


@pytest.fixture(scope="module")
def tokenizer_decode():
    def decode(ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")

    return decode


@pytest.fixture(scope="module")
def trainer(small_model, cai_config, tokenizer_encode, tokenizer_decode):
    return CAITrainer(small_model, cai_config, tokenizer_encode, tokenizer_decode)


@pytest.fixture
def prompt_ids():
    torch.manual_seed(0)
    return torch.randint(0, VOCAB_SIZE, (1, PROMPT_LEN))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


# 1. CAIConfig defaults
def test_cai_config_defaults():
    cfg = CAIConfig()
    assert cfg.n_critique_rounds == 3
    assert cfg.max_new_tokens_critique == 64
    assert cfg.max_new_tokens_revision == 128
    assert cfg.temperature == 1.0
    assert cfg.top_k == 50
    assert cfg.sft_lr == 1e-5
    assert cfg.sft_batch_size == 4


# 2. top_k_sample returns int within vocab range
def test_top_k_sample_returns_int_in_range():
    torch.manual_seed(7)
    logits = torch.randn(VOCAB_SIZE)
    token = top_k_sample(logits, top_k=10, temperature=1.0)
    assert isinstance(token, int)
    assert 0 <= token < VOCAB_SIZE


# 3. top_k_sample with top_k=1 always returns argmax
def test_top_k_sample_top_k_1_is_deterministic():
    torch.manual_seed(99)
    logits = torch.randn(VOCAB_SIZE)
    expected = int(logits.argmax().item())
    for _ in range(10):
        result = top_k_sample(logits, top_k=1, temperature=1.0)
        assert result == expected, "top_k=1 must always return argmax"


# 4. greedy_generate output shape is (1, max_new_tokens)
def test_greedy_generate_shape(small_model, prompt_ids):
    out = greedy_generate(small_model, prompt_ids, MAX_NEW_TOKENS)
    assert out.shape == (1, MAX_NEW_TOKENS)


# 5. greedy_generate is deterministic
def test_greedy_generate_deterministic(small_model, prompt_ids):
    out1 = greedy_generate(small_model, prompt_ids, MAX_NEW_TOKENS)
    out2 = greedy_generate(small_model, prompt_ids, MAX_NEW_TOKENS)
    assert torch.equal(out1, out2), "Greedy generation must be deterministic"


# 6. stochastic_generate output shape is (1, max_new_tokens)
def test_stochastic_generate_shape(small_model, prompt_ids):
    out = stochastic_generate(small_model, prompt_ids, MAX_NEW_TOKENS, top_k=10, temperature=1.0)
    assert out.shape == (1, MAX_NEW_TOKENS)


# 7. generate_critique_prompt is longer than prompt alone
def test_generate_critique_prompt_longer_than_prompt(trainer, prompt_ids):
    response_ids = torch.randint(0, VOCAB_SIZE, (1, MAX_NEW_TOKENS))
    critique_prompt = trainer.generate_critique_prompt(prompt_ids, response_ids)
    assert critique_prompt.shape[1] > prompt_ids.shape[1], (
        "Critique prompt must be longer than original prompt"
    )


# 8. generate_revision_prompt longer than critique_prompt
def test_generate_revision_prompt_longer_than_critique_prompt(trainer, prompt_ids):
    response_ids = torch.randint(0, VOCAB_SIZE, (1, MAX_NEW_TOKENS))
    critique_ids = torch.randint(0, VOCAB_SIZE, (1, MAX_NEW_TOKENS))
    critique_prompt = trainer.generate_critique_prompt(prompt_ids, response_ids)
    revision_prompt = trainer.generate_revision_prompt(prompt_ids, response_ids, critique_ids)
    assert revision_prompt.shape[1] > critique_prompt.shape[1], (
        "Revision prompt must be longer than critique prompt"
    )


# 9. run_critique_revision returns CritiqueRevisionPair with tensor fields
def test_run_critique_revision_returns_pair_with_tensors(trainer, prompt_ids):
    pair = trainer.run_critique_revision(prompt_ids)
    assert isinstance(pair, CritiqueRevisionPair)
    assert isinstance(pair.prompt_ids, torch.Tensor)
    assert isinstance(pair.initial_ids, torch.Tensor)
    assert isinstance(pair.critique_ids, torch.Tensor)
    assert isinstance(pair.revised_ids, torch.Tensor)


# 10. run_critique_revision revised_ids shape correct
def test_run_critique_revision_revised_ids_shape(trainer, cai_config, prompt_ids):
    pair = trainer.run_critique_revision(prompt_ids)
    assert pair.revised_ids.shape == (1, cai_config.max_new_tokens_revision), (
        f"Expected (1, {cai_config.max_new_tokens_revision}), got {pair.revised_ids.shape}"
    )


# 11. sft_loss returns scalar > 0
def test_sft_loss_returns_positive_scalar(trainer, prompt_ids):
    target_ids = torch.randint(0, VOCAB_SIZE, (1, MAX_NEW_TOKENS))
    loss = trainer.sft_loss(prompt_ids, target_ids)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert float(loss.item()) > 0.0, "SFT loss must be positive"


# 12. train_step returns correct keys
def test_train_step_returns_correct_keys(trainer, prompt_ids):
    result = trainer.train_step([prompt_ids])
    assert "loss" in result, "train_step must return 'loss' key"
    assert "n_pairs" in result, "train_step must return 'n_pairs' key"


# 13. train_step loss is finite
def test_train_step_loss_is_finite(trainer, prompt_ids):
    torch.manual_seed(1)
    p1 = torch.randint(0, VOCAB_SIZE, (1, PROMPT_LEN))
    p2 = torch.randint(0, VOCAB_SIZE, (1, PROMPT_LEN))
    result = trainer.train_step([p1, p2])
    assert math.isfinite(result["loss"]), f"Expected finite loss, got {result['loss']}"
    assert result["n_pairs"] == 2
