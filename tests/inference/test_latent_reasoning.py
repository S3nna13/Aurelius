"""Tests for latent reasoning module (Quiet-STaR / Thinking Tokens)."""

import torch
import torch.nn as nn

from src.inference.latent_reasoning import (
    LatentReasoningConfig,
    LatentReasoningDecoder,
    LatentReasoningLayer,
    ThoughtTokenEmbedding,
    compute_thought_supervision_loss,
    extract_answer_from_thoughts,
    measure_thought_benefit,
    prepend_thought_tokens,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
D_MODEL = 64
N_HEADS = 2
D_FF = 128
N_THOUGHT_TOKENS = 4
VOCAB_SIZE = 256
B = 2
T = 8
torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Tiny helpers for decoder tests
# ---------------------------------------------------------------------------


class _TinyLM(nn.Module):
    """Minimal language model: embed + linear to vocab."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(ids))


def _tok_encode(text: str) -> list[int]:
    return [ord(c) % VOCAB_SIZE for c in text[:4]] or [0]


def _tok_decode(ids: list[int]) -> str:
    return "".join(chr(i % 128) for i in ids)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_latent_reasoning_config_defaults():
    cfg = LatentReasoningConfig()
    assert cfg.n_thought_tokens == 8
    assert cfg.thought_token_id == 1
    assert cfg.max_answer_tokens == 64
    assert cfg.temperature == 1.0
    assert cfg.use_thought_supervision is True
    assert cfg.thought_start_id == 2
    assert cfg.thought_end_id == 3


# ---------------------------------------------------------------------------
# 2. ThoughtTokenEmbedding shape
# ---------------------------------------------------------------------------


def test_thought_token_embedding_shape():
    torch.manual_seed(0)
    emb = ThoughtTokenEmbedding(D_MODEL, N_THOUGHT_TOKENS)
    thought_ids = torch.arange(N_THOUGHT_TOKENS).unsqueeze(0).expand(B, -1)
    out = emb(thought_ids)
    assert out.shape == (B, N_THOUGHT_TOKENS, D_MODEL)


# ---------------------------------------------------------------------------
# 3. Gradient flow through ThoughtTokenEmbedding
# ---------------------------------------------------------------------------


def test_thought_token_embedding_gradient_flow():
    torch.manual_seed(0)
    emb = ThoughtTokenEmbedding(D_MODEL, N_THOUGHT_TOKENS)
    thought_ids = torch.arange(N_THOUGHT_TOKENS).unsqueeze(0).expand(B, -1)
    out = emb(thought_ids)
    out.sum().backward()
    grad = emb.thought_embeddings.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 4. prepend_thought_tokens shape
# ---------------------------------------------------------------------------


def test_prepend_thought_tokens_shape():
    torch.manual_seed(0)
    hidden = torch.randn(B, T, D_MODEL)
    thoughts = torch.randn(B, N_THOUGHT_TOKENS, D_MODEL)
    out = prepend_thought_tokens(hidden, thoughts)
    assert out.shape == (B, N_THOUGHT_TOKENS + T, D_MODEL)


# ---------------------------------------------------------------------------
# 5. extract_answer_removes_thoughts shape
# ---------------------------------------------------------------------------


def test_extract_answer_removes_thoughts():
    torch.manual_seed(0)
    full = torch.randn(B, N_THOUGHT_TOKENS + T, D_MODEL)
    out = extract_answer_from_thoughts(full, N_THOUGHT_TOKENS)
    assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 6. LatentReasoningLayer output shape
# ---------------------------------------------------------------------------


def test_latent_reasoning_layer_output_shape():
    torch.manual_seed(0)
    cfg = LatentReasoningConfig(n_thought_tokens=N_THOUGHT_TOKENS)
    layer = LatentReasoningLayer(D_MODEL, N_HEADS, D_FF, cfg)
    x = torch.randn(B, T, D_MODEL)
    out = layer(x)
    assert out.shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 7. LatentReasoningLayer gradient flow
# ---------------------------------------------------------------------------


def test_latent_reasoning_layer_gradient_flow():
    torch.manual_seed(0)
    cfg = LatentReasoningConfig(n_thought_tokens=N_THOUGHT_TOKENS)
    layer = LatentReasoningLayer(D_MODEL, N_HEADS, D_FF, cfg)
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    out = layer(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 8. compute_thought_supervision_loss is scalar
# ---------------------------------------------------------------------------


def test_compute_thought_supervision_loss_scalar():
    torch.manual_seed(0)
    logits = torch.randn(B, N_THOUGHT_TOKENS, VOCAB_SIZE)
    targets = torch.randint(0, VOCAB_SIZE, (B, N_THOUGHT_TOKENS))
    loss = compute_thought_supervision_loss(logits, targets)
    assert loss.shape == ()
    assert float(loss) > 0.0


# ---------------------------------------------------------------------------
# 9. All-ignore targets → loss == 0
# ---------------------------------------------------------------------------


def test_compute_thought_supervision_all_ignore():
    torch.manual_seed(0)
    logits = torch.randn(B, N_THOUGHT_TOKENS, VOCAB_SIZE)
    targets = torch.full((B, N_THOUGHT_TOKENS), -100, dtype=torch.long)
    loss = compute_thought_supervision_loss(logits, targets, ignore_index=-100)
    assert float(loss) == 0.0


# ---------------------------------------------------------------------------
# 10. LatentReasoningDecoder.generate_with_thoughts returns string
# ---------------------------------------------------------------------------


def test_latent_decoder_generate_returns_string():
    torch.manual_seed(0)
    cfg = LatentReasoningConfig(n_thought_tokens=N_THOUGHT_TOKENS, max_answer_tokens=4)
    model = _TinyLM()
    decoder = LatentReasoningDecoder(model, _tok_encode, _tok_decode, cfg)
    answer, thought_ids = decoder.generate_with_thoughts("hello")
    assert isinstance(answer, str)


# ---------------------------------------------------------------------------
# 11. thought_ids not empty
# ---------------------------------------------------------------------------


def test_latent_decoder_thought_ids_not_empty():
    torch.manual_seed(0)
    cfg = LatentReasoningConfig(n_thought_tokens=N_THOUGHT_TOKENS, max_answer_tokens=4)
    model = _TinyLM()
    decoder = LatentReasoningDecoder(model, _tok_encode, _tok_decode, cfg)
    _, thought_ids = decoder.generate_with_thoughts("hello")
    assert len(thought_ids) == N_THOUGHT_TOKENS


# ---------------------------------------------------------------------------
# 12. measure_thought_benefit returns correct keys
# ---------------------------------------------------------------------------


def test_measure_thought_benefit_keys():
    torch.manual_seed(0)
    cfg = LatentReasoningConfig(n_thought_tokens=N_THOUGHT_TOKENS, max_answer_tokens=4)
    model = _TinyLM()
    result = measure_thought_benefit(model, ["hi", "bye"], _tok_encode, _tok_decode, cfg)
    assert "thought_perplexity" in result
    assert "direct_perplexity" in result
    assert "relative_improvement" in result
    assert isinstance(result["thought_perplexity"], float)
    assert isinstance(result["direct_perplexity"], float)
    assert isinstance(result["relative_improvement"], float)
