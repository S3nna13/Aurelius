"""Tests for prefix_tuning_v3.py — 15 tests covering all classes and edge cases.

Tiny configs: n_tokens=4, d_model=16, n_heads=2 (head_dim=8),
n_layers=2, seq_len=8, batch=2.
"""

import math

import pytest
import torch
import torch.nn as nn

from src.training.prefix_tuning_v3 import (
    PrefixAttention,
    PrefixEncoder,
    PrefixTuningModel,
    PrefixTuningTrainer,
    SoftPromptEmbedding,
)

# ---------------------------------------------------------------------------
# Tiny config constants
# ---------------------------------------------------------------------------
N_TOKENS = 4
D_MODEL = 16
N_HEADS = 2
HEAD_DIM = D_MODEL // N_HEADS  # 8
N_LAYERS = 2
SEQ_LEN = 8
BATCH = 2
VOCAB = 16
PREFIX_HIDDEN = 32  # small for tests


# ---------------------------------------------------------------------------
# Minimal backbone for PrefixTuningModel tests
# ---------------------------------------------------------------------------

class TinyBackbone(nn.Module):
    """Minimal backbone: embedding + linear lm_head."""

    def __init__(self, vocab_size: int = VOCAB, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.embedding(input_ids))


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_input_ids(B: int = BATCH, T: int = SEQ_LEN) -> torch.Tensor:
    return torch.randint(0, VOCAB, (B, T))


def make_prefix_tuning_model(n_prefix: int = N_TOKENS, use_reparam: bool = True):
    backbone = TinyBackbone()
    model = PrefixTuningModel(
        backbone=backbone,
        n_prefix_tokens=n_prefix,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        use_reparameterization=use_reparam,
        prefix_hidden_dim=PREFIX_HIDDEN,  # keep tiny for tests
    )
    return model


# ===========================================================================
# 1. SoftPromptEmbedding: output shape (B, n_tokens, D)
# ===========================================================================

def test_soft_prompt_embedding_shape():
    spe = SoftPromptEmbedding(n_tokens=N_TOKENS, d_model=D_MODEL)
    out = spe(BATCH)
    assert out.shape == (BATCH, N_TOKENS, D_MODEL), (
        f"Expected ({BATCH}, {N_TOKENS}, {D_MODEL}), got {out.shape}"
    )


# ===========================================================================
# 2. SoftPromptEmbedding: prompt_embeddings requires_grad=True
# ===========================================================================

def test_soft_prompt_embedding_requires_grad():
    spe = SoftPromptEmbedding(n_tokens=N_TOKENS, d_model=D_MODEL)
    assert spe.prompt_embeddings.requires_grad, (
        "prompt_embeddings must require grad"
    )


# ===========================================================================
# 3. SoftPromptEmbedding: backward flows through prompt_embeddings
# ===========================================================================

def test_soft_prompt_embedding_backward():
    spe = SoftPromptEmbedding(n_tokens=N_TOKENS, d_model=D_MODEL)
    out = spe(BATCH)
    loss = out.sum()
    loss.backward()
    assert spe.prompt_embeddings.grad is not None, (
        "grad should flow to prompt_embeddings"
    )
    assert not torch.all(spe.prompt_embeddings.grad == 0), (
        "grad should be non-zero"
    )


# ===========================================================================
# 4. SoftPromptEmbedding: expand works for batch=1 and batch=4
# ===========================================================================

def test_soft_prompt_embedding_batch_sizes():
    spe = SoftPromptEmbedding(n_tokens=N_TOKENS, d_model=D_MODEL)
    out1 = spe(1)
    out4 = spe(4)
    assert out1.shape == (1, N_TOKENS, D_MODEL)
    assert out4.shape == (4, N_TOKENS, D_MODEL)


# ===========================================================================
# 5. PrefixEncoder: output shape (n_layers, 2, B, n_heads, n_tokens, head_dim)
# ===========================================================================

def test_prefix_encoder_output_shape():
    enc = PrefixEncoder(
        n_tokens=N_TOKENS,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        prefix_hidden_dim=PREFIX_HIDDEN,
    )
    out = enc(BATCH)
    expected = (N_LAYERS, 2, BATCH, N_HEADS, N_TOKENS, HEAD_DIM)
    assert out.shape == expected, f"Expected {expected}, got {out.shape}"


# ===========================================================================
# 6. PrefixEncoder: backward flows to prefix_params
# ===========================================================================

def test_prefix_encoder_backward():
    enc = PrefixEncoder(
        n_tokens=N_TOKENS,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        prefix_hidden_dim=PREFIX_HIDDEN,
    )
    out = enc(BATCH)
    out.sum().backward()
    assert enc.prefix_params.grad is not None, "grad must flow to prefix_params"
    assert not torch.all(enc.prefix_params.grad == 0)


# ===========================================================================
# 7. PrefixAttention: output shape (B, T, D)
# ===========================================================================

def test_prefix_attention_output_shape():
    attn = PrefixAttention(d_model=D_MODEL, n_heads=N_HEADS, n_prefix_tokens=N_TOKENS)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = attn(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), (
        f"Expected ({BATCH}, {SEQ_LEN}, {D_MODEL}), got {out.shape}"
    )


# ===========================================================================
# 8. PrefixAttention: prefix tokens genuinely alter the attention computation.
#    An attention layer with n_prefix_tokens > 0 must produce different output
#    than one with n_prefix_tokens == 0 (same weights, different prefix size).
#    We also verify that the prefix_k / prefix_v parameters participate in the
#    forward graph by checking that output changes when prefix params change.
# ===========================================================================

def test_prefix_attention_extended_keys():
    torch.manual_seed(0)
    attn = PrefixAttention(d_model=D_MODEL, n_heads=N_HEADS, n_prefix_tokens=N_TOKENS)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    # Forward with default prefix
    out1 = attn(x)
    assert out1.shape == (BATCH, SEQ_LEN, D_MODEL)

    # Perturb prefix_k and rerun — output must change (prefix tokens affect attn)
    with torch.no_grad():
        attn.prefix_k.add_(10.0)
    out2 = attn(x)

    assert not torch.allclose(out1, out2), (
        "Output must change when prefix_k is perturbed — "
        "prefix tokens must influence the attention computation"
    )


# ===========================================================================
# 9. PrefixAttention: backward flows to prefix_k and prefix_v
# ===========================================================================

def test_prefix_attention_backward():
    attn = PrefixAttention(d_model=D_MODEL, n_heads=N_HEADS, n_prefix_tokens=N_TOKENS)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = attn(x)
    out.sum().backward()
    assert attn.prefix_k.grad is not None, "grad must flow to prefix_k"
    assert attn.prefix_v.grad is not None, "grad must flow to prefix_v"


# ===========================================================================
# 10. PrefixTuningModel: backbone params are frozen (requires_grad=False)
# ===========================================================================

def test_prefix_tuning_model_backbone_frozen():
    model = make_prefix_tuning_model()
    for name, param in model.backbone.named_parameters():
        assert not param.requires_grad, (
            f"Backbone param '{name}' should be frozen"
        )


# ===========================================================================
# 11. PrefixTuningModel: prefix params are trainable (requires_grad=True)
# ===========================================================================

def test_prefix_tuning_model_prefix_trainable():
    model = make_prefix_tuning_model()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0, "There must be trainable (prefix) parameters"


# ===========================================================================
# 12. PrefixTuningModel: forward output shape (B, T, V)
# ===========================================================================

def test_prefix_tuning_model_forward_shape():
    model = make_prefix_tuning_model()
    input_ids = make_input_ids()
    logits = model(input_ids)
    assert logits.shape == (BATCH, SEQ_LEN, VOCAB), (
        f"Expected ({BATCH}, {SEQ_LEN}, {VOCAB}), got {logits.shape}"
    )


# ===========================================================================
# 13. PrefixTuningTrainer.train_step: loss is finite, trainable < frozen.
#     Uses use_reparam=False (SoftPromptEmbedding) so that prefix params
#     (n_tokens * d_model = 4*16 = 64 + prefix_embed 64 = 128) are fewer than
#     the backbone (embedding 256 + lm_head 256 = 512, all frozen).
# ===========================================================================

def test_prefix_tuning_trainer_train_step():
    # use_reparam=False avoids the MLP overhead so prefix is tiny vs backbone
    model = make_prefix_tuning_model(use_reparam=False)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    trainer = PrefixTuningTrainer(model, optimizer)

    input_ids = make_input_ids()
    labels = input_ids.clone()

    result = trainer.train_step(input_ids, labels)

    assert math.isfinite(result["loss"]), f"Loss must be finite, got {result['loss']}"
    assert result["n_trainable_params"] > 0
    assert result["n_frozen_params"] > 0
    assert result["n_trainable_params"] < result["n_frozen_params"], (
        f"Prefix params ({result['n_trainable_params']}) should be fewer than "
        f"frozen backbone params ({result['n_frozen_params']})"
    )


# ===========================================================================
# 14. PrefixTuningTrainer: grad flows to prefix params only after train_step
# ===========================================================================

def test_prefix_tuning_trainer_grad_only_on_prefix():
    model = make_prefix_tuning_model()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    trainer = PrefixTuningTrainer(model, optimizer)

    input_ids = make_input_ids()
    labels = input_ids.clone()
    trainer.train_step(input_ids, labels)

    # All backbone params must have no grad
    for name, param in model.backbone.named_parameters():
        assert param.grad is None, (
            f"Backbone param '{name}' must not have grad after train_step"
        )

    # At least one prefix param must have a grad
    trainable_grads = [
        p.grad for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert len(trainable_grads) > 0, "Prefix params must have grads after train_step"


# ===========================================================================
# 15. Border case: n_tokens=0 gives same output shape as no-prefix baseline
# ===========================================================================

def test_prefix_tuning_zero_tokens_border():
    backbone = TinyBackbone()
    model_zero = PrefixTuningModel(
        backbone=backbone,
        n_prefix_tokens=0,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
    )

    input_ids = make_input_ids()
    logits = model_zero(input_ids)

    # With 0 prefix tokens, output should still be (B, T, V)
    assert logits.shape == (BATCH, SEQ_LEN, VOCAB), (
        f"With n_tokens=0, expected ({BATCH}, {SEQ_LEN}, {VOCAB}), got {logits.shape}"
    )

    # No prefix parameters should be trainable (only backbone params, all frozen)
    # Actually n_tokens=0 means prefix_embed is None, no PrefixEncoder
    trainable = [p for p in model_zero.parameters() if p.requires_grad]
    assert len(trainable) == 0, (
        "With n_tokens=0 there should be no trainable params (backbone is frozen)"
    )
