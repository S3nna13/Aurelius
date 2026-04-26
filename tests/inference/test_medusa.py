"""Tests for Medusa multi-token prediction heads."""

import pytest
import torch

from src.inference.medusa import MedusaConfig, MedusaModel
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    return AureliusTransformer(cfg)


def test_medusa_forward_shapes(small_model):
    """forward() must return correct shapes for logits and head logits."""
    medusa_cfg = MedusaConfig(num_heads=3)
    mm = MedusaModel(small_model, medusa_cfg)

    ids = torch.randint(0, 256, (2, 16))
    loss, base_logits, head_logits = mm(ids, labels=ids)

    assert base_logits.shape == (2, 16, 256)
    assert len(head_logits) == 3
    for h in head_logits:
        assert h.shape == (2, 16, 256)
    assert loss is not None
    assert torch.isfinite(loss)


def test_medusa_forward_no_labels(small_model):
    """forward() without labels must return loss=None."""
    mm = MedusaModel(small_model)
    ids = torch.randint(0, 256, (1, 8))
    loss, base_logits, head_logits = mm(ids)
    assert loss is None
    assert base_logits.shape[0] == 1


def test_medusa_heads_trainable(small_model):
    """With freeze_base=True, only Medusa heads should be trainable."""
    mm = MedusaModel(small_model, freeze_base=True)
    trainable = [n for n, p in mm.named_parameters() if p.requires_grad]
    assert all("medusa_heads" in n for n in trainable)
    assert len(trainable) > 0


def test_medusa_generate_length(small_model):
    """generate() must produce more tokens than the prompt."""
    medusa_cfg = MedusaConfig(num_heads=2, max_new_tokens=10, temperature=1.0)
    mm = MedusaModel(small_model, medusa_cfg)

    prompt = torch.randint(0, 256, (1, 4))
    output = mm.generate(prompt)

    assert output.shape[0] == 1
    assert output.shape[1] > prompt.shape[1]
    assert output.shape[1] <= prompt.shape[1] + medusa_cfg.max_new_tokens + medusa_cfg.num_heads


def test_medusa_generate_preserves_prompt(small_model):
    """Output must start with the original prompt."""
    mm = MedusaModel(small_model)
    prompt = torch.randint(0, 256, (1, 6))
    output = mm.generate(prompt)
    assert torch.equal(output[:, : prompt.shape[1]], prompt)


def test_medusa_head_loss_trains(small_model):
    """A training step on Medusa heads must update their weights."""
    import torch.optim as optim

    mm = MedusaModel(small_model, freeze_base=True)
    optimizer = optim.AdamW([p for p in mm.parameters() if p.requires_grad], lr=1e-3)

    before = [p.clone() for p in mm.medusa_heads.parameters()]

    ids = torch.randint(0, 256, (2, 16))
    loss, _, _ = mm(ids, labels=ids)
    loss.backward()
    optimizer.step()

    after = list(mm.medusa_heads.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed, "Medusa head weights did not update"
