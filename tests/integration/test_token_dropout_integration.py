"""Integration tests for TokenDropout inside a tiny LM training step."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.training as training_pkg
from src.training.token_dropout import TokenDropout


def test_exposed_via_src_training() -> None:
    # Must be accessible via the package surface.
    assert hasattr(training_pkg, "TokenDropout")
    assert training_pkg.TokenDropout is TokenDropout


def test_prior_training_entries_intact() -> None:
    # FSDPLite and friends must still be exposed.
    for name in ("FSDPLite", "ShardSpec", "shard_tensor", "gather_tensor"):
        assert hasattr(training_pkg, name), f"missing {name}"


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 32, dim: int = 16) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.emb(ids))


def test_training_step_with_token_dropout() -> None:
    torch.manual_seed(0)
    vocab = 32
    model = _TinyLM(vocab=vocab, dim=16)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)

    td = TokenDropout(p=0.2, mask_token_id=0, mode="both", exclude_special_ids=(1,))

    ids = torch.randint(1, vocab, (4, 8))
    targets = ids.clone()

    gen = torch.Generator(device="cpu")
    gen.manual_seed(7)
    new_ids, loss_mask = td.apply(ids, rng=gen)

    logits = model(new_ids)
    loss_flat = F.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1),
        reduction="none",
    )
    loss = (loss_flat * loss_mask.reshape(-1)).sum() / loss_mask.sum().clamp(min=1.0)

    assert torch.isfinite(loss)
    loss.backward()
    # At least one parameter should have a non-zero gradient.
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    opt.step()
    opt.zero_grad()


def test_integration_determinism() -> None:
    ids = torch.randint(1, 50, (2, 16))
    td = TokenDropout(p=0.3, mask_token_id=0, mode="replace")

    g1 = torch.Generator(device="cpu")
    g1.manual_seed(2026)
    out1, m1 = td.apply(ids, rng=g1)

    g2 = torch.Generator(device="cpu")
    g2.manual_seed(2026)
    out2, m2 = td.apply(ids, rng=g2)

    assert torch.equal(out1, out2)
    assert torch.equal(m1, m2)
