"""Integration tests for ChunkedLocalAttention.

Verifies that (a) the class is exposed via ``src.model``, (b) adding the new
module did NOT modify the frozen ``src/model/attention.py`` file, and (c) an
end-to-end forward pass runs without error.
"""

from __future__ import annotations

import hashlib
import pathlib

import torch

FROZEN_ATTENTION_SHA256 = "e1edf0b235c89e7284ac64a75dd5a7ffcd95846da765714a8b32e96e75bacc49"


def _attention_path() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    # tests/integration/.. -> repo root
    root = here.parents[2]
    return root / "src" / "model" / "attention.py"


def test_exposed_via_src_model():
    import src.model as m
    from src.model import ChunkedLocalAttention  # noqa: F401

    assert hasattr(m, "ChunkedLocalAttention")
    assert "ChunkedLocalAttention" in m.__all__


def test_frozen_attention_file_unchanged():
    path = _attention_path()
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    assert got == FROZEN_ATTENTION_SHA256, (
        f"src/model/attention.py was modified — sha256 {got} (expected {FROZEN_ATTENTION_SHA256})"
    )


def test_end_to_end_forward():
    from src.model import ChunkedLocalAttention

    torch.manual_seed(1234)
    mod = ChunkedLocalAttention(
        d_model=32,
        n_heads=4,
        head_dim=8,
        chunk_size=8,
        window_size=4,
        dropout=0.0,
    )
    x = torch.randn(2, 32, 32)
    y = mod(x)
    assert y.shape == (2, 32, 32)
    assert torch.isfinite(y).all()

    # Gradient sanity.
    loss = y.pow(2).mean()
    loss.backward()
    for p in mod.parameters():
        assert p.grad is not None
