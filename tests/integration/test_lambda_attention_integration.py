"""Integration tests for LambdaAttention.

Verifies that (a) the class is exposed via ``src.model``, (b) adding the new
module did NOT modify the frozen files, (c) the existing registry is intact,
and (d) an end-to-end forward/backward pass runs without error.
"""

from __future__ import annotations

import hashlib
import pathlib

import torch


FROZEN_SHA256 = {
    "transformer.py":
        "eed4c0af0f87689d6b785eac5438c3891abe988c5cacba4ead64987e772d0d3b",
    "attention.py":
        "e1edf0b235c89e7284ac64a75dd5a7ffcd95846da765714a8b32e96e75bacc49",
    "ffn.py":
        "294c8f94059f50114ccdb90a10f3134f836fb6a8619d6211027b27efb53bc102",
    "rms_norm.py":
        "2c1e41972c7b4e3699b0d532c89464578055c28079de021e8e6310f2a5edda84",
}


def _model_dir() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    root = here.parents[2]
    return root / "src" / "model"


def test_exposed_via_src_model():
    from src.model import LambdaAttention  # noqa: F401

    import src.model as m
    assert hasattr(m, "LambdaAttention")
    assert "LambdaAttention" in m.__all__


def test_frozen_files_unchanged():
    d = _model_dir()
    for name, expected in FROZEN_SHA256.items():
        got = hashlib.sha256((d / name).read_bytes()).hexdigest()
        assert got == expected, (
            f"Frozen file src/model/{name} was modified: {got} != {expected}"
        )


def test_existing_registry_intact():
    import src.model as m
    # Sanity: previously registered names are still present.
    for name in (
        "AureliusConfig", "AureliusTransformer", "ChunkedLocalAttention",
        "GroupedQueryAttention", "ParallelAttentionBlock", "RMSNorm",
        "SwiGLUFFN", "TransformerBlock",
    ):
        assert hasattr(m, name), f"{name} missing from src.model"
        assert name in m.__all__


def test_end_to_end_forward_backward():
    from src.model import LambdaAttention

    torch.manual_seed(7)
    mod = LambdaAttention(
        d_model=32, n_heads=4, head_dim_key=4, head_dim_value=8,
        n_positions=64,
    )
    x = torch.randn(2, 32, 32, requires_grad=True)
    y = mod(x)
    assert y.shape == (2, 32, 32)
    assert torch.isfinite(y).all()

    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    for p in mod.parameters():
        assert p.grad is not None
