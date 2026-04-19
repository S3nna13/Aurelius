"""Integration tests for :class:`ParallelAttentionBlock`.

Checks:
  * Exposed via ``src.model`` (module attribute + ``__all__``).
  * Frozen files (``transformer.py``, ``attention.py``, ``ffn.py``,
    ``rms_norm.py``) are byte-for-byte unchanged (SHA256 pinned).
  * Existing model registry entries remain intact.
  * End-to-end forward + backward runs cleanly.
"""

from __future__ import annotations

import hashlib
import pathlib

import torch


FROZEN_SHA256 = {
    "transformer.py": "eed4c0af0f87689d6b785eac5438c3891abe988c5cacba4ead64987e772d0d3b",
    "attention.py":   "e1edf0b235c89e7284ac64a75dd5a7ffcd95846da765714a8b32e96e75bacc49",
    "ffn.py":         "294c8f94059f50114ccdb90a10f3134f836fb6a8619d6211027b27efb53bc102",
    "rms_norm.py":    "2c1e41972c7b4e3699b0d532c89464578055c28079de021e8e6310f2a5edda84",
}


def _model_dir() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    return here.parents[2] / "src" / "model"


def test_exposed_via_src_model():
    import src.model as m
    from src.model import ParallelAttentionBlock  # noqa: F401

    assert hasattr(m, "ParallelAttentionBlock")
    assert "ParallelAttentionBlock" in m.__all__


def test_existing_registry_entries_intact():
    """Adding the new symbol must not remove any prior public symbol."""
    import src.model as m

    must_have = {
        "AureliusConfig",
        "AureliusTransformer",
        "ChunkedLocalAttention",
        "GroupedQueryAttention",
        "RMSNorm",
        "SwiGLUFFN",
        "TransformerBlock",
        "apply_rope",
        "count_parameters",
        "precompute_rope_frequencies",
    }
    missing = must_have - set(m.__all__)
    assert not missing, f"missing prior exports from src.model.__all__: {missing}"
    for name in must_have:
        assert hasattr(m, name), f"src.model lost attribute {name}"


def test_frozen_files_unchanged():
    model_dir = _model_dir()
    for name, expected in FROZEN_SHA256.items():
        path = model_dir / name
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        assert got == expected, (
            f"{path} was modified — sha256 {got} (expected {expected})"
        )


def test_end_to_end_forward_and_backward():
    from src.model import ParallelAttentionBlock

    torch.manual_seed(1234)
    block = ParallelAttentionBlock(
        d_model=32, n_heads=4, head_dim=8, n_kv_heads=2, d_ff=64, dropout=0.0
    )
    x = torch.randn(2, 16, 32, requires_grad=True)
    y = block(x)

    assert y.shape == (2, 16, 32)
    assert torch.isfinite(y).all()

    y.pow(2).mean().backward()
    for n, p in block.named_parameters():
        assert p.grad is not None, f"no grad for {n}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {n}"
    assert x.grad is not None and torch.isfinite(x.grad).all()
