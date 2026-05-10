from __future__ import annotations

import torch

import rust_bridge


def test_rust_bridge_fallback_page_table_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(rust_bridge, "HAS_RUST", False)
    monkeypatch.setattr(rust_bridge, "_rust_page_table", None)

    page_table = rust_bridge.get_page_table(capacity=2, gpu_budget_mb=1)
    assert page_table.register_page(1, 0.5, 128, False) == "ok"
    assert page_table.access(1) in {"gpu", "cpu"}
    assert "pages=1" in page_table.stats()

    checkpoint = tmp_path / "checkpoint.pt"
    rust_bridge.save_checkpoint(str(checkpoint), {"x": torch.ones(2)})
    assert checkpoint.exists()

    memory = rust_bridge.estimate_memory(8, 16, 2, 4, 1)
    assert isinstance(memory, str)
    assert memory
