"""Integration tests for InstructionDatasetPacker via src.data surface."""

from __future__ import annotations

import torch

import src.data as data_pkg
from src.data.instruction_dataset_packer import (
    InstructionDatasetPacker,
    InstructionSample,
    PackedBatch,
)


def test_exposed_via_src_data_module_import():
    # Must be importable via dotted module path.
    from src.data import instruction_dataset_packer as mod  # noqa: F401

    assert hasattr(mod, "InstructionSample")
    assert hasattr(mod, "PackedBatch")
    assert hasattr(mod, "InstructionDatasetPacker")


def test_existing_data_entries_intact():
    # Sanity: pre-existing exports still present after our additions.
    for name in (
        "AURELIUS_MIX",
        "DataMixConfig",
        "FIMConfig",
        "fim_transform",
        "get_mix",
    ):
        assert hasattr(data_pkg, name), f"missing pre-existing export: {name}"


def test_pack_ten_samples_bound_on_total_tokens():
    max_seq_len = 64
    packer = InstructionDatasetPacker(max_seq_len=max_seq_len, pad_token_id=0)

    samples = []
    for i in range(10):
        plen = 3 + (i % 4)
        rlen = 4 + ((i * 2) % 5)
        base = 1000 + i * 100
        samples.append(
            InstructionSample(
                prompt_token_ids=list(range(base, base + plen)),
                response_token_ids=list(range(base + plen, base + plen + rlen)),
            )
        )

    out = packer.pack(samples)
    assert len(out) >= 1
    assert all(isinstance(pb, PackedBatch) for pb in out)

    total_tokens = sum(pb.input_ids.numel() for pb in out)
    assert total_tokens <= 10 * max_seq_len

    # Every bin is exactly max_seq_len.
    for pb in out:
        assert pb.input_ids.shape == (1, max_seq_len)
        assert pb.attention_mask.shape == (1, max_seq_len, max_seq_len)
        assert pb.loss_mask.shape == (1, max_seq_len)
        assert pb.segment_ids.shape == (1, max_seq_len)

    # All response tokens are accounted for in loss_mask across bins.
    total_response = sum(len(s.response_token_ids) for s in samples)
    total_loss_true = sum(int(pb.loss_mask.sum().item()) for pb in out)
    assert total_loss_true == total_response


def test_pack_iter_on_generator():
    packer = InstructionDatasetPacker(max_seq_len=32)

    def gen():
        for i in range(6):
            yield InstructionSample(
                prompt_token_ids=[i + 1, i + 2],
                response_token_ids=[i + 3, i + 4, i + 5],
            )

    batches = list(packer.pack_iter(gen(), batch_size=3))
    assert batches
    for pb in batches:
        assert pb.input_ids.dtype == torch.long
        assert pb.attention_mask.dtype == torch.bool
        assert pb.loss_mask.dtype == torch.bool
