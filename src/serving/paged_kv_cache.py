"""Paged KV cache management for serving-side helpers.

This is a lightweight, layer-agnostic page manager that mirrors the
vLLM-style block table model closely enough for local serving utilities
and smoke tests. It favors deterministic behavior and copy-on-write
prefix sharing over kernel-specific optimizations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Block:
    block_id: int
    num_tokens: int
    ref_count: int = 1


class BlockTable:
    """Logical-to-physical block table for KV cache paging."""

    def __init__(self, block_size: int = 16, num_physical_blocks: int = 16384):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if num_physical_blocks <= 0:
            raise ValueError("num_physical_blocks must be positive")
        self.block_size = block_size
        self.num_physical_blocks = num_physical_blocks
        self._blocks: dict[int, Block] = {}
        self._free_blocks: list[int] = list(range(num_physical_blocks))
        self._logical_to_physical: dict[int, list[int]] = {}

    def _take_block(self) -> int:
        if not self._free_blocks:
            raise MemoryError("Out of KV cache blocks: need 1, have 0")
        p_id = self._free_blocks.pop(0)
        self._blocks[p_id] = Block(block_id=p_id, num_tokens=self.block_size)
        return p_id

    def _release_block(self, block_id: int) -> None:
        block = self._blocks.get(block_id)
        if block is None:
            return
        block.ref_count -= 1
        if block.ref_count <= 0:
            del self._blocks[block_id]
            self._free_blocks.append(block_id)
            self._free_blocks.sort()

    def allocate(self, seq_id: int, num_logical_blocks: int) -> list[int]:
        if num_logical_blocks < 0:
            raise ValueError("num_logical_blocks must be non-negative")
        mapping = self._logical_to_physical.setdefault(seq_id, [])
        physical_ids: list[int] = []
        if num_logical_blocks == 0:
            return physical_ids
        if num_logical_blocks > len(self._free_blocks):
            raise MemoryError(
                f"Out of KV cache blocks: need {num_logical_blocks}, have {len(self._free_blocks)}"
            )
        for _ in range(num_logical_blocks):
            p_id = self._take_block()
            mapping.append(p_id)
            physical_ids.append(p_id)
        return physical_ids

    def free(self, seq_id: int) -> None:
        mapping = self._logical_to_physical.pop(seq_id, None)
        if mapping is None:
            return
        for p_id in mapping:
            self._release_block(p_id)

    def get_physical_id(self, seq_id: int, logical_idx: int) -> int | None:
        mapping = self._logical_to_physical.get(seq_id, [])
        if logical_idx < 0 or logical_idx >= len(mapping):
            return None
        return mapping[logical_idx]

    def share_prefix(self, src_seq: int, dst_seq: int, prefix_length: int) -> None:
        prefix_blocks = math.ceil(prefix_length / self.block_size)
        src_mapping = self._logical_to_physical.get(src_seq, [])
        shared = src_mapping[:prefix_blocks]
        if dst_seq in self._logical_to_physical:
            raise ValueError(f"Sequence {dst_seq} already exists")
        dst_mapping: list[int] = []
        for p_id in shared:
            block = self._blocks.get(p_id)
            if block is None:
                continue
            block.ref_count += 1
            dst_mapping.append(p_id)
        self._logical_to_physical[dst_seq] = dst_mapping

    @property
    def used_blocks(self) -> int:
        return self.num_physical_blocks - len(self._free_blocks)

    @property
    def utilization(self) -> float:
        return self.used_blocks / self.num_physical_blocks


class PagedKVCache:
    """Paged KV cache for serving-side helpers.

    The implementation is layer-agnostic for portability. `layer_idx` is
    accepted to preserve the public helper signature but does not alter the
    storage layout.
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_blocks: int = 16384,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cpu"),
    ):
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if max_blocks <= 0:
            raise ValueError("max_blocks must be positive")

        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks = max_blocks

        shape = (max_blocks, block_size, n_kv_heads, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=device)

        self.block_table = BlockTable(
            block_size=block_size,
            num_physical_blocks=max_blocks,
        )
        self._seq_lengths: dict[int, int] = {}

    def init_sequence(self, seq_id: int, num_tokens: int) -> None:
        if seq_id in self._seq_lengths:
            return
        num_blocks = max(1, math.ceil(num_tokens / self.block_size))
        self.block_table.allocate(seq_id, num_blocks)
        self._seq_lengths[seq_id] = 0

    def _ensure_block_for_position(self, seq_id: int, logical_idx: int) -> int:
        mapping = self.block_table._logical_to_physical.setdefault(seq_id, [])
        while logical_idx >= len(mapping):
            self.block_table.allocate(seq_id, 1)
        block_id = mapping[logical_idx]
        block = self.block_table._blocks[block_id]
        if block.ref_count > 1:
            new_block = self.block_table._take_block()
            self.k_cache[new_block].copy_(self.k_cache[block_id])
            self.v_cache[new_block].copy_(self.v_cache[block_id])
            mapping[logical_idx] = new_block
            block.ref_count -= 1
            block_id = new_block
        return block_id

    def append_tokens(
        self,
        seq_id: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        if seq_id not in self._seq_lengths:
            self.init_sequence(seq_id, 1)

        pos = self._seq_lengths[seq_id]
        logical_idx = pos // self.block_size
        offset = pos % self.block_size
        block_id = self._ensure_block_for_position(seq_id, logical_idx)
        self.k_cache[block_id, offset] = k
        self.v_cache[block_id, offset] = v
        self._seq_lengths[seq_id] = pos + 1

    def gather(
        self,
        seq_id: int,
        num_tokens: int,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        written = self._seq_lengths.get(seq_id, 0)
        count = min(num_tokens, written)
        if count <= 0:
            empty = torch.zeros(
                1,
                self.n_kv_heads,
                0,
                self.head_dim,
                dtype=self.k_cache.dtype,
                device=self.k_cache.device,
            )
            return empty, empty.clone()

        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []
        for pos in range(count):
            logical_idx = pos // self.block_size
            offset = pos % self.block_size
            block_id = self.block_table.get_physical_id(seq_id, logical_idx)
            if block_id is None:
                break
            k_parts.append(self.k_cache[block_id, offset].unsqueeze(0))
            v_parts.append(self.v_cache[block_id, offset].unsqueeze(0))

        if not k_parts:
            empty = torch.zeros(
                1,
                self.n_kv_heads,
                0,
                self.head_dim,
                dtype=self.k_cache.dtype,
                device=self.k_cache.device,
            )
            return empty, empty.clone()

        k = torch.cat(k_parts, dim=0).unsqueeze(0).transpose(1, 2)
        v = torch.cat(v_parts, dim=0).unsqueeze(0).transpose(1, 2)
        return k, v

    def free_sequence(self, seq_id: int) -> None:
        self.block_table.free(seq_id)
        self._seq_lengths.pop(seq_id, None)

    def _get_seq_len(self, seq_id: int) -> int:
        return self._seq_lengths.get(seq_id, 0)

    @property
    def utilization(self) -> float:
        return self.block_table.utilization
