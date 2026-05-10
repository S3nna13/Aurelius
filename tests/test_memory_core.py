from __future__ import annotations

import torch
from aurelius.memory_core import AurelianMemoryCore, GraphConsolidator


def test_aurelian_memory_core_forward_returns_memory_state():
    torch.manual_seed(0)
    core = AurelianMemoryCore(
        d_model=16,
        d_mem=8,
        episodic_slots=2,
        lts_capacity=4,
        consolidation_freq=1,
    )
    hidden = torch.randn(1, 3, 16)

    output, mem_state = core(hidden, return_mem_state=True)

    assert output.shape == hidden.shape
    assert set(mem_state) == {"surprise", "lambda", "mem_read"}
    assert mem_state["mem_read"].shape == (1, 3, 8)


def test_graph_consolidator_clusters_slot_embeddings():
    consolidator = GraphConsolidator(d_mem=8, threshold=0.0)
    slots = torch.randn(1, 3, 8)

    clustered = consolidator(slots)

    assert clustered.shape == slots.shape
