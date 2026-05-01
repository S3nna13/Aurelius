"""agent_registry.py — contract surface for agent components.

Contract: Agent components implement Observe→Think→Act→Reflect→Learn loop.
ToolFormerAdapter cross-attends to tool descriptions. PlanningModule runs MCTS.
Live path: agent_core.* and agent_loop.* → torch tensor operations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import torch

from agent_core import (
    ToolEmbedding as _ToolEmbedding,
    ToolFormerAdapter as _ToolFormerAdapter,
    PlanningModule as _PlanningModule,
    CriticHead as _CriticHead,
    ValueHead as _ValueHead,
    MCTSNode as _MCTSNode,
)
from agent_loop import (
    AgentLoopController as _AgentLoopController,
    AgentMemoryBridge as _AgentMemoryBridge,
    AgentContextManager as _AgentContextManager,
    ExperienceReplayBuffer as _ExperienceReplayBuffer,
)
from recursive_mas import (
    RecursiveLink as _RecursiveLink,
    RecursiveAgentWrapper as _RecursiveAgentWrapper,
    InnerOuterOptimizer as _InnerOuterOptimizer,
)


@dataclass
class RegistryEntry:
    name: str
    version: str = "0.1.0"
    contract: str = ""
    live: bool = True
    path: str = ""
    test_command: str = ""


AGENT_REGISTRY: dict[str, RegistryEntry] = {
    "tool_embedding": RegistryEntry(
        name="ToolEmbedding",
        contract="Embed tool IDs or descriptions into d_model space",
        path="agent_core.ToolEmbedding",
        test_command="python3 -m pytest tests.py -k test_tool_embedding -v",
    ),
    "tool_former_adapter": RegistryEntry(
        name="ToolFormerAdapter",
        contract="Cross-attend hidden states to tool embeddings + optional tool call head",
        path="agent_core.ToolFormerAdapter",
        test_command="python3 -m pytest tests.py -k test_tool_former_adapter -v",
    ),
    "tool_call_head": RegistryEntry(
        name="ToolCallHead",
        contract="Predict tool selection logits, param presence, and param values",
        path="agent_core.ToolCallHead",
        test_command="python3 -m pytest tests.py -k test_tool_call_head -v",
    ),
    "planning_module": RegistryEntry(
        name="PlanningModule",
        contract="MCTS-based planning with ValueHead lookahead",
        path="agent_core.PlanningModule",
        test_command="python3 -m pytest tests.py -k test_planning_module -v",
    ),
    "mcts_node": RegistryEntry(
        name="MCTSNode",
        contract="Monte Carlo Tree Search node with UCB scoring",
        path="agent_core.MCTSNode",
        test_command="python3 -m pytest tests.py -k test_mcts_node -v",
    ),
    "critic_head": RegistryEntry(
        name="CriticHead",
        contract="Score state-action pairs and suggest corrections",
        path="agent_core.CriticHead",
        test_command="python3 -m pytest tests.py -k test_critic_head -v",
    ),
    "value_head": RegistryEntry(
        name="ValueHead",
        contract="Predict scalar value from hidden state",
        path="agent_core.ValueHead",
        test_command="python3 -m pytest tests.py::test_value_head_shape -v",
    ),
    "cross_attention": RegistryEntry(
        name="ToolCrossAttention",
        contract="Gated cross-attention: query from hidden, key/value from tool embeds",
        path="agent_core.ToolCrossAttention",
        test_command="python3 -m pytest tests.py::test_cross_attention_shape -v",
    ),
    "tool_result_integrator": RegistryEntry(
        name="ToolResultIntegrator",
        contract="Gate tool results back into hidden state",
        path="agent_core.ToolResultIntegrator",
        test_command="python3 -m pytest tests.py::test_tool_result_integrator -v",
    ),
    "agent_loop_controller": RegistryEntry(
        name="AgentLoopController",
        contract="Full Observe→Think→Act→Reflect→Learn loop with skill library",
        path="agent_loop.AgentLoopController",
        test_command="python3 -m pytest tests.py -k test_agent_loop -v",
    ),
    "agent_memory_bridge": RegistryEntry(
        name="AgentMemoryBridge",
        contract="Bidirectional read/write between agent hidden state and episodic memory",
        path="agent_loop.AgentMemoryBridge",
        test_command="python3 -m pytest tests.py -k test_agent_memory -v",
    ),
    "context_manager": RegistryEntry(
        name="AgentContextManager",
        contract="Sliding-window context buffer with task summary",
        path="agent_loop.AgentContextManager",
        test_command="python3 -m pytest tests.py -k test_context_manager -v",
    ),
    "experience_replay": RegistryEntry(
        name="ExperienceReplayBuffer",
        contract="Fixed-capacity replay buffer for experience sampling",
        path="agent_loop.ExperienceReplayBuffer",
        test_command="python3 -m pytest tests.py -k test_experience_replay -v",
    ),
    "recursive_link": RegistryEntry(
        name="RecursiveLink",
        contract="Cross-agent latent state transfer with learned gating",
        path="recursive_mas.RecursiveLink",
        test_command="python3 -m pytest test_recursive_mas.py::test_recursive_link_shapes -v",
    ),
    "recursive_mas_loop": RegistryEntry(
        name="RecursiveAgentWrapper",
        contract="Multi-agent recursion orchestration with latent state accumulation",
        path="recursive_mas.RecursiveAgentWrapper",
        test_command="python3 -m pytest test_recursive_mas.py::test_recursive_mas_forward -v",
    ),
    "inner_outer_optimizer": RegistryEntry(
        name="InnerOuterOptimizer",
        contract="Inner-outer loop gradient-based credit assignment across recursion rounds",
        path="recursive_mas.InnerOuterOptimizer",
        test_command="python3 -m pytest test_recursive_mas.py::test_inner_outer_step_shapes -v",
    ),
}


def get_registry() -> dict[str, RegistryEntry]:
    return AGENT_REGISTRY


def lookup(name: str) -> RegistryEntry | None:
    return AGENT_REGISTRY.get(name)


class ToolEmbeddingContract:
    def __init__(self, d_model: int = 64, max_tools: int = 16, tool_desc_dim: int = 64):
        self._impl = _ToolEmbedding(d_model, max_tools, tool_desc_dim)
        self._d_model = d_model
        self._max_tools = max_tools
        self._tool_desc_dim = tool_desc_dim

    def verify_contract(self, batch: int = 2, time: int = 8) -> None:
        out = self._impl()
        assert out.shape == (1, self._d_model), f"no-arg shape: {out.shape}"
        ids = torch.randint(0, self._max_tools, (batch, time))
        out_ids = self._impl(tool_ids=ids)
        assert out_ids.shape == (batch, time, self._d_model), f"id shape: {out_ids.shape}"
        descs = torch.randn(batch, time, self._tool_desc_dim)
        out_desc = self._impl(tool_descs=descs)
        assert out_desc.shape == (batch, time, self._d_model), f"desc shape: {out_desc.shape}"


class ToolFormerAdapterContract:
    def __init__(self, d_model: int = 64, n_heads: int = 4, n_known_tools: int = 16):
        self._impl = _ToolFormerAdapter(d_model, n_heads, n_known_tools)
        self._d_model = d_model

    def verify_contract(self, batch: int = 2, time: int = 8) -> None:
        h = torch.randn(batch, time, self._d_model)
        out, call = self._impl(h)
        assert out.shape == (batch, time, self._d_model), f"forward shape: {out.shape}"
        assert call is None, "call should be None without return_call=True"
        out2, call2 = self._impl(h, return_call=True)
        assert call2 is not None, "call should not be None with return_call=True"
        logits, presence, params = call2
        assert logits.shape == (batch, 16)
        assert presence.shape == (batch, 8)
        assert params.shape == (batch, 64)


class PlanningModuleContract:
    def __init__(self, d_model: int = 64, n_simulations: int = 2, max_depth: int = 2):
        self._impl = _PlanningModule(d_model, n_simulations, max_depth)

    def verify_contract(self, batch: int = 2, time: int = 8) -> None:
        h = torch.randn(batch, time, 64)
        plan, value, tree = self._impl(h, n_actions=4)
        assert plan.shape[0] == batch
        assert value.shape[0] == batch


class AgentLoopControllerContract:
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_mem: int = 64,
                 n_known_tools: int = 8, n_simulations: int = 2):
        self._impl = _AgentLoopController(d_model, n_heads, d_mem, n_known_tools, n_simulations)

    def verify_contract(self, batch: int = 2, time: int = 8) -> None:
        h = torch.randn(batch, time, 64)
        out = self._impl.observe(h)
        assert out.shape == (batch, time, 64)
        plan, value = self._impl.think(h)
        assert plan.shape[0] == batch or plan.dim() == 2
        h_acted, tool_id, skill_idx = self._impl.act(h)
        score, suggestion = self._impl.reflect(h, plan)
        assert isinstance(score, float)


class AgentMemoryBridgeContract:
    def __init__(self, d_model: int = 64, d_mem: int = 64, episodic_slots: int = 16):
        self._impl = _AgentMemoryBridge(d_model, d_mem, episodic_slots)

    def verify_contract(self, batch: int = 2, time: int = 8) -> None:
        agent = torch.randn(batch, time, 64)
        mem = torch.randn(1, self._impl.episodic_slots, 64)
        out = self._impl.read_from_memory(agent, mem)
        assert out.shape == (batch, time, 64)
        out2 = self._impl.write_to_memory(agent, mem, slot_idx=0)
        assert out2.shape == mem.shape


class CriticHeadContract:
    def __init__(self, d_model: int = 64):
        self._impl = _CriticHead(d_model)

    def verify_contract(self, batch: int = 2, time: int = 8) -> None:
        state = torch.randn(batch, time, 64)
        action = torch.randn(batch, 64)
        score, suggestion = self._impl(state, action)
        assert score.shape == (batch,)
        assert suggestion.shape == (batch, 64)


class ValueHeadContract:
    def __init__(self, d_model: int = 64):
        self._impl = _ValueHead(d_model)

    def verify_contract(self, batch: int = 2, time: int = 8) -> None:
        h = torch.randn(batch, time, 64)
        out = self._impl(h)
        assert out.shape == (batch,)
