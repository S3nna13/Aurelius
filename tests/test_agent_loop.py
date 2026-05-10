from __future__ import annotations

import torch
from aurelius.agent_loop import AgentLoopController, AgentMemoryBridge, ExperienceReplayBuffer


def test_agent_loop_controller_smoke():
    torch.manual_seed(0)
    controller = AgentLoopController(
        d_model=16,
        n_heads=4,
        d_mem=8,
        n_known_tools=4,
        n_simulations=1,
    )
    hidden = torch.randn(1, 4, 16)
    tool_descs = torch.randn(1, 4, 64)

    out = controller(hidden, tool_descs=tool_descs, full_cycle=True)

    assert out["hidden"].shape == hidden.shape
    assert out["plan"].shape[-1] == 16
    assert "critic_score" in out
    assert "suggestion" in out


def test_agent_memory_bridge_round_trip():
    bridge = AgentMemoryBridge(d_model=16, d_mem=8, episodic_slots=2)
    hidden = torch.randn(1, 4, 16)
    episodic = torch.randn(1, 2, 8)

    out = bridge.read_from_memory(hidden, episodic)

    assert out.shape == hidden.shape


def test_experience_replay_buffer_samples_tensor_batches():
    buffer = ExperienceReplayBuffer(capacity=4)
    buffer.push(
        torch.ones(2),
        torch.zeros(2),
        1.0,
        torch.full((2,), 2.0),
        False,
    )

    states, actions, rewards, next_states, dones = buffer.sample(1)

    assert states.shape == (1, 2)
    assert actions.shape == (1, 2)
    assert rewards.shape == (1,)
    assert next_states.shape == (1, 2)
    assert dones.shape == (1,)
