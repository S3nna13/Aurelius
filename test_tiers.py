import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aurelius_model_7b import AureliusModel7B, BrainBridge as BrainBridge7B
from aurelius_model_14b import AureliusModel14B, BrainBridge as BrainBridge14B
from aurelius_model_32b import AureliusModel32B, BrainBridge as BrainBridge32B
from memory_core import AurelianMemoryCore, SurpriseGate, LTSMemory, GraphConsolidator
from agent_core import ToolFormerAdapter, PlanningModule, CriticHead, ValueHead, MCTSNode
from agent_loop import AgentLoopController, AgentMemoryBridge, ExperienceReplayBuffer
from skills import SkillLibrary, SkillRegistry
from brain_integrated import IntegratedNeuralBrain


def make_small_config(overrides=None):
    base = {
        'vocab_size': 1000, 'd_model': 128, 'n_heads': 4, 'd_ff': 512,
        'n_layers': 2, 'max_seq_len': 64, 'd_mem': 32,
        'episodic_slots': 32, 'lts_capacity': 64,
        'skill_dim': 32, 'n_known_tools': 16, 'n_simulations': 2,
        'gradient_checkpointing': False, 'd_brain': 64, 'n_actions': 4,
        'replay_capacity': 1000, 'agent_interval': 2,
    }
    if overrides:
        base.update(overrides)
    return base


class TestAureliusModel7B:
    def setup_method(self):
        self.config = make_small_config()
        self.model = AureliusModel7B(self.config)
        self.x = torch.randint(0, self.config['vocab_size'], (1, 16))

    def test_forward_basic(self):
        out = self.model(self.x, use_brain=False)
        assert 'logits' in out
        assert out['logits'].shape == (1, 16, self.config['vocab_size'])

    def test_forward_with_brain(self):
        out = self.model(self.x, use_brain=True, return_agent_state=True)
        assert 'logits' in out
        assert 'agent' in out
        assert 'brain' in out
        assert 'episodic' in out
        assert out['brain'] is not None
        assert 'value' in out['brain']
        assert 'epistemic_uncertainty' in out['brain']

    def test_forward_no_agent_state(self):
        out = self.model(self.x, use_brain=True, return_agent_state=False)
        assert 'logits' in out
        assert 'agent' not in out

    def test_generation(self):
        gen = self.model.generate(self.x, max_new_tokens=3, use_brain=False)
        assert gen.shape[0] == 1
        assert gen.shape[1] == 16 + 3

    def test_memory_in_blocks(self):
        for block in self.model.blocks:
            assert hasattr(block, 'memory')
            assert isinstance(block.memory, AurelianMemoryCore)

    def test_agent_in_every_4th_block(self):
        for i, block in enumerate(self.model.blocks):
            if i % 4 == 3:
                assert block.has_agent, f'Block {i} should have agent layer'
            else:
                assert not block.has_agent, f'Block {i} should not have agent layer'

    def test_gradient_checkpointing_toggle(self):
        self.model.gradient_checkpointing = True
        assert self.model.gradient_checkpointing is True
        self.model.gradient_checkpointing = False
        assert self.model.gradient_checkpointing is False

    def test_weight_tying(self):
        assert (self.model.token_embedding.weight.data == self.model.lm_head.weight.data).all()

    def test_count_parameters(self):
        params = self.model.count_parameters()
        assert params['total'] > 0
        assert params['total'] == params['trainable'] + sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        assert params['memory'] > 0
        assert params['agent'] > 0
        assert params['brain'] > 0
        assert params['transformer'] > 0

    def test_backward_pass(self):
        self.model.train()
        out = self.model(self.x, use_brain=False)
        loss = out['logits'].mean()
        loss.backward()
        grads_exist = any(p.grad is not None and p.grad.abs().sum() > 0
                         for p in self.model.parameters() if p.requires_grad)
        assert grads_exist

    def test_brain_bridge_standalone(self):
        bridge = BrainBridge7B(self.config['d_model'], self.config['d_brain'], 4)
        h = torch.randn(1, 16, self.config['d_model'])
        result = bridge(h)
        assert 'output' in result
        assert result['output'].shape == h.shape


class TestAureliusModel14B:
    def setup_method(self):
        self.config = make_small_config()
        self.model = AureliusModel14B(self.config)
        self.x = torch.randint(0, self.config['vocab_size'], (1, 16))

    def test_forward_basic(self):
        out = self.model(self.x, use_brain=False)
        assert out['logits'].shape == (1, 16, self.config['vocab_size'])

    def test_forward_with_brain(self):
        out = self.model(self.x, use_brain=True, return_agent_state=True)
        assert out['brain'] is not None
        assert 'value' in out['brain']

    def test_generation(self):
        gen = self.model.generate(self.x, max_new_tokens=3, use_brain=False)
        assert gen.shape[1] == 19

    def test_agent_interval(self):
        for i, block in enumerate(self.model.blocks):
            if i % 2 == 1:
                assert block.has_agent
            else:
                assert not block.has_agent


class TestAureliusModel32B:
    def setup_method(self):
        self.config = make_small_config()
        self.model = AureliusModel32B(self.config)
        self.x = torch.randint(0, self.config['vocab_size'], (1, 16))

    def test_forward_basic(self):
        out = self.model(self.x, use_brain=False)
        assert out['logits'].shape == (1, 16, self.config['vocab_size'])

    def test_forward_with_brain(self):
        out = self.model(self.x, use_brain=True, return_agent_state=True)
        assert out['brain'] is not None

    def test_generation(self):
        gen = self.model.generate(self.x, max_new_tokens=3, use_brain=False)
        assert gen.shape[1] == 19

    def test_count_parameters(self):
        params = self.model.count_parameters()
        assert params['total'] > 0
        assert params['total'] == params['memory'] + params['agent'] + params['brain'] + params['transformer']


class TestMemoryCore:
    def test_surprise_gate(self):
        gate = SurpriseGate(128, 32)
        h = torch.randn(1, 8, 128)
        s, lam = gate(h)
        assert s.shape == (1, 8, 32)
        assert lam.shape == (1, 8, 1)
        assert (s >= 0).all() and (s <= 1).all()

    def test_lts_memory(self):
        lts = LTSMemory(32, 64)
        q = torch.randn(1, 4, 32)
        out = lts.read(q)
        assert out.shape == (1, 4, 32)

    def test_graph_consolidator(self):
        gc = GraphConsolidator(32, threshold=0.5)
        slots = torch.randn(1, 16, 32)
        out = gc(slots)
        assert out.shape == (1, 16, 32)

    def test_full_memory_core(self):
        mem = AurelianMemoryCore(128, 32, episodic_slots=32, lts_capacity=64)
        h = torch.randn(1, 8, 128)
        out = mem(h)
        assert out.shape == (1, 8, 128)


class TestAgentCore:
    def test_tool_former(self):
        tf = ToolFormerAdapter(128, 4, 16)
        h = torch.randn(1, 8, 128)
        out, call = tf(h, None)
        assert out.shape == (1, 8, 128)

    def test_mcts_node(self):
        state = torch.randn(1, 128)
        node = MCTSNode(state)
        assert node.visits == 0
        assert node.value() == 0.0

    def test_planning_module(self):
        pm = PlanningModule(128, n_simulations=2)
        h = torch.randn(1, 4, 128)
        plan, value, root = pm(h, n_actions=4)
        assert plan.shape[-1] == 128
        assert value.shape[0] == 1


class TestSkillLibrary:
    def test_skill_registry(self):
        sr = SkillRegistry(128, skill_dim=32, max_skills=64, n_top_k=4)
        h = torch.randn(1, 8, 128)
        h_out, skill = sr(h)
        assert h_out.shape == (1, 8, 128)

    def test_skill_library(self):
        sl = SkillLibrary(128, skill_dim=32, max_skills=64)
        h = torch.randn(1, 8, 128)
        h_out, idx = sl(h)
        assert h_out.shape == (1, 8, 128)


class TestCrossTierConsistency:
    def test_all_tiers_have_same_interface(self):
        configs = {
            '7B': make_small_config(),
            '14B': make_small_config(),
            '32B': make_small_config(),
        }
        models = {
            '7B': AureliusModel7B(configs['7B']),
            '14B': AureliusModel14B(configs['14B']),
            '32B': AureliusModel32B(configs['32B']),
        }
        x = torch.randint(0, configs['7B']['vocab_size'], (1, 16))

        for name, model in models.items():
            out = model(x, use_brain=False)
            assert 'logits' in out, f'{name} missing logits'
            assert out['logits'].shape[-1] == configs['7B']['vocab_size'], f'{name} wrong vocab'

            out_full = model(x, use_brain=True, return_agent_state=True)
            assert 'agent' in out_full, f'{name} missing agent'
            assert 'brain' in out_full, f'{name} missing brain'
            assert 'episodic' in out_full, f'{name} missing episodic'

            gen = model.generate(x, max_new_tokens=3, use_brain=False)
            assert gen.shape[1] == 19, f'{name} generation length wrong'

    def test_config_yaml_loading(self):
        import yaml
        for fname in ['config_7b.yaml', 'config_14b.yaml', 'config_32b.yaml']:
            path = os.path.join(os.path.dirname(__file__), fname)
            assert os.path.exists(path), f'{fname} not found'
            with open(path) as f:
                config = yaml.safe_load(f)
            assert len(config) >= 1, f'{fname} should have at least one top-level key'
            key = list(config.keys())[0]
            model_config = config[key]
            for field in ['d_model', 'n_heads', 'd_ff', 'n_layers', 'vocab_size',
                         'max_seq_len', 'd_mem', 'episodic_slots', 'lts_capacity']:
                assert field in model_config, f'{fname} missing {field}'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])