import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import time

from aurelius_model_1b import AureliusModel1B
from agent_core import ToolFormerAdapter, PlanningModule, CriticHead, ValueHead
from skills import SkillLibrary
from agent_loop import AgentLoopController, AgentMemoryBridge, AgentEpisode, ExperienceReplayBuffer
import logging
logger = logging.getLogger("agent_train")



class AgentAureliusModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.base_model = AureliusModel1B(config)

        n_layers = config['n_layers']
        d_model = config['d_model']
        n_heads = config['n_heads']
        d_mem = config['d_mem']

        self.agent_controller = AgentLoopController(
            d_model=d_model, n_heads=n_heads, d_mem=d_mem,
            n_known_tools=config.get('n_known_tools', 64),
            n_simulations=config.get('n_simulations', 8),
        )

        self.memory_bridge = AgentMemoryBridge(
            d_model=d_model, d_mem=d_mem,
            episodic_slots=config.get('episodic_slots', 1024),
        )

        self.replay = ExperienceReplayBuffer(capacity=10000)

    def forward(self, input_ids: torch.Tensor, tool_descs: torch.Tensor | None = None,
                return_agent_state: bool = False) -> dict:
        logits, h = self.base_model(input_ids, return_mem_state=True, return_hidden=True)
        episodic = None
        for i, block in enumerate(self.base_model.blocks):
            _, ms = block.memory(h, return_mem_state=True)
            if i == 0:
                episodic = ms.get('mem_read', None)
        if episodic is not None:
            h = self.memory_bridge.read_from_memory(h, episodic)
        agent_out = self.agent_controller(h, tool_descs, full_cycle=True)
        if return_agent_state:
            return logits, agent_out
        return logits

    def act(self, input_ids: torch.Tensor, tool_descs: torch.Tensor | None = None) -> dict:
        with torch.no_grad():
            logits, agent_out = self.forward(input_ids, tool_descs, return_agent_state=True)
        return agent_out

    def learn_from_episode(self, episode: AgentEpisode, gamma: float = 0.99, lr: float = 1e-5):
        self.agent_controller.learn(episode)
        returns = []
        G = 0.0
        for step in reversed(episode.steps):
            G = step.reward + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        for step_idx, step in enumerate(episode.steps):
            if step.action is not None:
                self.replay.push(
                    step.observation, step.result if step.result is not None else step.observation,
                    returns[step_idx], step.observation, step_idx == len(episode.steps) - 1
                )

    def train_rl_step(self, batch_size: int = 16) -> dict:
        if len(self.replay) < batch_size:
            return {'loss': 0.0}
        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)
        with torch.no_grad():
            next_values = self.agent_controller.value_head(next_states[:, -1:])
        values = self.agent_controller.value_head(states[:, -1:])
        advantages = rewards + dones * next_values.squeeze() - values.squeeze()
        policy_loss = -(advantages * values.squeeze()).mean()
        value_loss = F.mse_loss(values.squeeze(), rewards)
        total = policy_loss + value_loss
        return {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item(), 'total': total.item()}


class AgentTrainer:
    def __init__(self, model: AgentAureliusModel, config: dict):
        self.model = model
        self.cfg = config['training']
        self.agent_cfg = config.get('agent', {})
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.get('learning_rate', 2e-4),
            weight_decay=self.cfg.get('weight_decay', 0.1),
        )
        self.step = 0
        self.episode_log = []

    def train_supervised(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()
        input_ids = batch['input_ids']
        labels = batch['labels']
        tool_labels = batch.get('tool_labels', None)
        logits, agent_out = self.model(
            input_ids,
            tool_descs=batch.get('tool_descs', None),
            return_agent_state=True,
        )
        lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss = lm_loss
        metrics = {'lm_loss': lm_loss.item()}
        if tool_labels is not None:
            tool_logits = agent_out.get('tool_id', None)
            if tool_logits is not None:
                tool_loss = F.cross_entropy(
                    tool_logits.unsqueeze(0).float(),
                    tool_labels[:, -1].long(),
                )
                total_loss = total_loss + 0.1 * tool_loss
                metrics['tool_loss'] = tool_loss.item()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.step += 1
        return metrics

    def train_rl(self, batch_size: int = 16) -> dict:
        return self.model.train_rl_step(batch_size)

    def log_episode(self, episode: AgentEpisode):
        self.episode_log.append({
            'steps': len(episode.steps),
            'reward': episode.total_reward,
            'step': self.step,
        })
        if len(self.episode_log) > 1000:
            self.episode_log.pop(0)

    def get_agent_summary(self) -> str:
        recent = self.episode_log[-10:] if self.episode_log else []
        avg_reward = sum(e['reward'] for e in recent) / max(len(recent), 1)
        avg_steps = sum(e['steps'] for e in recent) / max(len(recent), 1)
        top_skills = self.model.agent_controller.skill_lib.registry.get_top_skills(5)
        return (
            f"Agent: step={self.step} | avg_reward={avg_reward:.2f} | "
            f"avg_ep_len={avg_steps:.1f} | top_skills={top_skills.tolist() if hasattr(top_skills, 'tolist') else top_skills}"
        )


class ImitationDataset:
    def __init__(self, demonstrations: list[dict]):
        self.demos = demonstrations

    def sample_batch(self, batch_size: int) -> dict:
        import random
        batch = random.sample(self.demos, min(batch_size, len(self.demos)))
        input_ids = torch.stack([b['input_ids'] for b in batch])
        labels = torch.stack([b['labels'] for b in batch])
        tool_labels = torch.stack([b['tool_labels'] for b in batch]) if 'tool_labels' in batch[0] else None
        return {
            'input_ids': input_ids,
            'labels': labels,
            'tool_labels': tool_labels,
        }


def agent_training_loop(model: AgentAureliusModel, config: dict,
                         imitation_data: ImitationDataset | None = None,
                         n_epochs: int = 10):
    trainer = AgentTrainer(model, config)
    for epoch in range(n_epochs):
        if imitation_data and len(imitation_data.demos) > 0:
            batch = imitation_data.sample_batch(
                config['training'].get('batch_size', 8)
            )
            metrics = trainer.train_supervised(batch)
            logger.info(f"Epoch {epoch}: lm_loss={metrics['lm_loss']:.4f}")

        rl_metrics = trainer.train_rl(batch_size=16)
        if epoch % 5 == 0:
            logger.info(trainer.get_agent_summary())
    return trainer
