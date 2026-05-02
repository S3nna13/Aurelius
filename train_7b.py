import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.amp import GradScaler, autocast
import yaml
import os
import time
from functools import partial

from aurelius_model_7b import AureliusModel7B
from aurelius_model_14b import AureliusModel14B
from aurelius_model_32b import AureliusModel32B
from alignment_impl import DPOLoss, ProcessRewardModel
from memory_core import AurelianMemoryCore
import logging
logger = logging.getLogger("train_7b")



MODEL_REGISTRY = {
    '7b': AureliusModel7B,
    '14b': AureliusModel14B,
    '32b': AureliusModel32B,
}

BLOCK_REGISTRY = {
    '7b': 'AureliusBlock7B',
    '14b': 'AureliusBlock14B',
    '32b': 'AureliusBlock32B',
}


class SyntheticDataset(Dataset):
    def __init__(self, vocab_size=50257, seq_len=1024, num_samples=10000,
                 include_tool_descs=False, tool_dim=256):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.include_tool_descs = include_tool_descs
        self.tool_dim = tool_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        labels = torch.cat([input_ids[1:], torch.tensor([0], dtype=torch.long)])
        sample = {'input_ids': input_ids, 'labels': labels}
        if self.include_tool_descs:
            sample['tool_descs'] = torch.randn(self.seq_len, self.tool_dim)
        return sample


class AureliusTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.memory_loss_weight = config.get('memory_loss_weight', 0.1)
        self.consolidation_loss_weight = config.get('consolidation_loss_weight', 0.05)
        self.agent_rl_coef = config.get('agent_rl_coef', 0.3)
        self.imitation_coef = config.get('imitation_coef', 1.0)
        self.brain_loss_weight = config.get('brain_loss_weight', 0.15)

        lr = config['learning_rate']
        min_lr = config.get('min_lr', 1e-5)
        warmup_steps = config['warmup_steps']
        total_steps = config['total_steps']
        weight_decay = config.get('weight_decay', 0.1)
        beta1 = config.get('beta1', 0.9)
        beta2 = config.get('beta2', 0.95)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )

        warmup_sched = LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup_steps)
        cosine_sched = CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=min_lr)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

        self.grad_clip = config.get('grad_clip', 1.0)
        self.scaler = GradScaler('cpu')
        self.global_step = 0
        self.loss_history = {
            'total': [], 'ce': [], 'memory': [], 'agent': [],
            'brain': [], 'dpo': [], 'prm': [],
        }

        self.dpo_loss_fn = DPOLoss()
        self.prm_model = ProcessRewardModel(config.get('d_model', 3584)).to(self.device)
        self.prm_optimizer = AdamW(self.prm_model.parameters(), lr=lr * 0.1, weight_decay=weight_decay)

    def _compute_memory_loss(self, hidden_states_per_block):
        surprise_penalty = 0.0
        n_blocks = 0
        for block in self.model.blocks:
            h = hidden_states_per_block.get(block.layer_idx, None)
            if h is None:
                continue
            mem_state = block.memory.surprise
            if isinstance(mem_state, torch.Tensor):
                surprise_penalty = surprise_penalty + mem_state.pow(2).mean()
                n_blocks += 1
        if n_blocks > 0:
            surprise_penalty = surprise_penalty / n_blocks
        return surprise_penalty

    def _compute_memory_loss_from_steps(self):
        surprise_total = 0.0
        count = 0
        for block in self.model.blocks:
            if hasattr(block, 'memory') and hasattr(block.memory, 'surprise'):
                s = block.memory.surprise
                if isinstance(s, torch.Tensor):
                    surprise_total = surprise_total + s.pow(2).mean()
                    count += 1
        if count > 0:
            surprise_total = surprise_total / count
        return surprise_total

    def _compute_agent_loss(self, outputs, labels):
        if 'agent' not in outputs:
            return torch.tensor(0.0, device=self.device)
        agent_out = outputs['agent']
        if isinstance(agent_out, dict):
            if 'tool_logits' in agent_out:
                tool_logits = agent_out['tool_logits']
                tool_loss = F.cross_entropy(
                    tool_logits.view(-1, tool_logits.size(-1)),
                    labels.view(-1),
                )
                return self.agent_rl_coef * tool_loss
            if 'logits' in agent_out:
                agent_logits = agent_out['logits']
                ce_loss = F.cross_entropy(
                    agent_logits.view(-1, agent_logits.size(-1)),
                    labels.view(-1),
                )
                return self.imitation_coef * ce_loss
        if isinstance(agent_out, torch.Tensor) and agent_out.dim() >= 2:
            agent_loss = F.cross_entropy(
                agent_out.view(-1, agent_out.size(-1)),
                labels.view(-1),
            )
            return self.agent_rl_coef * agent_loss
        return torch.tensor(0.0, device=self.device)

    def _compute_brain_loss(self, outputs):
        if 'brain' not in outputs:
            return torch.tensor(0.0, device=self.device)
        brain = outputs['brain']
        if brain is None:
            return torch.tensor(0.0, device=self.device)
        value = brain.get('value', None)
        critic_score = brain.get('critic_score', None)
        if value is not None and critic_score is not None:
            if value.dim() > critic_score.dim():
                value = value.squeeze(-1) if value.dim() > 1 else value
            if critic_score.dim() > 1:
                critic_score = critic_score.squeeze(-1)
            min_len = min(value.shape[-1], critic_score.shape[-1]) if value.dim() > 0 else 1
            if value.dim() >= 1 and critic_score.dim() >= 1:
                v = value[..., :min_len].float()
                c = critic_score[..., :min_len].float()
                td_error = F.mse_loss(v, c.detach())
            else:
                td_error = F.mse_loss(value.float(), critic_score.float().detach())
            return self.brain_loss_weight * td_error
        epistemic = brain.get('epistemic_uncertainty', None)
        if epistemic is not None:
            return self.brain_loss_weight * epistemic.pow(2).mean()
        return torch.tensor(0.0, device=self.device)

    def _compute_dpo_loss(self, outputs):
        dpo = outputs.get('dpo', None)
        if dpo is not None and isinstance(dpo, dict):
            chosen_logps = dpo.get('chosen_logps', None)
            rejected_logps = dpo.get('rejected_logps', None)
            ref_chosen = dpo.get('ref_chosen_logps', None)
            ref_rejected = dpo.get('ref_rejected_logps', None)
            if all(x is not None for x in [chosen_logps, rejected_logps, ref_chosen, ref_rejected]):
                loss, _ = self.dpo_loss_fn(chosen_logps, rejected_logps, ref_chosen, ref_rejected)
                return loss
        return torch.tensor(0.0, device=self.device)

    def _compute_prm_loss(self, outputs, labels):
        hidden = outputs.get('hidden', None)
        if hidden is None:
            return torch.tensor(0.0, device=self.device)
        step_rewards = self.prm_model(hidden)
        target_rewards = torch.ones_like(step_rewards)
        prm_loss = F.mse_loss(step_rewards, target_rewards)
        return prm_loss

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        tool_descs = batch.get('tool_descs', None)
        if tool_descs is not None:
            tool_descs = tool_descs.to(self.device)

        use_amp = self.config.get('mixed_precision', 'bf16') == 'bf16'
        amp_dtype = torch.bfloat16 if use_amp else torch.float32
        amp_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

        with autocast(amp_device_type, dtype=amp_dtype, enabled=use_amp):
            outputs = self.model(
                input_ids=input_ids,
                tool_descs=tool_descs,
                use_brain=True,
                return_agent_state=True,
            )

            logits = outputs['logits']
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            memory_loss = self._compute_memory_loss_from_steps()

            agent_loss = self._compute_agent_loss(outputs, labels)

            brain_loss = self._compute_brain_loss(outputs)

            dpo_loss = self._compute_dpo_loss(outputs)

            prm_loss = self._compute_prm_loss(outputs, labels)

            total_loss = (
                ce_loss
                + self.memory_loss_weight * memory_loss
                + agent_loss
                + brain_loss
                + dpo_loss
                + 0.05 * prm_loss
            )

        if use_amp:
            self.scaler.scale(total_loss).backward()
            if self.config.get('grad_clip', 1.0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            if self.config.get('grad_clip', 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        self.scheduler.step()
        self.global_step += 1

        loss_dict = {
            'total': total_loss.item(),
            'ce': ce_loss.item(),
            'memory': memory_loss.item() if isinstance(memory_loss, torch.Tensor) else memory_loss,
            'agent': agent_loss.item() if isinstance(agent_loss, torch.Tensor) else agent_loss,
            'brain': brain_loss.item() if isinstance(brain_loss, torch.Tensor) else brain_loss,
            'dpo': dpo_loss.item() if isinstance(dpo_loss, torch.Tensor) else dpo_loss,
            'prm': prm_loss.item() if isinstance(prm_loss, torch.Tensor) else prm_loss,
        }

        for k, v in loss_dict.items():
            self.loss_history[k].append(v)

        return loss_dict

    @torch.no_grad()
    def evaluate(self, batch):
        self.model.eval()

        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        tool_descs = batch.get('tool_descs', None)
        if tool_descs is not None:
            tool_descs = tool_descs.to(self.device)

        use_amp = self.config.get('mixed_precision', 'bf16') == 'bf16'
        amp_dtype = torch.bfloat16 if use_amp else torch.float32
        amp_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

        with autocast(amp_device_type, dtype=amp_dtype, enabled=use_amp):
            outputs = self.model(
                input_ids=input_ids,
                tool_descs=tool_descs,
                use_brain=True,
                return_agent_state=True,
            )

            logits = outputs['logits']
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            memory_loss = self._compute_memory_loss_from_steps()
            agent_loss = self._compute_agent_loss(outputs, labels)
            brain_loss = self._compute_brain_loss(outputs)
            dpo_loss = self._compute_dpo_loss(outputs)
            prm_loss = self._compute_prm_loss(outputs, labels)

            total_loss = (
                ce_loss
                + self.memory_loss_weight * memory_loss
                + agent_loss
                + brain_loss
                + dpo_loss
                + 0.05 * prm_loss
            )

        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels).float()
        mask = labels != -100
        accuracy = (correct * mask).sum() / mask.sum().clamp(min=1)

        loss_dict = {
            'total': total_loss.item(),
            'ce': ce_loss.item(),
            'memory': memory_loss.item() if isinstance(memory_loss, torch.Tensor) else memory_loss,
            'agent': agent_loss.item() if isinstance(agent_loss, torch.Tensor) else agent_loss,
            'brain': brain_loss.item() if isinstance(brain_loss, torch.Tensor) else brain_loss,
            'dpo': dpo_loss.item() if isinstance(dpo_loss, torch.Tensor) else dpo_loss,
            'prm': prm_loss.item() if isinstance(prm_loss, torch.Tensor) else prm_loss,
            'accuracy': accuracy.item(),
        }

        return loss_dict

    def save_checkpoint(self, path):
        os.makedirs(path, exist_ok=True)
        model_state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'prm_model': self.prm_model.state_dict(),
            'global_step': self.global_step,
            'config': self.config,
            'loss_history': self.loss_history,
        }
        torch.save(model_state, os.path.join(path, 'checkpoint.pt'))

    def load_checkpoint(self, path):
        checkpoint_path = os.path.join(path, 'checkpoint.pt')
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.prm_model.load_state_dict(checkpoint['prm_model'])
        self.global_step = checkpoint['global_step']
        self.loss_history = checkpoint.get('loss_history', self.loss_history)
        self.config = checkpoint.get('config', self.config)


def setup_fsdp(model, config):
    if not dist.is_initialized():
        if not dist.is_available():
            return model
        try:
            dist.init_process_group(backend='nccl')
        except RuntimeError:
            return model

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)

    tier = config.get('model_tier', '7b')
    block_class_name = BLOCK_REGISTRY.get(tier, 'AureliusBlock7B')

    import aurelius_model_7b as mod_7b
    import aurelius_model_14b as mod_14b
    import aurelius_model_32b as mod_32b
    module_map = {
        'AureliusBlock7B': mod_7b,
        'AureliusBlock14B': mod_14b,
        'AureliusBlock32B': mod_32b,
    }

    block_module = module_map.get(block_class_name, mod_7b)
    block_cls = getattr(block_module, block_class_name, None)

    if block_cls is not None:
        wrap_policy = ModuleWrapPolicy({block_cls})
    else:
        wrap_policy = ModuleWrapPolicy({nn.TransformerEncoderLayer})

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        device_id=rank,
    )

    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_7b.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--tier', type=str, default=None, choices=['7b', '14b', '32b'])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    tier = args.tier
    if tier is None:
        for key in config:
            key_lower = key.lower()
            if '7b' in key_lower:
                tier = '7b'
            elif '14b' in key_lower:
                tier = '14b'
            elif '32b' in key_lower:
                tier = '32b'
        if tier is None:
            tier = '7b'

    config_key = list(config.keys())[0]
    model_config = config[config_key]
    train_config = config.get('training', {})
    data_config = config.get('data', {})

    full_config = {**model_config, **train_config, **data_config, 'model_tier': tier}

    model_cls = MODEL_REGISTRY[tier]
    model = model_cls(model_config)

    use_distributed = train_config.get('distributed_strategy', '').lower() == 'fsdp' and dist.is_available()

    if use_distributed:
        model = setup_fsdp(model, full_config)

    trainer = AureliusTrainer(model, full_config)

    if args.resume:
        resume_dir = os.path.realpath(os.path.abspath(args.resume))
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Invalid resume path (must be an existing directory): {args.resume}")
        trainer.load_checkpoint(resume_dir)

    dataset = SyntheticDataset(
        vocab_size=model_config.get('vocab_size', 50257),
        seq_len=model_config.get('max_seq_len', 16384),
        num_samples=train_config.get('total_steps', 700000) * train_config.get('batch_size', 2),
        include_tool_descs=True,
        tool_dim=model_config.get('n_known_tools', 256),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=train_config.get('batch_size', 2),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    total_steps = train_config.get('total_steps', 700000)
    eval_interval = data_config.get('eval_interval', 500)
    save_interval = data_config.get('save_interval', 5000)

    start_step = trainer.global_step
    data_iter = iter(dataloader)

    logger.info(f"Aurelius {tier.upper()} Training")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Starting from step {start_step}, running to {total_steps}")
    logger.info(f"Eval every {eval_interval} steps, save every {save_interval} steps")

    for step in range(start_step, total_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        loss_dict = trainer.train_step(batch)

        if step % eval_interval == 0 and step > 0:
            eval_batch = next(data_iter) if True else batch
            eval_metrics = trainer.evaluate(batch)
            lr = trainer.scheduler.get_last_lr()[0]
            logger.info(
                f"[Step {step}] "
                f"train_loss={loss_dict['total']:.4f} "
                f"ce={loss_dict['ce']:.4f} "
                f"mem={loss_dict['memory']:.4f} "
                f"agent={loss_dict['agent']:.4f} "
                f"brain={loss_dict['brain']:.4f} "
                f"dpo={loss_dict['dpo']:.4f} "
                f"prm={loss_dict['prm']:.4f} "
                f"lr={lr:.2e}"
            )
            if 'accuracy' in eval_metrics:
                logger.info(f"  eval: loss={eval_metrics['total']:.4f} acc={eval_metrics['accuracy']:.4f}")

        if step % save_interval == 0 and step > 0:
            checkpoint_dir = f"checkpoints/aurelius_{tier}_step{step}"
            if not use_distributed or dist.get_rank() == 0:
                tmp_path = checkpoint_dir + ".tmp"
                trainer.save_checkpoint(tmp_path)
                if os.path.exists(tmp_path):
                    os.rename(tmp_path, checkpoint_dir)
                    logger.info(f"Checkpoint saved: {checkpoint_dir}")

    final_dir = f"checkpoints/aurelius_{tier}_final"
    if not use_distributed or dist.get_rank() == 0:
        trainer.save_checkpoint(final_dir)
        logger.info(f"Training complete. Final checkpoint: {final_dir}")


if __name__ == '__main__':
    main()