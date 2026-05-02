import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from aurelius_model_3b import AureliusModel3B
from agent_train import AgentTrainer, ImitationDataset
from memory_optimizer import MixedPrecisionTrainer
import logging
logger = logging.getLogger("train_3b")



def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    by_component = {}
    for name, param in model.named_parameters():
        prefix = name.split('.')[0]
        by_component[prefix] = by_component.get(prefix, 0) + param.numel()
    return {'total': total, 'components': by_component}


if __name__ == '__main__':
    with open('config_3b.yaml') as f:
        config = yaml.safe_load(f)

    model_cfg = config['aurelius_3b']
    model = AureliusModel3B(model_cfg)
    stats = count_parameters(model)

    logger.info(f"\n=== AURELIUS 3B (Agent + Skills) ===")
    total_b = stats['total'] / 1e9
    logger.info(f"Total parameters: {stats['total']:,} ({total_b:.2f}B)")

    logger.info(f"\nComponent breakdown:")
    for comp, count in sorted(stats['components'].items(), key=lambda x: -x[1]):
        logger.info(f"  {comp}: {count:,} ({100*count/stats['total']:.1f}%)")

    model.gradient_checkpointing = True

    mp_trainer = MixedPrecisionTrainer(model, enabled=True)
    logger.info(f"\nMixed precision: enabled (BF16)")

    logger.info(f"\nAgent components:")
    n_known = model_cfg.get('n_known_tools', 128)
    n_sim = model_cfg.get('n_simulations', 16)
    skill_dim = model_cfg.get('skill_dim', 256)
    max_skills = model_cfg.get('max_skills', 8192)
    logger.info(f"  ToolFormer adapter: {n_known} known tools")
    logger.info(f"  Planning: {n_sim} MCTS simulations")
    logger.info(f"  Skill library: {max_skills} skills @ dim={skill_dim}")
    logger.info(f"  Skill retrieval: top-{model_cfg.get('n_top_k_skills', 16)}")

    mem = 0
    for n, p in model.named_parameters():
        if 'memory' in n or 'skill' in n or 'tool' in n or 'agent' in n:
            mem += p.numel()
    logger.info(f"  Agent+Memory+Skill params: {mem:,} ({100*mem/stats['total']:.1f}%)")

    logger.info(f"\nInfrastructure: {config['infrastructure']['n_gpus']}×{config['infrastructure']['gpu_type']}")
    mem_per_gpu = total_b * 2 * 2 / config['infrastructure']['n_gpus']
    act_mb = model_cfg['d_model'] * model_cfg['max_seq_len'] * model_cfg['n_layers'] * 2 * 8
    mem_per_gpu += act_mb / (1024**3) / config['infrastructure']['n_gpus']
    logger.info(f"  Estimated memory per GPU: {mem_per_gpu:.1f}GB")
    logger.info(f"  Strategy: {config['infrastructure']['distributed_strategy']}")

    logger.info(f"\nReady for agent training. Run:")
    logger.info(f"  trainer = AgentTrainer(model, config)")
    logger.info(f"  trainer.train_supervised(batch)")
    logger.info(f"  trainer.train_rl()")
