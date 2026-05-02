import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Optional, Dict, List, Tuple, Any

from aurelius_model_3b import AureliusModel3B


class PagedOptimizerState:
    def __init__(self, params, lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step = 0
        self.state = {}
        for p in self.params:
            self.state[p] = {
                'exp_avg': torch.zeros_like(p, pin_memory=True).cpu(),
                'exp_avg_sq': torch.zeros_like(p, pin_memory=True).cpu(),
            }

    def step(self):
        self.step += 1
        for p in self.params:
            if p.grad is None:
                continue
            state = self.state[p]
            exp_avg = state['exp_avg'].to(p.device, non_blocking=True)
            exp_avg_sq = state['exp_avg_sq'].to(p.device, non_blocking=True)
            grad = p.grad.data

            if self.weight_decay > 0:
                p.data.mul_(1 - self.lr * self.weight_decay)

            exp_avg.mul_(self.betas[0]).add_(grad, alpha=1 - self.betas[0])
            exp_avg_sq.mul_(self.betas[1]).add_(grad.pow(2), alpha=1 - self.betas[1])

            bias_corr1 = 1 - self.betas[0] ** self.step
            bias_corr2 = 1 - self.betas[1] ** self.step

            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(self.eps)
            step_size = self.lr / bias_corr1

            p.data.addcdiv_(exp_avg, denom, value=-step_size)

            state['exp_avg'].copy_(exp_avg.cpu(), non_blocking=True)
            state['exp_avg_sq'].copy_(exp_avg_sq.cpu(), non_blocking=True)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


class LoraLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int, r: int, alpha: float, dropout: float = 0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.empty(d_in, r))
        self.B = nn.Parameter(torch.zeros(r, d_out))
        self.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x) @ self.A @ self.B * (self.alpha / self.r)

    def merge_into_weight(self, original_weight: torch.Tensor) -> torch.Tensor:
        return original_weight + (self.A @ self.B) * (self.alpha / self.r)

    def reset(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)


class LoraMemoryModel(nn.Module):
    def __init__(self, config: dict, base_model: Optional[AureliusModel3B] = None):
        super().__init__()
        if base_model is None:
            self.model = AureliusModel3B(config)
        else:
            self.model = base_model
        self.freeze_base_model()
        self.lora_adapters = nn.ModuleDict()
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.lora_enabled = False

    def freeze_base_model(self):
        self.model = self.model.half()
        for p in self.model.parameters():
            p.requires_grad = False

    def _make_lora_hook(self, adapter_name: str):
        def hook(module, input, output):
            if self.lora_enabled:
                return output + self.lora_adapters[adapter_name](input[0])
            return output
        return hook

    def add_lora(self, r: int = 8, alpha: float = 16, dropout: float = 0.1,
                 target_modules: Optional[List[str]] = None):
        if target_modules is None:
            target_modules = ['qkv', 'mem_proj']
        self.lora_enabled = True

        for i, block in enumerate(self.model.blocks):
            if 'qkv' in target_modules:
                attn = block.attn
                name = f'block_{i}_qkv'
                self.lora_adapters[name] = LoraLayer(
                    attn.qkv.in_features, attn.qkv.out_features, r, alpha, dropout
                )
                self._hooks.append(
                    attn.qkv.register_forward_hook(self._make_lora_hook(name))
                )

            if 'mem_proj' in target_modules:
                mem = block.memory
                for proj_name, proj in [('q', mem.q_proj), ('k', mem.k_proj),
                                         ('v', mem.v_proj), ('out', mem.out_proj)]:
                    name = f'block_{i}_mem_{proj_name}'
                    self.lora_adapters[name] = LoraLayer(
                        proj.in_features, proj.out_features, r, alpha, dropout
                    )
                    self._hooks.append(
                        proj.register_forward_hook(self._make_lora_hook(name))
                    )

        self.lora_adapters = self.lora_adapters.to(
            next(self.model.parameters()).device
        )

    def disable_lora(self):
        self.lora_enabled = False

    def enable_lora(self):
        self.lora_enabled = True

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        return [p for p in self.lora_adapters.parameters() if p.requires_grad]

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        return self.model(input_ids, **kwargs)


class RewardModel(nn.Module):
    def __init__(self, base_model: AureliusModel3B, d_model: int):
        super().__init__()
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.lora = LoraLayer(d_model, 1, r=8, alpha=16, dropout=0.1)
        self.lora_enabled = True

    def disable_lora(self):
        self.lora_enabled = False

    def enable_lora(self):
        self.lora_enabled = True

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.base_model(input_ids, return_agent_state=True)
            hidden = out['hidden']
        last_hidden = hidden[:, -1, :]
        value = self.value_head(last_hidden)
        if self.lora_enabled:
            value = value + self.lora(last_hidden)
        return value.squeeze(-1)


class MemoryOffloadingRLHFCache:
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}

    def cache_rollout(self, key: str, activations: Dict[str, Any]):
        cpu_acts = {}
        for k, v in activations.items():
            if isinstance(v, torch.Tensor):
                cpu_acts[k] = v.detach().cpu()
            else:
                cpu_acts[k] = v
        self.cache[key] = cpu_acts

    def load_rollout(self, key: str) -> Optional[Dict[str, Any]]:
        return self.cache.get(key)

    def clear(self):
        self.cache.clear()


class PPOTrainer(nn.Module):
    def __init__(self, policy_model: LoraMemoryModel, d_model: int,
                 lr: float = 1e-5, gamma: float = 0.99, gae_lambda: float = 0.95):
        super().__init__()
        self.policy = policy_model
        self.value_head = nn.Linear(d_model, 1)
        self.optimizer = PagedOptimizerState(
            list(policy_model.get_trainable_parameters()) +
            list(self.value_head.parameters()),
            lr=lr,
        )
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._cache = MemoryOffloadingRLHFCache()
        self._ref_log_probs: Optional[torch.Tensor] = None

    def _log_probs(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def compute_ref_log_probs(self, input_ids: torch.Tensor,
                               response_start: int) -> torch.Tensor:
        self.policy.disable_lora()
        logits = self.policy(input_ids)['logits']
        labels = input_ids[:, 1:]
        log_probs = self._log_probs(logits[:, :-1], labels)
        ref = log_probs[:, response_start - 1:].cpu()
        self.policy.enable_lora()
        self._ref_log_probs = ref
        return ref

    def train_step(self, query: torch.Tensor, response: torch.Tensor,
                   reward_model: RewardModel, kl_coef: float = 0.1,
                   clip_range: float = 0.2) -> Dict[str, torch.Tensor]:
        self.train()
        response_len = response.shape[1]
        response_start = query.shape[1]
        input_ids = torch.cat([query, response], dim=-1)
        labels = input_ids[:, 1:]

        out = self.policy(input_ids, return_agent_state=True)
        logits = out['logits']
        hidden = out['hidden']

        log_probs = self._log_probs(logits[:, :-1], labels)
        response_log_probs = log_probs[:, response_start - 1:]

        values = self.value_head(hidden).squeeze(-1)
        response_values = values[:, response_start:]

        ref_log_probs = self._ref_log_probs.to(response_log_probs.device)
        ratio = torch.exp(response_log_probs - ref_log_probs)

        reward = reward_model(input_ids)
        reward = torch.clamp(reward, -10.0, 10.0)
        reward = (reward - reward.mean()) / (reward.std() + 1e-8)

        returns = reward.unsqueeze(-1).expand(-1, response_len)
        advantages = returns - response_values

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(response_values, returns.detach())

        kl_div = (torch.exp(response_log_probs - ref_log_probs) - 1
                  - (response_log_probs - ref_log_probs)).mean()

        total_loss = policy_loss + 0.5 * value_loss + kl_coef * kl_div

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'loss': total_loss.detach(),
            'policy_loss': policy_loss.detach(),
            'value_loss': value_loss.detach(),
            'kl_div': kl_div.detach(),
            'reward': reward.detach().mean(),
        }

    @torch.no_grad()
    def rollout(self, query: torch.Tensor, max_new_tokens: int = 128,
                temperature: float = 0.8, top_p: float = 0.9) -> torch.Tensor:
        self.policy.eval()
        generated = query.clone()
        for _ in range(max_new_tokens):
            out = self.policy(generated)
            next_token = sample_with_top_p_top_k(out['logits'][:, -1, :], temperature, top_k=0, top_p=top_p)
            generated = torch.cat([generated, next_token], dim=-1)
        return generated[:, query.shape[1]:]
