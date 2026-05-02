import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple, List, Union, Callable

from brain_layer import (
    InputEncoder, WorkingMemory, LongTermMemory, ReasoningCore,
    Planner, ToolController, AgentRouter, Verifier, Critic,
    ReflectionModule, ExecutiveController, UncertaintyEstimator,
    SelfImprovementLoop, RMSNorm, GatedResidual, MLP,
)

from reasoning_paper_impl import (
    SelfTaughtReasoner, TreeOfThoughtSearcher,
    QuietStarEngine, SkillLibraryVoyager,
)

from memory_moe_impl import (
    NoisyTopKRouter, ExpertChoiceRouter,
    CompressiveMemory, KNNAttentionLayer, MoEMemoryLayer,
)

from alignment_impl import (
    DPOLoss, ConstitutionalClassifier, RARRRetriever, ProcessRewardModel,
)

from efficiency_impl import (
    TiledFlashAttention, PagedKVManager,
    StreamingCache, ZeROOptimizer, DistributedTrainingManager,
)
import logging
logger = logging.getLogger(__name__)


class IntegratedNeuralBrain(nn.Module):
    def __init__(
        self,
        d_brain: int = 1024,
        d_model: int = 768,
        vocab_size: int = 10000,
        n_heads: int = 8,
        n_slots: int = 64,
        n_tools: int = 32,
        n_agents: int = 10,
        n_actions: int = 8,
        n_experts: int = 8,
        n_principles: int = 16,
        max_steps: int = 50,
    ):
        super().__init__()
        self.d_brain = d_brain
        self.d_model = d_model
        self.max_steps = max_steps

        hd = d_model // n_heads

        # ── Base brain modules (13) ──
        self.encoder = InputEncoder(d_brain, d_model)
        self.wm = WorkingMemory(d_brain, n_slots)
        self.ltm = LongTermMemory(d_brain)
        self.reasoning_core = ReasoningCore(d_brain)
        self.planner = Planner(d_brain)
        self.tool_ctrl = ToolController(d_brain, n_tools)
        self.agent_router = AgentRouter(d_brain, n_agents)
        self.verifier = Verifier(d_brain)
        self.critic = Critic(d_brain)
        self.reflection_mod = ReflectionModule(d_brain)
        self.executive = ExecutiveController(d_brain, n_actions)
        self.uncertainty_est = UncertaintyEstimator(d_brain)
        self.self_improve = SelfImprovementLoop(d_brain)

        # ── Reasoning paper modules (4) ──
        self.star_reasoner = SelfTaughtReasoner(
            vocab_size=vocab_size, d_model=d_model, n_layers=2,
            n_heads=n_heads, d_ff=d_model * 4, max_seq_len=1024,
        )
        self.tot_searcher = TreeOfThoughtSearcher(
            d_model=d_model, vocab_size=vocab_size,
            n_heads=n_heads, d_ff=d_model * 4, n_encoder_layers=2,
        )
        self.quiet_star = QuietStarEngine(
            vocab_size=vocab_size, d_model=d_model,
            n_layers_main=2, n_heads_main=n_heads,
            d_ff_main=d_model * 4, d_ff_thought=d_model * 2,
            max_thought_steps=32,
        )
        self.skill_library = SkillLibraryVoyager(
            d_model=d_model, d_skill=d_brain // 2, max_skills=1000,
            n_heads=n_heads, d_ff=d_model * 4, n_encoder_layers=2,
        )

        # ── Memory & MoE modules (5) ──
        self.noisy_router = NoisyTopKRouter(
            d_model=d_model, n_experts=n_experts, top_k=2,
        )
        self.expert_router = ExpertChoiceRouter(
            d_model=d_model, n_experts=n_experts,
        )
        self.compr_mem = CompressiveMemory(
            d_model=d_model, mem_len=256, cmem_len=128, compress_ratio=2,
        )
        self.knn_attn = KNNAttentionLayer(
            d_model=d_model, n_heads=n_heads, k=32,
        )
        self.moe_memory = MoEMemoryLayer(
            d_model=d_model, n_heads=n_heads, n_experts=n_experts,
        )

        # ── Alignment modules (4) ──
        self.dpo_loss = DPOLoss()
        self.constitutional = ConstitutionalClassifier(
            d_model=d_model, n_principles=n_principles,
        )
        self.rarr = RARRRetriever(d_model=d_model)
        self.process_reward = ProcessRewardModel(d_model=d_model)

        # ── Tool registry for dispatch ──
        self._tool_registry: Dict[int, Callable] = {}

        # ── Efficiency modules (5) ──
        self.flash_attn = TiledFlashAttention(
            d_model=d_model, n_heads=n_heads, block_size=128,
        )
        self.kv_manager = PagedKVManager(
            n_layers=4, n_heads=n_heads,
            head_dim=hd, block_size=16, max_blocks=4096,
        )
        self.stream_cache = StreamingCache(
            d_model=d_model, window_size=2048, n_sink=4,
        )
        self.dist_manager = DistributedTrainingManager(
            d_model=d_model, d_ff=d_model * 4, n_heads=n_heads,
        )

        # Cross-dim projection (d_model -> d_brain)
        self.out_proj = nn.Linear(d_model, d_brain)

        # Memory state for CompressiveMemory
        self.register_buffer('budget', torch.tensor(1.0))
        self._mem_state = None
        self._stream_cache_state = None
        self._moe_mem_state = None

    def forward(
        self,
        x: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> Dict[str, Any]:
        if x is not None:
            if isinstance(x, tuple):
                hidden_states, input_ids = x
            elif x.dtype == torch.long:
                input_ids = x
            else:
                hidden_states = x
        if hidden_states is None and input_ids is None:
            raise TypeError("At least one of hidden_states or input_ids must be provided")
        if hidden_states is None and input_ids is not None:
            hidden_states = self.star_reasoner._embed(input_ids)
        if input_ids is None and hidden_states is not None:
            input_ids = torch.zeros(hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device)

        b, seq_len, d = hidden_states.shape
        device = hidden_states.device

        losses = {}
        trajectory = []

        # ── 1. Encode ──
        s = self.encoder(hidden_states)
        goal = s.clone()

        # ── 2. Retrieve from LTM ──
        ltm_context = self.ltm.retrieve(s)
        ltm_embed = None
        if ltm_context:
            ltm_embed = torch.stack([c[0] for c in ltm_context]).to(device).mean(dim=0, keepdim=True)

        # ── 3. Working Memory ──
        wm_out = self.wm(s, goal)

        # ── 3b. Planner & Agent Router ──
        plan_subtasks, plan_deps = self.planner(goal, wm_out)
        trajectory.append({'stage': 'plan', 'subtasks': plan_subtasks, 'dependencies': plan_deps, 'wm': wm_out.detach().clone()})
        agent_routes = []
        for i in range(min(3, plan_subtasks.shape[1])):
            route = self.agent_router.route(plan_subtasks[:, i, :])
            agent_routes.append(route)
        trajectory.append({'stage': 'route', 'agent_routes': agent_routes, 'wm': wm_out.detach().clone()})

        # ── 4. TiledFlashAttention + KV Manager ──
        attn_out, k, v = self.flash_attn(hidden_states, return_kv=True)
        n_blocks = (seq_len + self.kv_manager.block_size - 1) // self.kv_manager.block_size
        if hasattr(self, '_kv_blocks') and self._kv_blocks:
            for blocks in self._kv_blocks:
                self.kv_manager.free(blocks)
        self._kv_blocks = []
        for layer_idx in range(self.kv_manager.n_layers):
            blocks = self.kv_manager.alloc(n_blocks)
            self._kv_blocks.append(blocks)
            for bi, blk_id in enumerate(blocks):
                start = bi * self.kv_manager.block_size
                end = min(start + self.kv_manager.block_size, seq_len)
                for pos in range(start, end):
                    self.kv_manager.write(layer_idx, blk_id, pos, k[0, :, pos, :], v[0, :, pos, :])

        # ── 5. Memory augmentations + MoE Routers ──
        knn_out = self.knn_attn(attn_out)
        if self._mem_state is None:
            fine_mem = [knn_out[0, i:i+1, :].detach() for i in range(min(seq_len, 64))]
            self._mem_state = (fine_mem, [])
        mem_out, recon_loss, self._mem_state = self.compr_mem(knn_out, self._mem_state)
        moe_out, moe_aux_loss, moe_mem_state = self.moe_memory(mem_out, self._moe_mem_state)
        self._moe_mem_state = moe_mem_state
        losses['recon'] = recon_loss
        losses['moe_aux'] = moe_aux_loss

        # Wire MoE routers on memory representation
        mem_repr = mem_out.mean(dim=1)
        noisy_dispatch, noisy_indices, noisy_aux = self.noisy_router(mem_repr)
        expert_dispatch, expert_assign, expert_scores, expert_indices, expert_load_loss, expert_importance_loss = self.expert_router(mem_repr)
        losses['noisy_aux'] = noisy_aux
        losses['expert_load'] = expert_load_loss
        losses['expert_importance'] = expert_importance_loss
        trajectory.append({'stage': 'moe_route', 'noisy_indices': noisy_indices, 'expert_indices': expert_indices, 'wm': moe_out.detach().clone()})

        # ── 6. QuietStar internal thought ──
        thought_result = self.quiet_star(input_ids, n_think_tokens=4, return_thought_info=True)
        trajectory.append({'stage': 'quiet_star', 'thought_mask': thought_result.get('thought_mask'), 'wm': wm_out.detach().clone()})

        # ── 7. ToT search (BFS) ──
        tot_result = self.tot_searcher(input_ids)
        trajectory.append({'stage': 'tot', 'best_path': tot_result['best_path'], 'scores': tot_result['value_scores'], 'wm': wm_out.detach().clone()})

        # ── 8. STaR reasoning ──
        wm_2d = wm_out if wm_out.dim() == 2 else wm_out.mean(dim=1) if wm_out.dim() > 2 else wm_out
        goal_2d = goal if goal.dim() == 2 else goal.mean(dim=1) if goal.dim() > 2 else goal
        star_out, confidence = self.reasoning_core(
            wm_2d.unsqueeze(0) if wm_2d.dim() == 1 else wm_2d,
            goal_2d.unsqueeze(0) if goal_2d.dim() == 1 else goal_2d,
            ltm_context,
        )
        trajectory.append({'stage': 'star', 'confidence': confidence, 'wm': wm_out.detach().clone()})

        # ── 9. Verify with ProcessRewardModel + ConstitutionalClassifier + Verifier + Critic ──
        step_rewards = self.process_reward(moe_out)
        violation_scores, violation_types = self.constitutional(moe_out.mean(dim=1))
        verified = moe_out
        if (violation_scores > 0.5).any():
            vios, suggestions = self.constitutional.critique(moe_out.mean(dim=1))
            verified = verified + 0.01 * suggestions.unsqueeze(1)

        # Wire Verifier and Critic (project to d_brain space)
        moe_brain = self.out_proj(moe_out)
        error_prob, error_types, conf_verifier, fix = self.verifier(moe_brain, wm_out.unsqueeze(0) if wm_out.dim() == 2 else wm_out)
        critic_score, critic_suggestion = self.critic(moe_brain, wm_out.unsqueeze(0) if wm_out.dim() == 2 else wm_out)
        trajectory.append({'stage': 'verify', 'rewards': step_rewards, 'violations': violation_scores, 'verifier_conf': conf_verifier, 'critic_score': critic_score, 'wm': moe_out.detach().clone()})

        # ── 10. RARR verify & revise ──
        rarr_evidence, rarr_confidence = self.rarr(verified.mean(dim=1))
        losses['rarr_confidence'] = rarr_confidence.mean() if isinstance(rarr_confidence, torch.Tensor) else torch.tensor(rarr_confidence, device=device)

        # ── 11. StreamingCache ──
        stream_out, self._stream_cache_state = self.stream_cache(verified, self._stream_cache_state)

        trajectory.append({'stage': 'stream', 'wm': wm_out.detach().clone()})
        # ── 12. Reflect with SkillLibraryVoyager ──
        task_embed = s.mean(dim=1)
        skills = self.skill_library(task_embed, top_k=3)
        discovered_skill = self.skill_library.discover_skill(verified)
        trajectory.append({'stage': 'reflect', 'skills': skills, 'wm': wm_out.detach().clone()})

        # ── 12b. Tool Controller ──
        tool_idx, tool_conf = self.tool_ctrl.select(wm_out, goal)
        if tool_conf.max().item() > 0.3:
            tool_id = int(tool_idx[0, 0].item()) if tool_idx.dim() > 1 else int(tool_idx[0].item())
            if tool_id in self._tool_registry:
                tool_result = self._tool_registry[tool_id](wm_out)
            else:
                tool_result = wm_out * 0.01
            wm_out = self.tool_ctrl.integrate(wm_out, tool_result)
        trajectory.append({'stage': 'tool', 'tool_idx': tool_idx, 'tool_conf': tool_conf, 'wm': wm_out.detach().clone()})

        # ── 13. ReflectionModule ──
        wm_snapshots = [t['wm'].mean(dim=1) if t['wm'].dim() == 3 else t['wm'] for t in trajectory if isinstance(t, dict) and 'wm' in t]
        if not wm_snapshots:
            wm_snapshots = [wm_out.detach().clone()]
        ref_out = self.reflection_mod(wm_snapshots, s.mean(dim=1))

        # ── 14. Uncertainty ──
        brain_out = self.out_proj(stream_out)
        epi_unc, alea_unc, entropy = self.uncertainty_est(brain_out)

        # ── 15. Executive - select final action ──
        action_idx, value = self.executive(s, wm_out, budget=self.budget.item())

        # ── Training losses (DPO placeholder — needs real paired preference data) ──
        if training:
            logit_for_dpo = self.out_proj(stream_out.mean(dim=1))
            pseudo_chosen = F.log_softmax(logit_for_dpo, dim=-1).mean(dim=-1)
            pseudo_rejected = pseudo_chosen - 0.5
            ref_chosen = pseudo_chosen.detach()
            ref_rejected = pseudo_rejected.detach()
            dpo_loss_val, dpo_acc = self.dpo_loss(
                pseudo_chosen, pseudo_rejected, ref_chosen, ref_rejected,
            )
            losses['dpo'] = dpo_loss_val
            losses['dpo_acc'] = dpo_acc

        return {
            'output': stream_out,
            'confidence': max(0.01, min(0.99, 1.0 - (epi_unc + alea_unc + entropy).mean().item())),
            'steps': len(trajectory),
            'trajectory': trajectory,
            'uncertainty': {
                'epistemic': epi_unc,
                'aleatoric': alea_unc,
                'entropy': entropy,
            },
            'reflection': {
                'summary': ref_out.get('summary', torch.zeros(1, self.d_brain, device=device)),
                'improvement': ref_out.get('improvement', torch.tensor(0.0, device=device)),
                'skills': skills,
                'discovered_skill': discovered_skill,
            },
            'losses': losses,
        }

    def learn(self, trajectory, reward, task_embed, action, logprob):
        return self.self_improve.update(trajectory, reward, task_embed, action, logprob)

    def register_tool(self, tool_id: int, fn: Callable[[torch.Tensor], torch.Tensor]):
        self._tool_registry[tool_id] = fn

    def reset(self):
        self.wm.reset()
        self.executive.reset()
        self._mem_state = None
        self._moe_mem_state = None
        self._stream_cache_state = None
        if hasattr(self, '_kv_blocks') and self._kv_blocks:
            for blocks in self._kv_blocks:
                self.kv_manager.free(blocks)
            self._kv_blocks = []
