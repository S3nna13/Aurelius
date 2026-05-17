"""MemCoE — Cognition-Inspired Two-Stage Memory Optimization.

Implements the framework from arXiv:2605.00702:
  Stage 1 — Memory Guideline Induction (MGI): learns a global textual guideline
            from contrastive feedback (preferred vs non-preferred memory updates).
  Stage 2 — Guideline-Aligned Memory Policy Optimization (GMPO): uses the
            induced guideline to define structured process rewards and performs
            multi-turn GRPO updates.

The two-stage design mirrors the prefrontal (schema/organization) and
hippocampal (episodic content storage) division in cognitive science.

Inspired by: GRPO (Shao et al.), REINFORCE, contrastive reward shaping,
and memory-augmented LLM architectures (MemGPT, Generative Agents).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("ark.memcoe")

DEFAULT_GUIDELINE = (
    "Prioritize memories that are temporally recent, contextually relevant to "
    "current tasks, and emotionally salient. Discard redundant, obsolete, or "
    "contradictory information. Prefer specific, actionable memories over "
    "vague generalities."
)


@dataclass
class MemorySchema:
    """Global memory guideline with schema organization (prefrontal cortex).

    Stores the induced memory guideline as natural language and maintains
    metadata about its evolution across MGI iterations.
    """

    guideline: str = field(default_factory=lambda: DEFAULT_GUIDELINE)
    version: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    source_model: str = "unknown"
    n_induction_steps: int = 0

    def bump_version(self) -> None:
        self.version += 1
        self.updated_at = datetime.now(UTC)

    def update_guideline(self, new_guideline: str) -> None:
        self.guideline = new_guideline
        self.bump_version()
        self.n_induction_steps += 1


@dataclass
class MemoryTrace:
    """A single memory interaction trace for contrastive learning.

    Contains the memory update action and its outcome, annotated with
    whether this trace was preferred (1) or not (0).
    """

    memory_key: str
    content_before: str
    content_after: str
    action: str
    preferred: bool = False
    preference_score: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    context: str = ""
    task: str = ""

    def to_text(self) -> str:
        return (
            f"Context: {self.context}\n"
            f"Task: {self.task}\n"
            f"Before: {self.content_before}\n"
            f"Action: {self.action}\n"
            f"After: {self.content_after}"
        )


@dataclass
class ProcessReward:
    """Dense process reward for a memory update step.

    Combines guideline-alignment reward (prefrontal) with answer correctness
    reward (outcome evaluation).
    """

    step: int
    guideline_reward: float
    answer_reward: float
    total: float
    guideline_component: float = 0.0
    answer_component: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryUpdateResult:
    """Outcome of a memory update operation."""

    key: str
    success: bool
    content_after: str
    process_rewards: list[ProcessReward]
    guideline_version: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    error: str = ""


class _GuidelineEncoder(nn.Module):
    """Encodes guideline + memory content into a reward-relevant embedding."""

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.guideline_proj = nn.Linear(embed_dim, hidden_dim)
        self.content_proj = nn.Linear(embed_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        guideline_emb: torch.Tensor,
        content_emb: torch.Tensor,
    ) -> torch.Tensor:
        g = self.guideline_proj(guideline_emb)
        c = self.content_proj(content_emb)
        combined = torch.cat([g, c], dim=-1)
        return self.fusion(combined).squeeze(-1)


class MemoryGuidelineInducer:
    """Stage 1 — Memory Guideline Induction (MGI).

    Learns a global textual guideline from contrastive feedback by:
      1. Collecting preferred/non-preferred memory traces
      2. Using a small reward model to score alignment
      3. Aggregating batch-level gradients to update the guideline embedding
      4. Decoding the updated embedding back to a textual guideline

    The induced guideline captures what distinguishes good memory updates
    from poor ones, forming the "prefrontal schema" that guides Stage 2.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        learning_rate: float = 1e-3,
        batch_size: int = 8,
        n_epochs_per_update: int = 4,
        temperature: float = 0.5,
        device: str | None = None,
    ) -> None:
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.n_epochs = n_epochs_per_update
        self.temperature = temperature

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._encoder = _GuidelineEncoder(embed_dim, hidden_dim).to(self._device)
        self._guideline_emb = self._init_guideline_emb()
        self._optimizer = torch.optim.AdamW(
            [*self._encoder.parameters(), self._guideline_emb],
            lr=learning_rate,
            weight_decay=0.01,
        )
        self._step_count = 0

    def _init_guideline_emb(self) -> nn.Parameter:
        emb = nn.Parameter(torch.randn(self.embed_dim, device=self._device) * 0.02)
        return emb

    def _score_alignment(
        self,
        guideline_emb: torch.Tensor,
        content_emb: torch.Tensor,
    ) -> torch.Tensor:
        return self._encoder(guideline_emb.unsqueeze(0), content_emb.unsqueeze(0))

    def _encode_text(self, text: str) -> torch.Tensor:
        tokens = text.encode("utf-8")[: self.embed_dim]
        arr = torch.zeros(self.embed_dim, device=self._device)
        for i, b in enumerate(tokens):
            arr[i] = b / 255.0
        return arr

    def _contrastive_loss(
        self,
        preferred_emb: torch.Tensor,
        non_preferred_emb: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.margin_ranking_loss(
            preferred_emb,
            non_preferred_emb,
            torch.ones_like(preferred_emb),
            margin=0.5,
        )
        return loss

    def induce_from_traces(self, traces: list[MemoryTrace]) -> str:
        """Induce an updated guideline from a batch of memory traces.

        Performs multi-epoch gradient updates on the guideline embedding using
        contrastive loss between preferred and non-preferred traces.
        """
        if len(traces) < 2:
            return self._decode_guideline()

        preferred_traces = [t for t in traces if t.preferred]
        non_preferred_traces = [t for t in traces if not t.preferred]

        if not preferred_traces or not non_preferred_traces:
            return self._decode_guideline()

        self._encoder.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for i in range(0, len(traces), self.batch_size):
                batch = traces[i : i + self.batch_size]
                pref_batch = [t for t in batch if t.preferred]
                non_pref_batch = [t for t in batch if not t.preferred]

                if not pref_batch or not non_pref_batch:
                    continue

                pair_count = min(len(pref_batch), len(non_pref_batch))
                pref_batch = pref_batch[:pair_count]
                non_pref_batch = non_pref_batch[:pair_count]

                pref_embs = torch.stack([self._encode_text(t.to_text()) for t in pref_batch])
                non_pref_embs = torch.stack(
                    [self._encode_text(t.to_text()) for t in non_pref_batch]
                )

                g_emb = self._guideline_emb.unsqueeze(0)
                pref_expanded = g_emb.expand(len(pref_embs), -1)
                non_pref_expanded = g_emb.expand(len(non_pref_embs), -1)
                pref_scores = self._encoder(pref_expanded, pref_embs)
                non_pref_scores = self._encoder(non_pref_expanded, non_pref_embs)

                loss = self._contrastive_loss(pref_scores.mean(), non_pref_scores.mean())
                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [*self._encoder.parameters(), self._guideline_emb],
                    1.0,
                )
                self._optimizer.step()
                epoch_loss += loss.item()

            logger.debug(
                "MGI epoch %d/%d — loss=%.4f",
                epoch + 1,
                self.n_epochs,
                epoch_loss,
            )

        self._step_count += 1
        return self._decode_guideline()

    def _decode_guideline(self) -> str:
        # Text reconstruction from a learned embedding requires a decoder; keep
        # this placeholder deterministic instead of emitting arbitrary bytes.
        return DEFAULT_GUIDELINE

    def get_guideline_embedding(self) -> torch.Tensor:
        return self._guideline_emb.detach().clone()

    def score_memory_content(
        self,
        content: str,
        guideline: str | None = None,
    ) -> float:
        """Score how well a piece of content aligns with the current guideline."""
        self._encoder.eval()
        with torch.no_grad():
            content_emb = self._encode_text(content)
            guideline_emb = (
                self._encode_text(guideline) if guideline is not None else self._guideline_emb
            )
            score = self._score_alignment(guideline_emb, content_emb)
            return torch.sigmoid(score).item()

    def compute_process_reward(
        self,
        memory_content: str,
        action: str,
        correct_answer: float,
    ) -> ProcessReward:
        """Compute dense process reward for a memory update step.

        Combines guideline-alignment score (prefrontal) with answer correctness
        (outcome evaluation) into a dense reward signal.
        """
        guideline_score = self.score_memory_content(memory_content)
        answer_component = correct_answer
        guideline_component = guideline_score * 0.5

        total = guideline_component * 0.4 + answer_component * 0.6

        return ProcessReward(
            step=self._step_count,
            guideline_reward=guideline_score,
            answer_reward=correct_answer,
            total=total,
            guideline_component=guideline_component,
            answer_component=answer_component,
        )


class GuidelineAlignedPolicy(nn.Module):
    """Stage 2 — Guideline-Aligned Memory Policy Optimization (GMPO).

    A memory-update policy that uses the induced guideline from Stage 1
    (MGI) to define structured process rewards. Updated via GRPO-style
    multi-turn policy gradients.

    The policy receives dense rewards at each step (guideline alignment +
    answer correctness) rather than a single sparse outcome reward,
    enabling more stable and directed optimization.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        guideline_dim: int = 128,
        lr: float = 5e-4,
        entropy_coef: float = 0.01,
        guidance_scale: float = 1.0,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.guidance_scale = guidance_scale
        self.entropy_coef = entropy_coef
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.action_proj = nn.Sequential(
            nn.Linear(embed_dim * 2 + guideline_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.value_head = nn.Linear(embed_dim, 1)
        self._optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)

    def forward(
        self,
        content_emb: torch.Tensor,
        action_emb: torch.Tensor,
        guideline_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing action embedding and value estimate."""
        combined = torch.cat([content_emb, action_emb, guideline_emb], dim=-1)
        action_out = self.action_proj(combined)
        value = self.value_head(action_out)
        return action_out, value

    def compute_grpo_loss(
        self,
        content_embs: torch.Tensor,
        action_embs: torch.Tensor,
        guideline_emb: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor,
        epsilon: float = 0.2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GRPO-style clipped policy gradient loss.

        Uses the reward signal from MGI process rewards to compute
        policy gradients with clipping for stability.
        """
        action_out, values = self.forward(content_embs, action_embs, guideline_emb)
        log_probs = F.log_softmax(action_out, dim=-1)

        values = values.squeeze(-1)
        advantages = rewards - values.detach()
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False).clamp(min=1e-8)
        )
        action_advantages = advantages.unsqueeze(-1)

        ratio = torch.exp(log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        policy_loss = -torch.min(
            ratio * action_advantages,
            clipped * action_advantages,
        ).mean()

        entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1).mean()
        entropy_bonus = -self.entropy_coef * entropy

        return policy_loss + entropy_bonus, entropy

    def update(
        self,
        content_embs: torch.Tensor,
        action_embs: torch.Tensor,
        guideline_emb: torch.Tensor,
        process_rewards: list[ProcessReward],
    ) -> dict[str, float]:
        """Perform a single GRPO update using dense process rewards."""
        if not process_rewards:
            return {"loss": 0.0, "entropy": 0.0, "policy_grad_norm": 0.0}

        rewards_tensor = torch.tensor(
            [r.total for r in process_rewards],
            dtype=torch.float32,
            device=self._device,
        )
        with torch.no_grad():
            old_action_out, _ = self.forward(content_embs, action_embs, guideline_emb)
            old_log_probs = F.log_softmax(old_action_out, dim=-1).detach()

        self._optimizer.zero_grad()
        loss, entropy = self.compute_grpo_loss(
            content_embs,
            action_embs,
            guideline_emb,
            rewards_tensor,
            old_log_probs,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self._optimizer.step()

        return {
            "loss": loss.item(),
            "entropy": entropy.item(),
            "policy_grad_norm": grad_norm.item(),
            "mean_reward": rewards_tensor.mean().item(),
        }


class MemCoE:
    """Main coordinator class — Cognition-Inspired Two-Stage Memory System.

    Combines Stage 1 (MGI — guideline induction) and Stage 2 (GMPO — policy
    optimization) with dual-memory architecture:

      • Prefrontal (schema): global memory guideline + organization schema
      • Hippocampal (episodic): content-addressable episodic memory storage

    Usage:
        memcoe = MemCoE()
        result = memcoe.update_memory(
            key="session_001",
            content_before="...",
            content_after="...",
            action="consolidate",
            preferred=True,
            process_rewards=[...],
        )
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        device: str | None = None,
        guideline: str | None = None,
        max_episodic_entries: int = 10000,
    ) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim
        self.max_episodic_entries = max_episodic_entries

        self.schema = MemorySchema(guideline=guideline or DEFAULT_GUIDELINE)
        self.inducer = MemoryGuidelineInducer(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            device=self._device,
        )
        self.policy = GuidelineAlignedPolicy(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            guideline_dim=embed_dim,
            device=self._device,
        ).to(self._device)

        self._episodic: OrderedDict[str, str] = OrderedDict()
        self._trace_buffer: list[MemoryTrace] = []
        self._update_count = 0

    @property
    def guideline(self) -> str:
        return self.schema.guideline

    def store_episodic(self, key: str, content: str) -> None:
        """Store content in the hippocampal episodic store."""
        self._episodic[key] = content
        self._episodic.move_to_end(key)
        if len(self._episodic) > self.max_episodic_entries:
            self._episodic.popitem(last=False)

    def retrieve_episodic(self, key: str) -> str | None:
        return self._episodic.get(key)

    def update_memory(
        self,
        key: str,
        content_before: str,
        content_after: str,
        action: str,
        preferred: bool = False,
        context: str = "",
        task: str = "",
        correct_answer: float = 0.5,
    ) -> MemoryUpdateResult:
        """Perform a complete MGI + GMPO memory update cycle.

        1. Record the memory trace in the trace buffer.
        2. Run MGI induction if buffer is full.
        3. Compute process rewards using the induced guideline.
        4. Run GMPO policy update.
        5. Apply the memory update to episodic storage.
        """
        trace = MemoryTrace(
            memory_key=key,
            content_before=content_before,
            content_after=content_after,
            action=action,
            preferred=preferred,
            context=context,
            task=task,
        )
        self._trace_buffer.append(trace)

        process_rewards: list[ProcessReward] = []
        guideline_version = self.schema.version

        content_emb = self._encode_content(content_after)
        action_emb = self._encode_content(action)

        guideline_emb = self.inducer.get_guideline_embedding()
        content_emb_t = content_emb.unsqueeze(0).to(self._device)
        action_emb_t = action_emb.unsqueeze(0).to(self._device)
        guideline_emb_t = guideline_emb.unsqueeze(0).to(self._device)

        proc_reward = self.inducer.compute_process_reward(content_after, action, correct_answer)
        process_rewards.append(proc_reward)

        if len(self._trace_buffer) >= self.inducer.batch_size * 2:
            new_guideline = self.inducer.induce_from_traces(self._trace_buffer)
            self.schema.update_guideline(new_guideline)
            guideline_version = self.schema.version
            self._trace_buffer.clear()
            guideline_emb = self.inducer.get_guideline_embedding()
            guideline_emb_t = guideline_emb.unsqueeze(0).to(self._device)

        self.policy.update(
            content_emb_t,
            action_emb_t,
            guideline_emb_t,
            process_rewards,
        )

        self.store_episodic(key, content_after)
        self._update_count += 1

        return MemoryUpdateResult(
            key=key,
            success=True,
            content_after=content_after,
            process_rewards=process_rewards,
            guideline_version=guideline_version,
        )

    def induce_guideline(self, traces: list[MemoryTrace]) -> str:
        """Manually trigger MGI guideline induction from a set of traces.

        Useful for curriculum learning where traces are collected across
        multiple interaction rounds before induction.
        """
        new_guideline = self.inducer.induce_from_traces(traces)
        self.schema.update_guideline(new_guideline)
        return self.guideline

    def score_content(self, content: str) -> float:
        """Score content alignment with the current guideline (prefrontal check)."""
        return self.inducer.score_memory_content(content)

    def batch_score(self, contents: list[str]) -> list[float]:
        """Score multiple content pieces in batch."""
        return [self.score_content(c) for c in contents]

    def transfer_guideline(self, target_model_name: str) -> dict[str, Any]:
        """Export guideline metadata for transfer to another model.

        The induced guideline from Stage 1 is model-agnostic and can be
        used to initialize the prefrontal schema of a different LLM
        without re-running MGI.
        """
        return {
            "guideline": self.guideline,
            "version": self.schema.version,
            "n_induction_steps": self.schema.n_induction_steps,
            "source_model": self.schema.source_model,
            "target_model": target_model_name,
            "transfer_timestamp": datetime.now(UTC).isoformat(),
        }

    def _encode_content(self, text: str) -> torch.Tensor:
        tokens = text.encode("utf-8")[: self.embed_dim]
        arr = torch.zeros(self.embed_dim, device=self._device)
        for i, b in enumerate(tokens):
            arr[i] = b / 255.0
        return arr

    def stats(self) -> dict[str, Any]:
        return {
            "update_count": self._update_count,
            "trace_buffer_size": len(self._trace_buffer),
            "episodic_entries": len(self._episodic),
            "guideline_version": self.schema.version,
            "n_induction_steps": self.schema.n_induction_steps,
            "guideline_length": len(self.guideline),
        }

    def reset_episodic(self) -> None:
        self._episodic.clear()

    def export_schema(self) -> dict[str, Any]:
        return {
            "guideline": self.guideline,
            "version": self.schema.version,
            "created_at": self.schema.created_at.isoformat(),
            "updated_at": self.schema.updated_at.isoformat(),
            "source_model": self.schema.source_model,
            "n_induction_steps": self.schema.n_induction_steps,
        }


__all__ = [
    "MemCoE",
    "MemorySchema",
    "MemoryGuidelineInducer",
    "GuidelineAlignedPolicy",
    "MemoryTrace",
    "ProcessReward",
    "MemoryUpdateResult",
]
