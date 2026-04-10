"""Constitutional AI: iterative self-critique and revision for harmlessness (Anthropic, 2022).

Also implements RLAIF self-critique loop with learned principles, principle compliance
scoring, and Constitutional AI training (Bai et al., 2022 - arXiv:2212.08073).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ConstitutionalPrinciple:
    """A single constitutional principle with paired critique and revision prompts."""

    critique_request: str
    revision_request: str
    name: str = ""


# ---------------------------------------------------------------------------
# Default harmlessness principles
# ---------------------------------------------------------------------------

HARMLESSNESS_PRINCIPLES: list[ConstitutionalPrinciple] = [
    ConstitutionalPrinciple(
        name="harmful_content",
        critique_request=(
            "Identify any harmful, dangerous, or potentially injurious content in the response."
        ),
        revision_request=(
            "Rewrite the response to remove any harmful or dangerous content, ensuring it is safe and helpful."
        ),
    ),
    ConstitutionalPrinciple(
        name="discrimination",
        critique_request=(
            "Identify any discriminatory, biased, or offensive language targeting individuals or groups in the response."
        ),
        revision_request=(
            "Rewrite the response to eliminate discriminatory or biased language, treating all people fairly and respectfully."
        ),
    ),
    ConstitutionalPrinciple(
        name="deception",
        critique_request=(
            "Identify any misleading, false, or deceptive claims made in the response."
        ),
        revision_request=(
            "Rewrite the response to be accurate and honest, correcting any misleading claims or acknowledging uncertainty where appropriate."
        ),
    ),
    ConstitutionalPrinciple(
        name="privacy",
        critique_request=(
            "Identify any content in the response that could violate personal privacy or expose sensitive information."
        ),
        revision_request=(
            "Rewrite the response to protect individual privacy and avoid disclosing or encouraging disclosure of sensitive personal information."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------

def format_critique_prompt(response: str, principle: ConstitutionalPrinciple) -> str:
    """Format a prompt asking the model to critique a response."""
    return f"Here is a response: {response}\n\n{principle.critique_request}\nCritique:"


def format_revision_prompt(
    response: str, critique: str, principle: ConstitutionalPrinciple
) -> str:
    """Format a prompt asking the model to revise a response based on a critique."""
    return (
        f"Here is a response: {response}\n\n"
        f"Critique: {critique}\n\n"
        f"{principle.revision_request}\nRevision:"
    )


# ---------------------------------------------------------------------------
# CAIStep dataclass
# ---------------------------------------------------------------------------

@dataclass
class CAIStep:
    """Record of a single critique-revision step."""

    principle: ConstitutionalPrinciple
    original: str
    critique: str
    revised: str
    step_num: int = 0


# ---------------------------------------------------------------------------
# ConstitutionalAILoop
# ---------------------------------------------------------------------------

class ConstitutionalAILoop:
    """Iterative self-critique and revision loop implementing the CAI pipeline.

    Args:
        generate_fn: A callable that maps a prompt string to a generated text string.
        principles: List of ConstitutionalPrinciple to apply.
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        principles: list[ConstitutionalPrinciple] | None = None,
    ) -> None:
        self.generate_fn = generate_fn
        self.principles = principles if principles is not None else HARMLESSNESS_PRINCIPLES

    def critique(self, response: str, principle: ConstitutionalPrinciple) -> str:
        """Generate a critique of response with respect to principle."""
        prompt = format_critique_prompt(response, principle)
        return self.generate_fn(prompt)

    def revise(self, response: str, critique: str, principle: ConstitutionalPrinciple) -> str:
        """Generate a revised response given a critique and principle."""
        prompt = format_revision_prompt(response, critique, principle)
        return self.generate_fn(prompt)

    def run_step(
        self,
        response: str,
        principle: ConstitutionalPrinciple,
        step_num: int = 0,
    ) -> CAIStep:
        """Perform one critique-revision step and return a CAIStep."""
        critique_text = self.critique(response, principle)
        revised_text = self.revise(response, critique_text, principle)
        return CAIStep(
            principle=principle,
            original=response,
            critique=critique_text,
            revised=revised_text,
            step_num=step_num,
        )

    def run(self, initial_response: str, n_revisions: int = 2) -> list[CAIStep]:
        """Cycle through principles for n_revisions total steps."""
        steps: list[CAIStep] = []
        current_response = initial_response

        for i in range(n_revisions):
            principle = self.principles[i % len(self.principles)]
            step = self.run_step(current_response, principle, step_num=i)
            steps.append(step)
            current_response = step.revised

        return steps

    def final_response(self, steps: list[CAIStep]) -> str:
        """Return the last revised response, or empty string if steps is empty."""
        return steps[-1].revised if steps else ""


# ---------------------------------------------------------------------------
# SyntheticCAIDataGenerator
# ---------------------------------------------------------------------------

class SyntheticCAIDataGenerator:
    """Generate (harmful_prompt, revised_response) pairs for SFT training."""

    def __init__(self, cai_loop: ConstitutionalAILoop) -> None:
        self.cai_loop = cai_loop

    def generate_pair(self, harmful_prompt: str, initial_response: str) -> dict:
        """Run the CAI loop and return a training pair dict."""
        steps = self.cai_loop.run(initial_response)
        revised = self.cai_loop.final_response(steps)
        return {
            "prompt": harmful_prompt,
            "original": initial_response,
            "revised": revised,
            "n_steps": len(steps),
        }

    def generate_dataset(self, prompts_and_responses: list[tuple[str, str]]) -> list[dict]:
        """Map generate_pair over all (prompt, response) tuples."""
        return [self.generate_pair(prompt, response) for prompt, response in prompts_and_responses]


# ---------------------------------------------------------------------------
# Reward heuristic
# ---------------------------------------------------------------------------

def cai_reward_score(original: str, revised: str, principle: ConstitutionalPrinciple) -> float:
    """Simple heuristic reward score based on keyword change between original and revised.

    Returns 0.0 if identical, 1.0 if completely different (no shared words).
    """
    if original == revised:
        return 0.0

    original_words = set(original.lower().split())
    revised_words = set(revised.lower().split())

    if not original_words:
        return 1.0

    removed = original_words - revised_words
    score = len(removed) / len(original_words)
    return float(min(max(score, 0.0), 1.0))


# ===========================================================================
# RLAIF / Constitutional AI training components
# ===========================================================================

# ---------------------------------------------------------------------------
# RLAIF Configuration
# ---------------------------------------------------------------------------

@dataclass
class ConstitutionalAIConfig:
    """Configuration for RLAIF Constitutional AI training loop."""

    n_principles: int = 4        # number of constitutional principles
    n_critique_rounds: int = 2   # how many rounds of critique/revision
    sft_loss_coeff: float = 1.0
    kl_coeff: float = 0.1
    max_seq_len: int = 128


# ---------------------------------------------------------------------------
# Principle dataclass (RLAIF variant with description field)
# ---------------------------------------------------------------------------

@dataclass
class Principle:
    """A constitutional principle for critiquing outputs (RLAIF variant)."""

    name: str
    description: str
    critique_prompt: str    # prompt template for critique
    revision_prompt: str    # prompt template for revision


def default_principles() -> list[Principle]:
    """Return a small set of default principles."""
    return [
        Principle("helpful", "Be helpful", "Is the response helpful?", "Make it more helpful."),
        Principle("harmless", "Avoid harm", "Does the response avoid harm?", "Remove harmful content."),
        Principle("honest", "Be honest", "Is the response honest?", "Correct any inaccuracies."),
        Principle("concise", "Be concise", "Is the response concise?", "Make it shorter."),
    ]


# ---------------------------------------------------------------------------
# score_principle_compliance
# ---------------------------------------------------------------------------

def score_principle_compliance(
    logits: Tensor,
    token_ids: Tensor,
    principle_idx: int,
    n_principles: int,
) -> Tensor:
    """Compute compliance score for a principle given logits and tokens.

    Heuristic: use log probability of the response tokens as proxy score.
    Returns scalar tensor (mean log-prob over sequence).
    """
    # Normalise to 3-D
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
        token_ids = token_ids.unsqueeze(0)

    B, T, V = logits.shape

    if T < 2:
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)
    target = token_ids[:, 1:]                               # (B, T-1)
    token_lp = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    # Slightly different scale per principle (heuristic proxy)
    scale = 1.0 - principle_idx / max(n_principles, 1) * 0.1
    return token_lp.mean() * scale


# ---------------------------------------------------------------------------
# compute_critique_loss
# ---------------------------------------------------------------------------

def compute_critique_loss(
    policy_logits: Tensor,       # (B, T, V)
    ref_logits: Tensor,          # (B, T, V) — reference model logits
    target_ids: Tensor,          # (B, T) — revised token ids
    principle_scores: Tensor,    # (B, n_principles)
    config: ConstitutionalAIConfig,
) -> tuple[Tensor, dict]:
    """Compute Constitutional AI training loss.

    Loss = sft_loss_coeff * CE(policy_logits, target_ids)
           + kl_coeff * KL(policy || ref)
           - mean(principle_scores)   [reward signal]

    Returns (loss, metrics_dict).
    metrics_dict keys: sft_loss, kl_loss, principle_reward, total_loss
    """
    B, T, V = policy_logits.shape

    # SFT cross-entropy loss
    if T >= 2:
        shift_logits = policy_logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        sft_loss = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=-100,
        )
    else:
        sft_loss = torch.tensor(0.0, dtype=policy_logits.dtype, device=policy_logits.device)

    # KL divergence: KL(policy || ref)
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    kl_loss = F.kl_div(
        ref_log_probs,
        policy_log_probs.detach().exp(),
        reduction="batchmean",
        log_target=False,
    ).clamp(min=0.0)

    # Principle reward
    principle_reward = principle_scores.mean()

    # Total loss
    total_loss = (
        config.sft_loss_coeff * sft_loss
        + config.kl_coeff * kl_loss
        - principle_reward
    )

    metrics: dict = {
        "sft_loss": sft_loss.detach().item(),
        "kl_loss": kl_loss.detach().item(),
        "principle_reward": principle_reward.detach().item(),
        "total_loss": total_loss.detach().item(),
    }

    return total_loss, metrics


# ---------------------------------------------------------------------------
# SelfCritiqueBuffer
# ---------------------------------------------------------------------------

class SelfCritiqueBuffer:
    """Stores (prompt, response, revised_response, principle_scores) tuples."""

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._prompts: list[Tensor] = []
        self._responses: list[Tensor] = []
        self._revised: list[Tensor] = []
        self._scores: list[Tensor] = []

    def add(
        self,
        prompt_ids: Tensor,
        response_ids: Tensor,
        revised_ids: Tensor,
        scores: Tensor,
    ) -> None:
        """Add a (prompt, response, revised, scores) tuple to the buffer."""
        if len(self._prompts) >= self.max_size:
            self._prompts.pop(0)
            self._responses.pop(0)
            self._revised.pop(0)
            self._scores.pop(0)

        self._prompts.append(prompt_ids.detach().cpu())
        self._responses.append(response_ids.detach().cpu())
        self._revised.append(revised_ids.detach().cpu())
        self._scores.append(scores.detach().cpu())

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        """Sample a random batch. Returns None if buffer too small."""
        if len(self._prompts) < batch_size:
            return None

        indices = torch.randperm(len(self._prompts))[:batch_size].tolist()
        prompts = torch.stack([self._prompts[i] for i in indices])
        responses = torch.stack([self._responses[i] for i in indices])
        revised = torch.stack([self._revised[i] for i in indices])
        scores = torch.stack([self._scores[i] for i in indices])
        return prompts, responses, revised, scores

    def __len__(self) -> int:
        return len(self._prompts)


# ---------------------------------------------------------------------------
# ConstitutionalAITrainer
# ---------------------------------------------------------------------------

class ConstitutionalAITrainer:
    """Trains a model using Constitutional AI self-critique."""

    def __init__(
        self,
        policy: nn.Module,
        ref_model: nn.Module,
        config: ConstitutionalAIConfig,
        optimizer: torch.optim.Optimizer,
        principles: list[Principle] | None = None,
    ) -> None:
        self.policy = policy
        self.ref_model = ref_model
        self.config = config
        self.optimizer = optimizer
        self.principles = principles if principles is not None else default_principles()

        # Freeze reference model
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    def critique_and_revise(
        self,
        prompt_ids: Tensor,
        response_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply critique rounds and return (revised_ids, principle_scores).

        Runs the policy model on the response to compute principle scores from logits.
        Returns (revised_ids same shape as response_ids, scores (B, n_principles)).
        """
        self.policy.eval()

        with torch.no_grad():
            _, logits, _ = self.policy(response_ids)

        n_principles = self.config.n_principles
        B = response_ids.shape[0]
        scores = torch.zeros(B, n_principles, dtype=logits.dtype, device=logits.device)

        for i in range(n_principles):
            score = score_principle_compliance(logits, response_ids, i, n_principles)
            scores[:, i] = score

        # revised_ids = response_ids (simplified; scores are the key signal)
        revised_ids = response_ids.clone()

        self.policy.train()
        return revised_ids, scores

    def train_step(self, prompt_ids: Tensor, response_ids: Tensor) -> dict:
        """Full CAI training step:

        1. critique_and_revise to get revised responses + principle scores
        2. compute_critique_loss
        3. backward + optimizer.step()

        Returns metrics dict with keys: sft_loss, kl_loss, principle_reward, total_loss
        """
        revised_ids, principle_scores = self.critique_and_revise(prompt_ids, response_ids)

        self.policy.train()
        _, policy_logits, _ = self.policy(revised_ids)

        with torch.no_grad():
            _, ref_logits, _ = self.ref_model(revised_ids)

        loss, metrics = compute_critique_loss(
            policy_logits,
            ref_logits,
            revised_ids,
            principle_scores,
            self.config,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return metrics
