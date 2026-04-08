"""Autonomous self-improvement loop for AureliusTransformer.

The model iteratively:
1. Generates synthetic instruction-response pairs via Magpie
2. Scores them with a reward model
3. Keeps the top-scoring fraction
4. Fine-tunes itself on the filtered data
5. Evaluates improvement and logs results

This loop is the core of the autonomous continuous improvement system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("aurelius.self_improve")


@dataclass
class SelfImproveConfig:
    n_synthetic_examples: int = 100   # examples to generate per round
    top_fraction: float = 0.7         # keep top 70% by reward score
    min_examples_to_train: int = 10   # skip round if fewer pass filter
    sft_steps: int = 50               # gradient steps per round
    sft_lr: float = 1e-5
    max_rounds: int = 5               # number of improvement rounds
    eval_interval_rounds: int = 1     # evaluate every N rounds
    reward_threshold: float = 0.0     # only keep examples with reward > threshold
    max_new_tokens: int = 64          # generation length for synthetic data


@dataclass
class RoundResult:
    """Results from one self-improvement round."""
    round_idx: int
    n_generated: int
    n_kept: int              # after reward filtering
    mean_reward_kept: float  # mean reward of kept examples
    sft_loss: float | None   # training loss (None if skipped)
    eval_perplexity: float | None  # perplexity after this round
    improved: bool | None    # vs previous round (None if no prev eval)


@dataclass
class SelfImproveReport:
    """Full report across all rounds."""
    config: SelfImproveConfig
    rounds: list[RoundResult] = field(default_factory=list)

    @property
    def best_perplexity(self) -> float | None:
        ppls = [r.eval_perplexity for r in self.rounds if r.eval_perplexity is not None]
        return min(ppls) if ppls else None

    @property
    def n_rounds_improved(self) -> int:
        return sum(1 for r in self.rounds if r.improved)


class SelfImprover:
    """Autonomous self-improvement engine.

    Args:
        model: AureliusTransformer to improve
        reward_model: Trained RewardModel for scoring
        cfg: Self-improvement configuration
        eval_loader: Optional held-out DataLoader for perplexity measurement
    """

    def __init__(
        self,
        model: nn.Module,
        reward_model: nn.Module,
        cfg: SelfImproveConfig | None = None,
        eval_loader: DataLoader | None = None,
    ) -> None:
        self.model = model
        self.reward_model = reward_model
        self.cfg = cfg or SelfImproveConfig()
        self.eval_loader = eval_loader
        self._prev_perplexity: float | None = None

    def generate_synthetic_batch(
        self,
        n: int,
        prompt_template: torch.Tensor,  # (S_prompt,) token ids to use as prefix
    ) -> list[torch.Tensor]:
        """Generate n synthetic responses using the model.

        Uses model.generate() with temperature=1.0, top_p=0.9.
        Returns list of n tensors, each (S_resp,).
        """
        self.model.eval()
        results: list[torch.Tensor] = []
        # prompt_template is (S_prompt,) -- add batch dim for generate
        prompt_ids = prompt_template.unsqueeze(0)  # (1, S_prompt)
        prompt_len = prompt_ids.shape[1]

        for _ in range(n):
            with torch.no_grad():
                generated = self.model.generate(
                    prompt_ids,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=1.0,
                    top_p=0.9,
                )
            # generated: (1, prompt_len + new_tokens)
            # Extract only the response tokens (after the prompt)
            response = generated[0, prompt_len:]  # (S_resp,)
            results.append(response)

        return results

    def score_with_reward_model(
        self,
        prompts: list[torch.Tensor],    # list of (S_prompt,)
        responses: list[torch.Tensor],  # list of (S_resp,)
    ) -> torch.Tensor:
        """Score each (prompt, response) pair with the reward model.

        Concatenates prompt + response, runs through reward model.
        Returns (N,) tensor of reward scores.
        """
        self.reward_model.eval()
        scores: list[float] = []

        with torch.no_grad():
            for prompt, response in zip(prompts, responses):
                # Concatenate prompt + response into one sequence
                seq = torch.cat([prompt, response], dim=0)  # (S_prompt + S_resp,)
                # Clamp to model's max_seq_len if needed
                backbone = getattr(self.reward_model, "backbone", self.reward_model)
                backbone_cfg = getattr(backbone, "config", None)
                if backbone_cfg is not None:
                    max_seq_len = backbone_cfg.max_seq_len
                    if seq.shape[0] > max_seq_len:
                        seq = seq[-max_seq_len:]

                input_ids = seq.unsqueeze(0)  # (1, S)
                reward_scores = self.reward_model(input_ids)  # (1,)
                scores.append(reward_scores[0].item())

        return torch.tensor(scores, dtype=torch.float32)

    def filter_by_reward(
        self,
        examples: list[tuple[torch.Tensor, torch.Tensor]],  # list of (input_ids, labels)
        rewards: torch.Tensor,   # (N,)
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], float]:
        """Keep top_fraction examples by reward score.

        Also filters out examples with reward < reward_threshold.
        Returns (filtered_examples, mean_reward_of_kept).
        """
        n = len(examples)
        if n == 0:
            return [], 0.0

        # Apply threshold filter mask
        above_threshold = rewards >= self.cfg.reward_threshold

        # Number to keep based on top_fraction
        n_keep = max(1, int(n * self.cfg.top_fraction))

        # Sort indices by reward descending
        sorted_indices = torch.argsort(rewards, descending=True)

        kept_examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        kept_rewards: list[float] = []

        for idx in sorted_indices:
            if len(kept_examples) >= n_keep:
                break
            if above_threshold[idx]:
                kept_examples.append(examples[idx.item()])
                kept_rewards.append(rewards[idx.item()].item())

        mean_reward = float(sum(kept_rewards) / len(kept_rewards)) if kept_rewards else 0.0
        return kept_examples, mean_reward

    def sft_step(
        self,
        examples: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> float:
        """Run SFT fine-tuning steps on the filtered examples.

        Creates a DataLoader, runs sft_steps gradient steps.
        Returns final training loss.
        """
        self.model.train()

        # Pad all sequences to the same length
        max_len = max(ids.shape[0] for ids, _ in examples)

        input_id_list: list[torch.Tensor] = []
        label_list: list[torch.Tensor] = []

        for input_ids, labels in examples:
            pad_len = max_len - input_ids.shape[0]
            if pad_len > 0:
                input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
            input_id_list.append(input_ids)
            label_list.append(labels)

        all_input_ids = torch.stack(input_id_list)  # (N, max_len)
        all_labels = torch.stack(label_list)         # (N, max_len)

        dataset = TensorDataset(all_input_ids, all_labels)
        batch_size = max(1, len(examples) // 4 + 1)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.sft_lr)

        final_loss = 0.0
        loader_iter = iter(loader)

        for _ in range(self.cfg.sft_steps):
            try:
                batch_ids, batch_labels = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch_ids, batch_labels = next(loader_iter)

            optimizer.zero_grad()
            loss, _, _ = self.model(input_ids=batch_ids, labels=batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            final_loss = loss.item()

        logger.debug("SFT completed %d steps, final_loss=%.4f", self.cfg.sft_steps, final_loss)
        return final_loss

    def check_perplexity(self) -> float | None:
        """Compute perplexity on eval_loader. Returns None if no eval_loader."""
        if self.eval_loader is None:
            return None
        from src.eval.perplexity import perplexity_on_dataset
        result = perplexity_on_dataset(self.model, self.eval_loader)
        return result.perplexity

    def run_round(
        self,
        round_idx: int,
        prompt_template: torch.Tensor,  # (S_prompt,) token ids
    ) -> RoundResult:
        """Execute one self-improvement round."""
        cfg = self.cfg

        logger.info("Round %d: generating %d synthetic examples...", round_idx, cfg.n_synthetic_examples)

        # Step 1: Generate synthetic responses
        responses = self.generate_synthetic_batch(cfg.n_synthetic_examples, prompt_template)
        n_generated = len(responses)

        # Build prompts list (same prompt template repeated)
        prompts = [prompt_template for _ in range(n_generated)]

        # Step 2: Score with reward model
        rewards = self.score_with_reward_model(prompts, responses)

        # Step 3: Prepare (input_ids, labels) pairs
        raw_examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for prompt, response in zip(prompts, responses):
            full_seq = torch.cat([prompt, response], dim=0)
            labels = full_seq.clone()
            labels[:len(prompt)] = -100
            raw_examples.append((full_seq, labels))

        # Step 4: Filter by reward
        filtered_examples, mean_reward_kept = self.filter_by_reward(raw_examples, rewards)
        n_kept = len(filtered_examples)

        logger.info(
            "Round %d: kept %d/%d examples (mean_reward=%.4f)",
            round_idx, n_kept, n_generated, mean_reward_kept,
        )

        # Step 5: SFT fine-tuning
        sft_loss: float | None = None
        if n_kept >= cfg.min_examples_to_train:
            sft_loss = self.sft_step(filtered_examples)
            logger.info("Round %d: SFT loss=%.4f", round_idx, sft_loss)
        else:
            logger.info(
                "Round %d: skipping SFT (only %d examples, need %d)",
                round_idx, n_kept, cfg.min_examples_to_train,
            )

        # Step 6: Check perplexity
        ppl: float | None = None
        improved: bool | None = None

        if round_idx % cfg.eval_interval_rounds == 0:
            ppl = self.check_perplexity()
            if ppl is not None:
                if self._prev_perplexity is not None:
                    improved = ppl < self._prev_perplexity
                self._prev_perplexity = ppl
                logger.info(
                    "Round %d: perplexity=%.4f improved=%s",
                    round_idx, ppl, improved,
                )

        return RoundResult(
            round_idx=round_idx,
            n_generated=n_generated,
            n_kept=n_kept,
            mean_reward_kept=mean_reward_kept,
            sft_loss=sft_loss,
            eval_perplexity=ppl,
            improved=improved,
        )

    def run(
        self,
        prompt_template: torch.Tensor,  # (S_prompt,) token ids
    ) -> SelfImproveReport:
        """Run the full self-improvement loop for max_rounds rounds."""
        cfg = self.cfg
        report = SelfImproveReport(config=cfg)

        logger.info(
            "Starting self-improvement loop: %d rounds, %d examples/round",
            cfg.max_rounds, cfg.n_synthetic_examples,
        )

        for round_idx in range(cfg.max_rounds):
            result = self.run_round(round_idx, prompt_template)
            report.rounds.append(result)
            logger.info(
                "Round %d complete: n_kept=%d sft_loss=%s ppl=%s improved=%s",
                round_idx,
                result.n_kept,
                f"{result.sft_loss:.4f}" if result.sft_loss is not None else "skipped",
                f"{result.eval_perplexity:.4f}" if result.eval_perplexity is not None else "N/A",
                result.improved,
            )

        logger.info(
            "Self-improvement complete. best_ppl=%s rounds_improved=%d",
            report.best_perplexity,
            report.n_rounds_improved,
        )
        return report
