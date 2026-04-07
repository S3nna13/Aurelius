"""Aurelius 1.3B — Direct Preference Optimization (DPO) training.

Native PyTorch implementation of DPO using UltraFeedback binarized data to
align the SFT model with human preferences. Filters pairs where
chosen_score - rejected_score >= 1.0.
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DPORunConfig:
    """Configuration for DPO alignment training."""

    # Model
    model_name_or_path: str = "checkpoints/aurelius-1.3b-sft/final"
    tokenizer_path: str = "tokenizers/aurelius-128k"
    ref_model_path: str | None = None  # None = use implicit reference (copy of model)

    # DPO hyperparameters
    beta: float = 0.1
    loss_type: str = "sigmoid"  # sigmoid (standard DPO) or hinge

    # Training
    learning_rate: float = 5e-7
    num_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    max_length: int = 1024
    max_prompt_length: int = 512

    # Precision
    bf16: bool = True

    # Checkpointing
    output_dir: str = "checkpoints/aurelius-1.3b-dpo"
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3

    # Logging
    logging_steps: int = 10
    wandb_project: str = "aurelius-1.3b-dpo"

    # Data filtering
    min_score_gap: float = 1.0  # chosen_score - rejected_score >= 1.0
    max_train_samples: int | None = None
    eval_split_ratio: float = 0.02  # 2% held out for eval

    # Seed
    seed: int = 42


# ---------------------------------------------------------------------------
# Dataset loading and filtering
# ---------------------------------------------------------------------------

def load_ultrafeedback(cfg: DPORunConfig) -> tuple[Dataset, Dataset]:
    """Load and filter UltraFeedback binarized dataset.

    Filters to pairs where the score gap between chosen and rejected
    responses is >= min_score_gap, ensuring high-quality preference signal.

    Returns:
        Tuple of (train_dataset, eval_dataset) with columns:
        prompt, chosen, rejected.
    """
    logger.info("Loading UltraFeedback binarized dataset...")
    ds = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
    )

    logger.info("Raw dataset size: %d", len(ds))

    # Filter by score gap
    def _has_sufficient_gap(example: dict) -> bool:
        chosen_score = example.get("chosen_rating", 0.0)
        rejected_score = example.get("rejected_rating", 0.0)
        return (chosen_score - rejected_score) >= cfg.min_score_gap

    ds = ds.filter(_has_sufficient_gap, desc="Filtering by score gap")
    logger.info("After score gap filter (>= %.1f): %d examples", cfg.min_score_gap, len(ds))

    # Format into DPO-compatible columns
    def _format_dpo(example: dict) -> dict[str, str]:
        prompt = example["prompt"]

        # Extract text from chosen/rejected message lists
        chosen_messages = example.get("chosen", [])
        rejected_messages = example.get("rejected", [])

        # The chosen/rejected fields are lists of {"role": ..., "content": ...}
        # We need the assistant response portion
        chosen_text = _extract_assistant_response(chosen_messages)
        rejected_text = _extract_assistant_response(rejected_messages)

        return {
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rejected_text,
        }

    ds = ds.map(_format_dpo, desc="Formatting for DPO")

    # Remove any examples where chosen == rejected
    ds = ds.filter(
        lambda x: x["chosen"].strip() != x["rejected"].strip(),
        desc="Removing identical pairs",
    )

    if cfg.max_train_samples is not None:
        ds = ds.select(range(min(cfg.max_train_samples, len(ds))))

    # Train/eval split
    ds = ds.shuffle(seed=cfg.seed)
    split = ds.train_test_split(test_size=cfg.eval_split_ratio, seed=cfg.seed)

    logger.info(
        "DPO dataset ready: %d train, %d eval",
        len(split["train"]), len(split["test"]),
    )
    return split["train"], split["test"]


def _extract_assistant_response(messages: list[dict[str, str]]) -> str:
    """Extract the assistant response from a list of conversation messages."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    # Fallback: return last message content
    if messages:
        return messages[-1].get("content", "")
    return ""


# ---------------------------------------------------------------------------
# Core DPO functions
# ---------------------------------------------------------------------------

def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sequence log probabilities for the response tokens.

    Args:
        model: Policy or reference model. Forward signature:
               (loss, logits, present_key_values) = model(input_ids)
        input_ids: Shape (B, seq_len) — full sequence (prompt + response).
        response_mask: Shape (B, seq_len) — 1 for response tokens, 0 for prompt tokens.

    Returns:
        Shape (B,) — sum of log probs over response tokens for each sequence.
    """
    _, logits, _ = model(input_ids)  # logits: (B, seq_len, vocab_size)

    # Shift: predict token t+1 from position t
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, seq_len-1, vocab_size)

    # Gather the log prob of the actual next token
    token_lp = log_probs.gather(
        2, input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)  # (B, seq_len-1)

    # Apply response mask (shifted by 1 to align with next-token predictions)
    mask = response_mask[:, 1:].float()  # (B, seq_len-1)

    return (token_lp * mask).sum(dim=-1)  # (B,)


def dpo_loss(
    policy: nn.Module,
    reference: nn.Module,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute the DPO loss (sigmoid variant).

    Args:
        policy: Trainable policy model.
        reference: Frozen reference model.
        chosen_ids: Shape (B, seq_len) — chosen sequences.
        rejected_ids: Shape (B, seq_len) — rejected sequences.
        chosen_mask: Shape (B, seq_len) — response mask for chosen.
        rejected_mask: Shape (B, seq_len) — response mask for rejected.
        beta: DPO temperature parameter.

    Returns:
        Scalar loss tensor.
    """
    pi_chosen = compute_log_probs(policy, chosen_ids, chosen_mask)
    pi_rejected = compute_log_probs(policy, rejected_ids, rejected_mask)

    with torch.no_grad():
        ref_chosen = compute_log_probs(reference, chosen_ids, chosen_mask)
        ref_rejected = compute_log_probs(reference, rejected_ids, rejected_mask)

    # DPO objective: maximise the margin between chosen and rejected log-ratio differences
    logits = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    return -F.logsigmoid(logits).mean()


# ---------------------------------------------------------------------------
# NativeDPORunner
# ---------------------------------------------------------------------------

class NativeDPORunner:
    """Native PyTorch DPO training runner."""

    def __init__(self, policy: nn.Module, cfg: DPORunConfig) -> None:
        self.policy = policy
        self.cfg = cfg
        # Deep copy for frozen reference
        self.reference = copy.deepcopy(policy)
        for p in self.reference.parameters():
            p.requires_grad_(False)

    def _build_batch(self, example: dict, tokenizer, max_len: int) -> dict[str, torch.Tensor]:
        """Tokenize a single DPO example into tensors.

        Input format: {"prompt": str, "chosen": str, "rejected": str}
        where chosen/rejected are already the assistant response text.

        Returns dict with chosen_ids, rejected_ids, chosen_mask, rejected_mask
        all padded/truncated to max_len.
        """
        prompt_text = example["prompt"]
        chosen_text = example["chosen"]
        rejected_text = example["rejected"]

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        chosen_response_ids = tokenizer.encode(chosen_text, add_special_tokens=False)
        rejected_response_ids = tokenizer.encode(rejected_text, add_special_tokens=False)

        def _build_sequence(prompt_tok, response_tok):
            full = prompt_tok + response_tok
            mask = [0] * len(prompt_tok) + [1] * len(response_tok)
            # Truncate to max_len
            if len(full) > max_len:
                full = full[:max_len]
                mask = mask[:max_len]
            # Pad to max_len
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            pad_len = max_len - len(full)
            full = full + [pad_id] * pad_len
            mask = mask + [0] * pad_len
            return full, mask

        chosen_ids, chosen_mask = _build_sequence(prompt_ids, chosen_response_ids)
        rejected_ids, rejected_mask = _build_sequence(prompt_ids, rejected_response_ids)

        return {
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "chosen_mask": torch.tensor(chosen_mask, dtype=torch.long),
            "rejected_mask": torch.tensor(rejected_mask, dtype=torch.long),
        }

    def train(self, tokenizer) -> None:
        """Run DPO training loop.

        Loads the UltraFeedback dataset, runs training for cfg.num_epochs,
        and saves the policy weights to cfg.output_dir/dpo_final.pt.
        """
        train_dataset, _ = load_ultrafeedback(self.cfg)

        optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.cfg.learning_rate,
        )

        self.policy.train()
        self.reference.eval()

        global_step = 0
        for epoch in range(self.cfg.num_epochs):
            logger.info("Epoch %d/%d", epoch + 1, self.cfg.num_epochs)

            chosen_ids_batch: list[torch.Tensor] = []
            rejected_ids_batch: list[torch.Tensor] = []
            chosen_mask_batch: list[torch.Tensor] = []
            rejected_mask_batch: list[torch.Tensor] = []

            for i, example in enumerate(train_dataset):
                tensors = self._build_batch(example, tokenizer, self.cfg.max_length)
                chosen_ids_batch.append(tensors["chosen_ids"])
                rejected_ids_batch.append(tensors["rejected_ids"])
                chosen_mask_batch.append(tensors["chosen_mask"])
                rejected_mask_batch.append(tensors["rejected_mask"])

                batch_size = self.cfg.per_device_train_batch_size
                if len(chosen_ids_batch) == batch_size:
                    chosen_ids = torch.stack(chosen_ids_batch)
                    rejected_ids = torch.stack(rejected_ids_batch)
                    chosen_mask = torch.stack(chosen_mask_batch)
                    rejected_mask = torch.stack(rejected_mask_batch)

                    loss = dpo_loss(
                        self.policy,
                        self.reference,
                        chosen_ids,
                        rejected_ids,
                        chosen_mask,
                        rejected_mask,
                        beta=self.cfg.beta,
                    )

                    loss = loss / self.cfg.gradient_accumulation_steps
                    loss.backward()

                    if (global_step + 1) % self.cfg.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                    global_step += 1

                    if global_step % self.cfg.logging_steps == 0:
                        logger.info(
                            "Step %d | loss=%.4f",
                            global_step,
                            loss.item() * self.cfg.gradient_accumulation_steps,
                        )

                    # Reset batch accumulators
                    chosen_ids_batch = []
                    rejected_ids_batch = []
                    chosen_mask_batch = []
                    rejected_mask_batch = []

        # Save final policy weights
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        save_path = os.path.join(self.cfg.output_dir, "dpo_final.pt")
        torch.save(self.policy.state_dict(), save_path)
        logger.info("DPO training complete. Policy saved to %s", save_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for DPO training."""
    import argparse

    parser = argparse.ArgumentParser(description="Aurelius 1.3B DPO Alignment")
    parser.add_argument(
        "--model", type=str, default="checkpoints/aurelius-1.3b-sft/final",
        help="Path to SFT model checkpoint",
    )
    parser.add_argument(
        "--tokenizer", type=str, default="tokenizers/aurelius-128k",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--ref-model", type=str, default=None,
        help="Path to reference model (default: implicit copy)",
    )
    parser.add_argument("--output-dir", type=str, default="checkpoints/aurelius-1.3b-dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--min-score-gap", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = DPORunConfig(
        model_name_or_path=args.model,
        tokenizer_path=args.tokenizer,
        ref_model_path=args.ref_model,
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        min_score_gap=args.min_score_gap,
        max_train_samples=args.max_samples,
        seed=args.seed,
    )

    # NativeDPORunner requires a pre-loaded model and tokenizer.
    # Use NativeDPORunner directly for programmatic usage.
    runner = NativeDPORunner(policy=None, cfg=cfg)  # type: ignore[arg-type]
    logger.warning("main() stub: provide a loaded model and tokenizer to runner.train(tokenizer)")


if __name__ == "__main__":
    main()
