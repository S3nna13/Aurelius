"""Aurelius 1.3B — Supervised Fine-Tuning (SFT) with native PyTorch + DoRA.

Trains the base model on instruction-following data using DoRA adapters.
Datasets: OASST2 (filtered by score > 0) and Dolly-15k.
Format: ChatML with <|system|>, <|user|>, <|assistant|>, <|end|> tokens.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from datasets import Dataset, concatenate_datasets, load_dataset

from src.alignment.dora import apply_dora_to_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ChatML formatting
# ---------------------------------------------------------------------------

CHATML_SYSTEM = "<|system|>"
CHATML_USER = "<|user|>"
CHATML_ASSISTANT = "<|assistant|>"
CHATML_END = "<|end|>"

DEFAULT_SYSTEM_PROMPT = (
    "You are Aurelius, a helpful, harmless, and honest AI assistant. "
    "Respond thoughtfully and accurately."
)


def format_chatml(
    system: str,
    user: str,
    assistant: str,
) -> str:
    """Format a single conversation turn into ChatML."""
    return (
        f"{CHATML_SYSTEM}\n{system}{CHATML_END}\n"
        f"{CHATML_USER}\n{user}{CHATML_END}\n"
        f"{CHATML_ASSISTANT}\n{assistant}{CHATML_END}"
    )


# ---------------------------------------------------------------------------
# SFT configuration
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning."""

    # Model
    model_name_or_path: str = "checkpoints/aurelius-1.3b/latest"
    tokenizer_path: str = "tokenizers/aurelius-128k"

    # DoRA / LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Training
    learning_rate: float = 2e-5
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048

    # Precision
    bf16: bool = True

    # Checkpointing
    output_dir: str = "checkpoints/aurelius-1.3b-sft"
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3

    # Logging
    logging_steps: int = 10
    wandb_project: str = "aurelius-1.3b-sft"

    # Data
    oasst2_min_score: float = 0.0  # filter: score > 0
    max_train_samples: int | None = None  # None = use all

    # Seed
    seed: int = 42


# ---------------------------------------------------------------------------
# Dataset loading and processing
# ---------------------------------------------------------------------------

def _load_oasst2(min_score: float) -> Dataset:
    """Load OASST2 dataset, filtering to assistant replies with score > min_score.

    Extracts (instruction, response) pairs from the conversation tree.
    """
    logger.info("Loading OASST2 dataset (min_score > %.1f)...", min_score)
    ds = load_dataset("OpenAssistant/oasst2", split="train")

    # OASST2 has a tree structure. We extract prompter->assistant pairs.
    # Group messages by parent_id to reconstruct turns.
    messages_by_id: dict[str | None, dict[str, Any]] = {}
    for row in ds:
        messages_by_id[row["message_id"]] = row

    pairs: list[dict[str, str]] = []
    for msg in ds:
        # Only take assistant messages with positive scores
        if msg["role"] != "assistant":
            continue
        if msg["rank"] is not None and msg["rank"] > 0:
            # rank 0 is the best; skip lower-ranked alternatives
            continue
        # Check score threshold: use labels if available
        labels = msg.get("labels", {})
        if labels:
            quality_score = labels.get("quality", 0.0)
            if quality_score <= min_score:
                continue

        # Find the parent (prompter) message
        parent = messages_by_id.get(msg["parent_id"])
        if parent is None or parent["role"] != "prompter":
            continue

        pairs.append({
            "instruction": parent["text"],
            "response": msg["text"],
        })

    logger.info("OASST2: extracted %d instruction-response pairs", len(pairs))
    return Dataset.from_list(pairs)


def _load_dolly() -> Dataset:
    """Load Databricks Dolly-15k dataset."""
    logger.info("Loading Dolly-15k dataset...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    pairs: list[dict[str, str]] = []
    for row in ds:
        instruction = row["instruction"]
        context = row.get("context", "")
        if context:
            instruction = f"{instruction}\n\nContext: {context}"
        pairs.append({
            "instruction": instruction,
            "response": row["response"],
        })

    logger.info("Dolly-15k: loaded %d instruction-response pairs", len(pairs))
    return Dataset.from_list(pairs)


def build_sft_dataset(cfg: SFTConfig) -> Dataset:
    """Build combined SFT dataset from OASST2 + Dolly-15k."""
    oasst2 = _load_oasst2(min_score=cfg.oasst2_min_score)
    dolly = _load_dolly()

    combined = concatenate_datasets([oasst2, dolly])
    combined = combined.shuffle(seed=cfg.seed)

    if cfg.max_train_samples is not None:
        combined = combined.select(range(min(cfg.max_train_samples, len(combined))))

    logger.info("Combined SFT dataset: %d examples", len(combined))
    return combined


def formatting_func(example: dict[str, str]) -> str:
    """Format a single example into ChatML."""
    return format_chatml(
        system=DEFAULT_SYSTEM_PROMPT,
        user=example["instruction"],
        assistant=example["response"],
    )


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_for_sft(text: str, tokenizer, max_len: int) -> dict[str, torch.Tensor]:
    """Tokenize a ChatML-formatted string with prompt masking.

    Tokens before the <|assistant|> boundary get labels=-100 (masked).
    Tokens from <|assistant|> onward get labels=input_ids (trained on).

    Args:
        text: Full ChatML-formatted conversation string.
        tokenizer: Tokenizer with an .encode() method.
        max_len: Maximum sequence length; output is truncated to this.

    Returns:
        Dict with "input_ids" and "labels", each a LongTensor of shape [max_len].
    """
    # Encode the full text
    token_ids = tokenizer.encode(text)
    token_ids = token_ids[:max_len]

    # Pad if shorter than max_len
    pad_len = max_len - len(token_ids)
    input_ids = torch.tensor(token_ids, dtype=torch.long)
    if pad_len > 0:
        input_ids = torch.cat([
            input_ids,
            torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long),
        ])

    # Find the split point: encode up to and including <|assistant|>
    assistant_marker = CHATML_ASSISTANT  # "<|assistant|>"
    split_idx = text.find(assistant_marker)
    if split_idx == -1:
        # No assistant marker: mask everything
        prompt_len = len(token_ids)
    else:
        # Include the marker itself in the prompt mask
        prompt_text = text[: split_idx + len(assistant_marker)]
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_len = min(len(prompt_tokens), max_len)

    # Build labels: -100 for prompt, copy input_ids for response
    labels = input_ids.clone()
    labels[:prompt_len] = -100

    return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_target_modules(model: nn.Module, short_names: tuple[str, ...]) -> list[str]:
    """Return full dotted module paths whose last component is in short_names.

    Args:
        model: The model to inspect.
        short_names: Short module names to match (e.g. ("q_proj", "v_proj")).

    Returns:
        List of full dotted paths such as ["layers.0.attn.q_proj", ...].
    """
    paths: list[str] = []
    for name, _ in model.named_modules():
        if not name:
            continue
        last_component = name.split(".")[-1]
        if last_component in short_names:
            paths.append(name)
    return paths


def _cosine_warmup(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Apply linear warmup then cosine decay to the optimizer's learning rate.

    Args:
        optimizer: The optimizer whose param groups will be updated.
        step: Current training step (0-indexed).
        total_steps: Total number of training steps.
        warmup_steps: Number of warmup steps.
        min_lr_ratio: Fraction of base_lr to decay to at end of training.

    Returns:
        The current learning rate as a float.
    """
    for group in optimizer.param_groups:
        base_lr: float = group.get("initial_lr", group["lr"])
        # Store initial lr on first call
        if "initial_lr" not in group:
            group["initial_lr"] = group["lr"]
            base_lr = group["lr"]

        if step < warmup_steps:
            # Linear warmup from 0 to base_lr
            lr = base_lr * (step / max(warmup_steps, 1))
        else:
            # Cosine decay from base_lr to min_lr_ratio * base_lr
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor)

        group["lr"] = lr

    return lr  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Native SFT Runner
# ---------------------------------------------------------------------------

class NativeSFTRunner:
    """Runs SFT training using native PyTorch with DoRA adapters."""

    def __init__(self, model: nn.Module, tokenizer, cfg: SFTConfig) -> None:
        """Initialize runner, applying DoRA to the model.

        Args:
            model: The base AureliusTransformer (or any nn.Module).
            tokenizer: Tokenizer with .encode() and .pad_token_id.
            cfg: SFTConfig controlling training hyperparameters.
        """
        self.tokenizer = tokenizer
        self.cfg = cfg

        # Resolve full module paths for DoRA target modules
        target_paths = _resolve_target_modules(model, cfg.lora_target_modules)
        logger.info("Applying DoRA to %d modules: %s", len(target_paths), target_paths)

        # Apply DoRA in-place; stores replaced modules
        apply_dora_to_model(
            model,
            target_modules=target_paths,
            rank=cfg.lora_r,
            alpha=float(cfg.lora_alpha),
        )

        self.model = model

    def prepare_dataset(self) -> list[dict]:
        """Load and tokenize the SFT dataset.

        Returns:
            List of {"input_ids": Tensor, "labels": Tensor} dicts.
        """
        raw_dataset = build_sft_dataset(self.cfg)
        tokenized: list[dict] = []
        for example in raw_dataset:
            text = formatting_func(example)
            item = tokenize_for_sft(text, self.tokenizer, self.cfg.max_seq_length)
            tokenized.append(item)
        logger.info("Tokenized %d examples", len(tokenized))
        return tokenized

    def train(self) -> None:
        """Run the full SFT training loop and save the final checkpoint."""
        tokenized = self.prepare_dataset()

        # Stack into tensors for DataLoader
        all_input_ids = torch.stack([d["input_ids"] for d in tokenized])  # (N, max_len)
        all_labels = torch.stack([d["labels"] for d in tokenized])        # (N, max_len)

        dataset = TensorDataset(all_input_ids, all_labels)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.per_device_train_batch_size,
            shuffle=True,
        )

        # Only optimize parameters that require gradients (DoRA params)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(
            "Trainable DoRA parameters: %d tensors, %.2fM params",
            len(trainable_params),
            sum(p.numel() for p in trainable_params) / 1e6,
        )

        optimizer = torch.optim.AdamW(trainable_params, lr=self.cfg.learning_rate)

        steps_per_epoch = len(loader)
        total_steps = steps_per_epoch * self.cfg.num_epochs
        warmup_steps = max(1, int(total_steps * self.cfg.warmup_ratio))

        global_step = 0
        for epoch in range(self.cfg.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_input_ids, batch_labels in loader:
                loss, _, _ = self.model(
                    input_ids=batch_input_ids,
                    labels=batch_labels,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                lr = _cosine_warmup(
                    optimizer,
                    step=global_step,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                )

                epoch_loss += loss.item()
                global_step += 1

                if global_step % self.cfg.logging_steps == 0:
                    logger.info(
                        "epoch=%d step=%d loss=%.4f lr=%.2e",
                        epoch + 1, global_step, loss.item(), lr,
                    )

            avg_loss = epoch_loss / max(steps_per_epoch, 1)
            logger.info("Epoch %d complete. avg_loss=%.4f", epoch + 1, avg_loss)

        # Save final checkpoint
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        ckpt_path = os.path.join(self.cfg.output_dir, "sft_final.pt")
        torch.save(self.model.state_dict(), ckpt_path)
        logger.info("Saved SFT checkpoint to %s", ckpt_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for SFT training."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Aurelius 1.3B SFT Training")
    parser.add_argument(
        "--model", type=str, default="checkpoints/aurelius-1.3b/latest",
        help="Path to base model checkpoint",
    )
    parser.add_argument(
        "--tokenizer", type=str, default="tokenizers/aurelius-128k",
        help="Path to tokenizer",
    )
    parser.add_argument("--output-dir", type=str, default="checkpoints/aurelius-1.3b-sft")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = SFTConfig(
        model_name_or_path=args.model,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        per_device_train_batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        max_train_samples=args.max_samples,
        seed=args.seed,
    )

    # Model and tokenizer must be provided externally for native training.
    # This CLI is a placeholder; integrate with your model loading code.
    logger.info("SFT config: %s", cfg)
    logger.info(
        "To run training, instantiate NativeSFTRunner(model, tokenizer, cfg).train()"
    )


if __name__ == "__main__":
    main()
