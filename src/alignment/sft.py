"""Aurelius 1.3B — Supervised Fine-Tuning (SFT) with TRL + Unsloth.

Trains the base model on instruction-following data using LoRA adapters.
Datasets: OASST2 (filtered by score > 0) and Dolly-15k.
Format: ChatML with <|system|>, <|user|>, <|assistant|>, <|end|> tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

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

    # LoRA
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

    # Unsloth
    use_unsloth: bool = True

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
    """Format a single example into ChatML for the SFTTrainer."""
    return format_chatml(
        system=DEFAULT_SYSTEM_PROMPT,
        user=example["instruction"],
        assistant=example["response"],
    )


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

def build_lora_config(cfg: SFTConfig) -> LoraConfig:
    """Build PEFT LoRA configuration."""
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_sft(cfg: SFTConfig | None = None) -> None:
    """Run the full SFT pipeline."""
    if cfg is None:
        cfg = SFTConfig()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load model and tokenizer
    logger.info("Loading model from %s", cfg.model_name_or_path)

    if cfg.use_unsloth:
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=cfg.model_name_or_path,
                max_seq_length=cfg.max_seq_length,
                dtype=None,  # auto-detect
                load_in_4bit=False,
            )
            # Apply LoRA via Unsloth's optimized path
            model = FastLanguageModel.get_peft_model(
                model,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=list(cfg.lora_target_modules),
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
            logger.info("Loaded model with Unsloth optimizations")
        except ImportError:
            logger.warning(
                "Unsloth not available, falling back to standard HuggingFace loading"
            )
            cfg.use_unsloth = False

    if not cfg.use_unsloth:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
        )

    # Ensure ChatML special tokens are in the tokenizer
    special_tokens = [CHATML_SYSTEM, CHATML_USER, CHATML_ASSISTANT, CHATML_END]
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in special_tokens if t not in existing_tokens]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Added %d ChatML special tokens to tokenizer", len(new_tokens))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset
    dataset = build_sft_dataset(cfg)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        report_to="wandb",
        run_name=f"aurelius-sft-r{cfg.lora_r}",
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Build LoRA config (used only for non-Unsloth path)
    peft_config = None if cfg.use_unsloth else build_lora_config(cfg)

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
        peft_config=peft_config,
        max_seq_length=cfg.max_seq_length,
        tokenizer=tokenizer,
        packing=True,  # pack multiple short examples into one sequence
    )

    # Train
    logger.info("Starting SFT training...")
    trainer.train()

    # Save final model
    final_dir = f"{cfg.output_dir}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("SFT training complete. Model saved to %s", final_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for SFT training."""
    import argparse

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
    parser.add_argument("--no-unsloth", action="store_true", help="Disable Unsloth")
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
        use_unsloth=not args.no_unsloth,
        max_train_samples=args.max_samples,
        seed=args.seed,
    )

    run_sft(cfg)


if __name__ == "__main__":
    main()
