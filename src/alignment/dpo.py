"""Aurelius 1.3B — Direct Preference Optimization (DPO) training.

Uses TRL DPOTrainer on UltraFeedback binarized data to align the SFT model
with human preferences. Filters pairs where chosen_score - rejected_score >= 1.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOConfig, DPOTrainer

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

    # LoRA (applied on top of SFT adapter)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

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
# Training
# ---------------------------------------------------------------------------

def run_dpo(cfg: DPORunConfig | None = None) -> None:
    """Run the full DPO alignment pipeline."""
    if cfg is None:
        cfg = DPORunConfig()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load tokenizer
    logger.info("Loading tokenizer from %s", cfg.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SFT model (may include LoRA adapters already merged)
    logger.info("Loading SFT model from %s", cfg.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )

    # Load reference model (or let DPOTrainer create an implicit one)
    ref_model = None
    if cfg.ref_model_path is not None:
        logger.info("Loading reference model from %s", cfg.ref_model_path)
        ref_model = AutoModelForCausalLM.from_pretrained(
            cfg.ref_model_path,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
        )

    # LoRA configuration for DPO adapter
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # Load dataset
    train_dataset, eval_dataset = load_ultrafeedback(cfg)

    # DPO-specific training arguments
    training_args = DPOConfig(
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
        run_name=f"aurelius-dpo-beta{cfg.beta}",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        # DPO-specific
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
    )

    # Initialize DPOTrainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Train
    logger.info(
        "Starting DPO training (beta=%.2f, lr=%.1e, epochs=%d)...",
        cfg.beta, cfg.learning_rate, cfg.num_epochs,
    )
    trainer.train()

    # Evaluate
    logger.info("Running final evaluation...")
    metrics = trainer.evaluate()
    logger.info("Final eval metrics: %s", metrics)

    # Save final model
    final_dir = f"{cfg.output_dir}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("DPO training complete. Model saved to %s", final_dir)


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

    run_dpo(cfg)


if __name__ == "__main__":
    main()
