"""Aurelius 1.3B — GRPO reasoning training.

Group Relative Policy Optimization (GRPO) from DeepSeekMath/DeepSeek-R1.
Unlike DPO, GRPO is an online RL method: it generates multiple completions
per prompt, scores them with a verifiable reward function, and trains using
group-normalized advantages — no reference model or critic needed.

This targets mathematical and coding reasoning using verifiable rewards
(exact-match answer checking). The model learns to reason rather than just
imitate, allowing improvement beyond the quality of the training data.

References:
    - DeepSeekMath: arXiv:2402.03300
    - DAPO improvements: arXiv:2503.14476
    - TRL GRPOTrainer: https://huggingface.co/docs/trl
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GRPORunConfig:
    """Configuration for GRPO reasoning training."""

    # Model
    model_name_or_path: str = "checkpoints/aurelius-1.3b-sft/final"
    tokenizer_path: str = "tokenizers/aurelius-128k"

    # GRPO hyperparameters
    num_generations: int = 8        # G: responses sampled per prompt
    max_new_tokens: int = 512       # max tokens per generated response
    temperature: float = 0.9        # sampling temperature
    top_p: float = 0.95

    # Training
    learning_rate: float = 1e-6
    num_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    max_prompt_length: int = 512

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Precision
    bf16: bool = True

    # Checkpointing
    output_dir: str = "checkpoints/aurelius-1.3b-grpo"
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 3

    # Logging
    logging_steps: int = 5
    wandb_project: str = "aurelius-1.3b-grpo"

    # Data
    dataset_name: str = "openai/gsm8k"     # default: GSM8K math reasoning
    dataset_config: str = "main"
    max_train_samples: int | None = None
    eval_split_ratio: float = 0.05

    # Seed
    seed: int = 42


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer from a chain-of-thought response.

    Looks for patterns like "#### 42", "The answer is 42", or just a trailing number.
    Returns the answer as a stripped string, or None if not found.
    """
    # GSM8K format: answer after ####
    match = re.search(r"####\s*([+-]?\d[\d,]*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    # "The answer is X" pattern
    match = re.search(r"(?:the answer is|answer:)\s*([+-]?\d[\d,]*(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip()

    return None


def gsm8k_reward(completions: list[str], ground_truths: list[str], **kwargs) -> list[float]:
    """Reward function for GSM8K: +1.0 if answer matches, 0.0 otherwise.

    Args:
        completions: List of model-generated responses.
        ground_truths: Corresponding ground truth answers.

    Returns:
        List of scalar rewards (one per completion).
    """
    rewards = []
    for completion, truth in zip(completions, ground_truths):
        predicted = extract_answer(completion)
        correct = extract_answer(truth) if truth else truth.strip()
        if predicted is not None and predicted == correct:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward function for chain-of-thought format quality.

    Gives a small bonus (+0.2) for responses that include step-by-step
    reasoning markers, encouraging structured thinking.
    """
    rewards = []
    for completion in completions:
        score = 0.0
        # Reward for showing work
        if "step" in completion.lower() or "=" in completion:
            score += 0.1
        # Reward for explicit answer
        if "####" in completion or "the answer is" in completion.lower():
            score += 0.1
        rewards.append(score)
    return rewards


def combined_reward(
    completions: list[str],
    ground_truths: list[str],
    **kwargs,
) -> list[float]:
    """Combined reward: correctness (primary) + format quality (secondary).

    correctness_reward + 0.2 * format_reward
    """
    correctness = gsm8k_reward(completions, ground_truths)
    formatting = format_reward(completions)
    return [c + 0.2 * f for c, f in zip(correctness, formatting)]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_gsm8k(cfg: GRPORunConfig) -> tuple[Dataset, Dataset]:
    """Load GSM8K and format for GRPO training.

    Returns datasets with columns: prompt, ground_truth
    """
    logger.info("Loading GSM8K dataset...")
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split="train")
    logger.info("GSM8K train size: %d", len(ds))

    def _format(example: dict) -> dict:
        return {
            "prompt": (
                "Solve this math problem step by step.\n\n"
                f"Problem: {example['question']}\n\n"
                "Solution:"
            ),
            "ground_truth": example["answer"],
        }

    ds = ds.map(_format, remove_columns=ds.column_names, desc="Formatting GSM8K")

    if cfg.max_train_samples is not None:
        ds = ds.select(range(min(cfg.max_train_samples, len(ds))))

    ds = ds.shuffle(seed=cfg.seed)
    split = ds.train_test_split(test_size=cfg.eval_split_ratio, seed=cfg.seed)
    logger.info(
        "GRPO dataset: %d train, %d eval", len(split["train"]), len(split["test"]),
    )
    return split["train"], split["test"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_grpo(cfg: GRPORunConfig | None = None) -> None:
    """Run the full GRPO reasoning training pipeline."""
    if cfg is None:
        cfg = GRPORunConfig()

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

    # Load model
    logger.info("Loading model from %s", cfg.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )

    # LoRA
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # Dataset
    train_dataset, eval_dataset = load_gsm8k(cfg)

    # Bind ground truths for reward function
    ground_truths = train_dataset["ground_truth"]

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        # ground_truths are passed via the dataset's "ground_truth" column
        # TRL GRPO passes them as kwargs when the column is present
        truths = kwargs.get("ground_truth", ground_truths[:len(completions)])
        return combined_reward(completions, list(truths))

    # GRPO training arguments
    training_args = GRPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
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
        run_name="aurelius-grpo",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        # GRPO-specific
        num_generations=cfg.num_generations,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_prompt_length=cfg.max_prompt_length,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    logger.info(
        "Starting GRPO training (G=%d, lr=%.1e, epochs=%d)...",
        cfg.num_generations, cfg.learning_rate, cfg.num_epochs,
    )
    trainer.train()

    final_dir = f"{cfg.output_dir}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("GRPO training complete. Model saved to %s", final_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Aurelius 1.3B GRPO Reasoning Training")
    parser.add_argument("--model", type=str, default="checkpoints/aurelius-1.3b-sft/final")
    parser.add_argument("--tokenizer", type=str, default="tokenizers/aurelius-128k")
    parser.add_argument("--output-dir", type=str, default="checkpoints/aurelius-1.3b-grpo")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="openai/gsm8k")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = GRPORunConfig(
        model_name_or_path=args.model,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        num_generations=args.num_generations,
        max_train_samples=args.max_samples,
        dataset_name=args.dataset,
        seed=args.seed,
    )
    run_grpo(cfg)


if __name__ == "__main__":
    main()
