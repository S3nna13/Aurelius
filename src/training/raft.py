"""RAFT: Retrieval-Augmented Fine-Tuning (Zhang et al., 2024).

Train model to use retrieved documents when relevant, and ignore them when irrelevant.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RAFTConfig:
    """Configuration for RAFT training."""

    p_oracle: float = 0.8  # probability of including the oracle document
    n_distractors: int = 3  # number of irrelevant documents to include
    max_seq_len: int = 512
    cot_training: bool = True  # include chain-of-thought reasoning in labels
    loss_only_on_answer: bool = True  # compute loss only on answer portion


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass
class RAFTExample:
    """A single RAFT training example."""

    question: str
    oracle_doc: str  # the relevant document containing the answer
    distractor_docs: list[str]  # irrelevant documents
    answer: str
    chain_of_thought: str = ""  # reasoning trace


# ---------------------------------------------------------------------------
# Prompt / target formatting
# ---------------------------------------------------------------------------


def format_raft_prompt(
    example: RAFTExample,
    include_oracle: bool,
    n_distractors: int,
    rng: random.Random,
) -> str:
    """Format a RAFT prompt with retrieved documents, question, and answer stub.

    Args:
        example: The RAFT example to format.
        include_oracle: Whether to include the oracle (relevant) document.
        n_distractors: Number of distractor documents to include.
        rng: Random number generator for document shuffling.

    Returns:
        Formatted prompt string ending with "Answer:".
    """
    k = min(n_distractors, len(example.distractor_docs))
    docs: list[str] = list(rng.sample(example.distractor_docs, k))

    if include_oracle:
        # Insert oracle at a random position among the distractors
        insert_pos = rng.randint(0, len(docs))
        docs.insert(insert_pos, example.oracle_doc)

    doc_block = "\n".join(f"[{doc}]" for doc in docs)
    return f"Documents: {doc_block}\nQuestion: {example.question}\nAnswer:"


def format_raft_target(example: RAFTExample, cot: bool) -> str:
    """Format the target (label) string for a RAFT example.

    Args:
        example: The RAFT example.
        cot: Whether to prepend chain-of-thought reasoning.

    Returns:
        Target string (CoT + answer, or just answer).
    """
    if cot and example.chain_of_thought:
        return f"{example.chain_of_thought}\n{example.answer}"
    return example.answer


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------


class RAFTDataCollator:
    """Collates RAFTExamples into (input_ids, labels) tensor pairs."""

    def __init__(
        self,
        config: RAFTConfig,
        tokenize_fn: Callable[[str], list[int]],
    ) -> None:
        self.config = config
        self.tokenize_fn = tokenize_fn

    def collate(
        self,
        examples: list[RAFTExample],
        seed: int = 42,
    ) -> list[tuple[Tensor, Tensor]]:
        """Collate examples into (input_ids, labels) pairs.

        For each example:
        - Decide whether to include the oracle document (Bernoulli(p_oracle)).
        - Format prompt + target, tokenize.
        - Build labels: -100 on prompt tokens if loss_only_on_answer.

        Args:
            examples: List of RAFT examples.
            seed: Random seed for reproducibility.

        Returns:
            List of (input_ids, labels) tensor tuples.
        """
        rng = random.Random(seed)  # noqa: S311
        results: list[tuple[Tensor, Tensor]] = []

        for example in examples:
            include_oracle = rng.random() < self.config.p_oracle
            prompt = format_raft_prompt(
                example,
                include_oracle=include_oracle,
                n_distractors=self.config.n_distractors,
                rng=rng,
            )
            target = format_raft_target(example, cot=self.config.cot_training)
            full_text = prompt + " " + target

            prompt_ids = self.tokenize_fn(prompt + " ")
            full_ids = self.tokenize_fn(full_text)

            # Truncate to max_seq_len
            full_ids = full_ids[: self.config.max_seq_len]
            prompt_len = min(len(prompt_ids), len(full_ids))

            input_ids = torch.tensor(full_ids, dtype=torch.long)

            if self.config.loss_only_on_answer:
                labels = input_ids.clone()
                labels[:prompt_len] = -100
            else:
                labels = input_ids.clone()

            results.append((input_ids, labels))

        return results


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class RAFTTrainer:
    """Trainer for RAFT fine-tuning."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: RAFTConfig,
        tokenize_fn: Callable[[str], list[int]],
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.collator = RAFTDataCollator(config, tokenize_fn)
        self.tokenize_fn = tokenize_fn

    def train_step(self, examples: list[RAFTExample]) -> dict[str, float]:
        """Run one training step on a list of examples.

        Collates examples, pads to equal length, runs forward pass,
        computes masked cross-entropy loss, and backpropagates.

        Args:
            examples: Batch of RAFT examples.

        Returns:
            Dict with keys: loss (float), n_examples (int), oracle_rate (float).
        """
        self.model.train()
        pairs = self.collator.collate(examples)

        # Determine oracle rate from included oracle docs (re-simulate with same seed)
        rng_check = random.Random(42)  # noqa: S311
        oracle_flags = [rng_check.random() < self.config.p_oracle for _ in examples]
        oracle_rate = sum(oracle_flags) / len(oracle_flags) if oracle_flags else 0.0

        # Pad all sequences to the same length
        max_len = max(ids.shape[0] for ids, _ in pairs)
        padded_ids: list[Tensor] = []
        padded_labels: list[Tensor] = []
        for input_ids, labels in pairs:
            pad_len = max_len - input_ids.shape[0]
            padded_ids.append(F.pad(input_ids, (0, pad_len), value=0))
            padded_labels.append(F.pad(labels, (0, pad_len), value=-100))

        input_ids_batch = torch.stack(padded_ids)  # (B, T)
        labels_batch = torch.stack(padded_labels)  # (B, T)

        # Forward pass — use logits for custom masked loss
        _loss, logits, _past_kv = self.model(input_ids_batch)
        # logits: (B, T, vocab_size); shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels_batch[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {
            "loss": loss.item(),
            "n_examples": len(examples),
            "oracle_rate": oracle_rate,
        }

    def evaluate(
        self,
        examples: list[RAFTExample],
        n_eval: int = 10,
    ) -> dict[str, float]:
        """Evaluate model by greedy-decoding answers and measuring output length.

        Args:
            examples: Examples to evaluate on.
            n_eval: Maximum number of examples to evaluate.

        Returns:
            Dict with key: mean_output_len (float).
        """
        self.model.eval()
        rng = random.Random(0)  # noqa: S311
        eval_examples = examples[:n_eval]
        output_lens: list[int] = []

        with torch.no_grad():
            for example in eval_examples:
                # Build prompt (always include oracle for eval, or use p_oracle)
                include_oracle = rng.random() < self.config.p_oracle
                prompt = format_raft_prompt(
                    example,
                    include_oracle=include_oracle,
                    n_distractors=self.config.n_distractors,
                    rng=rng,
                )
                prompt_ids = self.tokenize_fn(prompt + " ")
                prompt_ids = prompt_ids[: self.config.max_seq_len]

                input_ids = torch.tensor([prompt_ids], dtype=torch.long)
                generated: list[int] = []
                max_new_tokens = 32
                past_kv = None

                for _ in range(max_new_tokens):
                    _loss, logits, past_kv = self.model(input_ids, past_key_values=past_kv)
                    next_token = int(logits[0, -1, :].argmax(dim=-1).item())
                    generated.append(next_token)
                    input_ids = torch.tensor([[next_token]], dtype=torch.long)

                output_lens.append(len(generated))

        mean_len = sum(output_lens) / len(output_lens) if output_lens else 0.0
        return {"mean_output_len": mean_len}
