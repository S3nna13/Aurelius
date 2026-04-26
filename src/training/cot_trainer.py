"""Chain-of-thought scratchpad training (GSM8K-inspired).

Generates synthetic arithmetic reasoning problems internally (no external data),
trains on scratchpad->answer format with differential loss weighting so answer
tokens are penalized more than reasoning steps.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# -- Configuration -------------------------------------------------------------


@dataclass
class CoTConfig:
    """Configuration for chain-of-thought scratchpad training."""

    scratchpad_weight: float = 1.0  # loss weight on reasoning steps
    answer_weight: float = 5.0  # loss weight on final answer tokens
    answer_delimiter: str = "####"  # separator between scratchpad and answer
    max_seq_len: int = 512
    verification_reward: float = 1.0  # bonus applied when answer is correct
    min_answer_tokens: int = 1


# -- Data structure ------------------------------------------------------------


@dataclass
class CoTExample:
    """A single chain-of-thought example with question, scratchpad, and answer."""

    question: str
    scratchpad: str  # step-by-step reasoning
    answer: str  # final answer (e.g. "72")

    def full_text(self, delimiter: str = "####") -> str:
        return f"{self.question}\n{self.scratchpad}\n{delimiter} {self.answer}"


# -- Synthetic data generation -------------------------------------------------

_NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]
_OBJECTS = ["apples", "oranges", "books", "coins", "stickers", "marbles", "cookies"]


def _make_step_text(val: int, op: str, operand: int, result: int, obj: str) -> str:
    if op == "add":
        return f"After getting {operand} more {obj}, she has {val} + {operand} = {result} {obj}."
    else:
        return f"After giving away {operand} {obj}, she has {val} - {operand} = {result} {obj}."


def generate_arithmetic_example(n_steps: int = 3, max_val: int = 100) -> CoTExample:
    """Generate a random multi-step arithmetic word problem.

    Completely self-contained -- uses only random.randint, no external data.

    Args:
        n_steps: Number of arithmetic steps.
        max_val: Maximum value for initial quantity and operands.

    Returns:
        CoTExample with question, step-by-step scratchpad, and numeric answer.
    """
    name = random.choice(_NAMES)
    obj = random.choice(_OBJECTS)
    current = random.randint(1, max_val)

    question_parts = [f"{name} has {current} {obj}."]
    steps: list[str] = []

    for _ in range(n_steps):
        op = random.choice(["add", "sub"])
        if op == "add":
            operand = random.randint(1, max_val)
            new_val = current + operand
            question_parts.append(f"She gets {operand} more {obj}.")
            steps.append(_make_step_text(current, "add", operand, new_val, obj))
        else:
            operand = random.randint(1, max(1, current))
            new_val = current - operand
            question_parts.append(f"She gives away {operand} {obj}.")
            steps.append(_make_step_text(current, "sub", operand, new_val, obj))
        current = new_val

    question_parts.append(f"How many {obj} does she have now?")
    question = " ".join(question_parts)
    scratchpad = " ".join(steps)

    return CoTExample(question=question, scratchpad=scratchpad, answer=str(current))


def generate_arithmetic_dataset(
    n_examples: int,
    seed: int = 42,
    **kwargs,
) -> list[CoTExample]:
    """Generate n_examples with a fixed seed for reproducibility.

    Args:
        n_examples: Number of examples to generate.
        seed: Random seed.
        **kwargs: Forwarded to generate_arithmetic_example.

    Returns:
        List of CoTExample instances.
    """
    rng_state = random.getstate()
    random.seed(seed)
    examples = [generate_arithmetic_example(**kwargs) for _ in range(n_examples)]
    random.setstate(rng_state)
    return examples


# -- Loss computation ----------------------------------------------------------


def _find_subseq(seq: list[int], subseq: list[int]) -> int:
    """Return start index of the last occurrence of subseq in seq, or -1."""
    n, m = len(seq), len(subseq)
    last = -1
    for i in range(n - m + 1):
        if seq[i : i + m] == subseq:
            last = i
    return last


def build_cot_labels(
    input_ids: Tensor,
    delimiter_token_ids: list[int],
    cfg: CoTConfig,
) -> tuple[Tensor, Tensor]:
    """Build per-token labels and loss weights for a CoT sequence.

    Args:
        input_ids: 1-D token ID tensor of length T.
        delimiter_token_ids: Sequence of token IDs encoding the delimiter.
        cfg: CoTConfig with weight settings.

    Returns:
        (labels, weights) both shape (T,):
        - labels: same as input_ids with -100 on the question prefix (before delimiter).
        - weights: cfg.scratchpad_weight before delimiter, cfg.answer_weight after.
    """
    T = input_ids.shape[0]
    ids_list = input_ids.tolist()
    delim = list(delimiter_token_ids)

    delim_start = _find_subseq(ids_list, delim)

    labels = input_ids.clone()
    weights = torch.full((T,), cfg.scratchpad_weight, dtype=torch.float)

    if delim_start == -1:
        labels[:] = -100
        return labels, weights

    delim_end = delim_start + len(delim)

    # Mask question prefix (everything before the delimiter)
    labels[:delim_start] = -100

    # Answer tokens get higher weight
    weights[delim_end:] = cfg.answer_weight

    return labels, weights


def cot_loss(
    model,
    input_ids: Tensor,
    labels: Tensor,
    weights: Tensor,
) -> Tensor:
    """Compute weighted cross-entropy loss for a CoT batch.

    Runs a forward pass, computes per-token CE, multiplies by weights, and
    averages over non-ignored positions.

    Args:
        model: AureliusTransformer (forward returns (loss, logits, pkv)).
        input_ids: (B, T) token IDs.
        labels: (B, T) targets; -100 positions are ignored.
        weights: (B, T) per-token loss weights.

    Returns:
        Scalar loss tensor.
    """
    _loss, logits, _pkv = model(input_ids)
    B, T, V = logits.shape

    # Shift: logit[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = weights[:, 1:].contiguous()

    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(B, T - 1)

    mask = (shift_labels != -100).float()
    weighted = per_token_loss * shift_weights * mask

    n_active = mask.sum().clamp(min=1.0)
    return weighted.sum() / n_active


# -- CoTTrainer ----------------------------------------------------------------


class CoTTrainer:
    """Trainer for chain-of-thought scratchpad learning.

    Args:
        model: AureliusTransformer instance.
        optimizer: PyTorch optimizer (may be None for inference-only use).
        cfg: CoTConfig.
        tokenizer_encode: Callable mapping str -> list[int].
    """

    def __init__(
        self,
        model,
        optimizer,
        cfg: CoTConfig,
        tokenizer_encode: Callable[[str], list[int]],
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.tokenizer_encode = tokenizer_encode

    def _encode_delimiter(self) -> list[int]:
        return self.tokenizer_encode(self.cfg.answer_delimiter)

    def tokenize_example(self, example: CoTExample) -> tuple[Tensor, Tensor, Tensor]:
        """Tokenize one CoTExample and compute per-token labels and weights.

        Args:
            example: A CoTExample instance.

        Returns:
            (input_ids, labels, weights) each of shape (T,).
        """
        text = example.full_text(self.cfg.answer_delimiter)
        ids = self.tokenizer_encode(text)
        ids = ids[: self.cfg.max_seq_len]

        input_ids = torch.tensor(ids, dtype=torch.long)
        delim_ids = self._encode_delimiter()
        labels, weights = build_cot_labels(input_ids, delim_ids, self.cfg)
        return input_ids, labels, weights

    def train_step(self, examples: list[CoTExample]) -> dict[str, float]:
        """Perform one gradient update on a batch of examples.

        Args:
            examples: List of CoTExample instances (the batch).

        Returns:
            Dict with keys: "loss" (float), "n_examples" (int).
        """
        tokenized = [self.tokenize_example(ex) for ex in examples]

        max_len = max(t[0].shape[0] for t in tokenized)

        input_ids_list, labels_list, weights_list = [], [], []
        for ids, lbls, wts in tokenized:
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                ids = F.pad(ids, (0, pad_len), value=0)
                lbls = F.pad(lbls, (0, pad_len), value=-100)
                wts = F.pad(wts, (0, pad_len), value=0.0)
            input_ids_list.append(ids)
            labels_list.append(lbls)
            weights_list.append(wts)

        input_ids = torch.stack(input_ids_list)
        labels = torch.stack(labels_list)
        weights = torch.stack(weights_list)

        if self.optimizer is not None:
            self.optimizer.zero_grad()

        loss = cot_loss(self.model, input_ids, labels, weights)

        if self.optimizer is not None:
            loss.backward()
            self.optimizer.step()

        return {"loss": loss.item(), "n_examples": len(examples)}

    def train(
        self,
        dataset: list[CoTExample],
        n_steps: int,
        batch_size: int = 4,
    ) -> list[float]:
        """Training loop.

        Args:
            dataset: Full list of CoTExample instances.
            n_steps: Number of gradient steps.
            batch_size: Examples per step.

        Returns:
            List of per-step loss values (length n_steps).
        """
        losses: list[float] = []
        n = len(dataset)
        for step in range(n_steps):
            start = (step * batch_size) % n
            end = start + batch_size
            batch = dataset[start:end]
            if len(batch) < batch_size and n >= batch_size:
                batch = batch + dataset[: batch_size - len(batch)]
            metrics = self.train_step(batch)
            losses.append(metrics["loss"])
        return losses


# -- Answer verification -------------------------------------------------------


def extract_cot_answer(text: str, delimiter: str = "####") -> str | None:
    """Extract the answer from a CoT response.

    Args:
        text: Full generated text.
        delimiter: The answer delimiter string.

    Returns:
        Extracted answer string, or None if delimiter not found.
    """
    idx = text.rfind(delimiter)
    if idx == -1:
        return None
    after = text[idx + len(delimiter) :]
    answer = after.strip().strip(".,;:!?\"'")
    return answer if answer else None


def verify_answer_correct(predicted: str | None, gold: str) -> bool:
    """Check whether predicted matches gold answer.

    Args:
        predicted: Predicted answer string (may be None).
        gold: Ground-truth answer string.

    Returns:
        True if answers match, False otherwise.
    """
    if predicted is None:
        return False
    try:
        return float(predicted) == float(gold)
    except (ValueError, TypeError):
        pass
    return predicted.strip() == gold.strip()


def verified_cot_loss(
    model,
    examples: list[CoTExample],
    tokenizer_encode: Callable[[str], list[int]],
    tokenizer_decode: Callable[[list[int]], str],
    cfg: CoTConfig,
    correct_weight_multiplier: float = 2.0,
) -> tuple[Tensor, dict]:
    """Compute CoT loss with correctness-based weight boosting.

    For each example, greedily generates an answer, verifies correctness,
    and multiplies the loss weight by correct_weight_multiplier for correct
    examples.

    Args:
        model: AureliusTransformer.
        examples: Batch of CoTExample instances.
        tokenizer_encode: str -> list[int].
        tokenizer_decode: list[int] -> str.
        cfg: CoTConfig.
        correct_weight_multiplier: Weight multiplier for correct examples.

    Returns:
        (total_loss, info_dict) where info_dict has "n_correct" and "accuracy".
    """
    model.train(False)
    n_correct = 0
    all_input_ids: list[Tensor] = []
    all_labels: list[Tensor] = []
    all_weights: list[Tensor] = []

    delim_ids = list(tokenizer_encode(cfg.answer_delimiter))

    with torch.no_grad():
        for ex in examples:
            prefix_text = f"{ex.question}\n{ex.scratchpad}\n{cfg.answer_delimiter} "
            prefix_ids = tokenizer_encode(prefix_text)
            prefix_ids = prefix_ids[: cfg.max_seq_len - cfg.min_answer_tokens]

            generated = list(prefix_ids)
            for _ in range(32):
                inp = torch.tensor([generated], dtype=torch.long)
                _, logits, _ = model(inp)
                next_id = int(logits[0, -1].argmax())
                generated.append(next_id)
                if next_id in (0, 10):
                    break

            gen_text = tokenizer_decode(generated)
            predicted = extract_cot_answer(gen_text, cfg.answer_delimiter)
            is_correct = verify_answer_correct(predicted, ex.answer)
            if is_correct:
                n_correct += 1

            full_text = ex.full_text(cfg.answer_delimiter)
            full_ids = tokenizer_encode(full_text)[: cfg.max_seq_len]
            input_ids = torch.tensor(full_ids, dtype=torch.long)
            labels, weights = build_cot_labels(input_ids, delim_ids, cfg)

            if is_correct:
                weights = weights * correct_weight_multiplier

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_weights.append(weights)

    max_len = max(t.shape[0] for t in all_input_ids)
    padded_ids, padded_labels, padded_weights = [], [], []
    for ids, lbls, wts in zip(all_input_ids, all_labels, all_weights):
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            ids = F.pad(ids, (0, pad_len), value=0)
            lbls = F.pad(lbls, (0, pad_len), value=-100)
            wts = F.pad(wts, (0, pad_len), value=0.0)
        padded_ids.append(ids)
        padded_labels.append(lbls)
        padded_weights.append(wts)

    input_ids_batch = torch.stack(padded_ids)
    labels_batch = torch.stack(padded_labels)
    weights_batch = torch.stack(padded_weights)

    model.train(True)
    total_loss = cot_loss(model, input_ids_batch, labels_batch, weights_batch)

    accuracy = n_correct / len(examples) if examples else 0.0
    return total_loss, {"n_correct": n_correct, "accuracy": accuracy}
