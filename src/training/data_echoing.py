"""Data echoing utilities for repeating informative examples during training."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class EchoExample:
    example_id: str
    loss: float
    difficulty: float
    last_seen_step: int
    echo_count: int = 0


def echo_score(
    loss: torch.Tensor,
    difficulty: torch.Tensor,
    age: torch.Tensor,
    loss_weight: float = 1.0,
    difficulty_weight: float = 0.5,
    recency_weight: float = 0.1,
) -> torch.Tensor:
    """Score examples for replay based on loss, difficulty, and recency."""
    if loss.shape != difficulty.shape or loss.shape != age.shape:
        raise ValueError("loss, difficulty, and age must share the same shape")
    recency_bonus = torch.log1p(age.clamp_min(0))
    return loss_weight * loss + difficulty_weight * difficulty + recency_weight * recency_bonus


def echo_probabilities(
    examples: list[EchoExample],
    current_step: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Turn example scores into replay probabilities."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if not examples:
        return torch.empty(0)
    losses = torch.tensor([example.loss for example in examples], dtype=torch.float32)
    difficulty = torch.tensor([example.difficulty for example in examples], dtype=torch.float32)
    age = torch.tensor(
        [current_step - example.last_seen_step for example in examples], dtype=torch.float32
    )
    scores = echo_score(losses, difficulty, age)
    return torch.softmax(scores / temperature, dim=0)


def select_echoes(
    examples: list[EchoExample],
    current_step: int,
    n_select: int,
) -> list[EchoExample]:
    """Select the top-scoring examples to replay."""
    if n_select < 0:
        raise ValueError(f"n_select must be non-negative, got {n_select}")
    if not examples or n_select == 0:
        return []
    probs = echo_probabilities(examples, current_step)
    k = min(n_select, len(examples))
    indices = torch.topk(probs, k=k).indices.tolist()
    return [examples[index] for index in indices]


def update_echo_metadata(
    examples: list[EchoExample],
    echoed_ids: list[str],
    current_step: int,
) -> None:
    """Update replay counters after a batch uses echoed examples."""
    echoed = set(echoed_ids)
    for example in examples:
        if example.example_id in echoed:
            example.echo_count += 1
            example.last_seen_step = current_step


# ---------------------------------------------------------------------------
# Batch-repeat data echoing (Choi et al. 2019)
# ---------------------------------------------------------------------------


@dataclass
class EchoConfig:
    """Configuration for batch-repeat data echoing.

    Attributes:
        echo_factor: Number of gradient steps per unique batch (k).
        shuffle_echoes: If True, shuffle the batch dimension before each echo
            to reduce overfitting from seeing the same sequence order.
    """

    echo_factor: int = 4
    shuffle_echoes: bool = True


class DataEchoBuffer:
    """Wraps a data iterator to repeat each batch ``echo_factor`` times.

    Each call to ``__next__`` returns the current batch (optionally with a
    shuffled sequence order) and increments an internal echo counter.  Once
    the counter reaches ``echo_factor`` a new batch is fetched from the
    underlying iterator and the counter is reset.

    Args:
        data_iter: Iterator yielding ``(input_ids, labels)`` tensor pairs.
        config: :class:`EchoConfig` instance controlling repeat count and
            shuffle behaviour.
    """

    def __init__(self, data_iter, config: EchoConfig) -> None:
        self._iter: Iterator = iter(data_iter)
        self.config = config
        self._current_batch: tuple[torch.Tensor, torch.Tensor] | None = None
        self._echo_count: int = 0

    def __iter__(self) -> DataEchoBuffer:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the current batch (possibly shuffled) and advance state.

        Fetches a new batch from the underlying iterator when all echoes for
        the current batch have been emitted, or on the very first call.

        Raises:
            StopIteration: When the underlying iterator is exhausted and there
                are no more batches to echo.
        """
        if self._current_batch is None or self._echo_count >= self.config.echo_factor:
            # Fetch a fresh batch — propagate StopIteration to the caller.
            self._current_batch = next(self._iter)
            self._echo_count = 0

        input_ids, labels = self._current_batch
        self._echo_count += 1

        if self.config.shuffle_echoes:
            return self._shuffle_batch(input_ids, labels)
        return input_ids, labels

    def _shuffle_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shuffle sequence order along the batch dimension (dim=0) consistently.

        Both ``input_ids`` and ``labels`` receive the *same* permutation so
        that each (input, label) pair stays intact.

        Args:
            input_ids: ``(batch, seq_len)`` tensor.
            labels: ``(batch, seq_len)`` tensor.

        Returns:
            Tuple of ``(shuffled_input_ids, shuffled_labels)``.
        """
        perm = torch.randperm(input_ids.size(0), device=input_ids.device)
        return input_ids[perm], labels[perm]


class EchoTrainer:
    """Trainer that uses batch-repeat data echoing for efficient training.

    Instead of fetching a fresh batch every gradient step, each unique batch
    is reused ``echo_factor`` times with independent forward/backward passes
    and optimizer updates.

    Attributes:
        unique_batches_seen: Number of distinct batches pulled from the data
            iterator so far.
        total_steps: Total number of gradient update steps taken.

    Args:
        model: An :class:`~src.model.transformer.AureliusTransformer` (or any
            ``nn.Module`` whose forward returns ``(loss, logits, pkv)``).
        optimizer: PyTorch optimizer associated with ``model``.
        echo_config: :class:`EchoConfig` controlling echoing behaviour.
        tokenizer_encode: Optional callable ``str -> list[int]`` (unused by
            the trainer itself but stored for callers that need it).
        max_seq_len: Maximum sequence length used when building batches
            externally.
    """

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        echo_config: EchoConfig,
        tokenizer_encode=None,
        max_seq_len: int = 512,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.echo_config = echo_config
        self.tokenizer_encode = tokenizer_encode
        self.max_seq_len = max_seq_len

        self.unique_batches_seen: int = 0
        self.total_steps: int = 0

    def train_on_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> list[dict]:
        """Run ``echo_factor`` independent gradient steps on the same batch.

        For each echo the method:
        1. Optionally shuffles the batch (when ``echo_config.shuffle_echoes``).
        2. Runs a forward pass through the model.
        3. Computes cross-entropy loss manually (shift logits/labels by one
           position, ignoring positions labelled ``-100``).
        4. Runs a backward pass and calls ``optimizer.step()`` / ``zero_grad()``.

        Args:
            input_ids: ``(batch, seq_len)`` integer token tensor.
            labels: ``(batch, seq_len)`` integer label tensor.  Positions with
                value ``-100`` are ignored in the loss.

        Returns:
            List of ``{'loss': float, 'echo': int}`` dicts, one per echo.
        """
        self.unique_batches_seen += 1
        metrics: list[dict] = []

        for echo_idx in range(self.echo_config.echo_factor):
            if self.echo_config.shuffle_echoes:
                perm = torch.randperm(input_ids.size(0), device=input_ids.device)
                batch_ids = input_ids[perm]
                batch_labels = labels[perm]
            else:
                batch_ids = input_ids
                batch_labels = labels

            self.optimizer.zero_grad()

            # Forward pass — model returns (loss_or_none, logits, pkv).
            _, logits, _ = self.model(batch_ids)

            # Manual shifted cross-entropy so we control ignore_index behaviour.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch_labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            loss.backward()
            self.optimizer.step()

            self.total_steps += 1
            metrics.append({"loss": loss.item(), "echo": echo_idx})

        return metrics

    def train_steps(self, data_iter, n_steps: int) -> dict:
        """Run exactly ``n_steps`` gradient updates using echoed batches.

        Batches are drawn from ``data_iter`` via a :class:`DataEchoBuffer`.
        Each unique batch contributes ``echo_factor`` gradient steps.

        Args:
            data_iter: Iterable yielding ``(input_ids, labels)`` pairs.
            n_steps: Total number of gradient update steps to perform.

        Returns:
            Dictionary with keys:

            * ``total_steps`` — steps actually performed (≤ ``n_steps``).
            * ``unique_batches`` — unique batches consumed.
            * ``echo_ratio`` — ``total_steps / unique_batches``.
        """
        buffer = DataEchoBuffer(data_iter, self.echo_config)
        steps_done = 0

        while steps_done < n_steps:
            try:
                input_ids, labels = next(buffer)
            except StopIteration:
                break

            self.optimizer.zero_grad()

            _, logits, _ = self.model(input_ids)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            loss.backward()
            self.optimizer.step()

            steps_done += 1
            self.total_steps += 1

        # Count unique batches: each echo_factor steps = 1 unique batch,
        # but buffer may not be perfectly aligned.  We rely on buffer state.
        # Recompute from the buffer's internal echo counter.
        # full batches consumed = steps_done // echo_factor (integer div)
        full_batches = steps_done // self.echo_config.echo_factor
        partial = 1 if (steps_done % self.echo_config.echo_factor) > 0 else 0
        unique_batches = full_batches + partial
        self.unique_batches_seen += unique_batches

        echo_ratio = steps_done / unique_batches if unique_batches > 0 else 0.0
        return {
            "total_steps": steps_done,
            "unique_batches": unique_batches,
            "echo_ratio": echo_ratio,
        }


def compare_echo_vs_no_echo(
    model_factory,
    data_iter_factory,
    n_steps: int = 20,
    echo_factor: int = 4,
    lr: float = 1e-4,
) -> dict:
    """Train two identical models: one with echoing, one without.

    Both models perform exactly ``n_steps`` gradient updates.  The echo model
    consumes ``n_steps // echo_factor`` unique batches; the no-echo model
    consumes ``n_steps`` unique batches.

    Args:
        model_factory: Zero-argument callable returning a fresh model instance.
        data_iter_factory: Zero-argument callable returning a fresh data
            iterator yielding ``(input_ids, labels)`` pairs.
        n_steps: Total gradient steps for each model.
        echo_factor: Echo repeat count for the echo model.
        lr: Learning rate for both optimizers (SGD).

    Returns:
        Dictionary with keys:

        * ``echo_final_loss`` — loss on the last step of the echo model.
        * ``no_echo_final_loss`` — loss on the last step of the no-echo model.
        * ``echo_unique_batches`` — unique batches consumed by the echo model.
        * ``no_echo_unique_batches`` — unique batches consumed by the no-echo
          model (should equal ``n_steps``).
    """
    # --- Echo model ---
    echo_model = model_factory()
    echo_optimizer = torch.optim.SGD(echo_model.parameters(), lr=lr)
    echo_cfg = EchoConfig(echo_factor=echo_factor, shuffle_echoes=False)
    EchoTrainer(echo_model, echo_optimizer, echo_cfg)

    echo_metrics: list[dict] = []
    echo_buffer = DataEchoBuffer(data_iter_factory(), echo_cfg)
    echo_steps = 0
    while echo_steps < n_steps:
        try:
            input_ids, labels = next(echo_buffer)
        except StopIteration:
            break
        echo_optimizer.zero_grad()
        _, logits, _ = echo_model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        loss.backward()
        echo_optimizer.step()
        echo_metrics.append({"loss": loss.item()})
        echo_steps += 1

    echo_unique = echo_steps // echo_factor + (1 if echo_steps % echo_factor else 0)

    # --- No-echo model ---
    no_echo_model = model_factory()
    no_echo_optimizer = torch.optim.SGD(no_echo_model.parameters(), lr=lr)
    no_echo_cfg = EchoConfig(echo_factor=1, shuffle_echoes=False)
    no_echo_buffer = DataEchoBuffer(data_iter_factory(), no_echo_cfg)

    no_echo_metrics: list[dict] = []
    no_echo_steps = 0
    while no_echo_steps < n_steps:
        try:
            input_ids, labels = next(no_echo_buffer)
        except StopIteration:
            break
        no_echo_optimizer.zero_grad()
        _, logits, _ = no_echo_model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        loss.backward()
        no_echo_optimizer.step()
        no_echo_metrics.append({"loss": loss.item()})
        no_echo_steps += 1

    return {
        "echo_final_loss": echo_metrics[-1]["loss"] if echo_metrics else float("nan"),
        "no_echo_final_loss": no_echo_metrics[-1]["loss"] if no_echo_metrics else float("nan"),
        "echo_unique_batches": echo_unique,
        "no_echo_unique_batches": no_echo_steps,
    }
