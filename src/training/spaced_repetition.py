"""Spaced repetition learning scheduler for LLM training.

Schedules training examples based on model performance, using an adaptation of
the SM-2 algorithm (as used in Anki). Hard examples (high loss) are reviewed
more frequently; easy examples (low loss) are reviewed less often.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SM2Config:
    """Configuration for the SM-2 spaced repetition scheduler."""
    initial_interval: int = 1          # days (steps) until first review
    initial_easiness: float = 2.5      # starting easiness factor
    min_easiness: float = 1.3          # floor for easiness
    max_interval: int = 365            # maximum interval in steps
    performance_threshold: float = 0.6 # loss below this = "easy"


# ---------------------------------------------------------------------------
# Card state
# ---------------------------------------------------------------------------

@dataclass
class CardState:
    """State of a single training example (card) in the spaced repetition system."""
    card_id: str
    interval: int = 1               # steps until next review
    easiness: float = 2.5
    repetitions: int = 0
    last_loss: float = float('inf')
    next_review: int = 0            # step number when card is next due


# ---------------------------------------------------------------------------
# SM-2 Scheduler
# ---------------------------------------------------------------------------

class SM2Scheduler:
    """SM-2 algorithm adapted for LLM training.

    Cards represent training examples. After each forward pass the caller
    reports the observed loss via ``update()``. The scheduler then decides
    when the card should be reviewed again.
    """

    def __init__(self, config: SM2Config) -> None:
        self.config = config
        self._cards: dict[str, CardState] = {}
        self._step: int = 0

    def register(self, card_id: str) -> None:
        """Register a new training example."""
        if card_id not in self._cards:
            self._cards[card_id] = CardState(
                card_id=card_id,
                interval=self.config.initial_interval,
                easiness=self.config.initial_easiness,
                next_review=self._step,
            )

    def update(self, card_id: str, loss: float) -> None:
        """Update card state based on observed loss (SM-2 quality scoring)."""
        if card_id not in self._cards:
            self.register(card_id)

        card = self._cards[card_id]
        threshold = self.config.performance_threshold

        # Map loss to SM-2 quality score (0-5)
        if loss < threshold * 0.5:
            quality = 5   # very easy
        elif loss < threshold:
            quality = 3   # acceptable
        else:
            quality = 1   # hard / failed

        # Update easiness factor
        new_easiness = (
            card.easiness + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
        )
        new_easiness = max(self.config.min_easiness, min(5.0, new_easiness))

        # Update interval and repetitions
        if quality >= 3:
            if card.repetitions == 0:
                new_interval = 1
            elif card.repetitions == 1:
                new_interval = 6
            else:
                new_interval = int(card.interval * card.easiness)
            new_repetitions = card.repetitions + 1
        else:
            # Failed — reset
            new_interval = 1
            new_repetitions = 0

        new_interval = min(new_interval, self.config.max_interval)

        # Persist updates
        card.easiness = new_easiness
        card.interval = new_interval
        card.repetitions = new_repetitions
        card.last_loss = loss
        card.next_review = self._step + new_interval

    def due(self, n: Optional[int] = None) -> list[str]:
        """Return card_ids due for review at the current step.

        Args:
            n: If given, return at most n cards sorted by most-overdue first.

        Returns:
            List of card_ids whose next_review <= current step.
        """
        due_cards = [
            card for card in self._cards.values()
            if card.next_review <= self._step
        ]
        # Sort most overdue first (lowest next_review)
        due_cards.sort(key=lambda c: c.next_review)

        result = [c.card_id for c in due_cards]
        if n is not None:
            result = result[:n]
        return result

    def step(self) -> None:
        """Increment the internal step counter."""
        self._step += 1

    def stats(self) -> dict:
        """Return summary statistics about the scheduler state."""
        cards = list(self._cards.values())
        n_cards = len(cards)
        n_due = len(self.due())

        if n_cards > 0:
            mean_interval = sum(c.interval for c in cards) / n_cards
            mean_easiness = sum(c.easiness for c in cards) / n_cards
        else:
            mean_interval = 0.0
            mean_easiness = 0.0

        return {
            "n_cards": n_cards,
            "n_due": n_due,
            "mean_interval": mean_interval,
            "mean_easiness": mean_easiness,
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpacedRepetitionDataset:
    """Dataset wrapper that prioritises due (hard/overdue) examples.

    Examples are ``(card_id, token_ids)`` pairs. At each batch the scheduler
    is consulted and due cards are preferred over random sampling.
    """

    def __init__(
        self,
        examples: list[tuple[str, list[int]]],
        config: SM2Config,
    ) -> None:
        self._examples: dict[str, list[int]] = {}
        self.scheduler = SM2Scheduler(config)

        for card_id, token_ids in examples:
            self._examples[card_id] = token_ids
            self.scheduler.register(card_id)

        self._all_ids: list[str] = [card_id for card_id, _ in examples]

    def __len__(self) -> int:
        return len(self._examples)

    def get_batch(self, batch_size: int) -> list[tuple[str, list[int]]]:
        """Return a batch of examples, prioritising due cards.

        If there are fewer due cards than batch_size, pad with random samples
        from the full example pool.
        """
        due = self.scheduler.due(n=batch_size)
        selected = list(due)

        if len(selected) < batch_size:
            remaining_ids = [cid for cid in self._all_ids if cid not in set(selected)]
            needed = batch_size - len(selected)
            if remaining_ids:
                extra = random.sample(remaining_ids, min(needed, len(remaining_ids)))
                selected.extend(extra)

        return [(cid, self._examples[cid]) for cid in selected]

    def record_losses(self, card_ids: list[str], losses: list[float]) -> None:
        """Update scheduler with observed losses for a batch of cards."""
        for card_id, loss in zip(card_ids, losses):
            self.scheduler.update(card_id, loss)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SpacedRepetitionTrainer:
    """Trainer that uses spaced repetition to prioritise hard examples.

    On each ``train_step`` the trainer:
    1. Selects a batch from the dataset (preferring due cards).
    2. Runs a forward pass per example to compute per-example loss.
    3. Records losses back into the scheduler.
    4. Performs an optimizer step on the mean batch loss.
    """

    def __init__(
        self,
        model,
        optimizer,
        dataset: SpacedRepetitionDataset,
        config: SM2Config,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.config = config

    def train_step(self) -> dict:
        """Perform a single training step.

        Returns:
            dict with keys: loss, n_due, n_examples, mean_interval
        """
        batch = self.dataset.get_batch(batch_size=4)
        card_ids = [item[0] for item in batch]
        token_lists = [item[1] for item in batch]

        self.model.train()
        self.optimizer.zero_grad()

        per_example_losses: list[float] = []
        total_loss = torch.tensor(0.0)

        for token_ids in token_lists:
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            labels = input_ids.clone()

            loss, _logits, _pkv = self.model(input_ids, labels=labels)

            if loss is not None:
                total_loss = total_loss + loss
                per_example_losses.append(loss.item())
            else:
                # Fallback: compute loss from logits manually
                logits_for_loss = _logits[:, :-1, :].contiguous()
                targets = input_ids[:, 1:].contiguous()
                ex_loss = F.cross_entropy(
                    logits_for_loss.view(-1, _logits.shape[-1]),
                    targets.view(-1),
                )
                total_loss = total_loss + ex_loss
                per_example_losses.append(ex_loss.item())

        if per_example_losses:
            mean_loss = total_loss / len(per_example_losses)
            mean_loss.backward()
        else:
            mean_loss = total_loss

        self.optimizer.step()

        # Record per-example losses back into the scheduler
        self.dataset.record_losses(card_ids, per_example_losses)
        self.dataset.scheduler.step()

        stats = self.dataset.scheduler.stats()

        return {
            "loss": mean_loss.item(),
            "n_due": stats["n_due"],
            "n_examples": len(per_example_losses),
            "mean_interval": stats["mean_interval"],
        }

    def get_hard_examples(self, n: int = 10) -> list[str]:
        """Return the n examples with the lowest easiness scores (most difficult)."""
        cards = list(self.dataset.scheduler._cards.values())
        cards.sort(key=lambda c: c.easiness)
        return [c.card_id for c in cards[:n]]
