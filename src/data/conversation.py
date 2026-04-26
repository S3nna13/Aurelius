"""Multi-turn conversation dataset builder.

Formats and tokenizes multi-turn chat conversations for training.
Handles system prompts, turn alternation, and chat templates.
Produces SFT-ready (input_ids, labels) pairs where only assistant turns
are included in the loss (user turns masked to -100).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class Turn:
    """A single conversation turn."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class Conversation:
    """A complete multi-turn conversation."""

    turns: list[Turn]

    def __len__(self) -> int:
        return len(self.turns)

    def user_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "user"]

    def assistant_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "assistant"]

    def is_valid(self) -> bool:
        """A valid conversation alternates user/assistant (optionally starting with system)."""
        roles = [t.role for t in self.turns]
        # Remove system prefix if present
        if roles and roles[0] == "system":
            roles = roles[1:]
        # Must alternate user/assistant starting with user
        if not roles or roles[0] != "user":
            return False
        for i, role in enumerate(roles):
            expected = "user" if i % 2 == 0 else "assistant"
            if role != expected:
                return False
        return True


@dataclass
class ChatTemplate:
    """Template for formatting conversations as token sequences."""

    system_prefix: str = "<|system|>"
    system_suffix: str = "<|end|>\n"
    user_prefix: str = "<|user|>"
    user_suffix: str = "<|end|>\n"
    assistant_prefix: str = "<|assistant|>"
    assistant_suffix: str = "<|end|>\n"
    bos_token: str = "<|begin|>"
    eos_token: str = "<|end|>"

    def format_turn(self, turn: Turn) -> str:
        """Format a single turn as a string."""
        if turn.role == "system":
            return f"{self.system_prefix}{turn.content}{self.system_suffix}"
        elif turn.role == "user":
            return f"{self.user_prefix}{turn.content}{self.user_suffix}"
        elif turn.role == "assistant":
            return f"{self.assistant_prefix}{turn.content}{self.assistant_suffix}"
        else:
            raise ValueError(f"Unknown role: {turn.role!r}")

    def format_conversation(self, conv: Conversation) -> str:
        """Format full conversation as string."""
        return "".join(self.format_turn(t) for t in conv.turns)


def build_sft_labels(
    conversation: Conversation,
    template: ChatTemplate,
    tokenize_fn,  # callable: str → list[int]
    max_seq_len: int,
    loss_on_all_turns: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (input_ids, labels) for SFT training.

    By default (loss_on_all_turns=False):
    - User and system tokens have label=-100 (excluded from loss)
    - Only assistant tokens contribute to loss

    Args:
        conversation: Multi-turn conversation
        template: Chat formatting template
        tokenize_fn: Tokenization function (e.g., lambda s: [ord(c) % vocab for c in s])
        max_seq_len: Maximum sequence length (truncate if longer)
        loss_on_all_turns: If True, compute loss on all turns

    Returns:
        (input_ids, labels) — each shape (seq_len,)
    """
    all_ids: list[int] = []
    all_labels: list[int] = []

    for turn in conversation.turns:
        turn_str = template.format_turn(turn)
        ids = tokenize_fn(turn_str)
        if loss_on_all_turns or turn.role == "assistant":
            labels = ids[:]
        else:
            labels = [-100] * len(ids)
        all_ids.extend(ids)
        all_labels.extend(labels)

    # Truncate to max_seq_len
    all_ids = all_ids[:max_seq_len]
    all_labels = all_labels[:max_seq_len]

    input_ids = torch.tensor(all_ids, dtype=torch.long)
    labels = torch.tensor(all_labels, dtype=torch.long)
    return input_ids, labels


def conversations_from_pairs(
    pairs: list[tuple[str, str]],  # list of (user_message, assistant_response)
    system_prompt: str | None = None,
) -> list[Conversation]:
    """Convert (question, answer) pairs to single-turn Conversations."""
    result: list[Conversation] = []
    for user_msg, assistant_msg in pairs:
        turns: list[Turn] = []
        if system_prompt is not None:
            turns.append(Turn(role="system", content=system_prompt))
        turns.append(Turn(role="user", content=user_msg))
        turns.append(Turn(role="assistant", content=assistant_msg))
        result.append(Conversation(turns=turns))
    return result


def concatenate_conversations(
    convs: list[Conversation],
    max_turns: int = 10,
) -> Conversation:
    """Merge multiple single-turn conversations into one multi-turn conversation.

    Useful for building multi-turn training data from single-turn pairs.
    Truncates to max_turns total turns.
    """
    merged_turns: list[Turn] = []
    for conv in convs:
        for turn in conv.turns:
            # Skip duplicate system prompts after the first
            if turn.role == "system" and any(t.role == "system" for t in merged_turns):
                continue
            merged_turns.append(turn)
            if len(merged_turns) >= max_turns:
                break
        if len(merged_turns) >= max_turns:
            break
    return Conversation(turns=merged_turns)


class ConversationDataset:
    """Dataset of conversations ready for SFT training.

    Tokenizes conversations and produces (input_ids, labels) tensors.
    """

    def __init__(
        self,
        conversations: list[Conversation],
        template: ChatTemplate,
        tokenize_fn,  # callable: str → list[int]
        max_seq_len: int = 2048,
        loss_on_all_turns: bool = False,
    ) -> None:
        self._conversations = list(conversations)
        self._template = template
        self._tokenize_fn = tokenize_fn
        self._max_seq_len = max_seq_len
        self._loss_on_all_turns = loss_on_all_turns
        # Pre-build tensors for all conversations
        self._samples: list[tuple[torch.Tensor, torch.Tensor]] = [
            build_sft_labels(conv, template, tokenize_fn, max_seq_len, loss_on_all_turns)
            for conv in self._conversations
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Returns {"input_ids": Tensor(S,), "labels": Tensor(S,)}."""
        input_ids, labels = self._samples[idx]
        return {"input_ids": input_ids, "labels": labels}

    def filter_by_length(self, min_len: int = 4, max_len: int | None = None) -> ConversationDataset:
        """Return new dataset with conversations within length bounds."""
        filtered_convs = []
        for conv, (input_ids, _) in zip(self._conversations, self._samples):
            seq_len = len(input_ids)
            if seq_len < min_len:
                continue
            if max_len is not None and seq_len > max_len:
                continue
            filtered_convs.append(conv)
        return ConversationDataset(
            conversations=filtered_convs,
            template=self._template,
            tokenize_fn=self._tokenize_fn,
            max_seq_len=self._max_seq_len,
            loss_on_all_turns=self._loss_on_all_turns,
        )
