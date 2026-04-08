"""
MiniMax-M2-style interleaved thinking token format with SFT training support.

The model can emit <think>...</think> blocks containing chain-of-thought reasoning
before its final response. During training, thinking tokens are weighted differently
from answer tokens. During inference, thinking tokens are kept in KV cache but
optionally stripped from final output.
"""

import re
import torch
import torch.nn.functional as F

THINK_START = "<think>"
THINK_END = "</think>"

# Special token IDs (fixed integers for use without a real tokenizer)
THINK_START_TOKEN_ID = 200001
THINK_END_TOKEN_ID = 200002


class ThinkingFormat:
    """Handles formatting and parsing of thinking token sequences."""

    def wrap_thinking(self, thought: str, response: str) -> str:
        """Format: '<think>' + thought + '</think>' + response"""
        return f"{THINK_START}{thought}{THINK_END}{response}"

    def extract_thinking(self, text: str) -> tuple[str, str]:
        """Extract (thought, response) from formatted text.

        If no <think> block, thought='' and response=full text.
        """
        pattern = re.compile(
            r"^" + re.escape(THINK_START) + r"(.*?)" + re.escape(THINK_END),
            re.DOTALL,
        )
        m = pattern.match(text)
        if m:
            thought = m.group(1)
            response = text[m.end():]
            return thought, response
        return "", text

    def has_thinking(self, text: str) -> bool:
        """Check if text contains a thinking block."""
        return THINK_START in text and THINK_END in text


class ThinkingLossWeights:
    """Compute per-token loss weights for thinking token training."""

    def __init__(self, think_weight: float = 0.5, answer_weight: float = 1.0):
        """
        think_weight: weight for tokens inside <think>...</think>
        answer_weight: weight for tokens in final response
        """
        self.think_weight = think_weight
        self.answer_weight = answer_weight

    def compute_weights(self, token_ids: list[int]) -> torch.Tensor:
        """
        Given a list of token IDs (with THINK_START_TOKEN_ID and THINK_END_TOKEN_ID),
        return per-token float weights.

        Tokens before <think>: answer_weight (system/user prompt, though usually -100)
        Tokens inside <think>...</think>: think_weight
        Tokens after </think>: answer_weight
        THINK_START and THINK_END tokens themselves: 0.0 (structure tokens)
        """
        weights = []
        inside_think = False

        for tid in token_ids:
            if tid == THINK_START_TOKEN_ID:
                weights.append(0.0)
                inside_think = True
            elif tid == THINK_END_TOKEN_ID:
                weights.append(0.0)
                inside_think = False
            elif inside_think:
                weights.append(self.think_weight)
            else:
                weights.append(self.answer_weight)

        return torch.tensor(weights, dtype=torch.float32)


class ThinkingSFTDataset:
    """Prepare thinking token training examples."""

    def __init__(self, think_weight: float = 0.5, answer_weight: float = 1.0):
        self.think_weight = think_weight
        self.answer_weight = answer_weight
        self._loss_weights = ThinkingLossWeights(think_weight, answer_weight)

    def create_example(
        self,
        prompt_ids: list[int],
        thought_ids: list[int],
        response_ids: list[int],
    ) -> dict:
        """
        Build training example:
        input_ids = prompt_ids + [THINK_START] + thought_ids + [THINK_END] + response_ids
        labels = -100 for prompt_ids, then token_ids for the rest
        weights = per-token loss weights (think vs answer)

        Returns: {'input_ids': tensor, 'labels': tensor, 'weights': tensor}
        """
        completion_ids = (
            [THINK_START_TOKEN_ID] + thought_ids + [THINK_END_TOKEN_ID] + response_ids
        )
        input_ids = prompt_ids + completion_ids

        # Labels: mask prompt with -100, keep completion token ids
        labels = [-100] * len(prompt_ids) + completion_ids

        # Weights for full sequence (prompt gets answer_weight, but labels are -100 so ignored)
        weights = self._loss_weights.compute_weights(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "weights": weights,
        }

    def thinking_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted cross-entropy loss.
        logits: (B, S, vocab_size)
        labels: (B, S) with -100 for ignored positions
        weights: (B, S) per-token weights

        Returns: scalar loss
        """
        B, S, V = logits.shape

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_weights = weights[:, 1:].contiguous()

        # Flatten
        flat_logits = shift_logits.view(-1, V)
        flat_labels = shift_labels.view(-1)
        flat_weights = shift_weights.view(-1)

        # Per-token cross-entropy (reduction='none'), -100 positions give 0 loss
        per_token_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="none")

        # Zero out ignored positions in weights too
        mask = (flat_labels != -100).float()
        weighted_loss = per_token_loss * flat_weights * mask

        # Normalise by number of non-ignored, non-zero-weight tokens
        denom = (flat_weights * mask).sum()
        if denom == 0:
            return weighted_loss.sum()
        return weighted_loss.sum() / denom


class ThinkingInferenceHelper:
    """Helps with thinking token generation and output processing."""

    def __init__(self, max_think_tokens: int = 512):
        self.max_think_tokens = max_think_tokens

    def strip_thinking(self, text: str) -> str:
        """Remove <think>...</think> blocks from output."""
        return re.sub(
            re.escape(THINK_START) + r".*?" + re.escape(THINK_END),
            "",
            text,
            flags=re.DOTALL,
        )

    def split_thinking(self, text: str) -> dict:
        """Return {'thinking': str, 'response': str}"""
        fmt = ThinkingFormat()
        thought, response = fmt.extract_thinking(text)
        return {"thinking": thought, "response": response}

    def should_stop_thinking(self, generated_ids: list[int]) -> bool:
        """Return True if THINK_END_TOKEN_ID has been generated."""
        return THINK_END_TOKEN_ID in generated_ids

    def count_think_tokens(self, token_ids: list[int]) -> int:
        """Count tokens inside thinking blocks (excludes structure tokens)."""
        count = 0
        inside = False
        for tid in token_ids:
            if tid == THINK_START_TOKEN_ID:
                inside = True
            elif tid == THINK_END_TOKEN_ID:
                inside = False
            elif inside:
                count += 1
        return count
