"""Sequence-Level Knowledge Distillation.

Kim & Rush 2016 — https://arxiv.org/abs/1606.07947
Gu et al. 2023 (MiniLLM) — https://arxiv.org/abs/2306.08543
Agarwal et al. 2024 (GKD) — https://arxiv.org/abs/2306.13649

Distills at the sequence level rather than per-token logit matching.
"""

from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def sequence_log_probs(
    model_fn: Callable,
    prompt_ids: Tensor,
    seq_tokens: Tensor,
) -> Tensor:
    """Return mean log prob of *seq_tokens* given *prompt_ids* under *model_fn*.

    Args:
        model_fn: ``(input_ids: LongTensor(1, T)) -> (hidden, logits)`` where
            *logits* has shape ``(1, T, V)``.
        prompt_ids: ``(1, T_p)`` prompt token ids.
        seq_tokens: ``(L,)`` target token ids (new tokens only).

    Returns:
        Scalar tensor: mean log-probability of *seq_tokens*.
    """
    # Concatenate along sequence dimension: (1, T_p + L)
    full_ids = torch.cat([prompt_ids, seq_tokens.unsqueeze(0)], dim=1)
    _, logits = model_fn(full_ids)  # (1, T_p+L, V)

    # We need the logits at positions T_p-1 .. T_p+L-2 (predicting tokens at T_p .. T_p+L-1)
    T_p = prompt_ids.shape[1]
    L = seq_tokens.shape[0]

    # logits at positions [T_p-1 : T_p+L-1] predict tokens [T_p : T_p+L]
    pred_logits = logits[0, T_p - 1 : T_p + L - 1, :]  # (L, V)
    log_probs = F.log_softmax(pred_logits, dim=-1)       # (L, V)

    # Gather log probs at target token positions
    target = seq_tokens  # (L,)
    token_log_probs = log_probs[torch.arange(L), target]  # (L,)

    return token_log_probs.mean()


# ---------------------------------------------------------------------------
# TeacherSequenceSampler
# ---------------------------------------------------------------------------

class TeacherSequenceSampler:
    """Samples sequences from a teacher model.

    Args:
        teacher_fn: ``(input_ids: LongTensor(1, T)) -> (hidden, logits)``
        vocab_size: Vocabulary size.
        temperature: Sampling temperature.  Greedy decoding when <= 0.
    """

    def __init__(
        self,
        teacher_fn: Callable,
        vocab_size: int,
        temperature: float = 1.0,
    ) -> None:
        self.teacher_fn = teacher_fn
        self.vocab_size = vocab_size
        self.temperature = temperature

    def sample_sequence(self, prompt_ids: Tensor, max_length: int) -> Tensor:
        """Generate *max_length* new tokens from the teacher.

        Args:
            prompt_ids: ``(1, T_p)`` prompt token ids.
            max_length: Number of new tokens to generate.

        Returns:
            ``(max_length,)`` LongTensor of generated token ids.
        """
        generated: List[int] = []
        current_ids = prompt_ids.clone()

        with torch.no_grad():
            for _ in range(max_length):
                _, logits = self.teacher_fn(current_ids)  # (1, T, V)
                next_logits = logits[0, -1, :]             # (V,)

                if self.temperature <= 0:
                    next_token = next_logits.argmax().item()
                else:
                    probs = F.softmax(next_logits / self.temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()

                generated.append(int(next_token))
                new_tok = torch.tensor([[next_token]], dtype=torch.long)
                current_ids = torch.cat([current_ids, new_tok], dim=1)

        return torch.tensor(generated, dtype=torch.long)

    def beam_search(
        self,
        prompt_ids: Tensor,
        max_length: int,
        beam_width: int = 4,
    ) -> List[Tensor]:
        """Simple beam search returning *beam_width* sequences.

        Args:
            prompt_ids: ``(1, T_p)`` prompt token ids.
            max_length: Number of new tokens per sequence.
            beam_width: Number of beams.

        Returns:
            List of *beam_width* LongTensors each of shape ``(max_length,)``.
        """
        # Each beam: (cumulative_log_prob, list_of_token_ids, current_ids_tensor)
        beams: List[tuple] = [(0.0, [], prompt_ids.clone())]

        with torch.no_grad():
            for _ in range(max_length):
                candidates: List[tuple] = []
                for score, tokens, cur_ids in beams:
                    _, logits = self.teacher_fn(cur_ids)  # (1, T, V)
                    log_probs = F.log_softmax(logits[0, -1, :], dim=-1)  # (V,)

                    # Take top beam_width tokens
                    top_log_probs, top_tokens = log_probs.topk(beam_width)
                    for lp, tok in zip(top_log_probs.tolist(), top_tokens.tolist()):
                        new_ids = torch.cat(
                            [cur_ids, torch.tensor([[tok]], dtype=torch.long)], dim=1
                        )
                        candidates.append((score + lp, tokens + [tok], new_ids))

                # Keep top beam_width by cumulative score
                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = candidates[:beam_width]

        return [torch.tensor(tokens, dtype=torch.long) for _, tokens, _ in beams]


# ---------------------------------------------------------------------------
# SequenceLevelKDLoss
# ---------------------------------------------------------------------------

class SequenceLevelKDLoss:
    """Train the student on teacher-generated sequences (Kim & Rush 2016).

    Args:
        student_fn: ``(input_ids: LongTensor(1, T)) -> (hidden, logits)``
        reduction: Currently only 'mean' is supported.
    """

    def __init__(self, student_fn: Callable, reduction: str = "mean") -> None:
        self.student_fn = student_fn
        self.reduction = reduction

    def forward(
        self,
        prompt_ids: Tensor,
        teacher_sequences: List[Tensor],
    ) -> Tensor:
        """Compute mean NLL of *teacher_sequences* under the student.

        Args:
            prompt_ids: ``(1, T_p)`` prompt token ids.
            teacher_sequences: List of ``(L,)`` sequences generated by teacher.

        Returns:
            Scalar tensor: mean NLL.
        """
        nlls: List[Tensor] = []
        for seq in teacher_sequences:
            # sequence_log_probs returns mean log prob (≤ 0); NLL = -mean_log_prob
            lp = sequence_log_probs(self.student_fn, prompt_ids, seq)
            nlls.append(-lp)
        return torch.stack(nlls).mean()


# ---------------------------------------------------------------------------
# MiniLLMLoss
# ---------------------------------------------------------------------------

class MiniLLMLoss:
    """Reverse KL distillation: E_student[log p_student / p_teacher].

    Args:
        student_fn: ``(input_ids: LongTensor(1, T)) -> (hidden, logits)``
        teacher_fn: ``(input_ids: LongTensor(1, T)) -> (hidden, logits)``
    """

    def __init__(self, student_fn: Callable, teacher_fn: Callable) -> None:
        self.student_fn = student_fn
        self.teacher_fn = teacher_fn

    def forward(
        self,
        prompt_ids: Tensor,
        student_sequences: List[Tensor],
    ) -> Tensor:
        """Reverse KL loss on student-sampled sequences.

        Args:
            prompt_ids: ``(1, T_p)`` prompt token ids.
            student_sequences: List of ``(L,)`` sequences sampled from student.

        Returns:
            Scalar tensor: mean reverse KL.
        """
        rkl_terms: List[Tensor] = []
        for seq in student_sequences:
            log_s = sequence_log_probs(self.student_fn, prompt_ids, seq)
            log_t = sequence_log_probs(self.teacher_fn, prompt_ids, seq)
            # Reverse KL contribution: log p_student - log p_teacher
            rkl_terms.append(log_s - log_t)
        return torch.stack(rkl_terms).mean()


# ---------------------------------------------------------------------------
# GKDLoss
# ---------------------------------------------------------------------------

class GKDLoss:
    """Generalized Knowledge Distillation (Agarwal et al. 2024).

    Mixes on-policy (student-sampled) reverse KL with off-policy (teacher-
    generated) forward KL / NLL.

    Args:
        student_fn: ``(input_ids: LongTensor(1, T)) -> (hidden, logits)``
        teacher_fn: ``(input_ids: LongTensor(1, T)) -> (hidden, logits)``
        beta: Fraction of weight on reverse KL (student seqs).
              ``1 - beta`` weight on NLL (teacher seqs).
    """

    def __init__(
        self,
        student_fn: Callable,
        teacher_fn: Callable,
        beta: float = 0.5,
    ) -> None:
        self.student_fn = student_fn
        self.teacher_fn = teacher_fn
        self.beta = beta
        self._seq_kd = SequenceLevelKDLoss(student_fn)
        self._mini_llm = MiniLLMLoss(student_fn, teacher_fn)

    def forward(
        self,
        prompt_ids: Tensor,
        teacher_sequences: List[Tensor],
        student_sequences: List[Tensor],
    ) -> Tensor:
        """Combined GKD loss.

        Args:
            prompt_ids: ``(1, T_p)`` prompt token ids.
            teacher_sequences: List of ``(L,)`` teacher-generated sequences.
            student_sequences: List of ``(L,)`` student-sampled sequences.

        Returns:
            Scalar tensor: combined loss.
        """
        forward_kl = self._seq_kd.forward(prompt_ids, teacher_sequences)
        reverse_kl = self._mini_llm.forward(prompt_ids, student_sequences)
        return self.beta * reverse_kl + (1.0 - self.beta) * forward_kl
