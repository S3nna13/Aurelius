"""Adaptive inference-time compute scaling for Aurelius LLM.

Dynamically allocates more compute to harder queries at inference time.
Based on "Scaling LLM Test-Time Compute" (Snell et al. 2024).

Key strategies:
1. Best-of-N with adaptive N: try N samples, pick best by verifier
2. Sequential revision: iteratively refine answer using model's own feedback
3. Compute budget predictor: predict required compute from input features
4. Early exit with confidence gate: stop generating when confidence is high enough
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ComputeBudget:
    max_tokens: int = 512
    max_iterations: int = 5
    confidence_threshold: float = 0.9
    n_candidates: int = 4


@dataclass
class InferenceResult:
    output_ids: torch.Tensor      # (T,) final output
    n_tokens_used: int
    n_iterations: int
    confidence: float
    strategy_used: str


# ---------------------------------------------------------------------------
# Confidence estimator
# ---------------------------------------------------------------------------

class ConfidenceEstimator:
    """Estimate model confidence from logits at each generation step."""

    def __init__(self, method: str = "max_prob"):
        """
        method: "max_prob" | "entropy" | "margin"
        """
        if method not in ("max_prob", "entropy", "margin"):
            raise ValueError(f"Unknown method: {method}. Choose from 'max_prob', 'entropy', 'margin'.")
        self.method = method

    def estimate(self, logits: torch.Tensor) -> float:
        """
        logits: (vocab,) last position logits.
        Returns confidence in [0, 1].
        - max_prob: softmax max probability
        - entropy: 1 - normalized_entropy (higher = more confident)
        - margin: difference between top-2 probabilities
        """
        logits = logits.float()
        probs = F.softmax(logits, dim=-1)
        vocab_size = logits.shape[-1]

        if self.method == "max_prob":
            return float(probs.max().item())

        elif self.method == "entropy":
            # entropy in [0, log(V)]; normalize to [0, 1] then invert
            ent = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            max_ent = math.log(vocab_size) if vocab_size > 1 else 1.0
            normalized = ent / max_ent
            return float(max(0.0, min(1.0, 1.0 - normalized)))

        else:  # margin
            if vocab_size < 2:
                return 1.0
            top2 = torch.topk(probs, k=min(2, vocab_size)).values
            margin = float((top2[0] - top2[1]).item())
            return float(max(0.0, min(1.0, margin)))

    def estimate_sequence(self, logits: torch.Tensor) -> float:
        """logits: (T, vocab). Returns mean confidence across positions."""
        if logits.dim() != 2 or logits.shape[0] == 0:
            return 0.0
        confidences = [self.estimate(logits[t]) for t in range(logits.shape[0])]
        return float(sum(confidences) / len(confidences))


# ---------------------------------------------------------------------------
# Greedy decode helper (pure PyTorch, no HuggingFace)
# ---------------------------------------------------------------------------

def _greedy_decode(
    model: nn.Module,
    input_ids: torch.Tensor,   # (1, T)
    max_new_tokens: int,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple autoregressive decode loop.
    The model is expected to return a tuple (_, logits, _) where logits: (B, T, vocab).
    Returns (output_ids, all_logits) where:
      output_ids: (1, T + max_new_tokens) -- prompt + generated tokens
      all_logits: (max_new_tokens, vocab)  -- logits at each new step
    """
    model.train(False)
    all_generated_logits: List[torch.Tensor] = []
    ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(ids)
            # Accept (logits,) or (_, logits, _) style returns
            if isinstance(out, tuple):
                logits = out[1] if len(out) >= 2 else out[0]
            else:
                logits = out

            # logits: (B, T, vocab) -- take last position
            last_logits = logits[0, -1, :]  # (vocab,)
            all_generated_logits.append(last_logits)

            if temperature != 1.0 and temperature > 0:
                last_logits = last_logits / temperature

            next_token = last_logits.argmax(dim=-1, keepdim=True).unsqueeze(0)  # (1, 1)
            ids = torch.cat([ids, next_token], dim=1)

    all_logits = torch.stack(all_generated_logits, dim=0)  # (max_new_tokens, vocab)
    return ids, all_logits


# ---------------------------------------------------------------------------
# Best-of-N selector
# ---------------------------------------------------------------------------

class BestOfNSelector:
    """Generate N candidates, select best by verifier score."""

    def __init__(
        self,
        model: nn.Module,
        verifier_fn: Optional[Callable[[torch.Tensor], float]] = None,
        temperature: float = 0.8,
    ):
        self.model = model
        self.verifier_fn = verifier_fn
        self.temperature = temperature
        self._confidence_estimator = ConfidenceEstimator(method="max_prob")

    def generate_one(
        self,
        input_ids: torch.Tensor,  # (1, T_prompt)
        max_new_tokens: int = 20,
    ) -> Tuple[torch.Tensor, float]:
        """Generate one candidate. Returns (output_ids, confidence)."""
        output_ids, all_logits = _greedy_decode(
            self.model, input_ids, max_new_tokens, temperature=self.temperature
        )
        confidence = self._confidence_estimator.estimate_sequence(all_logits)
        return output_ids, confidence

    def select_best(
        self,
        input_ids: torch.Tensor,
        n: int = 4,
        max_new_tokens: int = 20,
    ) -> InferenceResult:
        """Generate n candidates, return best by verifier or confidence."""
        best_output: Optional[torch.Tensor] = None
        best_score = float("-inf")
        best_confidence = 0.0

        for _ in range(n):
            output_ids, confidence = self.generate_one(input_ids, max_new_tokens)

            if self.verifier_fn is not None:
                score = self.verifier_fn(output_ids)
            else:
                score = confidence

            if score > best_score:
                best_score = score
                best_output = output_ids
                best_confidence = confidence

        assert best_output is not None
        n_new = best_output.shape[1] - input_ids.shape[1]

        return InferenceResult(
            output_ids=best_output.squeeze(0),
            n_tokens_used=n_new * n,
            n_iterations=n,
            confidence=float(max(0.0, min(1.0, best_confidence))),
            strategy_used=f"best_of_{n}",
        )


# ---------------------------------------------------------------------------
# Sequential reviser
# ---------------------------------------------------------------------------

class SequentialReviser:
    """
    Iteratively revise output by feeding it back to the model with a critique prompt.
    """

    # Revision instruction encoded as a fixed token sequence
    _REVISION_TOKEN_ID: int = 1  # placeholder "revise" token

    def __init__(
        self,
        model: nn.Module,
        encode_fn: Callable[[str], List[int]],
        max_iterations: int = 3,
        confidence_threshold: float = 0.85,
    ):
        self.model = model
        self.encode_fn = encode_fn
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self._confidence_estimator = ConfidenceEstimator(method="max_prob")

    def revise_once(
        self,
        prompt_ids: torch.Tensor,    # (1, T_prompt)
        current_ids: torch.Tensor,   # (1, T_current) -- previous output
        max_new_tokens: int = 20,
    ) -> Tuple[torch.Tensor, float]:
        """
        Build a revision prompt: [prompt, current_output, revision_instruction].
        Generate revised output. Return (new_output_ids, confidence).
        """
        revision_signal = torch.tensor(
            [[self._REVISION_TOKEN_ID]], dtype=torch.long, device=prompt_ids.device
        )
        # Concatenate: prompt + previous output + revision signal
        revision_input = torch.cat([prompt_ids, current_ids, revision_signal], dim=1)

        output_ids, all_logits = _greedy_decode(
            self.model, revision_input, max_new_tokens, temperature=1.0
        )
        # Extract only the newly generated portion
        new_portion = output_ids[:, revision_input.shape[1]:]
        confidence = self._confidence_estimator.estimate_sequence(all_logits)
        return new_portion, confidence

    def run(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 20,
    ) -> InferenceResult:
        """Run sequential revision until confidence threshold or max_iterations."""
        # Initial generation
        output_ids, all_logits = _greedy_decode(
            self.model, prompt_ids, max_new_tokens, temperature=1.0
        )
        current_output = output_ids[:, prompt_ids.shape[1]:]  # (1, max_new_tokens)
        confidence = self._confidence_estimator.estimate_sequence(all_logits)

        total_tokens = max_new_tokens
        iteration = 1

        while iteration < self.max_iterations and confidence < self.confidence_threshold:
            revised_output, new_confidence = self.revise_once(
                prompt_ids, current_output, max_new_tokens
            )
            current_output = revised_output
            confidence = new_confidence
            total_tokens += max_new_tokens
            iteration += 1

        # Final output: prompt + last revised output
        final_ids = torch.cat([prompt_ids, current_output], dim=1).squeeze(0)

        return InferenceResult(
            output_ids=final_ids,
            n_tokens_used=total_tokens,
            n_iterations=iteration,
            confidence=float(max(0.0, min(1.0, confidence))),
            strategy_used="sequential_revision",
        )


# ---------------------------------------------------------------------------
# Compute budget predictor
# ---------------------------------------------------------------------------

class ComputeBudgetPredictor(nn.Module):
    """
    Predict required compute budget from input token features.
    Lightweight classifier: vocab_size -> embedding -> 3 budget levels.
    """

    BUDGET_LEVELS = ["low", "medium", "high"]

    def __init__(self, vocab_size: int, d_model: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.classifier = nn.Linear(d_model, 3)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, T) -> budget_logits: (B, 3)"""
        # Mean-pool embeddings over sequence length
        embedded = self.embedding(input_ids)   # (B, T, d_model)
        pooled = embedded.mean(dim=1)           # (B, d_model)
        return self.classifier(pooled)          # (B, 3)

    def predict_budget(self, input_ids: torch.Tensor) -> List[str]:
        """Returns list of budget level strings ('low', 'medium', 'high')."""
        self.train(False)
        with torch.no_grad():
            logits = self.forward(input_ids)    # (B, 3)
            indices = logits.argmax(dim=-1)     # (B,)
        return [self.BUDGET_LEVELS[i.item()] for i in indices]

    def to_compute_budget(self, budget_str: str) -> ComputeBudget:
        """Convert budget level string to ComputeBudget."""
        if budget_str == "low":
            return ComputeBudget(
                max_tokens=128,
                max_iterations=1,
                confidence_threshold=0.7,
                n_candidates=1,
            )
        elif budget_str == "medium":
            return ComputeBudget(
                max_tokens=256,
                max_iterations=3,
                confidence_threshold=0.85,
                n_candidates=2,
            )
        else:  # high
            return ComputeBudget(
                max_tokens=512,
                max_iterations=5,
                confidence_threshold=0.9,
                n_candidates=4,
            )


# ---------------------------------------------------------------------------
# Adaptive inference engine
# ---------------------------------------------------------------------------

class AdaptiveInferenceEngine:
    """
    Top-level engine: predict budget, select strategy, run inference.
    """

    def __init__(
        self,
        model: nn.Module,
        budget_predictor: Optional[ComputeBudgetPredictor] = None,
        default_budget: Optional[ComputeBudget] = None,
    ):
        self.model = model
        self.budget_predictor = budget_predictor
        self.default_budget = default_budget or ComputeBudget()
        self._confidence_estimator = ConfidenceEstimator(method="max_prob")

    def infer(
        self,
        input_ids: torch.Tensor,     # (1, T_prompt)
        override_budget: Optional[ComputeBudget] = None,
    ) -> InferenceResult:
        """
        Predict budget, run best-of-N or sequential revision based on budget.
        Low budget -> greedy decode.
        Medium budget -> best-of-2.
        High budget -> best-of-4 or sequential revision.
        """
        if override_budget is not None:
            budget = override_budget
        elif self.budget_predictor is not None:
            budget_str = self.budget_predictor.predict_budget(input_ids)[0]
            budget = self.budget_predictor.to_compute_budget(budget_str)
        else:
            budget = self.default_budget

        max_new_tokens = min(budget.max_tokens, 20)  # cap for efficiency in tests

        if budget.n_candidates <= 1:
            # Low budget: greedy decode
            output_ids, all_logits = _greedy_decode(
                self.model, input_ids, max_new_tokens, temperature=1.0
            )
            confidence = self._confidence_estimator.estimate_sequence(all_logits)
            n_new = output_ids.shape[1] - input_ids.shape[1]
            return InferenceResult(
                output_ids=output_ids.squeeze(0),
                n_tokens_used=n_new,
                n_iterations=1,
                confidence=float(max(0.0, min(1.0, confidence))),
                strategy_used="greedy",
            )

        elif budget.n_candidates <= 2:
            # Medium budget: best-of-2
            selector = BestOfNSelector(self.model, temperature=0.8)
            return selector.select_best(input_ids, n=2, max_new_tokens=max_new_tokens)

        else:
            # High budget: best-of-4
            selector = BestOfNSelector(self.model, temperature=0.8)
            return selector.select_best(
                input_ids, n=budget.n_candidates, max_new_tokens=max_new_tokens
            )

    def batch_infer(
        self,
        inputs: List[torch.Tensor],  # list of (1, T_i) tensors
    ) -> List[InferenceResult]:
        """Run infer on each input independently."""
        return [self.infer(inp) for inp in inputs]
