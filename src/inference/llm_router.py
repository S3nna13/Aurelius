"""LLM Router / Model Cascade: route queries to models based on complexity."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RouteDecision:
    model_id: str  # which model was chosen ("small", "large", etc.)
    confidence: float  # routing confidence [0, 1]
    features: dict[str, float]  # routing features used


@dataclass
class RouterConfig:
    n_models: int = 2
    model_names: list[str] = None  # default ["small", "large"]
    confidence_threshold: float = 0.7  # below this, escalate to larger model
    max_cascade_depth: int = 2  # max models to try in cascade

    def __post_init__(self):
        if self.model_names is None:
            self.model_names = ["small", "large"]


# ---------------------------------------------------------------------------
# LinearRouter
# ---------------------------------------------------------------------------


class LinearRouter(nn.Module):
    """
    Lightweight routing model: embedding + linear classifier.
    Routes input tokens to one of n_models.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_models: int = 2,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_models = n_models
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.classifier = nn.Linear(d_model, n_models)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, T) -> routing logits: (B, n_models)"""
        # Embed tokens and mean-pool over sequence dimension
        emb = self.embedding(input_ids)  # (B, T, d_model)
        pooled = emb.mean(dim=1)  # (B, d_model)
        logits = self.classifier(pooled)  # (B, n_models)
        return logits

    def route(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (model_indices: (B,), confidences: (B,))"""
        with torch.no_grad():
            logits = self.forward(input_ids)  # (B, n_models)
            probs = F.softmax(logits, dim=-1)  # (B, n_models)
            confidences, model_indices = probs.max(dim=-1)  # (B,), (B,)
        return model_indices, confidences


# ---------------------------------------------------------------------------
# QueryComplexityEstimator
# ---------------------------------------------------------------------------


class QueryComplexityEstimator:
    """
    Heuristic complexity scoring without a learned model.
    Features: token count, unique token ratio, question mark count,
    avg token id variance, nested clause depth (simple heuristic).
    """

    # Approximate token ids for common punctuation in byte-level tokenizers
    _QUESTION_MARK_IDS = {31, 63, 8263, 30}  # rough set -- tunable
    _COMMA_IDS = {11, 44, 1919}
    _SEMICOLON_IDS = {26, 59}

    def __init__(self, complexity_threshold: float = 0.5) -> None:
        self.complexity_threshold = complexity_threshold

    def extract_features(self, input_ids: list[int]) -> dict[str, float]:
        """Extract complexity features from token sequence."""
        n = len(input_ids)
        if n == 0:
            return {
                "token_count": 0.0,
                "unique_token_ratio": 0.0,
                "question_mark_count": 0.0,
                "token_id_variance": 0.0,
                "clause_depth": 0.0,
            }

        unique_count = len(set(input_ids))
        unique_ratio = unique_count / n

        # Count question-mark-like tokens (token id 63 = '?' in ASCII)
        q_count = sum(1 for t in input_ids if t in self._QUESTION_MARK_IDS)

        # Variance of token ids (high variance -> more diverse vocabulary)
        ids_t = torch.tensor(input_ids, dtype=torch.float)
        variance = ids_t.var().item() if n > 1 else 0.0

        # Simple clause depth: count commas + semicolons as clause separators
        clause_seps = sum(1 for t in input_ids if t in self._COMMA_IDS or t in self._SEMICOLON_IDS)
        clause_depth = clause_seps / max(n, 1)

        return {
            "token_count": float(n),
            "unique_token_ratio": float(unique_ratio),
            "question_mark_count": float(q_count),
            "token_id_variance": float(variance),
            "clause_depth": float(clause_depth),
        }

    def estimate_complexity(self, input_ids: list[int]) -> float:
        """Return complexity score in [0, 1]. Higher = harder query."""
        feats = self.extract_features(input_ids)
        n = feats["token_count"]

        # Normalise token count: squash with midpoint ~32 tokens
        length_score = min(n / 128.0, 1.0)

        # Unique ratio: higher diversity -> harder
        diversity_score = feats["unique_token_ratio"]

        # Question marks: more questions -> harder (cap at 1)
        q_score = min(feats["question_mark_count"] / 3.0, 1.0)

        # Variance: normalise to [0, 1] using expected max ~1e6 for vocab ids
        max_var = 1e6
        var_score = min(feats["token_id_variance"] / max_var, 1.0)

        # Clause depth already in [0, 1]
        clause_score = feats["clause_depth"]

        # Weighted combination
        score = (
            0.25 * length_score
            + 0.30 * diversity_score
            + 0.20 * q_score
            + 0.15 * var_score
            + 0.10 * clause_score
        )
        return float(max(0.0, min(1.0, score)))

    def should_use_large_model(self, input_ids: list[int]) -> bool:
        """True if complexity exceeds threshold."""
        return self.estimate_complexity(input_ids) >= self.complexity_threshold


# ---------------------------------------------------------------------------
# CascadeRouter
# ---------------------------------------------------------------------------


class CascadeRouter:
    """
    Try models in order (small -> large). Use large model if small model's
    confidence is below threshold.
    """

    def __init__(
        self,
        models: dict[str, nn.Module],  # name -> model
        config: RouterConfig = None,
        confidence_fn: Callable | None = None,  # custom confidence scorer
    ) -> None:
        self.models = models
        self.config = config or RouterConfig()
        self.confidence_fn = confidence_fn

        # Build ordered list: respect config.model_names order when possible
        cfg_names = self.config.model_names or list(models.keys())
        # Keep only names that actually exist in models dict
        self._order = [n for n in cfg_names if n in models]
        # Append any extra models not in config order
        for name in models:
            if name not in self._order:
                self._order.append(name)

    def _get_confidence(
        self,
        logits: torch.Tensor,  # (B, T, vocab)
    ) -> torch.Tensor:
        """Compute per-sample confidence: mean of top-1 softmax prob. Returns (B,)."""
        probs = F.softmax(logits, dim=-1)  # (B, T, vocab)
        top1_probs = probs.max(dim=-1).values  # (B, T)
        confidence = top1_probs.mean(dim=-1)  # (B,)
        return confidence

    def decode(
        self,
        input_ids: torch.Tensor,  # (B, T_prompt)
        max_new_tokens: int = 5,
    ) -> tuple[torch.Tensor, list[RouteDecision]]:
        """
        Cascade decode: try small -> large if needed.
        Returns (output_ids, route_decisions per sample).
        """
        B = input_ids.shape[0]
        threshold = self.config.confidence_threshold
        max_depth = min(self.config.max_cascade_depth, len(self._order))

        decisions: list[RouteDecision | None] = [None] * B
        pending = list(range(B))

        last_output_ids = None

        with torch.no_grad():
            for depth, model_name in enumerate(self._order[:max_depth]):
                if not pending:
                    break

                model = self.models[model_name]
                model.train(False)

                # Run model to get initial logits
                _, logits, _ = model(input_ids)  # (B, T, vocab)

                # Generate greedily for max_new_tokens
                generated = input_ids.clone()
                current_logits = logits
                for _ in range(max_new_tokens):
                    next_token = current_logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
                    _, current_logits, _ = model(generated)

                last_output_ids = generated

                # Confidence for pending samples
                conf = self._get_confidence(logits)  # (B,)

                still_pending = []
                for i in pending:
                    c = conf[i].item()
                    is_last = depth == max_depth - 1
                    if c >= threshold or is_last:
                        decisions[i] = RouteDecision(
                            model_id=model_name,
                            confidence=float(max(0.0, min(1.0, c))),
                            features={"depth": float(depth), "confidence": float(c)},
                        )
                    else:
                        still_pending.append(i)

                pending = still_pending

        # Fallback: any remaining pending get last model
        last_name = self._order[min(max_depth - 1, len(self._order) - 1)]
        for i in pending:
            decisions[i] = RouteDecision(
                model_id=last_name,
                confidence=0.0,
                features={"depth": float(max_depth - 1), "confidence": 0.0},
            )

        return last_output_ids, decisions  # type: ignore[return-value]

    def route_only(self, input_ids: torch.Tensor) -> list[RouteDecision]:
        """Just return routing decisions without decoding."""
        B = input_ids.shape[0]
        threshold = self.config.confidence_threshold
        max_depth = min(self.config.max_cascade_depth, len(self._order))

        decisions: list[RouteDecision | None] = [None] * B
        pending = list(range(B))

        with torch.no_grad():
            for depth, model_name in enumerate(self._order[:max_depth]):
                if not pending:
                    break

                model = self.models[model_name]
                model.train(False)

                _, logits, _ = model(input_ids)  # (B, T, vocab)
                conf = self._get_confidence(logits)  # (B,)

                still_pending = []
                for i in pending:
                    c = conf[i].item()
                    is_last = depth == max_depth - 1
                    if c >= threshold or is_last:
                        decisions[i] = RouteDecision(
                            model_id=model_name,
                            confidence=float(max(0.0, min(1.0, c))),
                            features={"depth": float(depth), "confidence": float(c)},
                        )
                    else:
                        still_pending.append(i)

                pending = still_pending

        last_name = self._order[min(max_depth - 1, len(self._order) - 1)]
        for i in pending:
            decisions[i] = RouteDecision(
                model_id=last_name,
                confidence=0.0,
                features={"depth": float(max_depth - 1), "confidence": 0.0},
            )

        return decisions  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# RouterTrainer
# ---------------------------------------------------------------------------


class RouterTrainer:
    """Train a LinearRouter on (input_ids, model_label) pairs."""

    def __init__(
        self,
        router: LinearRouter,
        lr: float = 1e-3,
    ) -> None:
        self.router = router
        self.optimizer = torch.optim.Adam(router.parameters(), lr=lr)

    def train_step(
        self,
        input_ids: torch.Tensor,  # (B, T)
        labels: torch.Tensor,  # (B,) int -- which model index is correct
    ) -> float:
        """Cross-entropy step. Returns loss value."""
        self.router.train()
        self.optimizer.zero_grad()
        logits = self.router(input_ids)  # (B, n_models)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Return routing accuracy."""
        self.router.train(False)
        with torch.no_grad():
            logits = self.router(input_ids)  # (B, n_models)
            preds = logits.argmax(dim=-1)  # (B,)
            accuracy = (preds == labels).float().mean().item()
        return float(accuracy)
