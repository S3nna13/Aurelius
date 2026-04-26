"""Safety classifier: lightweight binary classifier on top of transformer hidden states.

Architecture: mean-pool hidden states from model.norm -> Linear(d_model, 1) -> sigmoid.
Trained with BCE loss on (text, safe/unsafe) labeled pairs.
At inference: classify a generated response before returning it to the user.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class SafetyConfig:
    d_model: int = 2048
    threshold: float = 0.5  # probability threshold for "unsafe" classification
    freeze_backbone: bool = True  # if True, only train the safety head (not backbone)
    n_epochs: int = 5
    lr: float = 1e-3


class SafetyClassifier(nn.Module):
    """Binary safety classifier using transformer hidden states.

    Hooks into backbone.norm (final layer norm) to extract hidden states,
    applies mean pooling over sequence, then a linear head.

    Args:
        backbone: AureliusTransformer
        cfg: SafetyConfig

    Labels: 1 = unsafe, 0 = safe (consistent with safety literature)
    """

    def __init__(self, backbone: nn.Module, cfg: SafetyConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.safety_head = nn.Linear(cfg.d_model, 1, bias=True)
        nn.init.zeros_(self.safety_head.bias)
        nn.init.normal_(self.safety_head.weight, std=0.01)

        if cfg.freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, S)
        attention_mask: torch.Tensor | None = None,  # (B, S), 1=valid, 0=pad
    ) -> torch.Tensor:
        """Returns safety logits of shape (B,).

        Higher logit = more likely unsafe.
        """
        # Hook into backbone.norm to capture hidden states after final layer norm
        hidden: list[torch.Tensor] = []

        def _hook(m, i, o):
            hidden.append(o)

        hook = self.backbone.norm.register_forward_hook(_hook)
        try:
            _, _, _ = self.backbone(input_ids)
        finally:
            hook.remove()

        h = hidden[0]  # (B, S, D)

        # Mean pooling (with mask if provided)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
            h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)  # (B, D)

        return self.safety_head(h).squeeze(-1)  # (B,)

    def predict(
        self,
        input_ids: torch.Tensor,  # (B, S)
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (is_unsafe: BoolTensor (B,), probability: FloatTensor (B,))."""
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        return probs >= self.cfg.threshold, probs

    def safety_loss(
        self,
        logits: torch.Tensor,  # (B,)
        labels: torch.Tensor,  # (B,) - 1=unsafe, 0=safe
    ) -> torch.Tensor:
        """Binary cross-entropy loss."""
        return F.binary_cross_entropy_with_logits(logits, labels.float())


class SafetyTrainer:
    """Trains the safety classifier on labeled examples."""

    def __init__(self, classifier: SafetyClassifier) -> None:
        self.classifier = classifier
        self.optimizer = torch.optim.AdamW(
            [p for p in classifier.parameters() if p.requires_grad],
            lr=classifier.cfg.lr,
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """One epoch over dataloader yielding {"input_ids", "labels"} dicts.

        Returns mean loss.
        """
        self.classifier.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch.get("attention_mask", None)

            self.optimizer.zero_grad()
            logits = self.classifier(input_ids, attention_mask)
            loss = self.classifier.safety_loss(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.classifier.eval()
        return total_loss / max(n_batches, 1)

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Returns {"loss": float, "accuracy": float, "precision": float, "recall": float}."""
        self.classifier.eval()
        total_loss = 0.0
        n_batches = 0

        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attention_mask = batch.get("attention_mask", None)

                logits = self.classifier(input_ids, attention_mask)
                loss = self.classifier.safety_loss(logits, labels)
                total_loss += loss.item()
                n_batches += 1

                probs = torch.sigmoid(logits)
                preds = (probs >= self.classifier.cfg.threshold).long()
                all_preds.append(preds)
                all_labels.append(labels.long())

        mean_loss = total_loss / max(n_batches, 1)

        preds_cat = torch.cat(all_preds)  # (N,)
        labels_cat = torch.cat(all_labels)  # (N,)

        accuracy = (preds_cat == labels_cat).float().mean().item()

        # Precision = TP / (TP + FP), Recall = TP / (TP + FN)
        tp = ((preds_cat == 1) & (labels_cat == 1)).sum().float()
        fp = ((preds_cat == 1) & (labels_cat == 0)).sum().float()
        fn = ((preds_cat == 0) & (labels_cat == 1)).sum().float()

        precision = (tp / (tp + fp).clamp(min=1e-8)).item()
        recall = (tp / (tp + fn).clamp(min=1e-8)).item()

        return {
            "loss": mean_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }


def safety_filter(
    classifier: SafetyClassifier,
    input_ids: torch.Tensor,  # (B, S)
    generated_ids: torch.Tensor,  # (B, S_gen)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter generated responses through safety classifier.

    Concatenates prompt + response and classifies.

    Returns:
        (safe_mask: BoolTensor (B,), probs: FloatTensor (B,))
        safe_mask[i] = True means example i is safe (can be returned to user)
    """
    full_ids = torch.cat([input_ids, generated_ids], dim=1)
    is_unsafe, probs = classifier.predict(full_ids)
    return ~is_unsafe, probs
