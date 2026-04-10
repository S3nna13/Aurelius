"""Feature-level knowledge distillation with attention transfer and intermediate feature matching.

Implements:
- FeatDistillConfig: configuration dataclass
- FeatureAdapter: projects student features to teacher space
- extract_features: hook-based hidden state extraction per layer
- compute_feature_loss: MSE loss between student and teacher features at mapped layers
- compute_attention_transfer_loss: Attention Transfer (AT) loss from hidden states
- soft_label_loss: KL divergence with temperature scaling
- FeatureDistillTrainer: full training loop combining all losses
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FeatDistillConfig:
    """Configuration for feature-level knowledge distillation."""

    temperature: float = 4.0
    alpha: float = 0.5
    feature_loss_weight: float = 0.1
    attention_loss_weight: float = 0.1
    layer_mapping: list[tuple[int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# FeatureAdapter
# ---------------------------------------------------------------------------

class FeatureAdapter(nn.Module):
    """Project student features into teacher feature space.

    A single linear layer (no bias) from student_dim to teacher_dim.
    Used when the student and teacher have different hidden dimensions.

    Args:
        student_dim: Student hidden dimension.
        teacher_dim: Teacher hidden dimension.
    """

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Project x from student_dim to teacher_dim.

        Args:
            x: (B, T, student_dim)

        Returns:
            (B, T, teacher_dim)
        """
        return self.linear(x)


# ---------------------------------------------------------------------------
# Hook-based feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    model: nn.Module,
    input_ids: Tensor,
    layer_indices: list[int],
) -> dict[int, Tensor]:
    """Extract hidden states at specified transformer layers via forward hooks.

    Registers temporary forward hooks on model.layers[i] for each i in
    layer_indices. Handles layers that return (hidden, kv_cache) tuples by
    capturing output[0].

    Args:
        model: AureliusTransformer with a .layers ModuleList.
        input_ids: (B, T) token ids.
        layer_indices: Layer indices to capture.

    Returns:
        Dict mapping layer index → (B, T, D) hidden state tensor.
    """
    hidden_states: dict[int, Tensor] = {}
    hooks = []

    def make_hook(idx: int):
        def hook(module: nn.Module, inp: tuple, output) -> None:
            if isinstance(output, (tuple, list)):
                h = output[0]
            else:
                h = output
            hidden_states[idx] = h.detach()
        return hook

    for idx in layer_indices:
        handle = model.layers[idx].register_forward_hook(make_hook(idx))
        hooks.append(handle)

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for handle in hooks:
            handle.remove()

    return hidden_states


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def compute_feature_loss(
    student_feats: dict[int, Tensor],
    teacher_feats: dict[int, Tensor],
    layer_mapping: list[tuple[int, int]],
    adapters: dict | None = None,
) -> Tensor:
    """MSE loss between student and teacher features at mapped layer pairs.

    For each (student_layer, teacher_layer) pair in layer_mapping, computes
    MSE between the corresponding feature tensors. If adapters is provided,
    the student feature is first projected via adapters[student_layer].

    Args:
        student_feats: Dict of {student_layer_idx: (B, T, Ds)} tensors.
        teacher_feats: Dict of {teacher_layer_idx: (B, T, Dt)} tensors.
        layer_mapping: List of (student_layer, teacher_layer) index pairs.
        adapters: Optional dict of {student_layer_idx: FeatureAdapter}.

    Returns:
        Scalar MSE loss (mean over valid pairs). Zero tensor if no valid pairs.
    """
    device = next(iter(student_feats.values())).device if student_feats else torch.tensor(0.0).device
    total = torch.zeros(1, device=device).squeeze()
    n_pairs = 0

    for s_idx, t_idx in layer_mapping:
        s_feat = student_feats.get(s_idx)
        t_feat = teacher_feats.get(t_idx)
        if s_feat is None or t_feat is None:
            continue

        if adapters is not None and s_idx in adapters:
            s_feat = adapters[s_idx](s_feat)

        total = total + F.mse_loss(s_feat, t_feat.detach())
        n_pairs += 1

    if n_pairs > 0:
        total = total / n_pairs

    return total


def compute_attention_transfer_loss(
    student_hidden: Tensor,
    teacher_hidden: Tensor,
) -> Tensor:
    """Attention Transfer (AT) loss between hidden state feature maps.

    Computes spatial attention maps by averaging over the feature dimension D,
    then normalizes each map and computes MSE between student and teacher maps.

    student_hidden shape: (B, T, Ds)
    teacher_hidden shape: (B, T, Dt)

    Steps:
        1. Compute attention map per sample: mean over D → (B, T)
        2. Normalize each (B, T) map with L2 norm over T dimension
        3. MSE between student and teacher normalized maps

    Args:
        student_hidden: (B, T, Ds) student hidden states.
        teacher_hidden: (B, T, Dt) teacher hidden states.

    Returns:
        Scalar MSE loss.
    """
    # Attention map: mean over feature dimension → (B, T)
    s_attn = student_hidden.mean(dim=-1)   # (B, T)
    t_attn = teacher_hidden.mean(dim=-1)   # (B, T)

    # L2 normalize over T dimension
    s_norm = F.normalize(s_attn, p=2, dim=-1)  # (B, T)
    t_norm = F.normalize(t_attn, p=2, dim=-1)  # (B, T)

    return F.mse_loss(s_norm, t_norm.detach())


def soft_label_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float,
) -> Tensor:
    """KL divergence loss with temperature scaling.

    Computes KL(teacher_soft || student_soft) scaled by T^2, following
    the Hinton et al. convention.

    Args:
        student_logits: (B, T, V) student raw logits.
        teacher_logits: (B, T, V) teacher raw logits.
        temperature: Softening temperature T.

    Returns:
        Scalar KL divergence loss scaled by T^2.
    """
    B, S, V = student_logits.shape
    student_logits_flat = student_logits.reshape(-1, V)
    teacher_logits_flat = teacher_logits.reshape(-1, V)

    T = temperature
    student_log_soft = F.log_softmax(student_logits_flat / T, dim=-1)
    teacher_soft = F.softmax(teacher_logits_flat.detach() / T, dim=-1)

    kl = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean")
    return kl * (T ** 2)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class FeatureDistillTrainer:
    """Trains a student model using feature-level knowledge distillation.

    Combines:
    - Task loss: cross-entropy on student logits (next-token prediction)
    - Feature loss: MSE between student and teacher hidden states at mapped layers
    - Attention Transfer loss: AT loss on the first mapped layer pair's features
    - KL loss: soft label distillation with temperature scaling

    Total loss = alpha * kl_loss + (1 - alpha) * task_loss
               + feature_loss_weight * feature_loss
               + attention_loss_weight * attention_loss

    Args:
        student: Student model with .layers ModuleList.
        teacher: Teacher model (frozen on init).
        optimizer: Optimizer for student parameters.
        config: FeatDistillConfig.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: FeatDistillConfig,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.config = config

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        # Build adapters: one FeatureAdapter per layer pair
        # Detect dimensions by probing the student and teacher
        self._adapters: nn.ModuleDict = nn.ModuleDict()
        self._build_adapters()

    def _build_adapters(self) -> None:
        """Build FeatureAdapters for each layer pair in layer_mapping."""
        if not self.config.layer_mapping:
            return

        # Probe model hidden dims with a dummy forward
        device = next(self.student.parameters()).device
        dummy = torch.zeros(1, 4, dtype=torch.long, device=device)

        s_indices = [s for s, _ in self.config.layer_mapping]
        t_indices = [t for _, t in self.config.layer_mapping]

        with torch.no_grad():
            s_feats = {}
            s_hooks = []

            def make_s_hook(idx: int):
                def hook(module, inp, output):
                    h = output[0] if isinstance(output, (tuple, list)) else output
                    s_feats[idx] = h
                return hook

            for idx in set(s_indices):
                s_hooks.append(self.student.layers[idx].register_forward_hook(make_s_hook(idx)))
            try:
                self.student(dummy)
            finally:
                for h in s_hooks:
                    h.remove()

            t_feats = {}
            t_hooks = []

            def make_t_hook(idx: int):
                def hook(module, inp, output):
                    h = output[0] if isinstance(output, (tuple, list)) else output
                    t_feats[idx] = h
                return hook

            for idx in set(t_indices):
                t_hooks.append(self.teacher.layers[idx].register_forward_hook(make_t_hook(idx)))
            try:
                self.teacher(dummy)
            finally:
                for h in t_hooks:
                    h.remove()

        for s_idx, t_idx in self.config.layer_mapping:
            if s_idx in s_feats and t_idx in t_feats:
                s_dim = s_feats[s_idx].shape[-1]
                t_dim = t_feats[t_idx].shape[-1]
                if s_dim != t_dim:
                    self._adapters[str(s_idx)] = FeatureAdapter(s_dim, t_dim)

    def _get_features_and_logits(
        self, model: nn.Module, input_ids: Tensor, with_grad: bool = False
    ) -> tuple[dict[int, Tensor], Tensor]:
        """Extract features and logits from the model.

        Args:
            model: The model to run.
            input_ids: (B, T) input token ids.
            with_grad: If True, run without torch.no_grad context.

        Returns:
            Tuple of (features dict, logits tensor).
        """
        indices = list({idx for pair in self.config.layer_mapping for idx in pair})
        s_indices = [s for s, _ in self.config.layer_mapping]
        t_indices = [t for _, t in self.config.layer_mapping]

        hidden_states: dict[int, Tensor] = {}
        hooks = []

        def make_hook(idx: int):
            def hook(module, inp, output):
                h = output[0] if isinstance(output, (tuple, list)) else output
                hidden_states[idx] = h
            return hook

        # Decide which indices to hook based on the model
        if model is self.student:
            hook_indices = s_indices
        else:
            hook_indices = t_indices

        for idx in hook_indices:
            handle = model.layers[idx].register_forward_hook(make_hook(idx))
            hooks.append(handle)

        try:
            if with_grad:
                _, logits, _ = model(input_ids)
            else:
                with torch.no_grad():
                    _, logits, _ = model(input_ids)
        finally:
            for handle in hooks:
                handle.remove()

        return hidden_states, logits

    def train_step(self, input_ids: Tensor) -> dict:
        """One feature distillation training step.

        Computes:
        1. Student and teacher features at mapped layers
        2. Feature loss (MSE between feature maps)
        3. Attention transfer loss (AT loss)
        4. Soft label KL loss
        5. Task loss (student cross-entropy, next-token prediction)

        Combines losses with config weights, runs backward, steps optimizer.

        Args:
            input_ids: (B, T) input token ids.

        Returns:
            Dict with keys: "loss", "feature_loss", "attention_loss",
            "kl_loss", "task_loss" (all float).
        """
        self.student.train()
        self.optimizer.zero_grad()

        # --- Teacher forward (no grad) ---
        teacher_feats: dict[int, Tensor] = {}
        teacher_hooks = []
        t_indices = [t for _, t in self.config.layer_mapping]

        def make_t_hook(idx: int):
            def hook(module, inp, output):
                h = output[0] if isinstance(output, (tuple, list)) else output
                teacher_feats[idx] = h.detach()
            return hook

        for idx in t_indices:
            handle = self.teacher.layers[idx].register_forward_hook(make_t_hook(idx))
            teacher_hooks.append(handle)

        with torch.no_grad():
            try:
                _, teacher_logits, _ = self.teacher(input_ids)
            finally:
                for handle in teacher_hooks:
                    handle.remove()

        # --- Student forward (with grad) ---
        student_feats: dict[int, Tensor] = {}
        student_hooks = []
        s_indices = [s for s, _ in self.config.layer_mapping]

        def make_s_hook(idx: int):
            def hook(module, inp, output):
                h = output[0] if isinstance(output, (tuple, list)) else output
                student_feats[idx] = h
            return hook

        for idx in s_indices:
            handle = self.student.layers[idx].register_forward_hook(make_s_hook(idx))
            student_hooks.append(handle)

        try:
            _, student_logits, _ = self.student(input_ids)
        finally:
            for handle in student_hooks:
                handle.remove()

        B, T, V = student_logits.shape

        # Build adapters dict for compute_feature_loss
        adapters_for_loss = {}
        for s_idx, _ in self.config.layer_mapping:
            key = str(s_idx)
            if key in self._adapters:
                adapters_for_loss[s_idx] = self._adapters[key]

        # Feature loss
        feature_loss = compute_feature_loss(
            student_feats,
            teacher_feats,
            self.config.layer_mapping,
            adapters=adapters_for_loss if adapters_for_loss else None,
        )

        # Attention transfer loss — use first mapped pair
        attention_loss = torch.zeros(1, device=input_ids.device).squeeze()
        if self.config.layer_mapping:
            s_idx_first, t_idx_first = self.config.layer_mapping[0]
            s_h = student_feats.get(s_idx_first)
            t_h = teacher_feats.get(t_idx_first)
            if s_h is not None and t_h is not None:
                # If dims differ and adapter exists, apply adapter first
                s_h_for_attn = s_h
                adapter_key = str(s_idx_first)
                if adapter_key in self._adapters:
                    s_h_for_attn = self._adapters[adapter_key](s_h)
                attention_loss = compute_attention_transfer_loss(s_h_for_attn, t_h)

        # KL loss (soft labels)
        kl_loss = soft_label_loss(student_logits, teacher_logits, self.config.temperature)

        # Task loss: CE on next-token prediction
        task_loss = F.cross_entropy(
            student_logits[:, :-1].contiguous().view(-1, V),
            input_ids[:, 1:].contiguous().view(-1),
        )

        # Combine losses
        alpha = self.config.alpha
        total_loss = (
            (1.0 - alpha) * task_loss
            + alpha * kl_loss
            + self.config.feature_loss_weight * feature_loss
            + self.config.attention_loss_weight * attention_loss
        )

        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "feature_loss": feature_loss.item(),
            "attention_loss": attention_loss.item(),
            "kl_loss": kl_loss.item(),
            "task_loss": task_loss.item(),
        }

    def evaluate(self, input_ids: Tensor) -> dict:
        """Compute distillation metrics without updating weights.

        Args:
            input_ids: (B, T) input token ids.

        Returns:
            Dict with keys: "loss", "feature_loss", "attention_loss",
            "kl_loss", "task_loss" (all float).
        """
        self.student.eval()

        with torch.no_grad():
            # Teacher features
            teacher_feats: dict[int, Tensor] = {}
            teacher_hooks = []
            t_indices = [t for _, t in self.config.layer_mapping]

            def make_t_hook(idx: int):
                def hook(module, inp, output):
                    h = output[0] if isinstance(output, (tuple, list)) else output
                    teacher_feats[idx] = h
                return hook

            for idx in t_indices:
                handle = self.teacher.layers[idx].register_forward_hook(make_t_hook(idx))
                teacher_hooks.append(handle)

            try:
                _, teacher_logits, _ = self.teacher(input_ids)
            finally:
                for handle in teacher_hooks:
                    handle.remove()

            # Student features
            student_feats: dict[int, Tensor] = {}
            student_hooks = []
            s_indices = [s for s, _ in self.config.layer_mapping]

            def make_s_hook(idx: int):
                def hook(module, inp, output):
                    h = output[0] if isinstance(output, (tuple, list)) else output
                    student_feats[idx] = h
                return hook

            for idx in s_indices:
                handle = self.student.layers[idx].register_forward_hook(make_s_hook(idx))
                student_hooks.append(handle)

            try:
                _, student_logits, _ = self.student(input_ids)
            finally:
                for handle in student_hooks:
                    handle.remove()

            B, T, V = student_logits.shape

            adapters_for_loss = {}
            for s_idx, _ in self.config.layer_mapping:
                key = str(s_idx)
                if key in self._adapters:
                    adapters_for_loss[s_idx] = self._adapters[key]

            feature_loss = compute_feature_loss(
                student_feats,
                teacher_feats,
                self.config.layer_mapping,
                adapters=adapters_for_loss if adapters_for_loss else None,
            )

            attention_loss = torch.zeros(1, device=input_ids.device).squeeze()
            if self.config.layer_mapping:
                s_idx_first, t_idx_first = self.config.layer_mapping[0]
                s_h = student_feats.get(s_idx_first)
                t_h = teacher_feats.get(t_idx_first)
                if s_h is not None and t_h is not None:
                    s_h_for_attn = s_h
                    adapter_key = str(s_idx_first)
                    if adapter_key in self._adapters:
                        s_h_for_attn = self._adapters[adapter_key](s_h)
                    attention_loss = compute_attention_transfer_loss(s_h_for_attn, t_h)

            kl_loss = soft_label_loss(student_logits, teacher_logits, self.config.temperature)

            task_loss = F.cross_entropy(
                student_logits[:, :-1].contiguous().view(-1, V),
                input_ids[:, 1:].contiguous().view(-1),
            )

            alpha = self.config.alpha
            total_loss = (
                (1.0 - alpha) * task_loss
                + alpha * kl_loss
                + self.config.feature_loss_weight * feature_loss
                + self.config.attention_loss_weight * attention_loss
            )

        return {
            "loss": total_loss.item(),
            "feature_loss": feature_loss.item(),
            "attention_loss": attention_loss.item(),
            "kl_loss": kl_loss.item(),
            "task_loss": task_loss.item(),
        }
