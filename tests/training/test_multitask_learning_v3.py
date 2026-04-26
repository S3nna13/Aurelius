"""
Tests for multitask_learning_v3: shared backbone, task heads,
gradient surgery, dynamic task weighting, and MTLTrainer.
"""

import math

import pytest
import torch

from src.training.multitask_learning_v3 import (
    DynamicTaskWeighter,
    GradientSurgery,
    MTLConfig,
    MTLModel,
    MTLTrainer,
    SharedBackbone,
    TaskHead,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D_MODEL = 16
VOCAB_SIZE = 16
N_LAYERS = 2
B = 2
T = 4
N_TASKS = 2


def make_backbone() -> SharedBackbone:
    return SharedBackbone(d_model=D_MODEL, vocab_size=VOCAB_SIZE, n_layers=N_LAYERS)


def make_input() -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (B, T))


# ---------------------------------------------------------------------------
# TaskHead tests
# ---------------------------------------------------------------------------


def test_taskhead_lm_loss_finite_scalar():
    head = TaskHead(d_model=D_MODEL, output_size=VOCAB_SIZE, task_type="lm")
    # For lm, features are [B, T, d_model], targets are [B, T] of token ids
    features = torch.randn(B, T, D_MODEL)
    targets = torch.randint(0, VOCAB_SIZE, (B, T))
    loss = head.compute_loss(features, targets)
    assert loss.dim() == 0, "loss must be a scalar tensor"
    assert math.isfinite(loss.item()), "lm loss must be finite"


def test_taskhead_classification_loss_finite_scalar():
    n_classes = 5
    head = TaskHead(d_model=D_MODEL, output_size=n_classes, task_type="classification")
    features = torch.randn(B, D_MODEL)
    targets = torch.randint(0, n_classes, (B,))
    loss = head.compute_loss(features, targets)
    assert loss.dim() == 0
    assert math.isfinite(loss.item()), "classification loss must be finite"


def test_taskhead_regression_loss_finite_scalar():
    head = TaskHead(d_model=D_MODEL, output_size=1, task_type="regression")
    features = torch.randn(B, D_MODEL)
    targets = torch.randn(B)
    loss = head.compute_loss(features, targets)
    assert loss.dim() == 0
    assert math.isfinite(loss.item()), "regression loss must be finite"


def test_taskhead_invalid_type_raises():
    with pytest.raises(ValueError):
        TaskHead(d_model=D_MODEL, output_size=4, task_type="invalid")


# ---------------------------------------------------------------------------
# SharedBackbone tests
# ---------------------------------------------------------------------------


def test_shared_backbone_token_repr_shape():
    backbone = make_backbone()
    input_ids = make_input()
    token_repr, _ = backbone(input_ids)
    assert token_repr.shape == (B, T, D_MODEL), (
        f"Expected token_repr shape ({B}, {T}, {D_MODEL}), got {token_repr.shape}"
    )


def test_shared_backbone_pooled_shape():
    backbone = make_backbone()
    input_ids = make_input()
    _, pooled = backbone(input_ids)
    assert pooled.shape == (B, D_MODEL), (
        f"Expected pooled shape ({B}, {D_MODEL}), got {pooled.shape}"
    )


def test_shared_backbone_output_finite():
    backbone = make_backbone()
    input_ids = make_input()
    token_repr, pooled = backbone(input_ids)
    assert torch.all(torch.isfinite(token_repr)), "token_repr must be finite"
    assert torch.all(torch.isfinite(pooled)), "pooled must be finite"


# ---------------------------------------------------------------------------
# MTLModel tests
# ---------------------------------------------------------------------------


def _make_mtl_model():
    backbone = make_backbone()
    task_heads = {
        "lm_task": TaskHead(D_MODEL, VOCAB_SIZE, "lm"),
        "cls_task": TaskHead(D_MODEL, 3, "classification"),
    }
    return MTLModel(backbone=backbone, task_heads=task_heads)


def test_mtlmodel_forward_returns_features_and_logits():
    model = _make_mtl_model()
    input_ids = make_input()
    features, logits = model(input_ids, "lm_task")
    # For lm task, features should be [B, T, d_model]
    assert features.shape == (B, T, D_MODEL)
    assert logits.shape == (B, T, VOCAB_SIZE)


def test_mtlmodel_forward_cls_task():
    model = _make_mtl_model()
    input_ids = make_input()
    features, logits = model(input_ids, "cls_task")
    assert features.shape == (B, D_MODEL)
    assert logits.shape == (B, 3)


def test_mtlmodel_compute_all_losses_returns_dict_with_task_keys():
    model = _make_mtl_model()
    input_ids = make_input()
    targets_dict = {
        "lm_task": torch.randint(0, VOCAB_SIZE, (B, T)),
        "cls_task": torch.randint(0, 3, (B,)),
    }
    losses = model.compute_all_losses(input_ids, targets_dict)
    assert set(losses.keys()) == {"lm_task", "cls_task"}, f"Keys mismatch: {losses.keys()}"


def test_mtlmodel_compute_all_losses_all_finite():
    model = _make_mtl_model()
    input_ids = make_input()
    targets_dict = {
        "lm_task": torch.randint(0, VOCAB_SIZE, (B, T)),
        "cls_task": torch.randint(0, 3, (B,)),
    }
    losses = model.compute_all_losses(input_ids, targets_dict)
    for task_name, loss in losses.items():
        assert loss.dim() == 0, f"Loss for {task_name} must be scalar"
        assert math.isfinite(loss.item()), f"Loss for {task_name} must be finite"


# ---------------------------------------------------------------------------
# GradientSurgery tests
# ---------------------------------------------------------------------------


def _make_surgery_setup():
    model = _make_mtl_model()
    surgery = GradientSurgery(model)
    input_ids = make_input()
    targets_dict = {
        "lm_task": torch.randint(0, VOCAB_SIZE, (B, T)),
        "cls_task": torch.randint(0, 3, (B,)),
    }
    losses = model.compute_all_losses(input_ids, targets_dict)
    params = [p for p in model.backbone.parameters() if p.requires_grad]
    return surgery, losses, params


def test_gradient_surgery_compute_task_gradients_returns_grad_list_per_task():
    surgery, losses, params = _make_surgery_setup()
    task_grads = surgery.compute_task_gradients(losses, params)
    assert set(task_grads.keys()) == set(losses.keys())
    for task_name, grads in task_grads.items():
        assert len(grads) == len(params), (
            f"Task {task_name}: expected {len(params)} grad tensors, got {len(grads)}"
        )
        for g, p in zip(grads, params):
            assert g.shape == p.shape, "Gradient shape must match param shape"


def test_gradient_surgery_project_conflicting_returns_same_n_tensors():
    surgery, losses, params = _make_surgery_setup()
    task_grads = surgery.compute_task_gradients(losses, params)
    merged = surgery.project_conflicting(task_grads)
    assert len(merged) == len(params), f"Expected {len(params)} merged tensors, got {len(merged)}"


def test_gradient_surgery_cosine_similarity_in_range():
    surgery, losses, params = _make_surgery_setup()
    task_grads = surgery.compute_task_gradients(losses, params)
    task_names = list(task_grads.keys())
    cos_sim = surgery.cosine_similarity_grads(task_grads[task_names[0]], task_grads[task_names[1]])
    assert -1.0 - 1e-5 <= cos_sim <= 1.0 + 1e-5, f"Cosine similarity {cos_sim} out of [-1, 1]"


# ---------------------------------------------------------------------------
# DynamicTaskWeighter tests
# ---------------------------------------------------------------------------


def test_dynamic_task_weighter_uncertainty_forward_is_scalar():
    weighter = DynamicTaskWeighter(n_tasks=N_TASKS, method="uncertainty")
    losses = [torch.tensor(1.0), torch.tensor(0.5)]
    total = weighter(losses)
    assert total.dim() == 0, "Weighted loss must be scalar"
    assert math.isfinite(total.item()), "Weighted loss must be finite"


def test_dynamic_task_weighter_equal_is_mean():
    weighter = DynamicTaskWeighter(n_tasks=N_TASKS, method="equal")
    l1, l2 = torch.tensor(2.0), torch.tensor(4.0)
    total = weighter([l1, l2])
    expected = (l1 + l2) / 2.0
    assert abs(total.item() - expected.item()) < 1e-5, (
        f"Equal weighting should give mean: expected {expected.item()}, got {total.item()}"
    )


def test_dynamic_task_weighter_get_weights_all_positive():
    weighter = DynamicTaskWeighter(n_tasks=N_TASKS, method="uncertainty")
    weights = weighter.get_weights()
    assert weights.shape == (N_TASKS,)
    assert torch.all(weights > 0).item(), "All weights must be positive (exp is always > 0)"


def test_dynamic_task_weighter_gradnorm_forward_is_scalar():
    weighter = DynamicTaskWeighter(n_tasks=N_TASKS, method="gradnorm")
    losses = [torch.tensor(0.8), torch.tensor(1.2)]
    total = weighter(losses)
    assert total.dim() == 0
    assert math.isfinite(total.item())


# ---------------------------------------------------------------------------
# MTLTrainer tests
# ---------------------------------------------------------------------------


def test_mtl_trainer_train_step_returns_finite_total_loss():
    backbone = make_backbone()
    task_heads = {
        "lm_task": TaskHead(D_MODEL, VOCAB_SIZE, "lm"),
        "cls_task": TaskHead(D_MODEL, 3, "classification"),
    }
    model = MTLModel(backbone=backbone, task_heads=task_heads)
    weighter = DynamicTaskWeighter(n_tasks=N_TASKS, method="uncertainty")
    trainer = MTLTrainer(model=model, weighter=weighter, lr=1e-4)

    input_ids = make_input()
    targets_dict = {
        "lm_task": torch.randint(0, VOCAB_SIZE, (B, T)),
        "cls_task": torch.randint(0, 3, (B,)),
    }
    total_loss, per_task_losses = trainer.train_step(input_ids, targets_dict)

    assert math.isfinite(total_loss), f"total_loss is not finite: {total_loss}"
    assert isinstance(per_task_losses, dict)
    for task_name, loss_val in per_task_losses.items():
        assert math.isfinite(loss_val), f"Per-task loss for {task_name} is not finite"


def test_mtl_trainer_get_task_weights_returns_dict():
    backbone = make_backbone()
    task_heads = {
        "lm_task": TaskHead(D_MODEL, VOCAB_SIZE, "lm"),
        "cls_task": TaskHead(D_MODEL, 3, "classification"),
    }
    model = MTLModel(backbone=backbone, task_heads=task_heads)
    weighter = DynamicTaskWeighter(n_tasks=N_TASKS, method="uncertainty")
    trainer = MTLTrainer(model=model, weighter=weighter, lr=1e-4)
    weights = trainer.get_task_weights()
    assert isinstance(weights, dict)
    for val in weights.values():
        assert math.isfinite(val)
        assert val > 0.0


# ---------------------------------------------------------------------------
# MTLConfig defaults test
# ---------------------------------------------------------------------------


def test_mtlconfig_defaults():
    cfg = MTLConfig()
    assert cfg.d_model == 32
    assert cfg.vocab_size == 64
    assert cfg.n_layers == 2
    assert cfg.n_tasks == 3
    assert cfg.method == "uncertainty"
    assert cfg.lr == 1e-4
