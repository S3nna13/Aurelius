"""Tests for NEFTune: Noisy Embeddings Improve Instruction Finetuning."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from aurelius.training.neftune import NEFTuneTrainer, NEFTuneWrapper, NoisyEmbedding

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_noisy_emb(
    num_embeddings: int = 100,
    embedding_dim: int = 32,
    noise_alpha: float = 5.0,
) -> NoisyEmbedding:
    torch.manual_seed(0)
    return NoisyEmbedding(num_embeddings, embedding_dim, noise_alpha=noise_alpha)


def _random_input(batch: int = 2, seq_len: int = 8, vocab: int = 100) -> torch.Tensor:
    return torch.randint(0, vocab, (batch, seq_len))


# ---------------------------------------------------------------------------
# NoisyEmbedding tests
# ---------------------------------------------------------------------------


class TestNoisyEmbedding:
    """Tests for NoisyEmbedding (drop-in nn.Embedding replacement)."""

    def test_output_shape_matches_nn_embedding(self):
        """Output shape must match plain nn.Embedding."""
        B, T, V, D = 2, 8, 100, 32
        noisy = _make_noisy_emb(V, D)
        plain = nn.Embedding(V, D)
        x = _random_input(B, T, V)

        noisy.train()
        assert noisy(x).shape == plain(x).shape

    def test_training_mode_adds_noise(self):
        """In train mode the output must differ from the clean embedding."""
        torch.manual_seed(42)
        noisy = _make_noisy_emb()
        x = _random_input()

        noisy.train()
        clean = nn.Embedding(noisy.num_embeddings, noisy.embedding_dim)
        clean.weight = noisy.weight  # share weights

        out_noisy = noisy(x)
        out_clean = clean(x)
        assert not torch.allclose(out_noisy, out_clean), "noise should be non-zero"

    def test_eval_mode_equals_clean_embedding(self):
        """In eval mode (train(False)) the output must exactly equal the clean embedding."""
        torch.manual_seed(42)
        noisy = _make_noisy_emb()
        x = _random_input()

        # Use train(False) instead of .eval() — equivalent but avoids
        # confusion with Python's built-in eval()
        noisy.train(False)
        clean = nn.Embedding(noisy.num_embeddings, noisy.embedding_dim)
        clean.weight = noisy.weight

        out_noisy = noisy(x)
        out_clean = clean(x)
        assert torch.allclose(out_noisy, out_clean), "eval mode must be noise-free"

    def test_noise_scale_matches_uniform_std(self):
        """std of (noisy - clean) ≈ alpha / (sqrt(T·d) * sqrt(3)) for uniform noise."""
        torch.manual_seed(0)
        alpha = 5.0
        T, D = 16, 64
        noisy = NoisyEmbedding(200, D, noise_alpha=alpha)
        clean = nn.Embedding(200, D)
        clean.weight = noisy.weight

        x = _random_input(batch=128, seq_len=T, vocab=200)
        noisy.train()

        diffs = []
        for _ in range(50):
            diff = noisy(x) - clean(x)
            diffs.append(diff)

        all_diffs = torch.cat([d.reshape(-1) for d in diffs])
        expected_std = alpha / (math.sqrt(T * D) * math.sqrt(3))
        actual_std = all_diffs.std().item()
        # Allow 15% relative tolerance over many samples
        assert abs(actual_std - expected_std) / expected_std < 0.15, (
            f"Expected std ≈ {expected_std:.6f}, got {actual_std:.6f}"
        )

    def test_longer_sequence_smaller_noise_per_element(self):
        """Longer sequences should have smaller per-element noise magnitude."""
        alpha = 5.0
        D = 32
        vocab = 200

        def mean_abs_noise(T: int, n_trials: int = 100) -> float:
            noisy = NoisyEmbedding(vocab, D, noise_alpha=alpha)
            clean = nn.Embedding(vocab, D)
            clean.weight = noisy.weight
            noisy.train()
            x = _random_input(batch=4, seq_len=T, vocab=vocab)
            total = 0.0
            for _ in range(n_trials):
                total += (noisy(x) - clean(x)).abs().mean().item()
            return total / n_trials

        torch.manual_seed(1)
        short = mean_abs_noise(4)
        long_ = mean_abs_noise(64)
        assert short > long_, (
            f"Shorter seq (mean noise {short:.5f}) should have larger noise "
            f"than longer seq (mean noise {long_:.5f})"
        )

    def test_zero_noise_alpha_produces_no_noise(self):
        """noise_alpha=0 must produce no noise even in training mode."""
        torch.manual_seed(0)
        noisy = NoisyEmbedding(100, 32, noise_alpha=0.0)
        clean = nn.Embedding(100, 32)
        clean.weight = noisy.weight

        x = _random_input()
        noisy.train()

        assert torch.allclose(noisy(x), clean(x)), (
            "noise_alpha=0 must yield identical output to clean embedding"
        )

    def test_weight_identical_after_construction(self):
        """The weight tensor must equal the original nn.Embedding weight."""
        torch.manual_seed(0)
        plain = nn.Embedding(100, 32)
        noisy = NoisyEmbedding(100, 32)
        noisy.weight = plain.weight

        assert noisy.weight is plain.weight, "weights should be the same object"

    def test_gradient_flows_in_training_mode(self):
        """Gradients must reach the embedding weight during training."""
        noisy = _make_noisy_emb()
        x = _random_input()
        noisy.train()

        out = noisy(x)
        out.sum().backward()
        assert noisy.weight.grad is not None, "gradient should flow through NoisyEmbedding"

    def test_gradient_flows_in_inference_mode(self):
        """Gradients must reach the embedding weight in inference (non-training) mode too."""
        noisy = _make_noisy_emb()
        x = _random_input()
        # train(False) sets self.training = False — same as .eval()
        noisy.train(False)

        out = noisy(x)
        out.sum().backward()
        assert noisy.weight.grad is not None, "gradient should flow in non-training mode"


# ---------------------------------------------------------------------------
# NEFTuneWrapper tests
# ---------------------------------------------------------------------------


class _SimpleModel(nn.Module):
    """Tiny model with a single embedding attribute for testing."""

    def __init__(self, attr_path: str = "embed_tokens"):
        super().__init__()
        # Support nested paths like "encoder.embed"
        parts = attr_path.split(".")
        if len(parts) == 1:
            setattr(self, parts[0], nn.Embedding(50, 16))
        else:
            # One level of nesting
            sub = nn.Module()
            setattr(sub, parts[1], nn.Embedding(50, 16))
            setattr(self, parts[0], sub)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TestNEFTuneWrapper:
    """Tests for NEFTuneWrapper."""

    def test_apply_replaces_embedding_with_noisy_embedding(self):
        """apply() should swap the embedding for a NoisyEmbedding."""
        model = _SimpleModel()
        wrapper = NEFTuneWrapper(model, embedding_attr="embed_tokens")
        wrapper.apply()
        assert isinstance(model.embed_tokens, NoisyEmbedding), (
            "embed_tokens should be a NoisyEmbedding after apply()"
        )

    def test_remove_restores_plain_embedding(self):
        """remove() should restore the original nn.Embedding."""
        model = _SimpleModel()
        wrapper = NEFTuneWrapper(model, embedding_attr="embed_tokens")
        wrapper.apply()
        wrapper.remove()
        assert type(model.embed_tokens) is nn.Embedding, (
            "embed_tokens should be plain nn.Embedding after remove()"
        )

    def test_dot_path_attr_works(self):
        """Wrapper should support dot-path attributes like 'encoder.embed'."""
        model = _SimpleModel(attr_path="encoder.embed")
        wrapper = NEFTuneWrapper(model, embedding_attr="encoder.embed")
        wrapper.apply()
        assert isinstance(model.encoder.embed, NoisyEmbedding), (
            "dot-path attribute should resolve and wrap the embedding"
        )

    def test_train_mode_adds_noise_after_apply(self):
        """After apply(), model in train mode should produce noisy outputs."""
        torch.manual_seed(7)
        model = _SimpleModel()
        original_weight = model.embed_tokens.weight.clone()

        wrapper = NEFTuneWrapper(model, embedding_attr="embed_tokens")
        wrapper.apply()

        x = _random_input(seq_len=8, vocab=50)
        model.train()

        # Build a clean reference with the same weight
        clean_ref = nn.Embedding(50, 16)
        clean_ref.weight = nn.Parameter(original_weight.clone())

        out_noisy = model.embed_tokens(x)
        out_clean = clean_ref(x)
        assert not torch.allclose(out_noisy, out_clean), "train mode should add noise after apply()"

    def test_inference_mode_no_noise_after_apply(self):
        """After apply(), model in inference mode (train(False)) should not add noise."""
        torch.manual_seed(7)
        model = _SimpleModel()
        original_weight = model.embed_tokens.weight.clone()

        wrapper = NEFTuneWrapper(model, embedding_attr="embed_tokens")
        wrapper.apply()

        x = _random_input(seq_len=8, vocab=50)
        # Use train(False) to enter inference mode
        model.train(False)

        clean_ref = nn.Embedding(50, 16)
        clean_ref.weight = nn.Parameter(original_weight.clone())

        out_noisy = model.embed_tokens(x)
        out_clean = clean_ref(x)
        assert torch.allclose(out_noisy, out_clean), (
            "inference mode should not add noise after apply()"
        )


# ---------------------------------------------------------------------------
# NEFTuneTrainer tests
# ---------------------------------------------------------------------------


class _TinyLinearModel(nn.Module):
    """Tiny linear model: (B, T, d) → (B, T, num_classes)."""

    def __init__(self, d_model: int = 16, num_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _make_trainer(
    d_model: int = 16,
    num_classes: int = 10,
    noise_alpha: float = 5.0,
    lr: float = 1e-2,
) -> tuple[NEFTuneTrainer, _TinyLinearModel]:
    torch.manual_seed(0)
    model = _TinyLinearModel(d_model, num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    trainer = NEFTuneTrainer(model, optimizer, noise_alpha=noise_alpha)
    return trainer, model


class TestNEFTuneTrainer:
    """Tests for NEFTuneTrainer."""

    def test_train_step_returns_finite_loss(self):
        """train_step should return a finite float loss."""
        B, T, D, C = 2, 8, 16, 10
        trainer, _ = _make_trainer(D, C)

        embeddings = torch.randn(B, T, D)
        targets = torch.randint(0, C, (B, T))

        def loss_fn(logits, tgt):
            return nn.CrossEntropyLoss()(logits.view(-1, C), tgt.view(-1))

        loss_val = trainer.train_step(embeddings, targets, loss_fn)
        assert math.isfinite(loss_val), f"Expected finite loss, got {loss_val}"

    def test_gradient_step_decreases_loss(self):
        """Multiple training steps should reduce the loss on fixed data."""
        B, T, D, C = 4, 8, 16, 10
        torch.manual_seed(42)
        trainer, _ = _make_trainer(D, C, lr=0.1)

        embeddings = torch.randn(B, T, D)
        targets = torch.randint(0, C, (B, T))

        def loss_fn(logits, tgt):
            return nn.CrossEntropyLoss()(logits.view(-1, C), tgt.view(-1))

        losses = [trainer.train_step(embeddings.clone(), targets, loss_fn) for _ in range(20)]
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        )

    def test_parameters_updated_after_step(self):
        """Model parameters must change after a training step."""
        B, T, D, C = 2, 8, 16, 10
        trainer, model = _make_trainer(D, C)

        params_before = {k: v.clone() for k, v in model.named_parameters()}

        embeddings = torch.randn(B, T, D)
        targets = torch.randint(0, C, (B, T))

        def loss_fn(logits, tgt):
            return nn.CrossEntropyLoss()(logits.view(-1, C), tgt.view(-1))

        trainer.train_step(embeddings, targets, loss_fn)

        for name, param in model.named_parameters():
            assert not torch.allclose(param, params_before[name]), (
                f"Parameter '{name}' was not updated after train_step"
            )
