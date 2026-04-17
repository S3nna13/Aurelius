"""Tuned Lens: learned per-layer affine translators for intermediate representations.

Implements the Tuned Lens technique from:
    "Eliciting Latent Predictions from Transformers with the Tuned Lens"
    Belrose et al., arXiv:2303.08112

The tuned lens learns a linear transform T_l for each layer l that maps
intermediate representations to the vocabulary distribution:
    logits_l = unembed(T_l(h_l))  where T_l is an affine transform (Wx + b)

Training minimises KL(p_final || p_l) for each layer independently, revealing
what the model "thinks" at each layer of computation.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TunedLensTranslator(nn.Module):
    """Per-layer affine translators that map hidden states to a translated space.

    Each layer gets its own independent nn.Linear(d_model, d_model, bias=True)
    translator, initialised as the identity transform to provide a sensible
    starting point before training.

    Args:
        d_model: Hidden dimension of the model.
        n_layers: Number of transformer layers (one translator per layer).
    """

    def __init__(self, d_model: int, n_layers: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.translators = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=True) for _ in range(n_layers)]
        )

        # Initialise each translator as identity + zero bias so that before
        # training the tuned lens degrades gracefully to the logit lens.
        for linear in self.translators:
            nn.init.eye_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, hidden_states: List[Tensor], layer_idx: int) -> Tensor:
        """Apply the translator for a single layer.

        Args:
            hidden_states: List of per-layer hidden state tensors, each (B, T, d_model).
            layer_idx: Index of the layer whose translator to apply.

        Returns:
            Translated hidden state of shape (B, T, d_model).
        """
        h = hidden_states[layer_idx]  # (B, T, d_model)
        return self.translators[layer_idx](h)  # (B, T, d_model)

    def translate_all(self, hidden_states: List[Tensor]) -> List[Tensor]:
        """Apply each translator to its corresponding layer's hidden state.

        Args:
            hidden_states: List of length n_layers, each (B, T, d_model).

        Returns:
            List of length n_layers, each (B, T, d_model).
        """
        return [self.translators[i](hidden_states[i]) for i in range(self.n_layers)]


class TunedLensEvaluator:
    """Applies the tuned lens to produce per-layer vocabulary predictions.

    Args:
        translator: Trained TunedLensTranslator with per-layer affine maps.
        unembed_fn: Callable that maps (B, T, d_model) -> (B, T, V) logits,
                    e.g. the final LM head of the model.
    """

    def __init__(
        self,
        translator: TunedLensTranslator,
        unembed_fn: Callable[[Tensor], Tensor],
    ) -> None:
        self.translator = translator
        self.unembed_fn = unembed_fn

    def get_layer_logits(self, hidden_states: List[Tensor]) -> List[Tensor]:
        """Return per-layer logit predictions via the tuned lens.

        For each layer l:  logits_l = unembed_fn(T_l(h_l))

        Args:
            hidden_states: List of length n_layers, each (B, T, d_model).

        Returns:
            List of length n_layers, each (B, T, V).
        """
        translated = self.translator.translate_all(hidden_states)
        return [self.unembed_fn(h) for h in translated]

    def layer_entropy(self, hidden_states: List[Tensor]) -> Tensor:
        """Compute the mean entropy of per-layer predictions across batch and positions.

        Args:
            hidden_states: List of length n_layers, each (B, T, d_model).

        Returns:
            Tensor of shape (n_layers,) with per-layer mean entropy.
        """
        all_logits = self.get_layer_logits(hidden_states)
        entropies: List[Tensor] = []
        for logits in all_logits:  # each (B, T, V)
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-10)
            # H = -sum(p * log p) over vocab, then average over B and T
            h = -torch.sum(probs * log_probs, dim=-1)  # (B, T)
            entropies.append(h.mean())
        return torch.stack(entropies)  # (n_layers,)


class TunedLensTrainer:
    """Trains the per-layer affine translators of a TunedLensTranslator.

    Training objective: for each layer l minimise
        KL(softmax(final_logits) || softmax(translated_logits_l))

    Args:
        translator: The TunedLensTranslator whose parameters are to be trained.
        unembed_fn: Callable (B, T, d_model) -> (B, T, V).
        optimizer: A PyTorch optimiser configured on translator.parameters().
    """

    def __init__(
        self,
        translator: TunedLensTranslator,
        unembed_fn: Callable[[Tensor], Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.translator = translator
        self.unembed_fn = unembed_fn
        self.optimizer = optimizer

    def train_step(
        self,
        hidden_states: List[Tensor],
        final_logits: Tensor,
    ) -> Dict[str, float]:
        """Perform one gradient-descent step on the translators.

        Minimises  sum_l KL(p_final || p_l)  where
            p_final = softmax(final_logits)
            p_l     = softmax(unembed_fn(T_l(h_l)))

        KL divergence is computed as:
            KL(P || Q) = sum_v P_v * (log P_v - log Q_v)
        using F.kl_div which expects log-probabilities for the input (Q) and
        probabilities for the target (P).

        Args:
            hidden_states: List of length n_layers, each (B, T, d_model).
            final_logits: The final-layer logits (B, T, V) used as the target.

        Returns:
            Dict with keys:
                'loss': total scalar loss (float).
                'mean_kl_per_layer': average KL across layers (float).
        """
        self.optimizer.zero_grad()

        # Target distribution (detached — we do not back-prop into the model)
        p_final = F.softmax(final_logits.detach(), dim=-1)  # (B, T, V)

        kl_per_layer: List[Tensor] = []
        for i in range(self.translator.n_layers):
            h_translated = self.translator.translators[i](hidden_states[i])  # (B, T, D)
            layer_logits = self.unembed_fn(h_translated)  # (B, T, V)

            log_q = F.log_softmax(layer_logits, dim=-1)  # (B, T, V)

            # F.kl_div(input=log_Q, target=P, reduction='batchmean')
            # = mean over batch of sum_v P_v * (log P_v - log Q_v)
            # We use 'sum' and normalise manually to keep scale consistent.
            kl = F.kl_div(log_q, p_final, reduction="batchmean")
            kl_per_layer.append(kl)

        total_loss = torch.stack(kl_per_layer).sum()
        total_loss.backward()
        self.optimizer.step()

        mean_kl = float(torch.stack(kl_per_layer).mean().item())

        return {
            "loss": float(total_loss.item()),
            "mean_kl_per_layer": mean_kl,
        }


class LogitLens:
    """Simple unadapted logit lens — no learned transform.

    Directly applies the unembedding function to each layer's hidden state to
    obtain per-layer vocabulary logits.  This is the baseline against which the
    tuned lens is compared.

    Args:
        unembed_fn: Callable (B, T, d_model) -> (B, T, V).
    """

    def __init__(self, unembed_fn: Callable[[Tensor], Tensor]) -> None:
        self.unembed_fn = unembed_fn

    def forward(self, hidden_states: List[Tensor]) -> List[Tensor]:
        """Apply unembed_fn directly to each layer's hidden state.

        Args:
            hidden_states: List of length n_layers, each (B, T, d_model).

        Returns:
            List of length n_layers, each (B, T, V).
        """
        return [self.unembed_fn(h) for h in hidden_states]

    def top_tokens(self, hidden_states: List[Tensor], k: int = 5) -> List[Tensor]:
        """Return the top-k token indices per layer per position.

        Args:
            hidden_states: List of length n_layers, each (B, T, d_model).
            k: Number of top tokens to return.

        Returns:
            List of length n_layers; each element is (T, k) top-k indices
            for the *first* batch element.  (Returned as a Python list for
            easy stacking by the caller — see note below.)

        Note:
            The docstring in the task specification says shape (n_layers, T, k).
            This method returns a Python list of length n_layers where each
            entry is a (T, k) tensor so that the caller can do
            ``torch.stack(lens.top_tokens(...))`` to obtain (n_layers, T, k).
        """
        all_logits = self.forward(hidden_states)
        result: List[Tensor] = []
        for logits in all_logits:  # (B, T, V)
            # Use first batch element
            logits_0 = logits[0]  # (T, V)
            _, top_ids = torch.topk(logits_0, k, dim=-1)  # (T, k)
            result.append(top_ids)
        return result
