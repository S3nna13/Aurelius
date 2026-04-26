"""Model extraction attack for the Aurelius LLM research platform.

Implements a knockoff-nets style attack that steals a black-box model by
querying it, collecting soft labels, and distilling those predictions into a
clone model via KL-divergence minimisation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.model.transformer import AureliusTransformer


class ModelExtractor:
    """Black-box model extraction via soft-label distillation.

    The oracle model is treated as an opaque API: only its output logits are
    observed. A clone model is trained to match those logits using KL divergence,
    thereby reproducing the oracle's predictive behaviour without access to its
    weights or gradients.
    """

    def query_oracle(
        self,
        oracle_model: AureliusTransformer,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Query the oracle model and return its output logits as soft labels.

        The oracle is treated as a black box: no gradients are computed through
        it and its parameters are not updated.

        Args:
            oracle_model: The target model being stolen.
            input_ids: (batch, seq_len) integer token indices.

        Returns:
            Detached logits tensor of shape (batch, seq_len, vocab_size).
        """
        oracle_model.eval()
        with torch.no_grad():
            _loss, logits, _pkv = oracle_model(input_ids)
        return logits.detach()

    def extraction_step(
        self,
        clone_model: AureliusTransformer,
        optimizer: torch.optim.Optimizer,
        input_ids: torch.Tensor,
        oracle_logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> float:
        """Perform a single distillation step on the clone model.

        Minimises the KL divergence between the clone's output distribution and
        the oracle's soft-label distribution at the given temperature.

        Args:
            clone_model: The model being trained to mimic the oracle.
            optimizer: Optimiser attached to clone_model's parameters.
            input_ids: (batch, seq_len) integer token indices.
            oracle_logits: (batch, seq_len, vocab_size) detached oracle logits.
            temperature: Softmax temperature applied to both distributions.

        Returns:
            Scalar loss value as a Python float.
        """
        clone_model.train()
        optimizer.zero_grad()

        _loss, clone_logits, _pkv = clone_model(input_ids)

        # Scale both distributions by temperature then compute KL divergence.
        # KL(P_oracle || P_clone)
        oracle_probs = F.softmax(oracle_logits / temperature, dim=-1)
        clone_log_probs = F.log_softmax(clone_logits / temperature, dim=-1)

        loss = F.kl_div(clone_log_probs, oracle_probs, reduction="batchmean", log_target=False)

        loss.backward()
        optimizer.step()

        return loss.item()

    def extraction_fidelity(
        self,
        oracle_model: AureliusTransformer,
        clone_model: AureliusTransformer,
        test_ids: torch.Tensor,
    ) -> float:
        """Compute the token-level agreement rate between oracle and clone.

        Agreement is measured as the fraction of sequence positions where the
        oracle's argmax prediction matches the clone's argmax prediction.

        Args:
            oracle_model: The reference model.
            clone_model: The extracted clone model.
            test_ids: (batch, seq_len) integer token indices.

        Returns:
            Agreement rate in [0, 1] as a Python float.
        """
        oracle_logits = self.query_oracle(oracle_model, test_ids)

        clone_model.eval()
        with torch.no_grad():
            _loss, clone_logits, _pkv = clone_model(test_ids)

        oracle_preds = oracle_logits.argmax(dim=-1)
        clone_preds = clone_logits.argmax(dim=-1)

        agreement = (oracle_preds == clone_preds).float().mean().item()
        return float(agreement)

    def run_extraction(
        self,
        oracle: AureliusTransformer,
        clone: AureliusTransformer,
        dataset_ids: torch.Tensor,
        n_epochs: int,
        lr: float,
    ) -> list[float]:
        """Run the full extraction loop over the provided dataset.

        Each element of dataset_ids is treated as a separate input batch. The
        loop iterates over all batches for n_epochs epochs, collecting per-step
        losses.

        Args:
            oracle: The black-box oracle model (frozen, query-only).
            clone: The clone model to be trained.
            dataset_ids: (n_batches, seq_len) or (n_batches, batch, seq_len)
                integer token indices. If 2-D, each row is a single-item batch.
            n_epochs: Number of full passes over dataset_ids.
            lr: Learning rate for Adam optimiser.

        Returns:
            List of per-step scalar loss values (length = n_epochs * n_batches).
        """
        optimizer = torch.optim.Adam(clone.parameters(), lr=lr)

        # Normalise to 3-D: (n_batches, batch_size, seq_len)
        if dataset_ids.dim() == 2:
            dataset_ids = dataset_ids.unsqueeze(1)

        losses: list[float] = []

        for _epoch in range(n_epochs):
            for batch_ids in dataset_ids:
                oracle_logits = self.query_oracle(oracle, batch_ids)
                step_loss = self.extraction_step(clone, optimizer, batch_ids, oracle_logits)
                losses.append(step_loss)

        return losses
