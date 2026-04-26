from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ModelDelta:
    @staticmethod
    def compute(old: dict[str, torch.Tensor], new: dict[str, torch.Tensor]) -> ModelDelta:
        delta: ModelDelta = {}
        for key in old:
            if key in new:
                delta[key] = new[key] - old[key]
        return delta

    @staticmethod
    def apply(params: dict[str, torch.Tensor], delta: ModelDelta) -> dict[str, torch.Tensor]:
        result = {}
        for key in params:
            result[key] = params[key] + delta.get(key, torch.zeros_like(params[key]))
        return result


@dataclass
class ClientConfig:
    learning_rate: float = 0.01
    local_epochs: int = 1
    batch_size: int = 32
    client_id: str | None = None


class FederatedClient:
    def __init__(
        self, client_id: str, model: nn.Module, config: ClientConfig | None = None
    ) -> None:
        self.client_id = client_id
        self.model = model
        self.config = config or ClientConfig(client_id=client_id)
        self._epochs: int = 0
        self._samples: int = 0

    def train(
        self, data: torch.Tensor, lr: float | None = None, epochs: int | None = None
    ) -> ModelDelta:
        old_params = {k: v.data.clone() for k, v in self.model.named_parameters()}
        opt = torch.optim.SGD(self.model.parameters(), lr=lr or self.config.learning_rate)
        loss_fn = nn.MSELoss()
        n_epochs = epochs or self.config.local_epochs
        out_dim = self.model.out_features if hasattr(self.model, "out_features") else data.shape[-1]
        targets = torch.zeros(data.shape[0], out_dim)
        dataset = TensorDataset(data, targets)
        loader = DataLoader(dataset, batch_size=self.config.batch_size)

        for _ in range(n_epochs):
            for batch_x, batch_y in loader:
                opt.zero_grad()
                pred = self.model(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                opt.step()

        self._epochs += n_epochs
        self._samples += len(data)
        new_params = {k: v.data.clone() for k, v in self.model.named_parameters()}
        return ModelDelta.compute(old_params, new_params)

    def get_stats(self) -> dict[str, Any]:
        return {"client_id": self.client_id, "epochs": self._epochs, "samples": self._samples}
