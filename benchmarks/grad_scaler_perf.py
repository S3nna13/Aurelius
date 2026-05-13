#!/usr/bin/env python3
"""Micro‑benchmark for ``torch.cuda.amp.GradScaler``.

Runs a dummy forward/backward pass a few times and measures the overhead of
instantiating the scaler and calling ``scale``/``step``/``update``.
"""

import time

import torch
from torch.cuda.amp import GradScaler

# Simple linear model
model = torch.nn.Linear(1024, 1024)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

inputs = torch.randn(32, 1024)
targets = torch.randn(32, 1024)

scaler = GradScaler()

# Warm‑up
for _ in range(5):
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=False):  # use CPU fallback if no GPU
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

iterations = 100
start = time.perf_counter()
for _ in range(iterations):
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=False):
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
elapsed = time.perf_counter() - start
print(f"{iterations} steps in {elapsed:.3f}s => {iterations / elapsed:.2f} steps/sec")

if __name__ == "__main__":
    pass
