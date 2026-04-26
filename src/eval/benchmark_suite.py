"""Unified benchmark suite runner with task-specific metrics.

Supports three task types:
- perplexity: mean perplexity over token sequences
- multiple_choice: accuracy via lowest negative log-likelihood choice selection
- generation: exact-match rate via greedy decoding vs. reference strings
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkTask:
    """Specification for a single benchmark task."""

    name: str
    task_type: str  # "multiple_choice" | "generation" | "perplexity"
    n_samples: int = 100
    weight: float = 1.0


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark task."""

    task_name: str
    metric: float
    n_evaluated: int
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_perplexity_task(
    model: nn.Module,
    samples: list,
    max_len: int,
) -> BenchmarkResult:
    """Compute mean perplexity over a list of input_id tensors.

    Args:
        model: AureliusTransformer. Forward returns (loss, logits, pkv).
        samples: List of 1-D or 2-D (1, T) int tensors.
        max_len: Maximum sequence length; longer sequences are truncated.

    Returns:
        BenchmarkResult with metric=mean_perplexity.
    """
    model.train(False)
    device = next(model.parameters()).device

    nll_sum = 0.0
    token_count = 0
    per_sample_ppl = []

    for sample in samples:
        ids = sample.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)  # (1, T)
        ids = ids[:, :max_len]  # truncate

        T = ids.shape[1]
        if T < 2:
            per_sample_ppl.append(float("inf"))
            continue

        _, logits, _ = model(ids)  # logits: (1, T, vocab)

        shift_logits = logits[:, :-1, :].contiguous()  # (1, T-1, V)
        shift_targets = ids[:, 1:].contiguous()  # (1, T-1)

        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            reduction="sum",
        ).item()

        n_tokens = shift_targets.numel()
        nll_sum += nll
        token_count += n_tokens
        per_sample_ppl.append(math.exp(nll / max(n_tokens, 1)))

    if token_count == 0:
        mean_ppl = float("inf")
    else:
        mean_ppl = math.exp(nll_sum / token_count)

    return BenchmarkResult(
        task_name="perplexity",
        metric=mean_ppl,
        n_evaluated=len(samples),
        details={"per_sample_perplexity": per_sample_ppl},
    )


# ---------------------------------------------------------------------------
# Multiple-choice evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def _nll_of_choice(
    model: nn.Module,
    context_ids: torch.Tensor,
    choice_ids: torch.Tensor,
    device,
) -> float:
    """Return mean NLL of choice_ids conditioned on context_ids."""
    ctx = context_ids.to(device)
    ch = choice_ids.to(device)

    if ctx.dim() == 1:
        ctx = ctx.unsqueeze(0)
    if ch.dim() == 1:
        ch = ch.unsqueeze(0)

    full = torch.cat([ctx, ch], dim=1)  # (1, T_ctx + T_ch)

    _, logits, _ = model(full)  # (1, T, V)

    shift_logits = logits[:, :-1, :]  # (1, T-1, V)
    shift_targets = full[:, 1:]  # (1, T-1)

    nll_all = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        reduction="none",
    )  # (T-1,)

    n_ch = ch.shape[1]
    choice_nll = nll_all[-(n_ch):]
    return choice_nll.mean().item()


@torch.no_grad()
def evaluate_multiple_choice_task(
    model: nn.Module,
    questions: list,
    task_name: str,
) -> BenchmarkResult:
    """Evaluate accuracy on multiple-choice questions.

    Each question dict must have:
        "context_ids"  : Tensor of shape (T_ctx,) or (1, T_ctx)
        "choice_ids"   : list[Tensor], one per choice
        "correct_idx"  : int, index of the correct choice

    The choice with the lowest mean NLL is selected.

    Returns:
        BenchmarkResult with metric=accuracy (0.0 to 1.0).
    """
    model.train(False)
    device = next(model.parameters()).device

    n_correct = 0
    predictions = []

    for q in questions:
        context_ids = q["context_ids"]
        choice_ids_list = q["choice_ids"]
        correct_idx = int(q.get("correct_idx", 0))

        nlls = [_nll_of_choice(model, context_ids, ch, device) for ch in choice_ids_list]
        pred = int(min(range(len(nlls)), key=lambda i: nlls[i]))
        predictions.append(pred)
        if pred == correct_idx:
            n_correct += 1

    accuracy = n_correct / max(1, len(questions))

    return BenchmarkResult(
        task_name=task_name,
        metric=accuracy,
        n_evaluated=len(questions),
        details={"n_correct": n_correct, "predictions": predictions},
    )


# ---------------------------------------------------------------------------
# Generation / exact-match evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def _greedy_generate(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    device,
) -> list:
    """Greedy-decode up to max_new_tokens tokens from a prompt."""
    ids = prompt_ids.to(device)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)

    generated = []
    for _ in range(max_new_tokens):
        _, logits, _ = model(ids)
        next_token = logits[:, -1, :].argmax(dim=-1)  # (1,)
        generated.append(next_token.item())
        ids = torch.cat([ids, next_token.unsqueeze(0)], dim=1)

    return generated


@torch.no_grad()
def evaluate_generation_task(
    model: nn.Module,
    prompts: list,
    references: list,
    tokenizer_decode: Callable,
    max_new_tokens: int,
    task_name: str,
) -> BenchmarkResult:
    """Evaluate exact-match rate for a generation task.

    Args:
        model: AureliusTransformer.
        prompts: List of 1-D prompt tensors.
        references: List of reference strings (one per prompt).
        tokenizer_decode: Function mapping list[int] -> str.
        max_new_tokens: How many tokens to generate per prompt.
        task_name: Name for the returned BenchmarkResult.

    Returns:
        BenchmarkResult with metric=exact_match (0.0 to 1.0).
    """
    model.train(False)
    device = next(model.parameters()).device

    n_exact = 0
    generated_texts = []

    for prompt, ref in zip(prompts, references):
        gen_ids = _greedy_generate(model, prompt, max_new_tokens, device)
        gen_text = tokenizer_decode(gen_ids)
        generated_texts.append(gen_text)
        if gen_text == ref:
            n_exact += 1

    exact_match = n_exact / max(1, len(prompts))

    return BenchmarkResult(
        task_name=task_name,
        metric=exact_match,
        n_evaluated=len(prompts),
        details={"n_exact": n_exact, "generated": generated_texts},
    )


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------


class BenchmarkSuite:
    """Runs multiple benchmark tasks and aggregates results.

    Args:
        tasks: List of BenchmarkTask definitions.
        model: AureliusTransformer instance.
        tokenizer_decode: Decode function for generation tasks.
    """

    def __init__(
        self,
        tasks: list,
        model: nn.Module,
        tokenizer_decode: Callable | None = None,
    ) -> None:
        self.tasks = tasks
        self.model = model
        self.tokenizer_decode = tokenizer_decode or (
            lambda ids: bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")
        )

    def run_task(self, task: BenchmarkTask, data: dict) -> BenchmarkResult:
        """Dispatch a single task to the appropriate evaluation function.

        Args:
            task: BenchmarkTask specification.
            data: Task-specific data dict. Expected keys:
                  perplexity      -> "samples": list[Tensor], "max_len": int
                  multiple_choice -> "questions": list[dict]
                  generation      -> "prompts": list[Tensor],
                                     "references": list[str],
                                     "max_new_tokens": int

        Returns:
            BenchmarkResult for this task.
        """
        if task.task_type == "perplexity":
            result = evaluate_perplexity_task(
                model=self.model,
                samples=data["samples"],
                max_len=data.get("max_len", 512),
            )
            result.task_name = task.name
        elif task.task_type == "multiple_choice":
            result = evaluate_multiple_choice_task(
                model=self.model,
                questions=data["questions"],
                task_name=task.name,
            )
        elif task.task_type == "generation":
            result = evaluate_generation_task(
                model=self.model,
                prompts=data["prompts"],
                references=data["references"],
                tokenizer_decode=self.tokenizer_decode,
                max_new_tokens=data.get("max_new_tokens", 50),
                task_name=task.name,
            )
        else:
            raise ValueError(
                f"Unknown task_type {task.task_type!r}. "
                "Expected 'perplexity', 'multiple_choice', or 'generation'."
            )
        return result

    def run_all(self, data: dict) -> dict:
        """Run all configured tasks and aggregate scores.

        Args:
            data: Mapping of task_name -> task data dict.

        Returns:
            dict with keys:
                "results"         : list[BenchmarkResult]
                "aggregate_score" : float
                "task_scores"     : dict[str, float]
        """
        results = []
        for task in self.tasks:
            task_data = data.get(task.name, {})
            result = self.run_task(task, task_data)
            results.append(result)

        agg = self.aggregate_score(results)
        task_scores = {r.task_name: r.metric for r in results}

        return {
            "results": results,
            "aggregate_score": agg,
            "task_scores": task_scores,
        }

    def aggregate_score(self, results: list) -> float:
        """Compute weighted mean of task metrics.

        Weights are taken from the corresponding BenchmarkTask.weight.
        If a task result name is not found in self.tasks, weight defaults to 1.0.

        Args:
            results: List of BenchmarkResult instances.

        Returns:
            Weighted mean metric as a float.
        """
        weight_map = {t.name: t.weight for t in self.tasks}

        total_weight = 0.0
        weighted_sum = 0.0
        for result in results:
            w = weight_map.get(result.task_name, 1.0)
            weighted_sum += result.metric * w
            total_weight += w

        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight
