"""Neural program synthesis training via execution-guided REINFORCE reward.

Pipeline per step:
1. Build a prompt from a ProgramSpec (description + examples + function stub)
2. Sample n_samples programs by greedy token-by-token generation
3. Execute each program against all test cases in a sandboxed namespace
4. Score programs (fraction of test cases passed)
5. Update policy with REINFORCE: loss = -mean(log_prob * reward)
"""
from __future__ import annotations

import io
import sys
import threading
import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProgramSpec:
    """Specification for a program synthesis task."""
    task_id: str
    description: str
    test_cases: list[tuple[str, str]]  # (input_str, expected_output_str)
    language: str = "python"


@dataclass
class SynthesisConfig:
    """Hyperparameters for program synthesis training."""
    max_new_tokens: int = 32
    n_samples: int = 4
    temperature: float = 1.0
    execution_timeout: float = 1.0
    reward_correct: float = 1.0
    reward_partial: float = 0.3
    reward_wrong: float = 0.0


# ---------------------------------------------------------------------------
# Safe execution helpers
# ---------------------------------------------------------------------------

_SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "range": range,
    "int": int,
    "str": str,
    "float": float,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "sum": sum,
    "max": max,
    "min": min,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "__import__": None,
    "None": None,
    "True": True,
    "False": False,
}


def execute_program_safe(
    code: str,
    test_input: str,
    timeout: float,
) -> tuple[str, bool]:
    """Execute Python code string safely and capture stdout.

    The code runs inside a restricted namespace where __builtins__ is
    limited to a safe subset.  stdout is captured via io.StringIO.
    A threading.Timer is used to enforce the timeout.

    Returns:
        (output_str, success_bool)
    """
    result_holder: list = [None, False]

    def _run() -> None:
        buf = io.StringIO()
        namespace: dict = {
            "__builtins__": _SAFE_BUILTINS,
            "__name__": "__main__",
            "_test_input": test_input,
        }
        old_stdout = sys.stdout
        try:
            sys.stdout = buf
            exec(code, namespace)  # noqa: S102
            output = buf.getvalue()
            result_holder[0] = output
            result_holder[1] = True
        except Exception:
            result_holder[0] = ""
            result_holder[1] = False
        finally:
            sys.stdout = old_stdout

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return ("", False)

    output = result_holder[0] if result_holder[0] is not None else ""
    success = result_holder[1]
    return (output, success)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_program(
    code: str,
    spec: ProgramSpec,
    config: SynthesisConfig,
) -> float:
    """Execute code against every test case and return mean reward.

    Each test case contributes reward_correct if output matches expected,
    otherwise reward_wrong.
    """
    if not spec.test_cases:
        return 0.0

    total = 0.0
    for test_input, expected in spec.test_cases:
        output, success = execute_program_safe(
            code, test_input, config.execution_timeout
        )
        if success and output.strip() == expected.strip():
            total += config.reward_correct
        else:
            total += config.reward_wrong

    return total / len(spec.test_cases)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_program(
    model: torch.nn.Module,
    prompt_ids: list[int],
    config: SynthesisConfig,
) -> tuple[list[int], float]:
    """Generate tokens one-by-one using the model forward pass.

    Returns:
        (generated_ids, total_log_prob) where generated_ids are the newly
        generated tokens (not including prompt) and total_log_prob is the
        sum of log-probs of each chosen token.
    """
    model.eval()
    generated: list[int] = []
    total_log_prob: float = 0.0

    ids = list(prompt_ids)

    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            input_tensor = torch.tensor([ids], dtype=torch.long)
            output = model(input_tensor)
            logits = output[1]  # (1, seq_len, vocab_size)

            next_logits = logits[0, -1, :]  # (vocab_size,)

            if config.temperature > 0:
                next_logits = next_logits / config.temperature

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            log_prob = float(torch.log(probs[next_token] + 1e-9))

            generated.append(int(next_token))
            total_log_prob += log_prob
            ids.append(int(next_token))

    return generated, total_log_prob


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ProgramSynthesisTrainer:
    """REINFORCE-based trainer for neural program synthesis.

    Args:
        model: AureliusTransformer (or compatible) language model.
        optimizer: PyTorch optimizer.
        config: SynthesisConfig hyper-parameters.
        tokenizer_encode: str -> list[int]
        tokenizer_decode: list[int] -> str
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: SynthesisConfig,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode

    def build_prompt(self, spec: ProgramSpec) -> str:
        """Format spec as a text prompt ending with 'def solution('."""
        lines = [f"# Task: {spec.description}", "# Examples:"]
        for inp, out in spec.test_cases:
            lines.append(f"{inp}\u2192{out}")
        lines.append("def solution(")
        return "\n".join(lines)

    def train_step(self, spec: ProgramSpec) -> dict:
        """Sample programs, score, compute REINFORCE loss, backprop.

        Returns:
            dict with keys: loss, mean_reward, best_reward, n_samples
        """
        prompt = self.build_prompt(spec)
        prompt_ids = self.tokenizer_encode(prompt)

        samples: list[tuple[list[int], float, float]] = []

        self.model.eval()
        with torch.no_grad():
            for _ in range(self.config.n_samples):
                generated_ids, log_prob = generate_program(
                    self.model, prompt_ids, self.config
                )
                code = self.tokenizer_decode(generated_ids)
                reward = score_program(code, spec, self.config)
                samples.append((generated_ids, log_prob, reward))

        rewards = [r for _, _, r in samples]
        mean_reward = sum(rewards) / len(rewards)
        best_reward = max(rewards)

        # REINFORCE: recompute log-probs with gradients
        self.model.train()
        self.optimizer.zero_grad()

        loss_terms: list[torch.Tensor] = []
        for generated_ids, _, reward in samples:
            ids = prompt_ids + generated_ids
            if len(ids) < 2:
                continue
            input_tensor = torch.tensor([ids[:-1]], dtype=torch.long)
            target_ids = ids[1:]

            output = self.model(input_tensor)
            logits = output[1]  # (1, seq_len-1, vocab_size)

            gen_start = max(len(prompt_ids) - 1, 0)
            gen_logits = logits[0, gen_start:, :]
            gen_targets = torch.tensor(target_ids[gen_start:], dtype=torch.long)

            if gen_logits.shape[0] == 0 or gen_targets.shape[0] == 0:
                continue

            min_len = min(gen_logits.shape[0], gen_targets.shape[0])
            gen_logits = gen_logits[:min_len]
            gen_targets = gen_targets[:min_len]

            log_probs_tensor = F.log_softmax(
                gen_logits / max(self.config.temperature, 1e-6), dim=-1
            )
            token_log_probs = log_probs_tensor[
                torch.arange(min_len), gen_targets
            ]
            seq_log_prob = token_log_probs.sum()
            loss_terms.append(-seq_log_prob * reward)

        if loss_terms:
            loss = torch.stack(loss_terms).mean()
        else:
            loss = torch.tensor(0.0, requires_grad=True)

        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "mean_reward": mean_reward,
            "best_reward": best_reward,
            "n_samples": len(samples),
        }

    def generate_best(self, spec: ProgramSpec) -> str:
        """Sample n_samples programs and return the highest-scoring decoded string."""
        prompt = self.build_prompt(spec)
        prompt_ids = self.tokenizer_encode(prompt)

        best_code = ""
        best_reward = -1.0

        self.model.eval()
        with torch.no_grad():
            for _ in range(self.config.n_samples):
                generated_ids, _ = generate_program(
                    self.model, prompt_ids, self.config
                )
                code = self.tokenizer_decode(generated_ids)
                reward = score_program(code, spec, self.config)
                if reward > best_reward:
                    best_reward = reward
                    best_code = code

        return best_code
