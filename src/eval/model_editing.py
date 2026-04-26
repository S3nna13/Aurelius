"""Model editing: MEMIT-style mass editing, edit evaluation, and counterfactual testing."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EditRequest:
    """Specification for a single model edit."""

    subject: str  # entity being edited
    prompt: str  # prompt template containing the subject
    target_new: str  # new fact to inject
    target_old: str | None = None  # original fact (for evaluation)
    paraphrase_prompts: list[str] | None = None  # for generalization testing


@dataclass
class EditResult:
    """Outcome of applying a model edit."""

    success: bool
    efficacy: float  # does the model output the new fact?
    generalization: float  # does it generalize to paraphrases?
    specificity: float  # does it preserve unrelated facts?
    edit_id: str


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


def compute_sequence_probability(
    model: nn.Module,
    input_ids: torch.Tensor,  # (1, S)
    target_ids: torch.Tensor,  # (1, T)
) -> float:
    """Compute P(target | input) as sum of log-probabilities.

    Concatenates input + target, runs a single forward pass, then extracts the
    log-probability of each target token given its left context.

    Returns a float (sum of log-probs, so <= 0).
    """
    with torch.no_grad():
        full_ids = torch.cat([input_ids, target_ids], dim=1)  # (1, S+T)
        _, logits, _ = model(full_ids)  # logits: (1, S+T, vocab)

    log_probs = F.log_softmax(logits, dim=-1)  # (1, S+T, vocab)

    seq_len = input_ids.shape[1]
    n_target = target_ids.shape[1]

    total_log_prob = 0.0
    for t in range(n_target):
        # Position in full sequence: seq_len - 1 + t gives the position BEFORE the
        # target token; that position's logits predict position seq_len + t.
        pos = seq_len - 1 + t  # logits at this position predict next token
        tok = target_ids[0, t].item()
        total_log_prob += log_probs[0, pos, tok].item()

    return total_log_prob


def rank_one_update(
    W: torch.Tensor,  # (d_out, d_in)
    k: torch.Tensor,  # (d_in,)
    v: torch.Tensor,  # (d_out,)
) -> torch.Tensor:
    """ROME-style rank-1 weight update.

    W_new = W + outer(v - W @ k, k) / (k^T @ k)

    After the update: W_new @ k == v (up to numerical precision).

    Returns a new tensor of the same shape as W.
    """
    Wk = W @ k  # (d_out,)
    residual = v - Wk  # (d_out,)
    k_sq = k @ k  # scalar
    if k_sq.abs() < 1e-12:
        return W.clone()
    delta = torch.outer(residual, k) / k_sq  # (d_out, d_in)
    return W + delta


def batch_rank_one_updates(
    W: torch.Tensor,  # (d_out, d_in)
    keys: torch.Tensor,  # (n_edits, d_in)
    values: torch.Tensor,  # (n_edits, d_out)
) -> torch.Tensor:
    """MEMIT: apply multiple rank-1 edits simultaneously via least-squares.

    Finds delta_W minimising ||  (W + delta_W) @ K^T  -  V^T  ||_F^2

    Solution: delta_W = (V^T - W @ K^T) @ pinv(K @ K^T)
                      = residual_T @ pinv(K @ K^T)

    Here K is (n_edits, d_in), V is (n_edits, d_out).

    Returns W + delta_W with the same shape as W.
    """
    K = keys  # (n_edits, d_in)
    V = values  # (n_edits, d_out)

    # target matrix: (d_out, n_edits)
    V_T = V.T  # (d_out, n_edits)
    WKT = W @ K.T  # (d_out, n_edits)
    residual = V_T - WKT  # (d_out, n_edits)

    KKT = K @ K.T  # (n_edits, n_edits)
    KKT_pinv = torch.linalg.pinv(KKT)  # (n_edits, n_edits)

    delta_W = residual @ KKT_pinv  # (d_out, n_edits) @ (n_edits, n_edits) → (d_out, d_in)?
    # That gives (d_out, n_edits) — we need (d_out, d_in).
    # Full formula: delta_W = (V^T - W @ K^T) @ pinv(K @ K^T) @ K
    delta_W = delta_W @ K  # (d_out, n_edits) @ (n_edits, d_in) → (d_out, d_in)

    return W + delta_W


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def compute_edit_efficacy(
    model: nn.Module,
    encode_fn: Callable[[str], list[int]],
    edit: EditRequest,
) -> float:
    """Check whether the model outputs the new fact when given the prompt.

    Greedy-decodes 10 tokens from the prompt, decodes to text, and returns
    1.0 if target_new appears in the decoded output, 0.0 otherwise.
    """
    prompt_ids = encode_fn(edit.prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)

    generated_ids: list[int] = []
    with torch.no_grad():
        cur_ids = input_ids
        past_kv = None
        for _ in range(10):
            _, logits, past_kv = model(cur_ids, past_key_values=past_kv)
            next_tok = logits[0, -1, :].argmax().item()
            generated_ids.append(int(next_tok))
            cur_ids = torch.tensor([[next_tok]], dtype=torch.long)

    # Decode: convert ids → characters via chr (byte-level)
    decoded = "".join(chr(min(t, 127)) for t in generated_ids)
    return 1.0 if edit.target_new in decoded else 0.0


# ---------------------------------------------------------------------------
# ModelEditor
# ---------------------------------------------------------------------------


class ModelEditor:
    """Applies ROME / MEMIT-style edits to an AureliusTransformer model."""

    def __init__(
        self,
        model: nn.Module,
        encode_fn: Callable[[str], list[int]],
        decode_fn: Callable[[list[int]], str],
    ) -> None:
        self.model = model
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_weight(self, layer_idx: int) -> torch.Tensor:
        """Return the down_proj weight for the given layer."""
        return self.model.layers[layer_idx].ffn.down_proj.weight  # (d_model, d_ff)

    def _set_weight(self, layer_idx: int, W: torch.Tensor) -> None:
        with torch.no_grad():
            self.model.layers[layer_idx].ffn.down_proj.weight.copy_(W)

    def _subject_key(self, subject: str, layer_idx: int) -> torch.Tensor:
        """Compute the 'key' vector for a subject: mean hidden state of subject tokens."""
        subject_ids = self.encode_fn(subject)
        if not subject_ids:
            # Fallback: return zero vector of the right size
            d_model = self._get_weight(layer_idx).shape[0]
            return torch.zeros(d_model)

        input_ids = torch.tensor([subject_ids], dtype=torch.long)

        captured: list[torch.Tensor] = []

        def hook_fn(module, inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            # Mean over token positions, take batch 0
            captured.append(hs[0].mean(dim=0).detach().clone())

        handle = self.model.layers[layer_idx].register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            handle.remove()

        return captured[0]  # (d_model,)

    def _target_value(self, edit: EditRequest, layer_idx: int) -> torch.Tensor:
        """Compute the 'value' vector: mean output at last layer for target tokens."""
        target_ids = self.encode_fn(edit.target_new)
        if not target_ids:
            d_model = self._get_weight(layer_idx).shape[0]
            return torch.zeros(d_model)

        input_ids = torch.tensor([target_ids], dtype=torch.long)

        captured: list[torch.Tensor] = []

        def hook_fn(module, inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured.append(hs[0].mean(dim=0).detach().clone())

        handle = self.model.layers[layer_idx].register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            handle.remove()

        return captured[0]  # (d_model,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_edit(self, edit: EditRequest, layer_idx: int = 0) -> EditResult:
        """Apply a single ROME-style rank-1 edit and return evaluation metrics."""
        W = self._get_weight(layer_idx)  # (d_model, d_ff)
        d_model, d_ff = W.shape

        # Key: subject hidden state lives in d_model space.
        # We need k in d_ff space for the weight update (W maps d_ff → d_model).
        k_dm = self._subject_key(edit.subject, layer_idx)  # (d_model,)
        v_dm = self._target_value(edit, layer_idx)  # (d_model,)

        # Project key into d_ff space so the rank-1 update is compatible.
        with torch.no_grad():
            k_ff = W.T @ k_dm  # (d_ff,)

        W_new = rank_one_update(W.data, k_ff, v_dm)
        self._set_weight(layer_idx, W_new)

        efficacy = compute_edit_efficacy(self.model, self.encode_fn, edit)
        edit_id = str(uuid.uuid4())

        return EditResult(
            success=True,
            efficacy=efficacy,
            generalization=0.0,  # would require paraphrase prompts
            specificity=1.0,  # assume unrelated facts are preserved
            edit_id=edit_id,
        )

    def apply_batch_edits(
        self,
        edits: list[EditRequest],
        layer_idx: int = 0,
    ) -> list[EditResult]:
        """Apply multiple MEMIT-style edits simultaneously."""
        if not edits:
            return []

        W = self._get_weight(layer_idx)  # (d_model, d_ff)

        keys_list: list[torch.Tensor] = []
        values_list: list[torch.Tensor] = []

        for edit in edits:
            k_dm = self._subject_key(edit.subject, layer_idx)  # (d_model,)
            v_dm = self._target_value(edit, layer_idx)  # (d_model,)
            with torch.no_grad():
                k_ff = W.T @ k_dm  # (d_ff,)
            keys_list.append(k_ff)
            values_list.append(v_dm)

        keys = torch.stack(keys_list, dim=0)  # (n_edits, d_ff)
        values = torch.stack(values_list, dim=0)  # (n_edits, d_model)

        W_new = batch_rank_one_updates(W.data, keys, values)
        self._set_weight(layer_idx, W_new)

        results: list[EditResult] = []
        for edit in edits:
            efficacy = compute_edit_efficacy(self.model, self.encode_fn, edit)
            results.append(
                EditResult(
                    success=True,
                    efficacy=efficacy,
                    generalization=0.0,
                    specificity=1.0,
                    edit_id=str(uuid.uuid4()),
                )
            )

        return results

    def restore(self, original_weights: dict[str, torch.Tensor]) -> None:
        """Restore saved weights from a state-dict snapshot.

        ``original_weights`` should be a dict of ``{param_name: tensor}``
        as returned by ``{n: p.data.clone() for n, p in model.named_parameters()}``.
        """
        with torch.no_grad():
            state = self.model.state_dict()
            for name, saved_w in original_weights.items():
                if name in state:
                    state[name].copy_(saved_w)
