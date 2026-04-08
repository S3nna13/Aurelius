"""ROME: Rank-one model editing for factual associations.

Edits the fact (subject → new_object) by computing a rank-1 update
to the down_proj (second linear layer) of a specific MLP layer.

Algorithm:
1. Run forward pass on "[subject] [relation]" prompt
2. Extract hidden state k at the subject's LAST token position, at target layer
3. Compute target v: the hidden state that would produce new_object as output
4. Compute rank-1 update: ΔW = outer(v - W_out @ k, k) / (k^T @ k)
5. Apply: W_out += ΔW

Reference: Meng et al. 2022 "Locating and Editing Factual Associations in GPT"
           (arXiv:2202.05262)
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ROMEConfig:
    layer_idx: int = 12            # which layer to edit (middle layers usually best)
    n_gradient_steps: int = 20     # steps to compute target vector v
    v_lr: float = 0.1              # learning rate for v optimization
    kl_factor: float = 0.0625      # KL penalty weight (prevent distribution shift)


@dataclass
class ROMEEdit:
    """Specification for a single fact edit."""
    prompt: str              # "[subject] [relation]" e.g. "The Eiffel Tower is located in"
    target_new: str          # new target token(s) e.g. " Berlin"
    target_true: str | None = None  # original target (for computing KL penalty)
    subject: str = ""        # subject string for locating hidden state


def get_subject_hidden_state(
    model: nn.Module,
    input_ids: torch.Tensor,   # (1, S)
    subject_last_token_pos: int,
    layer_idx: int,
) -> torch.Tensor:
    """Extract hidden state k at subject's last token, after the target layer.

    Returns (D,) vector — the "key" for the rank-1 edit.
    """
    captured: list[torch.Tensor] = []

    def hook_fn(module, input, output):
        # TransformerBlock returns (hidden_state, kv_tuple)
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # hidden: (B, S, D) — capture the subject token position
        captured.append(hidden[0, subject_last_token_pos, :].detach().clone())

    layer = model.layers[layer_idx]
    handle = layer.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    return captured[0]


def compute_target_vector(
    model: nn.Module,
    edit: ROMEEdit,
    tokenizer_fn,              # callable: str → list[int]
    k: torch.Tensor,           # (D,) key vector
    layer_idx: int,
    n_steps: int = 20,
    lr: float = 0.1,
) -> torch.Tensor:
    """Optimize v such that the model outputs target_new given the patched hidden state.

    Simple approach: directly optimize v to maximize log P(target_token | v_patched).
    Returns (D,) target value vector.
    """
    # Tokenize prompt and get target token id
    prompt_ids = tokenizer_fn(edit.prompt)
    target_ids = tokenizer_fn(edit.target_new)
    target_token_id = target_ids[0]  # use first token of target

    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
    patch_pos = len(prompt_ids) - 1  # patch at last token of prompt

    # Get W_out (down_proj): shape (d_model, d_ff)
    w_out = model.layers[layer_idx].ffn.down_proj.weight  # (d_model, d_ff)

    # Initialize v as the current output of W_out applied to k
    # k is (D,) = (d_model,); but W_out maps d_ff -> d_model
    # For the rank-1 update, k is in d_model space and v is also d_model
    # We optimize v directly in d_model space
    v = k.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([v], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()

        # Patch hook: replace layer output at patch_pos with v
        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                hidden, rest = output[0], output[1:]
                patched = hidden.clone()
                patched[0, patch_pos, :] = v
                return (patched,) + rest
            else:
                patched = output.clone()
                patched[0, patch_pos, :] = v
                return patched

        handle = model.layers[layer_idx].register_forward_hook(patch_hook)
        try:
            _, logits, _ = model(prompt_tensor)
        finally:
            handle.remove()

        # logits: (1, S, vocab_size) — use last token position
        last_logits = logits[0, -1, :]
        log_probs = F.log_softmax(last_logits, dim=-1)
        loss = -log_probs[target_token_id]
        loss.backward()
        optimizer.step()

    return v.detach()


def apply_rome_edit(
    model: nn.Module,
    k: torch.Tensor,     # (D,) key vector
    v: torch.Tensor,     # (D,) target value vector
    layer_idx: int,
) -> None:
    """Apply rank-1 update to down_proj of FFN at layer_idx.

    ΔW = outer(v - W_out @ k, k) / (k^T @ k)
    W_out += ΔW

    W_out is model.layers[layer_idx].ffn.down_proj with shape (d_model, d_ff).
    k and v are both in d_model space.

    Since W_out maps d_ff → d_model, but ROME operates in residual (d_model) space,
    we treat k and v as d_model vectors and update W_out as an (d_model, d_model)
    effective map via the rank-1 correction in output space:

        ΔW_col = outer(v - W_out_eff @ k, k) / (k^T k)

    where W_out_eff is the (d_model, d_model) identity for the hidden state path.
    Concretely: the rank-1 update is applied directly to the weight matrix
    treating k as a right-vector in d_model and computing the residual in d_model.
    """
    w_out = model.layers[layer_idx].ffn.down_proj.weight  # (d_model, d_ff)

    # Project k into d_ff space using w_out^T so the rank-1 update is compatible
    # w_out: (d_model, d_ff)
    # We want: w_out @ k_ff ≈ v, so k_ff = w_out^T @ k (pseudo-inverse approach)
    # rank-1 update: ΔW = outer(v - w_out @ k_ff, k_ff) / (k_ff^T k_ff)
    with torch.no_grad():
        k_ff = w_out.T @ k          # (d_ff,) — project k into d_ff space
        w_k = w_out @ k_ff           # (d_model,) — current output
        residual = v - w_k           # (d_model,) — correction needed
        k_sq = k_ff @ k_ff           # scalar
        if k_sq.abs() < 1e-10:
            return
        delta_w = torch.outer(residual, k_ff) / k_sq  # (d_model, d_ff)
        w_out.add_(delta_w)


def rome_edit(
    model: nn.Module,
    edit: ROMEEdit,
    tokenizer_fn,          # callable: str → list[int]
    cfg: ROMEConfig | None = None,
) -> None:
    """Full ROME edit pipeline. Modifies model in place.

    1. Tokenize prompt + locate subject token position
    2. Extract key vector k from target layer
    3. Optimize target value vector v
    4. Apply rank-1 update to down_proj
    """
    if cfg is None:
        cfg = ROMEConfig()

    # Tokenize prompt
    prompt_ids = tokenizer_fn(edit.prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)

    # Find subject's last token position in the prompt
    subject_last_token_pos = len(prompt_ids) - 1
    if edit.subject:
        subject_ids = tokenizer_fn(edit.subject)
        # Search for subject tokens in prompt_ids
        for i in range(len(prompt_ids) - len(subject_ids), -1, -1):
            if prompt_ids[i : i + len(subject_ids)] == subject_ids:
                subject_last_token_pos = i + len(subject_ids) - 1
                break

    # Step 1: Extract key vector k
    k = get_subject_hidden_state(
        model=model,
        input_ids=input_ids,
        subject_last_token_pos=subject_last_token_pos,
        layer_idx=cfg.layer_idx,
    )

    # Step 2: Compute target value vector v
    v = compute_target_vector(
        model=model,
        edit=edit,
        tokenizer_fn=tokenizer_fn,
        k=k,
        layer_idx=cfg.layer_idx,
        n_steps=cfg.n_gradient_steps,
        lr=cfg.v_lr,
    )

    # Step 3: Apply rank-1 update
    apply_rome_edit(
        model=model,
        k=k,
        v=v,
        layer_idx=cfg.layer_idx,
    )
