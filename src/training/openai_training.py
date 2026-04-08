"""OpenAI-inspired training algorithms for Aurelius.

Implements algorithms INSPIRED BY OpenAI datasets (gsm8k, graphwalks, mmmlu,
coval, healthbench) using pure PyTorch only -- no external APIs, no network
calls, no HuggingFace datasets library.

Sections:
  1. GSM8K-inspired: Math Answer Verifier + Scratchpad Loss Weighting
  2. MMMLU-inspired: Multilingual Multiple-Choice Evaluator
  3. CoVal-inspired: Borda Count Preference Aggregation
  4. GraphWalks-inspired: Graph Reasoning Evaluator
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


# -- 1. GSM8K-inspired: Math Answer Verifier ----------------------------------


def extract_final_answer(text: str, delimiter: str = "####") -> str | None:
    """Extract everything after the last occurrence of *delimiter*.

    Strips whitespace, commas, and dollar signs from the result.
    Returns None if the delimiter is not found.
    """
    idx = text.rfind(delimiter)
    if idx == -1:
        return None
    raw = text[idx + len(delimiter):]
    cleaned = raw.strip().replace(",", "").replace("$", "")
    return cleaned


def verify_numeric_answer(
    predicted: str,
    gold: str,
    tolerance: float = 1e-6,
) -> bool:
    """Parse both strings as floats and compare within *tolerance*.

    Comma-separated numbers (e.g. "1,234") are accepted by stripping
    commas before parsing.  Returns False if either value cannot be parsed.
    """
    def _parse(s: str):
        try:
            return float(s.replace(",", "").strip())
        except (ValueError, AttributeError):
            return None

    p_val = _parse(predicted)
    g_val = _parse(gold)
    if p_val is None or g_val is None:
        return False
    return abs(p_val - g_val) <= tolerance


def answer_verification_reward(
    predicted_text: str,
    gold_answer: str,
    format_reward: float = 0.1,
    answer_reward: float = 1.0,
) -> float:
    """Return a scalar reward based on answer correctness.

    * answer_reward  -- correct answer (with or without #### delimiter).
    * format_reward  -- #### delimiter present but answer is wrong.
    * 0.0            -- no delimiter and wrong answer.
    """
    extracted = extract_final_answer(predicted_text)
    has_format = extracted is not None

    if has_format and verify_numeric_answer(extracted, gold_answer):
        return answer_reward
    if has_format:
        return format_reward
    return 0.0


# -- 2. GSM8K-inspired: Scratchpad Loss Weighting ----------------------------


def split_scratchpad_answer_mask(
    input_ids: Tensor,
    delimiter_id: int,
    answer_weight: float = 5.0,
) -> Tensor:
    """Build a per-token loss weight mask.

    * Positions before the last delimiter_id token -> weight 1.0 (scratchpad).
    * Positions at or after the last delimiter_id token -> weight answer_weight.
    * Rows where delimiter_id is absent -> uniform weight 1.0.

    Args:
        input_ids:      (B, T) integer token tensor.
        delimiter_id:   Token id representing the #### delimiter.
        answer_weight:  Weight assigned to answer tokens.

    Returns:
        (B, T) float tensor of per-token weights.
    """
    B, T = input_ids.shape
    mask = torch.ones(B, T, dtype=torch.float, device=input_ids.device)

    for b in range(B):
        positions = (input_ids[b] == delimiter_id).nonzero(as_tuple=False)
        if positions.numel() == 0:
            continue
        last_pos = int(positions[-1].item())
        mask[b, last_pos:] = answer_weight

    return mask


def weighted_lm_loss(
    logits: Tensor,
    labels: Tensor,
    weights: Tensor,
) -> Tensor:
    """Weighted cross-entropy language-modelling loss.

    Args:
        logits:  (B, T, V) unnormalised logit tensor.
        labels:  (B, T) token ids; positions with -100 are ignored.
        weights: (B, T) per-token loss weight scalars.

    Returns:
        Scalar loss (mean over non-ignored positions after weighting).
    """
    B, T, V = logits.shape

    per_token_ce = F.cross_entropy(
        logits.reshape(B * T, V),
        labels.reshape(B * T),
        ignore_index=-100,
        reduction="none",
    ).reshape(B, T)

    valid_mask = (labels != -100).float()
    weighted_ce = per_token_ce * weights * valid_mask

    n_valid = valid_mask.sum().clamp(min=1.0)
    return weighted_ce.sum() / n_valid


# -- 3. MMMLU-inspired: Multilingual Multiple-Choice Evaluator ---------------


@dataclass
class MultipleChoiceResult:
    """Aggregated results from evaluate_multiple_choice."""

    n_correct: int
    n_total: int
    accuracy: float
    by_subject: dict[str, float] = field(default_factory=dict)
    by_language: dict[str, float] = field(default_factory=dict)


def _format_single_mc(question: str, choices: dict[str, str]) -> str:
    lines = [f"Question: {question}"]
    for letter in ["A", "B", "C", "D"]:
        if letter in choices:
            lines.append(f"{letter}) {choices[letter]}")
    lines.append("Answer:")
    return "\n".join(lines) + "\n"


def format_multiple_choice_prompt(
    question: str,
    choices: dict[str, str],
    few_shot_examples: list[dict] | None = None,
) -> str:
    """Format a multiple-choice question as a text prompt.

    Output format:
        Question: {question}
        A) {choice_a}
        B) {choice_b}
        C) {choice_c}
        D) {choice_d}
        Answer:

    Few-shot examples (each a dict with question, choices, answer keys)
    are prepended when provided.
    """
    parts: list[str] = []

    if few_shot_examples:
        for ex in few_shot_examples:
            parts.append(_format_single_mc(ex["question"], ex["choices"]))
            parts.append(f"Answer: {ex['answer']}\n")

    parts.append(_format_single_mc(question, choices))
    return "".join(parts)


def evaluate_multiple_choice(
    model,
    samples: list[dict],
    tokenizer_encode: Callable,
    tokenizer_decode: Callable,
    batch_size: int = 8,
    few_shot_prefix: str = "",
) -> MultipleChoiceResult:
    """Evaluate model on multiple-choice samples by log-prob scoring.

    For each sample the four option tokens A, B, C, D are scored by their
    log-probability at the last position of the prompt.  The option with the
    highest log-prob is taken as the prediction.
    """
    model.eval()

    option_ids: dict[str, int] = {}
    for letter in ["A", "B", "C", "D"]:
        ids = tokenizer_encode(letter)
        option_ids[letter] = ids[0]

    n_correct = 0
    by_subject: dict[str, list[bool]] = {}
    by_language: dict[str, list[bool]] = {}

    with torch.no_grad():
        for sample in samples:
            prompt_text = few_shot_prefix + format_multiple_choice_prompt(
                question=sample["question"],
                choices=sample["choices"],
            )

            ids = tokenizer_encode(prompt_text)
            input_ids = torch.tensor([ids], dtype=torch.long)

            logits = model(input_ids)
            last_logits = logits[0, -1, :]
            log_probs = torch.log_softmax(last_logits, dim=-1)

            best_letter = max(
                [ltr for ltr in ["A", "B", "C", "D"] if ltr in sample["choices"]],
                key=lambda ltr: log_probs[option_ids[ltr]].item(),
            )

            correct = best_letter == sample.get("answer", "")

            if correct:
                n_correct += 1

            subj = sample.get("subject", "unknown")
            by_subject.setdefault(subj, []).append(correct)

            lang = sample.get("language", "")
            if lang:
                by_language.setdefault(lang, []).append(correct)

    n_total = len(samples)
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    return MultipleChoiceResult(
        n_correct=n_correct,
        n_total=n_total,
        accuracy=accuracy,
        by_subject={s: sum(v) / len(v) for s, v in by_subject.items()},
        by_language={lng: sum(v) / len(v) for lng, v in by_language.items()},
    )


# -- 4. CoVal-inspired: Borda Count Preference Aggregation -------------------


def parse_ranking_string(ranking: str) -> list[list[str]]:
    """Parse a ranking string like "B>A>C=D" into ordered groups.

    Ties within a rank level (=) are placed in the same inner list.

    Examples:
        parse_ranking_string("B>A>C=D") -> [["B"], ["A"], ["C", "D"]]
        parse_ranking_string("A>B")     -> [["A"], ["B"]]
    """
    groups: list[list[str]] = []
    for rank_group in ranking.split(">"):
        tied = [item.strip() for item in rank_group.split("=") if item.strip()]
        if tied:
            groups.append(tied)
    return groups


def borda_count(
    rankings: list[list[list[str]]],
    candidates: list[str],
) -> dict[str, float]:
    """Compute Borda count scores from a list of annotator rankings.

    Each ranking is a list[list[str]] where inner lists represent tied groups
    in preference order (best first).  Tied candidates share the average of
    the points their positions would otherwise receive.

    Args:
        rankings:   List of annotator rankings (each from parse_ranking_string).
        candidates: Full list of candidate names.

    Returns:
        {candidate: total_score} dict.
    """
    n = len(candidates)
    scores: dict[str, float] = {c: 0.0 for c in candidates}

    for ranking in rankings:
        pos = 0
        for group in ranking:
            group_size = len(group)
            group_points = sum(n - 1 - (pos + k) for k in range(group_size))
            avg_points = group_points / group_size
            for candidate in group:
                if candidate in scores:
                    scores[candidate] += avg_points
            pos += group_size

    return scores


def majority_vote_preference(
    rankings: list[list[list[str]]],
    candidate_a: str,
    candidate_b: str,
) -> tuple[str, float]:
    """Count how many annotators prefer candidate_a over candidate_b.

    An annotator prefers A over B if A appears in an earlier group than B.
    Ties (same group) count as neither preference.

    Returns:
        (winner, confidence) where confidence is the fraction of decisive
        preferences that favour the winner.  If every ranking is a tie the
        first candidate is returned with confidence 0.5.
    """
    prefer_a = 0
    prefer_b = 0

    for ranking in rankings:
        pos_a = pos_b = None
        for idx, group in enumerate(ranking):
            if candidate_a in group:
                pos_a = idx
            if candidate_b in group:
                pos_b = idx

        if pos_a is None or pos_b is None:
            continue

        if pos_a < pos_b:
            prefer_a += 1
        elif pos_b < pos_a:
            prefer_b += 1

    total_decisive = prefer_a + prefer_b
    if total_decisive == 0:
        return candidate_a, 0.5

    if prefer_a >= prefer_b:
        return candidate_a, prefer_a / total_decisive
    return candidate_b, prefer_b / total_decisive


def aggregate_to_dpo_pairs(
    responses: dict[str, str],
    rankings: list[list[list[str]]],
    min_confidence: float = 0.6,
) -> list[tuple[str, str]]:
    """Derive DPO training pairs from aggregated annotator rankings.

    Uses borda_count to rank all responses, then pairs the highest-ranked
    response against the lowest-ranked response, provided the
    majority_vote_preference confidence between them exceeds min_confidence.

    Args:
        responses:      {label: text} mapping, e.g. {"A": "...", "B": "..."}.
        rankings:       Annotator rankings (list of parsed ranking strings).
        min_confidence: Minimum majority-vote confidence to include a pair.

    Returns:
        List of (chosen_text, rejected_text) pairs.
    """
    candidates = list(responses.keys())
    if len(candidates) < 2:
        return []

    scores = borda_count(rankings, candidates)
    sorted_candidates = sorted(candidates, key=lambda c: scores[c], reverse=True)

    best = sorted_candidates[0]
    worst = sorted_candidates[-1]

    winner, confidence = majority_vote_preference(rankings, best, worst)
    if confidence < min_confidence:
        return []

    chosen_text = responses[winner]
    loser = worst if winner == best else best
    rejected_text = responses[loser]
    return [(chosen_text, rejected_text)]


# -- 5. GraphWalks-inspired: Graph Reasoning Evaluator -----------------------


def graphwalks_f1(predicted_nodes: list[str], gold_nodes: list[str]) -> float:
    """Set-based F1 between two node lists (from the GraphWalks paper).

    Both empty -> 1.0 (vacuously correct).
    One empty, one non-empty -> 0.0.
    """
    if len(predicted_nodes) == 0 and len(gold_nodes) == 0:
        return 1.0

    pred_set = set(predicted_nodes)
    gold_set = set(gold_nodes)

    if not pred_set or not gold_set:
        return 0.0

    n_overlap = len(pred_set & gold_set)
    precision = n_overlap / len(pred_set)
    recall = n_overlap / len(gold_set)

    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def extract_graph_answer_nodes(text: str) -> list[str]:
    """Extract the node list from a "Final Answer: [node1, node2, ...]" pattern.

    Returns an empty list if the pattern is not found.
    """
    match = re.search(r"Final Answer:\s*\[([^\]]*)\]", text)
    if not match:
        return []
    content = match.group(1).strip()
    if not content:
        return []
    return [n.strip() for n in content.split(",") if n.strip()]


def evaluate_graph_reasoning(
    model,
    prompts: list[str],
    gold_answers: list[list[str]],
    tokenizer_encode: Callable,
    tokenizer_decode: Callable,
    max_new_tokens: int = 256,
) -> dict[str, float]:
    """Evaluate model on graph-reasoning prompts using set-based F1.

    The model is expected to output a "Final Answer: [node1, node2, ...]"
    string which is parsed by extract_graph_answer_nodes.

    Args:
        model:             Model with generate(input_ids, max_new_tokens) method.
        prompts:           Plain-text graph problem prompts.
        gold_answers:      Corresponding gold node lists.
        tokenizer_encode:  str -> list[int] callable.
        tokenizer_decode:  Tensor | list[int] -> str callable.
        max_new_tokens:    Maximum tokens to generate per prompt.

    Returns:
        {"mean_f1": float, "exact_match_rate": float, "n_samples": int}
    """
    model.eval()
    f1_scores: list[float] = []
    exact_matches: list[bool] = []

    with torch.no_grad():
        for prompt, gold in zip(prompts, gold_answers):
            ids = tokenizer_encode(prompt)
            input_ids = torch.tensor([ids], dtype=torch.long)

            output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
            new_ids = output_ids[0, len(ids):]
            generated_text = tokenizer_decode(new_ids)

            predicted = extract_graph_answer_nodes(generated_text)
            f1 = graphwalks_f1(predicted, gold)
            f1_scores.append(f1)
            exact_matches.append(set(predicted) == set(gold))

    n = len(f1_scores)
    return {
        "mean_f1": sum(f1_scores) / n if n > 0 else 0.0,
        "exact_match_rate": sum(exact_matches) / n if n > 0 else 0.0,
        "n_samples": n,
    }
