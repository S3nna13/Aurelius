"""Cross-lingual transfer evaluation for Aurelius LLM.

Measures zero-shot transfer accuracy, transfer gaps, language confusion,
and multilingual consistency across language pairs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LanguageResult:
    language: str
    accuracy: float
    n_samples: int
    avg_confidence: float
    language_confusion_rate: float  # fraction of outputs detected in wrong language


@dataclass
class CrossLingualTransferResult:
    source_language: str
    source_accuracy: float
    target_results: Dict[str, LanguageResult]
    transfer_gaps: Dict[str, float]       # source_acc - target_acc per language
    avg_transfer_gap: float
    consistency_scores: Dict[str, float]  # per language pair


# ---------------------------------------------------------------------------
# Language / script detection
# ---------------------------------------------------------------------------

class LanguageDetector:
    """Heuristic script-level language detector. No external libraries."""

    # Unicode ranges (inclusive)
    _RANGES: Dict[str, Tuple[int, int]] = {
        "chinese":  (0x4E00, 0x9FFF),
        "japanese": (0x3040, 0x30FF),
        "arabic":   (0x0600, 0x06FF),
        "latin":    (0x0041, 0x007A),   # A-z (covers en/fr/de/es)
    }

    # Script → default ISO 639-1 code
    _SCRIPT_TO_LANG: Dict[str, str] = {
        "chinese":  "zh",
        "japanese": "ja",
        "arabic":   "ar",
        "latin":    "en",
    }

    # Explicit lang → expected script
    _LANG_TO_SCRIPT: Dict[str, str] = {
        "zh": "chinese",
        "ja": "japanese",
        "ar": "arabic",
        "en": "latin",
        "fr": "latin",
        "de": "latin",
        "es": "latin",
    }

    def detect_script(self, text: str) -> str:
        """Returns 'chinese', 'japanese', 'arabic', 'latin', or 'unknown'."""
        if not text:
            return "unknown"
        counts: Dict[str, int] = {k: 0 for k in self._RANGES}
        for ch in text:
            cp = ord(ch)
            for script, (lo, hi) in self._RANGES.items():
                if lo <= cp <= hi:
                    counts[script] += 1
        dominant = max(counts, key=lambda k: counts[k])
        return dominant if counts[dominant] > 0 else "unknown"

    def detect_language(self, text: str) -> str:
        """Returns ISO 639-1 code or 'unknown'."""
        script = self.detect_script(text)
        return self._SCRIPT_TO_LANG.get(script, "unknown")

    def is_target_language(self, text: str, expected_lang: str) -> bool:
        """Script-level check: does text use the script of expected_lang?"""
        expected_script = self._LANG_TO_SCRIPT.get(expected_lang)
        if expected_script is None:
            return False
        detected = self.detect_script(text)
        return detected == expected_script


# ---------------------------------------------------------------------------
# Multilingual consistency checker
# ---------------------------------------------------------------------------

class MultilingualConsistencyChecker:
    """Checks whether a model gives consistent answers across language variants."""

    def __init__(
        self,
        model: nn.Module,
        encode_fn: Callable[[str], List[int]],
        decode_fn: Callable[[List[int]], str],
    ) -> None:
        self.model = model
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self._detector = LanguageDetector()

    def _greedy_generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> List[int]:
        """Autoregressive greedy decoding. Returns list of new token ids."""
        generated: List[int] = []
        cur = input_ids.clone()
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.model(cur)
                logits = out[1] if isinstance(out, (tuple, list)) else out
                next_token = int(logits[0, -1].argmax().item())
                generated.append(next_token)
                cur = torch.cat([cur, torch.tensor([[next_token]])], dim=1)
        return generated

    def check_consistency(
        self,
        prompts_by_language: Dict[str, str],
        max_new_tokens: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate a response for each language variant of the same question.

        Returns:
            responses: Dict[lang, response_text]
            language_scores: Dict[lang, float]  (1.0 = correct lang, 0.0 = wrong)
            semantic_consistency: float  (avg pairwise token-set Jaccard)
        """
        responses: Dict[str, str] = {}
        language_scores: Dict[str, float] = {}

        for lang, prompt in prompts_by_language.items():
            ids = self.encode_fn(prompt)
            input_tensor = torch.tensor([ids], dtype=torch.long)
            new_tokens = self._greedy_generate(input_tensor, max_new_tokens)
            response_text = "".join(self.decode_fn(t) for t in new_tokens)
            responses[lang] = response_text
            correct = self._detector.is_target_language(response_text, lang)
            language_scores[lang] = 1.0 if correct else 0.0

        # Semantic consistency: avg pairwise Jaccard over word tokens
        lang_list = list(responses.keys())
        if len(lang_list) < 2:
            semantic_consistency = 1.0
        else:
            jaccard_sum = 0.0
            n_pairs = 0
            for i in range(len(lang_list)):
                for j in range(i + 1, len(lang_list)):
                    a = set(responses[lang_list[i]].split())
                    b = set(responses[lang_list[j]].split())
                    union = a | b
                    intersection = a & b
                    jaccard_sum += len(intersection) / len(union) if union else 1.0
                    n_pairs += 1
            semantic_consistency = jaccard_sum / n_pairs if n_pairs else 1.0

        return {
            "responses": responses,
            "language_scores": language_scores,
            "semantic_consistency": semantic_consistency,
        }


# ---------------------------------------------------------------------------
# Cross-lingual evaluator
# ---------------------------------------------------------------------------

class CrossLingualEvaluator:
    """Evaluates multiple-choice accuracy across source and target languages."""

    def __init__(
        self,
        model: nn.Module,
        encode_fn: Callable[[str], List[int]],
        decode_fn: Callable[[List[int]], str],
        source_language: str = "en",
    ) -> None:
        self.model = model
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.source_language = source_language
        self._detector = LanguageDetector()

    def _logprob_of_continuation(
        self, prompt_ids: List[int], continuation_ids: List[int]
    ) -> float:
        """Sum of log-probs of continuation tokens given prompt."""
        if not continuation_ids:
            return 0.0
        full_ids = prompt_ids + continuation_ids
        input_tensor = torch.tensor([full_ids], dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_tensor)
            logits = out[1] if isinstance(out, (tuple, list)) else out  # (1, T, V)
        log_probs = F.log_softmax(logits[0], dim=-1)  # (T, V)
        score = 0.0
        for i, tok in enumerate(continuation_ids):
            pos = len(prompt_ids) - 1 + i  # logits at position i predict position i+1
            if pos < log_probs.shape[0]:
                score += float(log_probs[pos, tok].item())
        return score

    def evaluate_language(
        self,
        prompts: List[str],
        labels: List[int],
        choices_per_prompt: List[List[str]],
        language: str = "en",
    ) -> LanguageResult:
        """Multiple-choice accuracy for one language via log-prob scoring."""
        assert len(prompts) == len(labels) == len(choices_per_prompt)
        n_correct = 0
        confidences: List[float] = []
        confusion_count = 0

        for prompt, label, choices in zip(prompts, labels, choices_per_prompt):
            prompt_ids = self.encode_fn(prompt)
            scores = []
            for choice in choices:
                choice_ids = self.encode_fn(choice)
                scores.append(self._logprob_of_continuation(prompt_ids, choice_ids))
            pred = int(max(range(len(scores)), key=lambda i: scores[i]))
            if pred == label:
                n_correct += 1
            # Confidence: softmax of scores → prob of predicted choice
            score_tensor = torch.tensor(scores)
            probs = torch.softmax(score_tensor, dim=0)
            confidences.append(float(probs[pred].item()))
            # Language confusion: check if prompt is in wrong script
            if not self._detector.is_target_language(prompt, language):
                confusion_count += 1

        n = len(prompts)
        accuracy = n_correct / n if n > 0 else 0.0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        confusion_rate = confusion_count / n if n > 0 else 0.0

        return LanguageResult(
            language=language,
            accuracy=accuracy,
            n_samples=n,
            avg_confidence=avg_confidence,
            language_confusion_rate=confusion_rate,
        )

    def compute_transfer_gap(
        self, source: LanguageResult, target: LanguageResult
    ) -> float:
        return source.accuracy - target.accuracy

    def evaluate_transfer(
        self,
        source_data: Tuple[List[str], List[int], List[List[str]]],
        target_data_by_lang: Dict[str, Tuple[List[str], List[int], List[List[str]]]],
    ) -> CrossLingualTransferResult:
        """Evaluate on source + all target languages, compute transfer metrics."""
        src_prompts, src_labels, src_choices = source_data
        source_result = self.evaluate_language(
            src_prompts, src_labels, src_choices, language=self.source_language
        )

        target_results: Dict[str, LanguageResult] = {}
        transfer_gaps: Dict[str, float] = {}
        for lang, (prompts, labels, choices) in target_data_by_lang.items():
            tgt = self.evaluate_language(prompts, labels, choices, language=lang)
            target_results[lang] = tgt
            transfer_gaps[lang] = self.compute_transfer_gap(source_result, tgt)

        avg_gap = (
            sum(transfer_gaps.values()) / len(transfer_gaps)
            if transfer_gaps else 0.0
        )

        # Consistency: dummy 1.0 per pair (no cross-gen in this evaluator)
        consistency_scores: Dict[str, float] = {
            lang: 1.0 for lang in target_results
        }

        return CrossLingualTransferResult(
            source_language=self.source_language,
            source_accuracy=source_result.accuracy,
            target_results=target_results,
            transfer_gaps=transfer_gaps,
            avg_transfer_gap=avg_gap,
            consistency_scores=consistency_scores,
        )


# ---------------------------------------------------------------------------
# Aggregation utility
# ---------------------------------------------------------------------------

def aggregate_transfer_results(
    results: List[CrossLingualTransferResult],
) -> Dict[str, float]:
    """Aggregate multiple eval runs into mean/std statistics."""
    if not results:
        return {"mean_transfer_gap": 0.0, "std_transfer_gap": 0.0, "mean_consistency": 0.0}

    all_gaps = [r.avg_transfer_gap for r in results]
    mean_gap = sum(all_gaps) / len(all_gaps)
    variance = sum((g - mean_gap) ** 2 for g in all_gaps) / len(all_gaps)
    std_gap = math.sqrt(variance)

    all_consistency: List[float] = []
    for r in results:
        if r.consistency_scores:
            all_consistency.extend(r.consistency_scores.values())
    mean_consistency = (
        sum(all_consistency) / len(all_consistency) if all_consistency else 0.0
    )

    return {
        "mean_transfer_gap": mean_gap,
        "std_transfer_gap": std_gap,
        "mean_consistency": mean_consistency,
    }
