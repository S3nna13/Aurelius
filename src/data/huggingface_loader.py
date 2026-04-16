"""Generic HuggingFace dataset loader supporting multiple public datasets.

Datasets covered:
  - Anthropic/hh-rlhf           (preference pairs for DPO/RLHF)
  - tatsu-lab/alpaca             (instruction-following)
  - OpenAssistant/oasst1         (multi-turn assistant conversations)
  - GAIR/lima                    (high-quality instruction data)
  - openbmb/UltraFeedback        (multi-model preference feedback)

The ``datasets`` library (HuggingFace) is used only when actually loading from
the Hub; all parsing logic works on plain Python dicts so tests never need a
network connection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Config & data-class definitions
# ---------------------------------------------------------------------------

@dataclass
class HFDatasetConfig:
    dataset_name: str
    split: str = "train"
    max_samples: int = 1000
    cache_dir: str | None = None


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    source: str = ""


@dataclass
class InstructionSample:
    instruction: str
    input_context: str = ""
    output: str = ""
    source: str = ""
    conversation: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-dataset parsers
# ---------------------------------------------------------------------------

def parse_hh_rlhf_sample(raw: dict) -> PreferencePair:
    """Split chosen/rejected at last '\\n\\nAssistant:' to get prompt vs response.

    Both the chosen and rejected strings share the same conversation prefix up
    to the final assistant turn.  We split on the last occurrence of the marker
    in the *chosen* string to extract the shared prompt, then strip each
    response of the leading newlines that follow the marker.
    """
    marker = "\n\nAssistant:"
    chosen_full: str = raw.get("chosen", "")
    rejected_full: str = raw.get("rejected", "")

    idx = chosen_full.rfind(marker)
    if idx == -1:
        # Fallback: no marker found — treat entire string as response
        return PreferencePair(
            prompt="",
            chosen=chosen_full,
            rejected=rejected_full,
            source="hh-rlhf",
        )

    split_point = idx + len(marker)
    prompt = chosen_full[:split_point]
    chosen_response = chosen_full[split_point:].lstrip(" ")

    # Extract only the response part from the rejected string
    rej_idx = rejected_full.rfind(marker)
    if rej_idx != -1:
        rejected_response = rejected_full[rej_idx + len(marker):].lstrip(" ")
    else:
        rejected_response = rejected_full

    return PreferencePair(
        prompt=prompt,
        chosen=chosen_response,
        rejected=rejected_response,
        source="hh-rlhf",
    )


def parse_alpaca_sample(raw: dict) -> InstructionSample:
    """Parse a tatsu-lab/alpaca record into an InstructionSample."""
    return InstructionSample(
        instruction=raw.get("instruction", ""),
        input_context=raw.get("input", "") or "",
        output=raw.get("output", ""),
        source="alpaca",
        conversation=[],
    )


def parse_oasst_sample(raw: dict) -> InstructionSample | None:
    """Return None if raw['deleted'] is True or role != 'assistant'."""
    if raw.get("deleted", False):
        return None
    if raw.get("role", "") != "assistant":
        return None
    return InstructionSample(
        instruction=raw.get("text", ""),
        input_context="",
        output=raw.get("text", ""),
        source="oasst1",
        conversation=[{"role": raw.get("role", ""), "text": raw.get("text", "")}],
    )


def parse_lima_sample(raw: dict) -> InstructionSample:
    """conversations[0]=user turn, conversations[1]=assistant turn."""
    convs: list[str] = raw.get("conversations", [])
    instruction = convs[0] if len(convs) > 0 else ""
    output = convs[1] if len(convs) > 1 else ""
    conversation = [
        {"role": ("user" if i % 2 == 0 else "assistant"), "text": t}
        for i, t in enumerate(convs)
    ]
    return InstructionSample(
        instruction=instruction,
        input_context="",
        output=output,
        source=raw.get("source", "lima"),
        conversation=conversation,
    )


def parse_ultrafeedback_sample(raw: dict) -> list[PreferencePair]:
    """Pair first vs last completion as chosen vs rejected.

    Returns a list because a single UltraFeedback record can in principle
    generate multiple pairs, but the default implementation produces exactly
    one pair (first completion as *chosen*, last as *rejected*) when there
    are at least two completions.  Returns an empty list when fewer than two
    completions are present.
    """
    instruction: str = raw.get("instruction", "")
    completions: list[dict] = raw.get("completions", [])
    if len(completions) < 2:
        return []

    chosen_completion = completions[0]
    rejected_completion = completions[-1]

    chosen_response = chosen_completion.get("response", "")
    rejected_response = rejected_completion.get("response", "")

    return [
        PreferencePair(
            prompt=instruction,
            chosen=chosen_response,
            rejected=rejected_response,
            source="ultrafeedback",
        )
    ]


# ---------------------------------------------------------------------------
# Loader class
# ---------------------------------------------------------------------------

_DATASET_PARSERS: dict[str, str] = {
    "Anthropic/hh-rlhf": "hh_rlhf",
    "tatsu-lab/alpaca": "alpaca",
    "OpenAssistant/oasst1": "oasst1",
    "GAIR/lima": "lima",
    "openbmb/UltraFeedback": "ultrafeedback",
}


class HuggingFaceLoader:
    """Thin wrapper around HuggingFace ``datasets`` with a dataset-agnostic API.

    Parameters
    ----------
    config:
        Configuration dataclass specifying which dataset to load.
    """

    def __init__(self, config: HFDatasetConfig) -> None:
        self.config = config
        self._data: list[dict] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> list[dict]:
        """Load the dataset from the HuggingFace Hub.

        Raises ``ImportError`` if the ``datasets`` package is not installed.
        """
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required to load HuggingFace datasets. "
                "Install it with: pip install datasets"
            ) from exc

        kwargs: dict = {"split": self.config.split}
        if self.config.cache_dir is not None:
            kwargs["cache_dir"] = self.config.cache_dir

        ds = load_dataset(self.config.dataset_name, **kwargs)
        if self.config.max_samples is not None and self.config.max_samples > 0:
            ds = ds.select(range(min(self.config.max_samples, len(ds))))

        self._data = [dict(sample) for sample in ds]
        return self._data

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _dataset_type(self) -> str:
        return _DATASET_PARSERS.get(self.config.dataset_name, "unknown")

    def as_preference_pairs(self) -> list[PreferencePair]:
        """Convert loaded data to ``PreferencePair`` objects."""
        dtype = self._dataset_type()
        pairs: list[PreferencePair] = []

        for raw in self._data:
            if dtype == "hh_rlhf":
                pairs.append(parse_hh_rlhf_sample(raw))
            elif dtype == "ultrafeedback":
                pairs.extend(parse_ultrafeedback_sample(raw))
            # Other datasets don't yield preference pairs natively

        return pairs

    def as_instruction_samples(self) -> list[InstructionSample]:
        """Convert loaded data to ``InstructionSample`` objects."""
        dtype = self._dataset_type()
        samples: list[InstructionSample] = []

        for raw in self._data:
            if dtype == "alpaca":
                samples.append(parse_alpaca_sample(raw))
            elif dtype == "oasst1":
                result = parse_oasst_sample(raw)
                if result is not None:
                    samples.append(result)
            elif dtype == "lima":
                samples.append(parse_lima_sample(raw))
            elif dtype == "hh_rlhf":
                # Treat the chosen conversation as an instruction sample
                pair = parse_hh_rlhf_sample(raw)
                samples.append(
                    InstructionSample(
                        instruction=pair.prompt,
                        output=pair.chosen,
                        source="hh-rlhf",
                    )
                )

        return samples

    def to_tokenized(
        self,
        tokenizer_fn: Callable[[str], list[int]],
        max_len: int = 512,
    ) -> list[dict]:
        """Tokenize preference pairs and return input_ids / labels tensors.

        Parameters
        ----------
        tokenizer_fn:
            A callable that maps a string to a list of integer token IDs.
        max_len:
            Maximum sequence length; sequences are truncated on the right.

        Returns
        -------
        list[dict]
            Each element is ``{"input_ids": Tensor, "labels": Tensor}`` where
            both tensors are 1-D int64 tensors of length ``<= max_len``.
        """
        pairs = self.as_preference_pairs()
        results: list[dict] = []

        for pair in pairs:
            text = pair.prompt + pair.chosen
            token_ids = tokenizer_fn(text)[:max_len]
            id_tensor = torch.tensor(token_ids, dtype=torch.long)
            results.append({"input_ids": id_tensor, "labels": id_tensor.clone()})

        return results


# ---------------------------------------------------------------------------
# Mock data generators (matching real HuggingFace field schemas)
# ---------------------------------------------------------------------------

def mock_hh_rlhf_data(n: int = 4) -> list[dict]:
    """Generate *n* synthetic hh-rlhf records with real field names."""
    records: list[dict] = []
    for i in range(n):
        shared_prefix = (
            f"\n\nHuman: Question number {i}\n\nAssistant:"
        )
        records.append(
            {
                "chosen": shared_prefix + f" This is a helpful answer {i}.",
                "rejected": shared_prefix + f" This is a less helpful answer {i}.",
            }
        )
    return records


def mock_alpaca_data(n: int = 4) -> list[dict]:
    """Generate *n* synthetic alpaca records with real field names."""
    records: list[dict] = []
    for i in range(n):
        instruction = f"Explain concept number {i}."
        inp = f"Context {i}" if i % 2 == 0 else ""
        output = f"Explanation for concept {i}."
        text = (
            f"Below is an instruction.\n\n### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n### Response:\n{output}"
        )
        records.append(
            {
                "instruction": instruction,
                "input": inp,
                "output": output,
                "text": text,
            }
        )
    return records


def mock_oasst_data(n: int = 4) -> list[dict]:
    """Generate *n* synthetic oasst1 records with real field names.

    Alternates between prompter and assistant roles; every third record is
    marked as deleted to exercise filtering logic.
    """
    records: list[dict] = []
    roles = ["prompter", "assistant"]
    for i in range(n):
        deleted = i % 3 == 2  # every third record deleted
        records.append(
            {
                "message_id": f"msg-{i:04d}",
                "parent_id": f"msg-{i - 1:04d}" if i > 0 else None,
                "text": f"Message text number {i}.",
                "role": roles[i % 2],
                "lang": "en",
                "deleted": deleted,
                "rank": i,
                "message_tree_id": f"tree-{i // 4:04d}",
            }
        )
    return records


def mock_lima_data(n: int = 4) -> list[dict]:
    """Generate *n* synthetic LIMA records with real field names."""
    records: list[dict] = []
    sources = ["stackexchange", "reddit", "wikiHow", "textbooks"]
    for i in range(n):
        records.append(
            {
                "source": sources[i % len(sources)],
                "conversations": [
                    f"User question number {i}.",
                    f"Assistant answer number {i}.",
                ],
            }
        )
    return records


def mock_ultrafeedback_data(n: int = 4) -> list[dict]:
    """Generate *n* synthetic UltraFeedback records with real field names."""
    model_names = ["gpt-4", "claude-2", "llama-2-70b", "mistral-7b"]
    records: list[dict] = []
    for i in range(n):
        completions = [
            {
                "model": model_names[j % len(model_names)],
                "response": f"Response from model {j} for instruction {i}.",
                "annotations": {
                    "helpfulness": {"Rating": str(5 - j), "Rationale": "Good."},
                },
            }
            for j in range(4)
        ]
        records.append(
            {
                "instruction": f"Instruction number {i}.",
                "models": [c["model"] for c in completions],
                "completions": completions,
            }
        )
    return records
