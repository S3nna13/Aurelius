"""Ollama Modelfile generator for local M1 Pro serving of Aurelius 1.3B.

Generates a valid Modelfile targeting GGUF Q4_K_M quantization with ChatML
template, tuned for Apple Silicon inference.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    """Configuration for Ollama Modelfile generation."""

    model_path: str
    """Path to the GGUF model file (Q4_K_M quantized)."""

    model_name: str = "aurelius"
    """Name to register with Ollama."""

    system_prompt: str = (
        "You are Aurelius, a helpful, harmless, and honest AI assistant. "
        "Answer questions clearly and concisely. If you are unsure about "
        "something, say so rather than guessing. Refuse requests for harmful, "
        "illegal, or unethical content."
    )

    context_size: int = 8192
    """Maximum context window in tokens."""

    temperature: float = 0.7
    """Sampling temperature."""

    top_p: float = 0.9
    """Nucleus sampling threshold."""

    top_k: int = 40
    """Top-k sampling threshold."""

    repeat_penalty: float = 1.1
    """Repetition penalty factor."""

    num_gpu: int = 1
    """Number of GPU layers to offload (M1 Pro unified memory)."""

    num_thread: int = 8
    """CPU threads for M1 Pro (8 performance cores)."""

    stop_tokens: list[str] = field(
        default_factory=lambda: ["<|im_end|>", "<|im_start|>"]
    )

    def generate_modelfile(self) -> str:
        """Generate a complete Ollama Modelfile string.

        Returns:
            A valid Modelfile ready to write to disk or pass to `ollama create`.
        """
        stop_lines = "\n".join(
            f'PARAMETER stop "{token}"' for token in self.stop_tokens
        )

        # ChatML template with proper escaping for Ollama's Go template syntax
        chat_template = textwrap.dedent("""\
            {{- if .System }}<|im_start|>system
            {{ .System }}<|im_end|>
            {{ end }}
            {{- range .Messages }}<|im_start|>{{ .Role }}
            {{ .Content }}<|im_end|>
            {{ end }}<|im_start|>assistant
            """).strip()

        modelfile = textwrap.dedent(f"""\
            # Aurelius 1.3B - Ollama Modelfile
            # Quantization: GGUF Q4_K_M
            # Target: Apple M1 Pro (16GB unified memory)

            FROM {self.model_path}

            # --- System prompt ---
            SYSTEM \"\"\"{self.system_prompt}\"\"\"

            # --- ChatML template ---
            TEMPLATE \"\"\"
            {chat_template}
            \"\"\"

            # --- Generation parameters ---
            PARAMETER temperature {self.temperature}
            PARAMETER top_p {self.top_p}
            PARAMETER top_k {self.top_k}
            PARAMETER repeat_penalty {self.repeat_penalty}
            PARAMETER num_ctx {self.context_size}
            PARAMETER num_gpu {self.num_gpu}
            PARAMETER num_thread {self.num_thread}

            # --- Stop tokens ---
            {stop_lines}
        """)

        return modelfile

    def write_modelfile(self, output_path: Path | str) -> Path:
        """Write the Modelfile to disk.

        Args:
            output_path: Destination file path.

        Returns:
            The resolved output path.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.generate_modelfile(), encoding="utf-8")
        return path

    @classmethod
    def from_defaults(cls, model_path: str) -> Self:
        """Create config with sensible defaults for Aurelius 1.3B on M1 Pro."""
        return cls(model_path=model_path)


def generate_default_modelfile(
    model_path: str,
    output_path: str | Path = "configs/ollama.Modelfile",
) -> Path:
    """Convenience function: generate a default Modelfile for Aurelius.

    Args:
        model_path: Path to the GGUF Q4_K_M model file.
        output_path: Where to write the Modelfile.

    Returns:
        Path to the written Modelfile.
    """
    config = OllamaConfig.from_defaults(model_path)
    return config.write_modelfile(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Ollama Modelfile for Aurelius 1.3B"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to GGUF Q4_K_M model file",
    )
    parser.add_argument(
        "--output",
        default="configs/ollama.Modelfile",
        help="Output Modelfile path (default: configs/ollama.Modelfile)",
    )
    args = parser.parse_args()

    out = generate_default_modelfile(args.model_path, args.output)
    print(f"Modelfile written to {out}")
