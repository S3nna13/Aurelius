"""Convert a HuggingFace checkpoint to GGUF Q4_K_M format.

Workflow:
  1. Run llama.cpp's convert_hf_to_gguf.py to produce an F16 GGUF
  2. Run llama-quantize to produce Q4_K_M
  3. Validate the output file size (~800MB for a 1.3B parameter model)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("aurelius.convert_gguf")

# Expected file size range for a 1.3B Q4_K_M model (in bytes)
EXPECTED_MIN_SIZE_MB: int = 600
EXPECTED_MAX_SIZE_MB: int = 1100


@dataclass(frozen=True, slots=True)
class ConversionConfig:
    """Configuration for HF-to-GGUF conversion."""

    hf_model_path: Path
    """Path to the HuggingFace model directory (or Hub ID)."""

    output_dir: Path = Path("models/gguf")
    """Directory to store output GGUF files."""

    llama_cpp_dir: Path = Path("vendor/llama.cpp")
    """Path to the llama.cpp repository root."""

    quantization_type: str = "Q4_K_M"
    """GGUF quantization type."""

    model_name: str = "aurelius-1.3b"
    """Base name for output files."""

    @property
    def f16_gguf_path(self) -> Path:
        return self.output_dir / f"{self.model_name}-f16.gguf"

    @property
    def quantized_gguf_path(self) -> Path:
        return self.output_dir / f"{self.model_name}-{self.quantization_type.lower()}.gguf"

    @property
    def convert_script(self) -> Path:
        return self.llama_cpp_dir / "convert_hf_to_gguf.py"

    @property
    def quantize_binary(self) -> Path:
        return self.llama_cpp_dir / "build" / "bin" / "llama-quantize"


def _run(cmd: list[str], description: str) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with logging and error handling."""
    logger.info("Running: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.error("STDOUT:\n%s", result.stdout)
        logger.error("STDERR:\n%s", result.stderr)
        msg = f"{description} failed with return code {result.returncode}"
        raise RuntimeError(msg)
    return result


def validate_prerequisites(config: ConversionConfig) -> None:
    """Verify that required tools and paths exist.

    Raises:
        FileNotFoundError: If a required file or directory is missing.
    """
    if not config.hf_model_path.exists():
        raise FileNotFoundError(
            f"HuggingFace model path not found: {config.hf_model_path}"
        )
    if not config.convert_script.exists():
        raise FileNotFoundError(
            f"llama.cpp convert script not found: {config.convert_script}. "
            f"Clone llama.cpp to {config.llama_cpp_dir}"
        )

    # Check for quantize binary; fall back to searching PATH
    if not config.quantize_binary.exists():
        if shutil.which("llama-quantize") is None:
            raise FileNotFoundError(
                f"llama-quantize binary not found at {config.quantize_binary} "
                "or in PATH. Build llama.cpp first."
            )


def convert_hf_to_gguf_f16(config: ConversionConfig) -> Path:
    """Step 1: Convert HuggingFace checkpoint to F16 GGUF.

    Returns:
        Path to the F16 GGUF file.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(config.convert_script),
        str(config.hf_model_path),
        "--outfile", str(config.f16_gguf_path),
        "--outtype", "f16",
    ]
    _run(cmd, "HF-to-GGUF conversion")

    if not config.f16_gguf_path.exists():
        msg = f"Expected F16 GGUF not found at {config.f16_gguf_path}"
        raise FileNotFoundError(msg)

    size_mb = config.f16_gguf_path.stat().st_size / (1024 * 1024)
    logger.info("F16 GGUF created: %s (%.1f MB)", config.f16_gguf_path, size_mb)
    return config.f16_gguf_path


def quantize_gguf(config: ConversionConfig) -> Path:
    """Step 2: Quantize F16 GGUF to Q4_K_M.

    Returns:
        Path to the quantized GGUF file.
    """
    quantize_bin = str(config.quantize_binary)
    if not config.quantize_binary.exists():
        # Fall back to PATH
        found = shutil.which("llama-quantize")
        if found is None:
            raise FileNotFoundError("llama-quantize not found")
        quantize_bin = found

    cmd = [
        quantize_bin,
        str(config.f16_gguf_path),
        str(config.quantized_gguf_path),
        config.quantization_type,
    ]
    _run(cmd, "GGUF quantization")

    if not config.quantized_gguf_path.exists():
        msg = f"Expected quantized GGUF not found at {config.quantized_gguf_path}"
        raise FileNotFoundError(msg)

    return config.quantized_gguf_path


def validate_output(config: ConversionConfig) -> None:
    """Step 3: Validate the quantized GGUF file size.

    For a 1.3B parameter model with Q4_K_M quantization,
    we expect approximately 800MB (600-1100MB acceptable range).

    Raises:
        ValueError: If file size is outside expected range.
    """
    path = config.quantized_gguf_path
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    logger.info(
        "Quantized GGUF: %s (%.1f MB, %d bytes)",
        path,
        size_mb,
        size_bytes,
    )

    if size_mb < EXPECTED_MIN_SIZE_MB:
        raise ValueError(
            f"Output file too small: {size_mb:.1f} MB "
            f"(expected >= {EXPECTED_MIN_SIZE_MB} MB for 1.3B Q4_K_M). "
            "Conversion may have failed silently."
        )

    if size_mb > EXPECTED_MAX_SIZE_MB:
        raise ValueError(
            f"Output file too large: {size_mb:.1f} MB "
            f"(expected <= {EXPECTED_MAX_SIZE_MB} MB for 1.3B Q4_K_M). "
            "Wrong quantization type or model size?"
        )

    logger.info("Validation passed: %.1f MB is within expected range", size_mb)


def convert(config: ConversionConfig) -> Path:
    """Run the full HF-to-GGUF Q4_K_M conversion pipeline.

    Args:
        config: Conversion configuration.

    Returns:
        Path to the final quantized GGUF file.
    """
    logger.info("=== Aurelius GGUF Conversion Pipeline ===")
    logger.info("Model: %s", config.hf_model_path)
    logger.info("Target quantization: %s", config.quantization_type)

    validate_prerequisites(config)
    convert_hf_to_gguf_f16(config)
    quantize_gguf(config)
    validate_output(config)

    logger.info("Conversion complete: %s", config.quantized_gguf_path)

    # Clean up intermediate F16 file
    if config.f16_gguf_path.exists():
        logger.info("Removing intermediate F16 GGUF: %s", config.f16_gguf_path)
        config.f16_gguf_path.unlink()

    return config.quantized_gguf_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Convert HuggingFace checkpoint to GGUF Q4_K_M"
    )
    parser.add_argument(
        "--hf-model-path",
        required=True,
        type=Path,
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--output-dir",
        default="models/gguf",
        type=Path,
        help="Output directory for GGUF files",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        default="vendor/llama.cpp",
        type=Path,
        help="Path to llama.cpp repository",
    )
    parser.add_argument(
        "--model-name",
        default="aurelius-1.3b",
        help="Base name for output files",
    )
    parser.add_argument(
        "--keep-f16",
        action="store_true",
        help="Keep intermediate F16 GGUF file",
    )
    args = parser.parse_args()

    cfg = ConversionConfig(
        hf_model_path=args.hf_model_path,
        output_dir=args.output_dir,
        llama_cpp_dir=args.llama_cpp_dir,
        model_name=args.model_name,
    )

    result_path = convert(cfg)
    print(f"\nDone. Quantized model: {result_path}")
