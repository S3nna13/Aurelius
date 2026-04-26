"""Pure-PyTorch Whisper-inspired mel spectrogram extractor and audio encoder.

No librosa, no torchaudio — everything is implemented from scratch using
torch.stft and hand-built mel filterbanks.

Architecture overview:
    waveform (B, T)
        -> MelSpectrogramExtractor  -> (B, n_mels, T_frames)
        -> WhisperStyleEncoder      -> (B, T_frames//2, output_dim)

The output of WhisperStyleEncoder is compatible with MultiModalProjector
(it expects features of shape (B, N, d) where d = output_dim).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MelConfig:
    """Configuration for mel spectrogram extraction."""

    sample_rate: int = 16000  # audio sample rate in Hz
    n_fft: int = 400  # FFT window size
    hop_length: int = 160  # samples between consecutive STFT windows
    n_mels: int = 128  # number of mel filterbank bins (Whisper v3 uses 128)
    fmin: float = 0.0  # minimum mel frequency in Hz
    fmax: float = 8000.0  # maximum mel frequency in Hz
    max_frames: int = 3000  # maximum number of time frames (30s at 10ms hop)
    normalize: bool = True  # normalize log-mel values to [-1, 1]


# ---------------------------------------------------------------------------
# Hz <-> Mel conversion (HTK formula)
# ---------------------------------------------------------------------------


def hz_to_mel(hz: float | Tensor) -> float | Tensor:
    """Convert frequency in Hz to mel scale using HTK formula.

    Formula: mel = 2595 * log10(1 + hz / 700)

    Args:
        hz: scalar float or Tensor of frequencies in Hz.

    Returns:
        Mel-scale value(s) with same type as input.
    """
    if isinstance(hz, Tensor):
        return 2595.0 * torch.log10(1.0 + hz / 700.0)
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float | Tensor) -> float | Tensor:
    """Convert mel scale back to Hz (inverse of hz_to_mel).

    Formula: hz = 700 * (10^(mel / 2595) - 1)

    Args:
        mel: scalar float or Tensor of mel values.

    Returns:
        Frequency in Hz with same type as input.
    """
    if isinstance(mel, Tensor):
        return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


# ---------------------------------------------------------------------------
# Mel filterbank construction
# ---------------------------------------------------------------------------


def build_mel_filterbank(cfg: MelConfig) -> Tensor:
    """Build a triangular mel filterbank matrix.

    Each row corresponds to one mel filter. Filters are triangular in mel
    space, uniformly spaced between fmin and fmax. The peak of each triangle
    is normalised to 1.0.

    Args:
        cfg: MelConfig instance.

    Returns:
        Float32 tensor of shape (n_mels, n_fft // 2 + 1).
    """
    n_freqs = cfg.n_fft // 2 + 1

    # Linearly spaced mel points: n_mels + 2 points (includes fmin and fmax edges)
    mel_min = hz_to_mel(cfg.fmin)
    mel_max = hz_to_mel(cfg.fmax)
    mel_points = torch.linspace(mel_min, mel_max, cfg.n_mels + 2)  # (n_mels+2,)
    hz_points = mel_to_hz(mel_points)  # (n_mels+2,) in Hz

    # Map hz_points to FFT bin indices
    # FFT bin k corresponds to frequency k * sample_rate / n_fft
    bin_points = (cfg.n_fft + 1) * hz_points / cfg.sample_rate  # fractional bins
    bin_points = bin_points.long()  # floor to integer bin indices

    # Build filterbank: (n_mels, n_freqs)
    filterbank = torch.zeros(cfg.n_mels, n_freqs, dtype=torch.float32)
    for m in range(cfg.n_mels):
        f_left = bin_points[m]  # left edge (frequency bin index)
        f_center = bin_points[m + 1]  # center
        f_right = bin_points[m + 2]  # right edge

        # Rising slope: left edge -> center (peak = 1.0)
        if f_center > f_left:
            for k in range(f_left, f_center + 1):
                if 0 <= k < n_freqs:
                    filterbank[m, k] = (k - f_left) / (f_center - f_left)

        # Falling slope: center -> right edge
        if f_right > f_center:
            for k in range(f_center, f_right + 1):
                if 0 <= k < n_freqs:
                    filterbank[m, k] = (f_right - k) / (f_right - f_center)

    # Ensure max of each row is exactly 1.0.
    # If a row is all-zero (degenerate filter when bins collapse at very low frequencies),
    # place a unit spike at the center bin so the row is still valid.
    for m in range(cfg.n_mels):
        row_max = filterbank[m].max()
        if row_max > 0:
            filterbank[m] = filterbank[m] / row_max
        else:
            # Degenerate filter: set unit spike at the center bin
            center_bin = int(bin_points[m + 1].clamp(0, n_freqs - 1).item())
            filterbank[m, center_bin] = 1.0

    return filterbank


def _build_mel_filterbank_vectorized(cfg: MelConfig) -> Tensor:
    """Vectorised version of build_mel_filterbank for use in nn.Module buffers.

    Same semantics as build_mel_filterbank but avoids Python loops for speed.
    """
    n_freqs = cfg.n_fft // 2 + 1

    mel_min = hz_to_mel(cfg.fmin)
    mel_max = hz_to_mel(cfg.fmax)
    mel_pts = torch.linspace(mel_min, mel_max, cfg.n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)

    # FFT frequency axis
    fft_freqs = torch.linspace(0, cfg.sample_rate / 2, n_freqs)  # (n_freqs,)

    # Build filterbank using broadcasting
    # Lower and upper slopes for each mel band
    lower = hz_pts[:-2]  # (n_mels,) — left edge
    center = hz_pts[1:-1]  # (n_mels,) — center / peak
    upper = hz_pts[2:]  # (n_mels,) — right edge

    # fft_freqs: (n_freqs,), lower/center/upper: (n_mels,)
    # Broadcast to (n_mels, n_freqs)
    f = fft_freqs.unsqueeze(0)  # (1, n_freqs)
    lo = lower.unsqueeze(1)  # (n_mels, 1)
    ce = center.unsqueeze(1)  # (n_mels, 1)
    up = upper.unsqueeze(1)  # (n_mels, 1)

    rising = (f - lo) / (ce - lo + 1e-10)
    falling = (up - f) / (up - ce + 1e-10)

    filterbank = torch.clamp(torch.minimum(rising, falling), min=0.0)

    # Normalise each row to have max = 1.0
    row_maxes = filterbank.max(dim=1, keepdim=True).values  # (n_mels, 1)

    # For degenerate rows (all-zero when lower == center), place a unit spike at center bin
    degenerate = row_maxes.squeeze(1) == 0  # (n_mels,)
    if degenerate.any():
        # Find the FFT bin closest to each degenerate filter's center frequency
        center_hz = center[degenerate]  # frequencies in Hz for degenerate filters
        # Nearest bin: round(freq * n_fft / sample_rate)
        center_bins = (center_hz * (n_freqs - 1) * 2 / cfg.sample_rate).long().clamp(0, n_freqs - 1)
        degenerate_indices = degenerate.nonzero(as_tuple=True)[0]
        for i, bin_idx in zip(degenerate_indices, center_bins):
            filterbank[i, bin_idx] = 1.0
        # Recompute row_maxes after fixing degenerate rows
        row_maxes = filterbank.max(dim=1, keepdim=True).values

    row_maxes = row_maxes.clamp(min=1e-10)
    filterbank = filterbank / row_maxes

    return filterbank.float()


# ---------------------------------------------------------------------------
# Hann window
# ---------------------------------------------------------------------------


def build_hann_window(n_fft: int) -> Tensor:
    """Build a periodic Hann window of length n_fft.

    Args:
        n_fft: FFT window size.

    Returns:
        Float32 tensor of shape (n_fft,).
    """
    return torch.hann_window(n_fft, periodic=True)


# ---------------------------------------------------------------------------
# STFT
# ---------------------------------------------------------------------------


def stft(waveform: Tensor, n_fft: int, hop_length: int, window: Tensor) -> Tensor:
    """Compute the magnitude spectrogram via Short-Time Fourier Transform.

    Args:
        waveform: (B, T) batched or (T,) single waveform of raw audio samples.
        n_fft: FFT window size.
        hop_length: number of samples between frames.
        window: Hann window tensor of shape (n_fft,).

    Returns:
        Magnitude spectrogram:
            - (B, n_fft//2+1, n_frames) if input was (B, T)
            - (n_fft//2+1, n_frames) if input was (T,)
    """
    batched = waveform.ndim == 2
    if not batched:
        waveform = waveform.unsqueeze(0)  # (1, T)

    B, T = waveform.shape

    # Pad waveform to handle edge frames (centre padding like librosa)
    pad = n_fft // 2
    waveform = torch.nn.functional.pad(waveform, (pad, pad), mode="reflect")

    # Run STFT per sample; torch.stft expects (T,) or uses batch dim
    specs = []
    for b in range(B):
        complex_spec = torch.stft(
            waveform[b],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )  # (n_fft//2+1, n_frames)
        specs.append(complex_spec.abs())

    result = torch.stack(specs, dim=0)  # (B, n_fft//2+1, n_frames)

    if not batched:
        result = result.squeeze(0)  # (n_fft//2+1, n_frames)

    return result


# ---------------------------------------------------------------------------
# Full pipeline: waveform -> log-mel spectrogram
# ---------------------------------------------------------------------------


def waveform_to_log_mel(waveform: Tensor, cfg: MelConfig) -> Tensor:
    """Convert raw waveform to normalised log-mel spectrogram.

    Pipeline:
        1. STFT magnitude spectrogram
        2. Apply mel filterbank (matrix multiply)
        3. Log10 with floor clamping
        4. Whisper-style normalisation to [-1, 1]

    Args:
        waveform: (B, T) float32 tensor of raw audio in [-1, 1].
        cfg: MelConfig.

    Returns:
        (B, n_mels, n_frames) float32 log-mel spectrogram.
    """
    batched = waveform.ndim == 2
    if not batched:
        waveform = waveform.unsqueeze(0)

    window = build_hann_window(cfg.n_fft).to(waveform.device)
    filterbank = _build_mel_filterbank_vectorized(cfg).to(waveform.device)

    # STFT magnitude: (B, n_fft//2+1, n_frames)
    mag = stft(waveform, cfg.n_fft, cfg.hop_length, window)

    # Power spectrogram
    power = mag**2  # (B, freq_bins, n_frames)

    # Apply mel filterbank: (n_mels, freq_bins) @ (B, freq_bins, n_frames)
    # -> (B, n_mels, n_frames)
    mel_spec = torch.einsum("mf,bft->bmt", filterbank, power)

    # Log compression with floor
    log_mel = torch.log10(torch.clamp(mel_spec, min=1e-10))

    if cfg.normalize:
        # Whisper-style normalisation: subtract per-sample max, divide by 4, shift to [-1,1]
        # x_norm = (x - max(x)) / 4.0 + 1.0
        # This maps [max-8, max] -> [-1, 1] (values below max-8 are clamped)
        # We apply it globally over (n_mels, n_frames) per sample
        B = log_mel.shape[0]
        log_mel_flat = log_mel.reshape(B, -1)
        max_vals = log_mel_flat.max(dim=1).values.reshape(B, 1, 1)
        log_mel = (log_mel - max_vals) / 4.0 + 1.0
        # Clamp to ensure values stay within [-1, 1]
        log_mel = log_mel.clamp(-1.0, 1.0)

    if not batched:
        log_mel = log_mel.squeeze(0)

    return log_mel


# ---------------------------------------------------------------------------
# MelSpectrogramExtractor (nn.Module with registered buffers)
# ---------------------------------------------------------------------------


class MelSpectrogramExtractor(nn.Module):
    """Differentiable mel spectrogram extraction as an nn.Module.

    Registers mel filterbank and Hann window as non-trainable buffers so they
    move to the correct device automatically when the module is transferred.

    Args:
        cfg: MelConfig specifying all audio processing parameters.
    """

    def __init__(self, cfg: MelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        filterbank = _build_mel_filterbank_vectorized(cfg)
        window = build_hann_window(cfg.n_fft)
        self.register_buffer("filterbank", filterbank)  # (n_mels, n_fft//2+1)
        self.register_buffer("window", window)  # (n_fft,)

    def forward(self, waveform: Tensor) -> Tensor:
        """Compute log-mel spectrogram.

        Args:
            waveform: (B, T) batched or (T,) single waveform.

        Returns:
            (B, n_mels, n_frames) or (n_mels, n_frames) log-mel spectrogram.
        """
        batched = waveform.ndim == 2
        if not batched:
            waveform = waveform.unsqueeze(0)

        cfg = self.cfg
        pad = cfg.n_fft // 2
        padded = torch.nn.functional.pad(waveform, (pad, pad), mode="reflect")

        # STFT per sample
        specs = []
        for b in range(padded.shape[0]):
            complex_spec = torch.stft(
                padded[b],
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                win_length=cfg.n_fft,
                window=self.window,
                center=False,
                normalized=False,
                onesided=True,
                return_complex=True,
            )  # (n_fft//2+1, n_frames)
            specs.append(complex_spec.abs())

        mag = torch.stack(specs, dim=0)  # (B, n_fft//2+1, n_frames)
        power = mag**2

        # Mel filterbank application
        mel_spec = torch.einsum("mf,bft->bmt", self.filterbank, power)

        # Log compression
        log_mel = torch.log10(torch.clamp(mel_spec, min=1e-10))

        if cfg.normalize:
            B = log_mel.shape[0]
            log_mel = log_mel.contiguous()
            max_vals = log_mel.reshape(B, -1).max(dim=1).values.reshape(B, 1, 1)
            log_mel = (log_mel - max_vals) / 4.0 + 1.0
            log_mel = log_mel.clamp(-1.0, 1.0)

        if not batched:
            log_mel = log_mel.squeeze(0)

        return log_mel


# ---------------------------------------------------------------------------
# WhisperStyleEncoder
# ---------------------------------------------------------------------------


class WhisperStyleEncoder(nn.Module):
    """Whisper-inspired audio encoder: waveform -> dense sequence embeddings.

    Architecture:
        1. MelSpectrogramExtractor -> (B, n_mels, T_frames)
        2. Conv1d(n_mels, d_model, kernel=3, padding=1) -> GELU
        3. Conv1d(d_model, d_model, kernel=3, stride=2, padding=1) -> GELU  [2x downsample]
        4. Transpose to (B, T_frames//2, d_model)
        5. Linear projection d_model -> output_dim

    The final output (B, T_frames//2, output_dim) is compatible with
    MultiModalProjector.project_modality(features).

    Args:
        mel_cfg: MelConfig for the spectrogram extractor.
        d_model: internal convolutional feature dimension (default 256).
        output_dim: final output dimension fed to MultiModalProjector (default 768).
    """

    def __init__(
        self,
        mel_cfg: MelConfig,
        d_model: int = 256,
        output_dim: int = 768,
    ) -> None:
        super().__init__()
        self.mel_cfg = mel_cfg
        self.d_model = d_model
        self.output_dim = output_dim

        self.mel_extractor = MelSpectrogramExtractor(mel_cfg)

        # Two 1D convolutions operating on the time axis
        # Input: (B, n_mels, T_frames)
        self.conv1 = nn.Conv1d(
            in_channels=mel_cfg.n_mels,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.act = nn.GELU()

        # Final linear projection to output_dim
        self.proj = nn.Linear(d_model, output_dim)

    def forward(self, waveform: Tensor) -> Tensor:
        """Encode audio waveform to a sequence of dense embeddings.

        Args:
            waveform: (B, T) raw audio samples, float32.

        Returns:
            (B, T_frames//2, output_dim) feature tensor.
        """
        # Mel spectrogram: (B, n_mels, T_frames)
        mel = self.mel_extractor(waveform)
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)  # handle single-sample case

        # Convolutional feature extraction
        x = self.act(self.conv1(mel))  # (B, d_model, T_frames)
        x = self.act(self.conv2(x))  # (B, d_model, T_frames//2)

        # Transpose for sequence-first format
        x = x.transpose(1, 2)  # (B, T_frames//2, d_model)

        # Project to output_dim
        x = self.proj(x)  # (B, T_frames//2, output_dim)

        return x

    def output_seq_len(self, n_audio_samples: int) -> int:
        """Compute the output sequence length for a given number of audio samples.

        This is deterministic and matches the actual forward pass output.

        Args:
            n_audio_samples: number of audio samples in the input.

        Returns:
            Integer output sequence length (number of time tokens).
        """
        cfg = self.mel_cfg
        # Padded length after reflect padding
        padded = n_audio_samples + cfg.n_fft  # pad = n_fft // 2 on each side
        # Number of STFT frames (center=False, same formula as torch.stft)
        n_frames = (padded - cfg.n_fft) // cfg.hop_length + 1
        # After stride-2 conv2: floor((n_frames + 2*pad - dilation*(kernel-1) - 1) / stride + 1)
        # kernel=3, padding=1, stride=2, dilation=1
        # -> floor((n_frames + 2 - 2 - 1) / 2 + 1) = floor((n_frames - 1) / 2 + 1)
        out_len = (n_frames - 1) // 2 + 1
        return out_len
