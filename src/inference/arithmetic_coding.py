"""Arithmetic Coding compression using LM probability distributions.

Implements lossless arithmetic coding where the coding distribution is
supplied by a language model's next-token probability output.  The
compressed bitstream length approaches the cross-entropy of the sequence
under the model (Shannon's source-coding theorem).

Reference: Witten, Neal & Cleary (1987), "Arithmetic Coding for Data
Compression", Communications of the ACM.

Classes:
    ArithmeticEncoder  — encodes token_ids given per-step prob tensors.
    ArithmeticDecoder  — decodes a bitstream given a callable probs_fn.
    LMArithmeticCoder  — high-level wrapper (compress / decompress /
                         bits_per_token).
"""
from __future__ import annotations

import math
from typing import Callable

import torch

# ---------------------------------------------------------------------------
# Integer-precision constants  (32-bit interval arithmetic)
# ---------------------------------------------------------------------------

PRECISION: int = 32
FULL: int = 1 << PRECISION          # 2^32  — exclusive upper bound
HALF: int = 1 << (PRECISION - 1)    # 2^31
QUARTER: int = 1 << (PRECISION - 2) # 2^30

EPSILON: float = 1e-9


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(probs: torch.Tensor) -> torch.Tensor:
    """Clip near-zero probabilities and renormalise to sum = 1."""
    p = probs.float().clamp(min=EPSILON)
    return p / p.sum()


def _build_cdf(probs: torch.Tensor):
    """Return integer CDF boundaries for *precision*-bit arithmetic.

    Returns a 1-D LongTensor of length vocab+1 where
        cdf[i]  = floor(sum(probs[:i]) * FULL)
    with cdf[0]=0 and cdf[-1] forced to FULL.
    """
    p = _normalise(probs)
    # cumsum in float64 for accuracy
    cum = torch.cumsum(p.double(), dim=0)
    # scale to integer range
    cdf = (cum * FULL).long()
    # prepend 0
    cdf = torch.cat([torch.zeros(1, dtype=torch.long), cdf])
    # guarantee strict monotonicity: each bucket ≥ 1
    for i in range(1, len(cdf)):
        if cdf[i] <= cdf[i - 1]:
            cdf[i] = cdf[i - 1] + 1
    # clamp top to FULL
    cdf[-1] = FULL
    return cdf


# ---------------------------------------------------------------------------
# ArithmeticEncoder
# ---------------------------------------------------------------------------

class ArithmeticEncoder:
    """Arithmetic encoder that takes pre-computed probability tensors.

    Usage::

        enc = ArithmeticEncoder()
        bits = enc.encode(token_ids, probs)   # list[int], list[Tensor]
    """

    def encode(
        self,
        token_ids: list[int],
        probs: list[torch.Tensor],
    ) -> bytes:
        """Encode *token_ids* using the supplied probability distributions.

        Args:
            token_ids: Sequence of token indices to compress.
            probs:     Per-step probability tensors (shape ``[vocab]``).
                       ``len(probs)`` must equal ``len(token_ids)``.

        Returns:
            Compressed bitstream as a :class:`bytes` object.

        Raises:
            ValueError: If ``len(probs) != len(token_ids)`` or any token
                        index is out of range for its probability tensor.
        """
        if len(token_ids) != len(probs):
            raise ValueError(
                f"len(token_ids)={len(token_ids)} != len(probs)={len(probs)}"
            )
        if len(token_ids) == 0:
            return b""

        low: int = 0
        high: int = FULL
        pending: int = 0  # pending (follow) bits
        bits: list[int] = []

        def _emit(bit: int) -> None:
            bits.append(bit)
            # emit all pending opposite bits
            for _ in range(pending):
                bits.append(1 - bit)

        for t, (tok, p) in enumerate(zip(token_ids, probs)):
            cdf = _build_cdf(p)
            vocab_size = len(cdf) - 1
            if not (0 <= tok < vocab_size):
                raise ValueError(
                    f"token_ids[{t}]={tok} out of range [0, {vocab_size})"
                )
            cum_low = int(cdf[tok].item())
            cum_high = int(cdf[tok + 1].item())

            width = high - low
            high = low + (width * cum_high) // FULL
            low = low + (width * cum_low) // FULL

            # Renormalise
            while True:
                if high <= HALF:
                    # both in lower half → emit 0
                    _emit(0)
                    low <<= 1
                    high <<= 1
                    pending = 0
                elif low >= HALF:
                    # both in upper half → emit 1
                    _emit(1)
                    low = (low - HALF) << 1
                    high = (high - HALF) << 1
                    pending = 0
                elif low >= QUARTER and high <= 3 * QUARTER:
                    # straddle middle
                    pending += 1
                    low = (low - QUARTER) << 1
                    high = (high - QUARTER) << 1
                else:
                    break

        # --- flush ---
        pending += 1
        if low < QUARTER:
            _emit(0)
        else:
            _emit(1)

        return _bits_to_bytes(bits)


# ---------------------------------------------------------------------------
# ArithmeticDecoder
# ---------------------------------------------------------------------------

class ArithmeticDecoder:
    """Arithmetic decoder that uses a callable to obtain probabilities.

    Usage::

        dec = ArithmeticDecoder()
        tokens = dec.decode(bitstream, probs_fn, n_tokens)
    """

    def decode(
        self,
        bitstream: bytes,
        probs_fn: Callable[[list[int]], torch.Tensor],
        n_tokens: int,
    ) -> list[int]:
        """Decode *n_tokens* symbols from *bitstream*.

        Args:
            bitstream: Bytes produced by :meth:`ArithmeticEncoder.encode`.
            probs_fn:  Callable ``probs_fn(decoded_so_far) -> Tensor[vocab]``.
            n_tokens:  Number of tokens to decode.

        Returns:
            List of decoded token indices.
        """
        if n_tokens == 0:
            return []

        bits = _bytes_to_bits(bitstream)
        # pad with zeros if needed
        bits_len = len(bits)

        def _read_bit(pos: int) -> int:
            if pos < bits_len:
                return bits[pos]
            return 0  # implicit trailing zeros

        # Initialise decoder value
        value: int = 0
        for i in range(PRECISION):
            value = (value << 1) | _read_bit(i)
        bit_pos: int = PRECISION

        low: int = 0
        high: int = FULL
        decoded: list[int] = []

        for _ in range(n_tokens):
            p = probs_fn(decoded)
            cdf = _build_cdf(p)
            vocab_size = len(cdf) - 1

            # Find symbol: scale value into [0, FULL)
            width = high - low
            # Avoid division by zero
            if width == 0:
                width = 1
            scaled = ((value - low + 1) * FULL - 1) // width

            # Binary search for symbol
            tok = _bisect(cdf, scaled)
            decoded.append(tok)

            cum_low = int(cdf[tok].item())
            cum_high = int(cdf[tok + 1].item())

            high = low + (width * cum_high) // FULL
            low = low + (width * cum_low) // FULL

            # Renormalise
            while True:
                if high <= HALF:
                    low <<= 1
                    high <<= 1
                    value <<= 1
                    value |= _read_bit(bit_pos)
                    bit_pos += 1
                elif low >= HALF:
                    low = (low - HALF) << 1
                    high = (high - HALF) << 1
                    value = ((value - HALF) << 1) | _read_bit(bit_pos)
                    bit_pos += 1
                elif low >= QUARTER and high <= 3 * QUARTER:
                    low = (low - QUARTER) << 1
                    high = (high - QUARTER) << 1
                    value = ((value - QUARTER) << 1) | _read_bit(bit_pos)
                    bit_pos += 1
                else:
                    break

        return decoded


# ---------------------------------------------------------------------------
# LMArithmeticCoder
# ---------------------------------------------------------------------------

class LMArithmeticCoder:
    """High-level wrapper around :class:`ArithmeticEncoder` /
    :class:`ArithmeticDecoder`.

    Args:
        model: Optional language model.  When *None* the caller must supply
               pre-computed probability tensors to :meth:`compress` /
               :meth:`decompress`.
    """

    def __init__(self, model=None) -> None:
        self.model = model
        self._encoder = ArithmeticEncoder()
        self._decoder = ArithmeticDecoder()

    # ------------------------------------------------------------------
    def compress(
        self,
        token_ids: list[int],
        probs: list[torch.Tensor],
    ) -> bytes:
        """Losslessly compress *token_ids* using *probs*.

        Args:
            token_ids: Sequence of token indices.
            probs:     Per-step probability tensors ``[vocab]``.

        Returns:
            Compressed :class:`bytes`.
        """
        return self._encoder.encode(token_ids, probs)

    # ------------------------------------------------------------------
    def decompress(
        self,
        bitstream: bytes,
        probs_fn: Callable[[list[int]], torch.Tensor],
        n_tokens: int,
    ) -> list[int]:
        """Losslessly decompress *bitstream* into token ids.

        Args:
            bitstream: Bytes from :meth:`compress`.
            probs_fn:  Callable that returns ``Tensor[vocab]`` given tokens
                       decoded so far.
            n_tokens:  Number of tokens to recover.

        Returns:
            List of token indices.
        """
        return self._decoder.decode(bitstream, probs_fn, n_tokens)

    # ------------------------------------------------------------------
    def bits_per_token(
        self,
        token_ids: list[int],
        probs: list[torch.Tensor],
    ) -> float:
        """Empirical bits-per-token for *token_ids* under *probs*.

        For a sequence of length *n* the value approaches the
        per-token cross-entropy (in bits) as *n* → ∞.

        Returns 0.0 for an empty sequence.
        """
        if len(token_ids) == 0:
            return 0.0
        bs = self._encoder.encode(token_ids, probs)
        n_bits = len(bs) * 8
        return n_bits / len(token_ids)


# ---------------------------------------------------------------------------
# Bit / byte utilities
# ---------------------------------------------------------------------------

def _bits_to_bytes(bits: list[int]) -> bytes:
    """Pack a list of bits (MSB first) into a :class:`bytes` object.

    The first byte stores the count of *padding* bits added at the tail
    (0–7) so the decoder can ignore them if needed.  This gives a
    1-byte overhead but allows exact length accounting.
    """
    # pad to multiple of 8
    pad = (8 - len(bits) % 8) % 8
    bits = bits + [0] * pad
    out = bytearray()
    out.append(pad)  # header: padding count
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        out.append(byte)
    return bytes(out)


def _bytes_to_bits(data: bytes) -> list[int]:
    """Inverse of :func:`_bits_to_bytes`.  Strips the padding header."""
    if len(data) == 0:
        return []
    pad = data[0]
    bits: list[int] = []
    for byte in data[1:]:
        for j in range(7, -1, -1):
            bits.append((byte >> j) & 1)
    # remove trailing padding
    if pad > 0:
        bits = bits[:-pad]
    return bits


def _bisect(cdf: torch.Tensor, scaled: int) -> int:
    """Return the largest index i such that cdf[i] <= scaled.

    Performs a binary search over the integer CDF array.
    """
    lo, hi = 0, len(cdf) - 2  # search in [0, vocab-1]
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if int(cdf[mid].item()) <= scaled:
            lo = mid
        else:
            hi = mid - 1
    return lo
