"""
Hash Embedding Compression
==========================
Memory-efficient embeddings using hashing tricks, feature hashing for large
vocabularies, and subword hash embeddings.

No external dependencies beyond stdlib and PyTorch.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _poly_hash(token_ids: torch.Tensor, seed: int, vocab_size: int) -> torch.Tensor:
    """
    Deterministic polynomial hash.

    h(x) = (a*x + b) % vocab_size
    where a and b are derived from `seed` to be non-zero constants.

    Args:
        token_ids: integer tensor of any shape
        seed:      integer seed controlling a/b
        vocab_size: modulus

    Returns:
        Tensor of same shape with values in [0, vocab_size).
    """
    # Large primes make good multipliers / offsets
    a = (seed * 1_000_003 + 7) | 1  # ensure odd
    b = (seed * 999_983 + 13) & 0xFFFF_FFFF
    return ((token_ids.long() * a + b) % vocab_size).long()


def _sign_bit(token_ids: torch.Tensor, bit_index: int) -> torch.Tensor:
    """Extract bit `bit_index` of each token_id and return ±1."""
    return (((token_ids.long() >> bit_index) & 1) * 2 - 1).float()


# ---------------------------------------------------------------------------
# HashEmbedding
# ---------------------------------------------------------------------------


class HashEmbedding(nn.Module):
    """
    Embedding via multiple hash tables (the "hash embedding" trick).

    Instead of one large Embedding(V, d) we maintain `num_hashes` smaller
    tables each of size `hash_vocab_size` × d.  For each token_id we:
      1. Map it to a bucket in each table via a polynomial hash.
      2. Multiply the looked-up vector by a ±1 sign (CountSketch style).
      3. Sum across hashes.

    Total parameters: num_hashes × hash_vocab_size × d_model
    vs. standard:     vocab_size × d_model  (much larger for big vocabs)

    Args:
        num_hashes:      number of independent hash tables
        hash_vocab_size: size of each hash table (bucket count)
        d_model:         embedding dimension
    """

    def __init__(self, num_hashes: int, hash_vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.num_hashes = num_hashes
        self.hash_vocab_size = hash_vocab_size
        self.d_model = d_model

        self.tables = nn.ModuleList(
            [nn.Embedding(hash_vocab_size, d_model) for _ in range(num_hashes)]
        )
        # signs is a list of seed integers; sign for hash i is derived from seed i
        self.signs: list[int] = list(range(num_hashes))

        # Kaiming-style initialisation
        std = math.sqrt(2.0 / (d_model * num_hashes))
        for table in self.tables:
            nn.init.normal_(table.weight, mean=0.0, std=std)

    # ------------------------------------------------------------------
    def hash_fn(self, token_ids: torch.Tensor, seed: int) -> torch.Tensor:
        """Deterministic hash: token_ids → bucket indices in [0, hash_vocab_size)."""
        return _poly_hash(token_ids, seed, self.hash_vocab_size)

    # ------------------------------------------------------------------
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, T] long tensor

        Returns:
            [B, T, d_model] float tensor
        """
        out = torch.zeros(
            *token_ids.shape, self.d_model, device=token_ids.device, dtype=torch.float32
        )
        for i, table in enumerate(self.tables):
            idx = self.hash_fn(token_ids, seed=i)  # [B, T]
            sign = _sign_bit(token_ids, i).unsqueeze(-1)  # [B, T, 1]
            out = out + sign * table(idx)  # [B, T, d]
        return out


# ---------------------------------------------------------------------------
# FeatureHasher
# ---------------------------------------------------------------------------


class FeatureHasher(nn.Module):
    """
    The hashing trick for sparse high-dimensional inputs → dense embedding.

    Given a sparse feature set represented as (indices, values), accumulate:
        out[b] = sum_j  sign(index_j) * embedding[index_j % n_features] * value_j

    This is the standard feature hashing / random projection approach used in
    e.g. Vowpal Wabbit and sklearn's HashingVectorizer.

    Args:
        n_features: number of hash buckets (embedding table size)
        d_model:    output embedding dimension
    """

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.embedding = nn.Embedding(n_features, d_model)
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(d_model))

    # ------------------------------------------------------------------
    def forward(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: [B, N] long tensor of (possibly large) feature indices
            values:  [B, N] float tensor of feature values

        Returns:
            [B, d_model] dense embedding
        """
        bucket = (indices % self.n_features).long()  # [B, N]
        sign = (2 * (indices % 2) - 1).float()  # [B, N]  ±1
        looked_up = self.embedding(bucket)  # [B, N, d]
        weighted = sign.unsqueeze(-1) * looked_up * values.unsqueeze(-1)
        return weighted.sum(dim=1)  # [B, d]


# ---------------------------------------------------------------------------
# SubwordHashEmbedding
# ---------------------------------------------------------------------------


class SubwordHashEmbedding(nn.Module):
    """
    Combines a standard word-level embedding with character-level hash
    embeddings to handle OOV tokens and subword structure (similar in spirit
    to FastText but using hash tables instead of an explicit character n-gram
    vocabulary).

    Architecture:
        word_emb  = Embedding(vocab_size, d_model)
        char_hash = HashEmbedding(n_char_hashes, char_vocab, d_model//4)
        char_proj = Linear(d_model//4, d_model)
        output    = word_emb + char_proj(mean_pool(char_hash(char_ids)))

    Args:
        vocab_size:    word vocabulary size
        d_model:       model embedding dimension
        n_char_hashes: number of hash tables in the character hasher
        char_vocab:    character vocabulary size (typically 256 for bytes)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_char_hashes: int = 4,
        char_vocab: int = 256,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.char_dim = max(d_model // 4, 1)

        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.char_hash_emb = HashEmbedding(n_char_hashes, char_vocab, self.char_dim)
        self.char_to_word_proj = nn.Linear(self.char_dim, d_model)

        nn.init.normal_(self.word_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.char_to_word_proj.weight)

    # ------------------------------------------------------------------
    def forward(
        self,
        token_ids: torch.Tensor,
        char_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: [B, T] long tensor of word/subword token ids
            char_ids:  [B, T, max_word_len] long tensor of character ids
                       for each token position

        Returns:
            [B, T, d_model]
        """
        B, T = token_ids.shape
        max_word_len = char_ids.shape[2]

        # Word embedding: [B, T, d_model]
        word_emb = self.word_embedding(token_ids)

        # Character hash embedding
        # Flatten [B, T, L] → [B*T, L] for hash embedding
        flat_char = char_ids.view(B * T, max_word_len)  # [B*T, L]
        char_emb = self.char_hash_emb(flat_char)  # [B*T, L, char_dim]
        char_emb = char_emb.mean(dim=1)  # [B*T, char_dim]
        char_emb = char_emb.view(B, T, self.char_dim)  # [B, T, char_dim]

        # Project to d_model and add
        char_contrib = self.char_to_word_proj(char_emb)  # [B, T, d_model]
        return word_emb + char_contrib


# ---------------------------------------------------------------------------
# CompressedEmbeddingLayer
# ---------------------------------------------------------------------------


class CompressedEmbeddingLayer(nn.Module):
    """
    Drop-in replacement for a standard Embedding layer that uses hash
    embedding compression to drastically reduce parameter count.

    The compressed vocabulary size is ``max(int(vocab_size * compression_ratio), 256)``.
    A lightweight linear re-mix (``rank_adapt``) follows the hash lookup to
    allow the model to learn better representations within the compressed space.

    Args:
        vocab_size:        original vocabulary size
        d_model:           embedding dimension
        compression_ratio: fraction of vocab_size used for hash tables
                           (e.g. 0.1 ⟹ 10 % of parameters of a full embedding)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        compression_ratio: float = 0.1,
        num_hashes: int = 4,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.compression_ratio = compression_ratio
        self.num_hashes = num_hashes

        self.compressed_vocab = max(int(vocab_size * compression_ratio), 256)
        self.hash_emb = HashEmbedding(num_hashes, self.compressed_vocab, d_model)
        self.rank_adapt = nn.Linear(d_model, d_model)

        nn.init.eye_(self.rank_adapt.weight)  # start as identity
        nn.init.zeros_(self.rank_adapt.bias)

    # ------------------------------------------------------------------
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, T] long tensor

        Returns:
            [B, T, d_model]
        """
        h = self.hash_emb(token_ids)  # [B, T, d_model]
        return self.rank_adapt(h)  # [B, T, d_model]

    # ------------------------------------------------------------------
    def compression_factor(self) -> float:
        """
        Ratio of hash embedding parameters to standard embedding parameters.
        Values < 1.0 indicate memory savings.
        """
        standard = self.vocab_size * self.d_model
        compressed = (
            self.num_hashes * self.compressed_vocab * self.d_model
            + self.d_model * self.d_model  # rank_adapt weight
            + self.d_model  # rank_adapt bias
        )
        return compressed / standard


# ---------------------------------------------------------------------------
# EmbeddingCompressionBenchmark
# ---------------------------------------------------------------------------


class EmbeddingCompressionBenchmark:
    """
    Utility class with static methods for analysing the trade-offs of hash
    embedding compression.
    """

    @staticmethod
    def standard_param_count(vocab_size: int, d_model: int) -> int:
        """Number of parameters in a standard Embedding(vocab_size, d_model)."""
        return vocab_size * d_model

    @staticmethod
    def hash_param_count(num_hashes: int, hash_vocab_size: int, d_model: int) -> int:
        """Number of parameters in a HashEmbedding with the given settings."""
        return num_hashes * hash_vocab_size * d_model

    @staticmethod
    def collision_rate_estimate(vocab_size: int, hash_vocab_size: int) -> float:
        """
        Birthday-paradox estimate of the probability that at least two tokens
        collide into the same bucket when hashing `vocab_size` items into
        `hash_vocab_size` buckets.

            P ≈ 1 - exp(-n*(n-1) / (2*m))

        where n = vocab_size, m = hash_vocab_size.
        """
        n = vocab_size
        m = hash_vocab_size
        if m <= 0:
            return 1.0
        exponent = -n * (n - 1) / (2.0 * m)
        rate = 1.0 - math.exp(exponent)
        return float(max(0.0, min(1.0, rate)))

    @staticmethod
    def reconstruction_error(
        emb1: torch.Tensor,
        emb2: torch.Tensor,
    ) -> float:
        """
        Mean L2 distance between two embedding tables.

        Args:
            emb1: [N, d] embedding table
            emb2: [N, d] embedding table

        Returns:
            Mean L2 distance (≥ 0).
        """
        assert emb1.shape == emb2.shape, "Embedding tables must have the same shape"  # noqa: S101
        diff = emb1.float() - emb2.float()
        distances = diff.norm(dim=-1)  # [N]
        return float(distances.mean().item())


# ---------------------------------------------------------------------------
# HashEmbeddingConfig
# ---------------------------------------------------------------------------


@dataclass
class HashEmbeddingConfig:
    """Configuration dataclass for hash embedding experiments."""

    num_hashes: int = 4
    hash_vocab_size: int = 1024
    d_model: int = 32
    n_char_hashes: int = 4
    compression_ratio: float = 0.1
