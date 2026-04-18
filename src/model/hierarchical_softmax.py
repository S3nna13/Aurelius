"""
Hierarchical Softmax for efficient large-vocabulary language modeling.

Implements:
  - HuffmanNode / HuffmanTree  – Huffman-coded output vocabulary
  - HierarchicalSoftmax        – HSoftmax nn.Module (log_prob, loss, sample)
  - AdaptiveSoftmax            – Adaptive softmax (Grave et al., 2017)
  - SoftmaxConfig              – Dataclass of default hyper-parameters
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Huffman tree
# ---------------------------------------------------------------------------

class HuffmanNode:
    """A single node in a Huffman binary tree."""

    def __init__(
        self,
        freq: float,
        word_id: Optional[int] = None,
        left: Optional["HuffmanNode"] = None,
        right: Optional["HuffmanNode"] = None,
    ) -> None:
        self.word_id: Optional[int] = word_id
        self.freq: float = freq
        self.left: Optional[HuffmanNode] = left
        self.right: Optional[HuffmanNode] = right
        # Filled in by HuffmanTree.build()
        self.path: List[int] = []         # binary codes from root (0=left, 1=right)
        self.inner_nodes: List[int] = []  # indices of inner nodes along path

    # Make nodes comparable for heapq (break ties by word_id, then id)
    def __lt__(self, other: "HuffmanNode") -> bool:
        if self.freq != other.freq:
            return self.freq < other.freq
        a = self.word_id if self.word_id is not None else -1
        b = other.word_id if other.word_id is not None else -1
        return a < b


class HuffmanTree:
    """
    Builds a Huffman binary tree from a frequency tensor and encodes the
    vocabulary into binary paths over inner nodes.

    Parameters
    ----------
    frequencies : Tensor of shape [vocab_size]
        Non-negative frequency / probability for each word.  Words with
        higher frequency receive shorter paths.
    """

    def __init__(self, frequencies: Tensor) -> None:
        self.frequencies = frequencies
        self.vocab_size: int = int(frequencies.shape[0])
        self.root: Optional[HuffmanNode] = None
        # {word_id: (codes, inner_node_indices)}
        self._encoding: Dict[int, Tuple[List[int], List[int]]] = {}
        self._n_inner: int = 0
        self.build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build the Huffman tree from self.frequencies and annotate paths."""
        freq_list = self.frequencies.tolist()

        # Create leaf nodes
        heap: List[HuffmanNode] = [
            HuffmanNode(freq=freq_list[i], word_id=i)
            for i in range(self.vocab_size)
        ]
        heapq.heapify(heap)

        # Edge case: single word
        if len(heap) == 1:
            node = heapq.heappop(heap)
            self.root = node
            node.path = []
            node.inner_nodes = []
            self._encoding = {node.word_id: ([], [])}
            return

        # Assign indices to inner nodes as they are created
        inner_idx = 0

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            parent = HuffmanNode(
                freq=left.freq + right.freq,
                word_id=None,
                left=left,
                right=right,
            )
            parent.inner_nodes = [inner_idx]  # store this node's index temporarily
            inner_idx += 1
            heapq.heappush(heap, parent)

        self.root = heapq.heappop(heap)
        self._n_inner = inner_idx

        # DFS to assign paths to leaf nodes
        self._assign_paths(self.root, [], [])

        # Build encoding dict
        self._build_encoding(self.root)

    def get_path(self, word_id: int) -> Tuple[List[int], List[int]]:
        """Return (codes, inner_nodes) for *word_id*."""
        return self._encoding[word_id]

    def max_depth(self) -> int:
        """Maximum depth (number of edges from root to any leaf)."""
        return self._max_depth(self.root)

    def encode_vocab(self) -> Dict[int, Tuple[List[int], List[int]]]:
        """Return {word_id: (codes, inner_nodes)} for all words."""
        return dict(self._encoding)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assign_paths(
        self,
        node: Optional[HuffmanNode],
        codes: List[int],
        nodes: List[int],
    ) -> None:
        if node is None:
            return
        if node.word_id is not None:
            # Leaf
            node.path = list(codes)
            node.inner_nodes = list(nodes)
            return
        # Inner node – retrieve its assigned index (stored in inner_nodes[0])
        idx = node.inner_nodes[0]
        self._assign_paths(node.left,  codes + [0], nodes + [idx])
        self._assign_paths(node.right, codes + [1], nodes + [idx])

    def _build_encoding(self, node: Optional[HuffmanNode]) -> None:
        if node is None:
            return
        if node.word_id is not None:
            self._encoding[node.word_id] = (node.path, node.inner_nodes)
            return
        self._build_encoding(node.left)
        self._build_encoding(node.right)

    def _max_depth(self, node: Optional[HuffmanNode]) -> int:
        if node is None:
            return 0
        if node.left is None and node.right is None:
            return 0
        return 1 + max(self._max_depth(node.left), self._max_depth(node.right))


# ---------------------------------------------------------------------------
# HierarchicalSoftmax
# ---------------------------------------------------------------------------

class HierarchicalSoftmax(nn.Module):
    """
    Hierarchical Softmax output layer using a Huffman coding tree.

    Parameters
    ----------
    d_model     : int  – Hidden dimension of input representations.
    vocab_size  : int  – Number of output classes.
    frequencies : Optional Tensor [vocab_size] – Word frequencies used to
                  build the Huffman tree.  Defaults to uniform.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        frequencies: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        if frequencies is None:
            frequencies = torch.ones(vocab_size)

        self.tree = HuffmanTree(frequencies)
        n_inner = self.tree._n_inner

        # One weight vector per inner node
        self.inner_weights = nn.Parameter(
            torch.empty(n_inner, d_model).normal_(0, 0.01)
        )

        # Pre-compute path tensors for fast batch lookup
        self._paths_codes: List[List[int]] = []
        self._paths_nodes: List[List[int]] = []
        encoding = self.tree.encode_vocab()
        for wid in range(vocab_size):
            codes, nodes = encoding[wid]
            self._paths_codes.append(codes)
            self._paths_nodes.append(nodes)

    # ------------------------------------------------------------------
    # Forward methods
    # ------------------------------------------------------------------

    def log_prob(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Compute log-probability of each target word given the input vector.

        Parameters
        ----------
        input  : [B, d_model]
        target : [B]  long tensor of word indices

        Returns
        -------
        log_p  : [B]  log-probabilities (≤ 0)
        """
        B = input.shape[0]
        log_probs = input.new_zeros(B)

        for b in range(B):
            wid = int(target[b].item())
            codes = self._paths_codes[wid]
            nodes = self._paths_nodes[wid]

            if len(codes) == 0:
                # Single-word vocabulary – probability 1
                continue

            h = input[b]  # [d_model]
            w = self.inner_weights[nodes]  # [L, d_model]
            scores = (w @ h)              # [L]

            # For bit=0 (left): log σ(score); for bit=1 (right): log σ(−score)
            signs = torch.tensor(
                [1.0 if c == 0 else -1.0 for c in codes],
                dtype=input.dtype,
                device=input.device,
            )
            log_probs[b] = F.logsigmoid(signs * scores).sum()

        return log_probs

    def loss(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Negative mean log-probability.

        Parameters
        ----------
        input  : [B, d_model]
        target : [B]

        Returns
        -------
        scalar loss
        """
        return -self.log_prob(input, target).mean()

    def sample(self, input: Tensor) -> Tensor:
        """
        Greedy decode by following the higher-sigmoid branch at each node.

        Parameters
        ----------
        input : [B, d_model]

        Returns
        -------
        word_ids : [B]  long tensor
        """
        B = input.shape[0]
        results = torch.zeros(B, dtype=torch.long, device=input.device)

        for b in range(B):
            h = input[b]
            node = self.tree.root
            # Walk until leaf
            while node is not None and node.word_id is None:
                if node.left is None and node.right is None:
                    break
                # Get inner node index (stored in first inner_nodes entry
                # for either child)
                left_idx = node.left.inner_nodes[0] if (node.left is not None and node.left.inner_nodes) else None
                right_idx = node.right.inner_nodes[0] if (node.right is not None and node.right.inner_nodes) else None

                # The current inner node index is the one that points here.
                # We derive it from the path of the *left* child (last element).
                if node.left is not None and node.left.inner_nodes:
                    cur_idx = node.left.inner_nodes[-1]
                elif node.right is not None and node.right.inner_nodes:
                    cur_idx = node.right.inner_nodes[-1]
                else:
                    # Leaf children with empty inner_nodes – shouldn't happen
                    # for vocab_size > 1
                    break

                score = h @ self.inner_weights[cur_idx]
                sig = torch.sigmoid(score)
                # sig > 0.5 → left (code=0), else → right (code=1)
                if sig.item() > 0.5:
                    node = node.left
                else:
                    node = node.right

            if node is not None and node.word_id is not None:
                results[b] = node.word_id
            # else stays 0 (fallback)

        return results


# ---------------------------------------------------------------------------
# AdaptiveSoftmax
# ---------------------------------------------------------------------------

class AdaptiveSoftmax(nn.Module):
    """
    Adaptive Softmax (Grave et al., 2017).

    Splits the vocabulary by frequency into a *head* cluster and one or more
    *tail* clusters of decreasing capacity.

    Parameters
    ----------
    d_model   : int          – Input hidden dimension.
    vocab_size: int          – Total vocabulary size.
    cutoffs   : list[int]    – Boundaries between clusters.
                               e.g. [100, 1000] gives:
                               head=[0:100], tail0=[100:1000], tail1=[1000:]
    factor    : float        – Dimension reduction factor per tail cluster.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        cutoffs: List[int],
        factor: float = 4.0,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.factor = factor

        # Build cluster boundaries
        boundaries = [0] + sorted(cutoffs) + [vocab_size]
        boundaries = [b for b in boundaries if b <= vocab_size]
        # Deduplicate & clamp
        seen: List[int] = []
        for b in boundaries:
            if not seen or b != seen[-1]:
                seen.append(b)
        boundaries = seen
        self.boundaries: List[int] = boundaries
        self.n_clusters: int = len(boundaries) - 1
        self.n_tails: int = self.n_clusters - 1  # tail clusters

        # Head cluster size = words in [0, boundaries[1]) + one token per tail
        head_size = (boundaries[1] - boundaries[0]) + self.n_tails
        self.head = nn.Linear(d_model, head_size, bias=False)

        # Tail clusters
        self.tails = nn.ModuleList()
        self.projections = nn.ModuleList()
        for i in range(self.n_tails):
            tail_dim = max(1, int(d_model // (factor ** (i + 1))))
            tail_size = boundaries[i + 2] - boundaries[i + 1]
            self.projections.append(nn.Linear(d_model, tail_dim, bias=False))
            self.tails.append(nn.Linear(tail_dim, tail_size, bias=False))

    # ------------------------------------------------------------------
    # Forward methods
    # ------------------------------------------------------------------

    def log_prob(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Compute log-probabilities for each target word.

        Parameters
        ----------
        input  : [B, d_model]
        target : [B]

        Returns
        -------
        log_p  : [B]
        """
        B = input.shape[0]
        head_logits = self.head(input)  # [B, head_size]
        # head_size = head_words + n_tails
        head_words = self.boundaries[1] - self.boundaries[0]
        # Log-softmax over entire head (head words + cluster tokens)
        head_log_probs = F.log_softmax(head_logits, dim=-1)  # [B, head_size]

        log_p = input.new_zeros(B)

        for b in range(B):
            wid = int(target[b].item())
            # Determine which cluster this word belongs to
            cluster = -1
            for c in range(self.n_clusters):
                if self.boundaries[c] <= wid < self.boundaries[c + 1]:
                    cluster = c
                    break
            if cluster == -1:
                cluster = self.n_clusters - 1  # clamp

            if cluster == 0:
                # Head cluster – direct log-prob
                local_id = wid - self.boundaries[0]
                log_p[b] = head_log_probs[b, local_id]
            else:
                # Tail cluster – log p(cluster) + log p(word | cluster)
                tail_idx = cluster - 1
                cluster_token_pos = head_words + tail_idx
                log_p_cluster = head_log_probs[b, cluster_token_pos]

                proj = self.projections[tail_idx](input[b : b + 1])  # [1, tail_dim]
                tail_logits = self.tails[tail_idx](proj)              # [1, tail_size]
                tail_log_probs = F.log_softmax(tail_logits, dim=-1)   # [1, tail_size]

                local_id = wid - self.boundaries[cluster]
                log_p[b] = log_p_cluster + tail_log_probs[0, local_id]

        return log_p

    def loss(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Negative mean log-probability.

        Parameters
        ----------
        input  : [B, d_model]
        target : [B]

        Returns
        -------
        scalar loss
        """
        return -self.log_prob(input, target).mean()


# ---------------------------------------------------------------------------
# SoftmaxConfig
# ---------------------------------------------------------------------------

@dataclass
class SoftmaxConfig:
    """Default hyper-parameters for softmax variants."""

    d_model: int = 32
    vocab_size: int = 64
    cutoffs: List[int] = field(default_factory=lambda: [16, 32])
    factor: float = 2.0
