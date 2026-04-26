"""
Graph Neural Network Language Model

Processes token sequences as graphs where tokens are nodes and structural
relations (sequential, window, dependency) define edges.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# TokenGraph
# ---------------------------------------------------------------------------


class TokenGraph:
    """Holds node features and an edge index for a single token graph."""

    def __init__(
        self,
        node_features: torch.Tensor,  # [N, d]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr: torch.Tensor | None = None,  # [E, d_e]
    ) -> None:
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    @property
    def n_nodes(self) -> int:
        return self.node_features.size(0)

    @property
    def n_edges(self) -> int:
        return self.edge_index.size(1)

    # ------------------------------------------------------------------
    # Static edge-construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def add_sequential_edges(seq_len: int) -> torch.Tensor:
        """Each token → next token: 0→1, 1→2, …   shape [2, seq_len-1]."""
        if seq_len <= 1:
            return torch.zeros(2, 0, dtype=torch.long)
        src = torch.arange(seq_len - 1, dtype=torch.long)  # 0,1,...,T-2
        dst = src + 1  # 1,2,...,T-1
        return torch.stack([src, dst], dim=0)  # [2, T-1]

    @staticmethod
    def add_window_edges(seq_len: int, window: int = 3) -> torch.Tensor:
        """Each token connects to tokens within ±window (excluding self)."""
        srcs: list[int] = []
        dsts: list[int] = []
        for i in range(seq_len):
            lo = max(0, i - window)
            hi = min(seq_len - 1, i + window)
            for j in range(lo, hi + 1):
                if j != i:
                    srcs.append(i)
                    dsts.append(j)
        if not srcs:
            return torch.zeros(2, 0, dtype=torch.long)
        return torch.stack(
            [torch.tensor(srcs, dtype=torch.long), torch.tensor(dsts, dtype=torch.long)],
            dim=0,
        )

    @staticmethod
    def add_dependency_edges(seq_len: int) -> torch.Tensor:
        """All previous tokens → current token (complete causal DAG)."""
        srcs: list[int] = []
        dsts: list[int] = []
        for dst in range(1, seq_len):  # token 0 has no predecessors
            for src in range(dst):
                srcs.append(src)
                dsts.append(dst)
        if not srcs:
            return torch.zeros(2, 0, dtype=torch.long)
        return torch.stack(
            [torch.tensor(srcs, dtype=torch.long), torch.tensor(dsts, dtype=torch.long)],
            dim=0,
        )


# ---------------------------------------------------------------------------
# GNNLayer
# ---------------------------------------------------------------------------


class GNNLayer(nn.Module):
    """
    Basic message-passing GNN layer.

    h_i = σ( W_self · x_i  +  W_neighbor · agg{ x_j : j→i } )

    aggregation: "mean" | "max" | "sum"
    """

    def __init__(self, d_in: int, d_out: int, aggregation: str = "mean") -> None:
        super().__init__()
        assert aggregation in ("mean", "max", "sum"), f"Unknown aggregation '{aggregation}'"  # noqa: S101
        self.aggregation = aggregation
        self.W_self = nn.Linear(d_in, d_out)
        self.W_neighbor = nn.Linear(d_in, d_out)

    def forward(
        self,
        x: torch.Tensor,  # [N, d_in]
        edge_index: torch.Tensor,  # [2, E]
    ) -> torch.Tensor:  # [N, d_out]
        N = x.size(0)
        self_term = self.W_self(x)  # [N, d_out]

        if edge_index.size(1) == 0:
            # No edges — no neighbor contribution
            neighbor_term = torch.zeros_like(self_term)
        else:
            src, dst = edge_index[0], edge_index[1]  # each [E]
            # Gather source features
            x_src = x[src]  # [E, d_in]
            msg = self.W_neighbor(x_src)  # [E, d_out]

            d_out = self_term.size(1)
            if self.aggregation == "mean":
                agg = torch.zeros(N, d_out, device=x.device, dtype=x.dtype)
                count = torch.zeros(N, 1, device=x.device, dtype=x.dtype)
                agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
                count.scatter_add_(
                    0,
                    dst.unsqueeze(1),
                    torch.ones(dst.size(0), 1, device=x.device, dtype=x.dtype),
                )
                count = count.clamp(min=1.0)
                neighbor_term = agg / count
            elif self.aggregation == "sum":
                agg = torch.zeros(N, d_out, device=x.device, dtype=x.dtype)
                agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
                neighbor_term = agg
            else:  # max
                # Initialize with -inf so nodes with no neighbors get 0 after clamp
                agg = torch.full((N, d_out), float("-inf"), device=x.device, dtype=x.dtype)
                # scatter max
                for e in range(dst.size(0)):
                    d_i = dst[e].item()
                    agg[d_i] = torch.max(agg[d_i], msg[e])
                # Replace -inf (isolated nodes) with 0
                agg = agg.masked_fill(agg == float("-inf"), 0.0)
                neighbor_term = agg

        return F.relu(self_term + neighbor_term)


# ---------------------------------------------------------------------------
# GATLayer
# ---------------------------------------------------------------------------


class GATLayer(nn.Module):
    """
    Graph Attention Network layer (Veličković et al., 2018).

    Multi-head attention:
        e_ij = LeakyReLU( a^T [W·h_i || W·h_j] )
        α_ij = softmax_j( e_ij )
        h'_i = σ( Σ_j α_ij · W·h_j )   (concat across heads, then project)
    """

    def __init__(self, d_in: int, d_out: int, n_heads: int = 4) -> None:
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"  # noqa: S101
        self.n_heads = n_heads
        self.d_head = d_out // n_heads

        # One linear per head (weight sharing across edges via per-node projections)
        self.W = nn.Linear(d_in, d_out, bias=False)  # [d_in → n_heads*d_head]
        # Attention vector: 2*d_head per head, stored as [n_heads, 2*d_head]
        self.attn_vec = nn.Parameter(torch.Tensor(n_heads, 2 * self.d_head))
        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.out_proj = nn.Linear(d_out, d_out)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,  # [N, d_in]
        edge_index: torch.Tensor,  # [2, E]
    ) -> torch.Tensor:  # [N, d_out]
        N = x.size(0)
        H = self.n_heads
        d_h = self.d_head
        d_out = H * d_h

        # Project all nodes: [N, H, d_h]
        Wx = self.W(x).view(N, H, d_h)

        if edge_index.size(1) == 0:
            # No edges — return self-projection
            return F.elu(self.out_proj(Wx.reshape(N, d_out)))

        src, dst = edge_index[0], edge_index[1]  # [E]
        E = src.size(0)

        # Features at source and destination: [E, H, d_h]
        h_src = Wx[src]  # [E, H, d_h]
        h_dst = Wx[dst]  # [E, H, d_h]

        # Attention logit: [E, H]
        # attn_vec: [H, 2*d_h] → split into a_i [H, d_h] and a_j [H, d_h]
        a_i = self.attn_vec[:, :d_h]  # [H, d_h]
        a_j = self.attn_vec[:, d_h:]  # [H, d_h]

        # e_ij = LeakyReLU( (h_src · a_i) + (h_dst · a_j) )
        # h_src: [E, H, d_h], a_i: [H, d_h] → einsum → [E, H]
        e = self.leaky(
            (h_src * a_i.unsqueeze(0)).sum(-1) + (h_dst * a_j.unsqueeze(0)).sum(-1)
        )  # [E, H]

        # Softmax per destination node per head
        # Build [E, H] attention weights via scatter softmax
        # 1. Compute exp(e) — subtract per-dst max for stability
        alpha = torch.zeros(E, H, device=x.device, dtype=x.dtype)

        # Per (dst, head) max for numerical stability
        e_max = torch.full((N, H), float("-inf"), device=x.device, dtype=x.dtype)
        e_max.scatter_reduce_(
            0,
            dst.unsqueeze(1).expand(E, H),
            e,
            reduce="amax",
            include_self=True,
        )
        e_shifted = e - e_max[dst]  # [E, H]
        exp_e = torch.exp(e_shifted)  # [E, H]

        # Sum of exps per destination
        exp_sum = torch.zeros(N, H, device=x.device, dtype=x.dtype)
        exp_sum.scatter_add_(0, dst.unsqueeze(1).expand(E, H), exp_e)
        # Normalise
        alpha = exp_e / (exp_sum[dst] + 1e-16)  # [E, H]

        # Weighted aggregation: [N, H, d_h]
        agg = torch.zeros(N, H, d_h, device=x.device, dtype=x.dtype)
        # alpha: [E, H, 1] * h_src: [E, H, d_h] → weighted messages [E, H, d_h]
        weighted = alpha.unsqueeze(-1) * h_src  # [E, H, d_h]
        dst_exp = dst.unsqueeze(1).unsqueeze(2).expand(E, H, d_h)
        agg.scatter_add_(0, dst_exp, weighted)

        # Reshape and project: [N, d_out]
        out = agg.reshape(N, d_out)
        return F.elu(self.out_proj(out))


# ---------------------------------------------------------------------------
# GraphTransformerLayer
# ---------------------------------------------------------------------------


class GraphTransformerLayer(nn.Module):
    """
    Transformer self-attention layer masked by graph adjacency.

    For each node i, attention is restricted to its graph neighbours (and itself).
    Positions with no edge receive -inf before softmax.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0  # noqa: S101
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,  # [N, d_model]
        edge_index: torch.Tensor,  # [2, E]
    ) -> torch.Tensor:  # [N, d_model]
        N, d_model = x.shape
        H = self.n_heads
        d_h = self.d_head

        # Build adjacency mask [N, N]: True where attention is BLOCKED
        # Each node attends to itself + its neighbours
        mask = torch.ones(N, N, dtype=torch.bool, device=x.device)  # all blocked
        # Self-connections allowed
        mask.fill_diagonal_(False)
        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]
            # dst attends to src
            mask[dst, src] = False

        # Multi-head attention
        Q = self.q_proj(x).view(N, H, d_h).transpose(0, 1)  # [H, N, d_h]
        K = self.k_proj(x).view(N, H, d_h).transpose(0, 1)  # [H, N, d_h]
        V = self.v_proj(x).view(N, H, d_h).transpose(0, 1)  # [H, N, d_h]

        # Attention scores [H, N, N]
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [H, N, N]
        # Apply mask: blocked positions → -inf
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)

        # Handle all-masked rows (isolated tokens that don't even have self-loop)
        # In practice self-connection is always allowed above, so this is a safeguard
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.bmm(attn, V)  # [H, N, d_h]
        out = out.transpose(0, 1).reshape(N, d_model)  # [N, d_model]
        out = self.out_proj(out)
        return self.norm(x + out)


# ---------------------------------------------------------------------------
# GNNConfig
# ---------------------------------------------------------------------------


@dataclass
class GNNConfig:
    d_model: int = 32
    vocab_size: int = 64
    n_layers: int = 2
    n_heads: int = 4
    gnn_type: str = "gat"  # "gnn" | "gat" | "graph_transformer"
    graph_type: str = "window"  # "sequential" | "window" | "dependency"
    window_size: int = 3


# ---------------------------------------------------------------------------
# GNNLanguageModel
# ---------------------------------------------------------------------------


class GNNLanguageModel(nn.Module):
    """
    Language model that processes token sequences as graphs.

    Graph construction per sequence:
        - sequential: each token → next
        - window:     each token ↔ tokens within ±window_size
        - dependency: complete causal DAG

    Batching: graphs are concatenated with offset node indices; a batch_vector
    maps each node to its sequence index.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_layers: int = 2,
        gnn_type: str = "gat",
        graph_type: str = "window",
        n_heads: int = 4,
        window_size: int = 3,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.gnn_type = gnn_type
        self.graph_type = graph_type
        self.window_size = window_size

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            if gnn_type == "gnn":
                self.gnn_layers.append(GNNLayer(d_model, d_model, aggregation="mean"))
            elif gnn_type == "gat":
                self.gnn_layers.append(GATLayer(d_model, d_model, n_heads=n_heads))
            elif gnn_type == "graph_transformer":
                self.gnn_layers.append(GraphTransformerLayer(d_model, n_heads=n_heads))
            else:
                raise ValueError(f"Unknown gnn_type: '{gnn_type}'")

        self.lm_head = nn.Linear(d_model, vocab_size)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(
        self,
        input_ids: torch.Tensor,  # [B, T]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Construct per-sequence graphs and concatenate them for batched GNN.

        Returns:
            edge_index  [2, total_E]  — node indices offset per sequence
            batch_vector [B*T]        — maps each node to its sequence index
        """
        B, T = input_ids.shape
        all_src: list[torch.Tensor] = []
        all_dst: list[torch.Tensor] = []
        batch_parts: list[torch.Tensor] = []

        for b in range(B):
            offset = b * T
            if self.graph_type == "sequential":
                ei = TokenGraph.add_sequential_edges(T)
            elif self.graph_type == "window":
                ei = TokenGraph.add_window_edges(T, window=self.window_size)
            elif self.graph_type == "dependency":
                ei = TokenGraph.add_dependency_edges(T)
            else:
                raise ValueError(f"Unknown graph_type: '{self.graph_type}'")

            if ei.size(1) > 0:
                all_src.append(ei[0] + offset)
                all_dst.append(ei[1] + offset)

            batch_parts.append(torch.full((T,), b, dtype=torch.long, device=input_ids.device))

        batch_vector = torch.cat(batch_parts, dim=0)  # [B*T]

        if all_src:
            edge_src = torch.cat(all_src, dim=0)
            edge_dst = torch.cat(all_dst, dim=0)
            edge_index = torch.stack([edge_src, edge_dst], dim=0)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=input_ids.device)

        return edge_index, batch_vector

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:  # [B, T, vocab_size]
        B, T = input_ids.shape

        # Node features: flatten batch, embed
        x = self.embedding(input_ids.view(B * T))  # [B*T, d_model]

        edge_index, _ = self.build_graph(input_ids)

        for layer in self.gnn_layers:
            x = layer(x, edge_index)  # [B*T, d_model]

        logits = self.lm_head(x)  # [B*T, vocab_size]
        return logits.view(B, T, self.vocab_size)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        input_ids: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:  # scalar
        """
        Standard next-token prediction loss.
        Predicts token t+1 from token t; uses tokens 0..T-2 as inputs,
        tokens 1..T-1 as targets.
        """
        logits = self.forward(input_ids)  # [B, T, vocab_size]
        # Shift: predict next token
        shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, vocab]
        shift_labels = input_ids[:, 1:].contiguous()  # [B, T-1]
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )
        return loss
