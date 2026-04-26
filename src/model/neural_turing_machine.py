"""
Neural Turing Machine (NTM)
A differentiable memory-augmented network.
Reference: Graves et al., "Neural Turing Machines" (2014), arXiv:1410.5401
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# NTMMemory
# ---------------------------------------------------------------------------


class NTMMemory:
    """External memory bank shared across an NTM batch."""

    def __init__(self, n_locations: int, word_size: int) -> None:
        self.n_locations = n_locations
        self.word_size = word_size
        # memory is registered as a plain tensor; managed externally
        self.memory: torch.Tensor = torch.zeros(n_locations, word_size)

    # ------------------------------------------------------------------
    def read(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Weighted read from memory.

        Args:
            weights: [B, n_locations]  (attention distribution, sums to 1)

        Returns:
            read_vec: [B, word_size]
        """
        # [B, n_loc] x [n_loc, word_size] → [B, word_size]
        return torch.matmul(weights, self.memory)

    # ------------------------------------------------------------------
    def write(
        self,
        weights: torch.Tensor,  # [B, n_locations]
        erase: torch.Tensor,  # [B, word_size]
        add: torch.Tensor,  # [B, word_size]
    ) -> None:
        """
        Differentiable erase + add write operation.

        M_i ← M_i * (1 - w_i * e) + w_i * a   for each location i

        The update is averaged over the batch dimension so that gradients
        flow properly when B > 1.
        """
        # weights: [B, n_loc], erase/add: [B, word_size]
        # Outer products → [B, n_loc, word_size]
        erase_mat = torch.bmm(weights.unsqueeze(2), erase.unsqueeze(1))  # [B, n_loc, word_size]
        add_mat = torch.bmm(weights.unsqueeze(2), add.unsqueeze(1))  # [B, n_loc, word_size]

        # Average over batch (allows gradient flow; standard NTM uses single sample or mean)
        erase_mean = erase_mat.mean(0)  # [n_loc, word_size]
        add_mean = add_mat.mean(0)  # [n_loc, word_size]

        self.memory = self.memory * (1.0 - erase_mean) + add_mean

    # ------------------------------------------------------------------
    def reset(self, batch_size: int) -> None:
        """Re-initialise memory to a small constant value (1e-6)."""
        self.memory = torch.full(
            (self.n_locations, self.word_size),
            fill_value=1e-6,
        )

    # ------------------------------------------------------------------
    def size(self) -> tuple[int, int]:
        return self.n_locations, self.word_size


# ---------------------------------------------------------------------------
# ContentAddressing
# ---------------------------------------------------------------------------


class ContentAddressing:
    """Content-based addressing via cosine similarity."""

    @staticmethod
    def cosine_similarity(
        key: torch.Tensor,  # [B, word_size]
        memory: torch.Tensor,  # [n_loc, word_size]
    ) -> torch.Tensor:
        """
        Returns cosine similarity between each key vector and every memory row.

        Output: [B, n_loc]
        """
        # key:    [B, W]    → [B, 1, W]
        # memory: [n_loc, W] → [1, n_loc, W]
        key_norm = F.normalize(key, p=2, dim=-1)  # [B, W]
        memory_norm = F.normalize(memory, p=2, dim=-1)  # [n_loc, W]
        # [B, W] @ [W, n_loc] → [B, n_loc]
        return torch.matmul(key_norm, memory_norm.t())

    @staticmethod
    def content_weights(
        key: torch.Tensor,  # [B, word_size]
        memory: torch.Tensor,  # [n_loc, word_size]
        beta: torch.Tensor,  # [B, 1]  key strength (sharpening)
    ) -> torch.Tensor:
        """
        Returns softmax-normalised content attention.

        Output: [B, n_loc]
        """
        sim = ContentAddressing.cosine_similarity(key, memory)  # [B, n_loc]
        return F.softmax(beta * sim, dim=-1)  # [B, n_loc]


# ---------------------------------------------------------------------------
# LocationAddressing
# ---------------------------------------------------------------------------


class LocationAddressing:
    """Location-based addressing: interpolation, shift, sharpen."""

    @staticmethod
    def interpolate(
        w_prev: torch.Tensor,  # [B, n_loc]
        w_content: torch.Tensor,  # [B, n_loc]
        gate: torch.Tensor,  # [B, 1]   in (0, 1)
    ) -> torch.Tensor:
        """
        Interpolate between previous and content weights.

        w_g = gate * w_content + (1 - gate) * w_prev

        Output: [B, n_loc]
        """
        return gate * w_content + (1.0 - gate) * w_prev

    @staticmethod
    def shift(
        w_g: torch.Tensor,  # [B, n_loc]
        shift_weights: torch.Tensor,  # [B, 3]   distribution over {-1, 0, +1}
    ) -> torch.Tensor:
        """
        Circular convolution with a width-3 shift kernel.

        shift_weights[:, 0] → shift left  (-1)
        shift_weights[:, 1] → no shift     (0)
        shift_weights[:, 2] → shift right (+1)

        Output: [B, n_loc]
        """
        B, n_loc = w_g.shape
        # Pad for circular convolution: replicate ends
        # Shift -1 means each location gets contribution from the one to its right
        # We implement via explicit roll summation for clarity and differentiability
        w_shifted = (
            shift_weights[:, 0:1] * torch.roll(w_g, 1, dims=1)  # shift left  (look from +1)
            + shift_weights[:, 1:2] * w_g  # no shift
            + shift_weights[:, 2:3] * torch.roll(w_g, -1, dims=1)  # shift right (look from -1)
        )
        return w_shifted  # [B, n_loc]

    @staticmethod
    def sharpen(
        w_shifted: torch.Tensor,  # [B, n_loc]
        gamma: torch.Tensor,  # [B, 1]  ≥ 1
    ) -> torch.Tensor:
        """
        Sharpening: w^gamma / sum(w^gamma).

        Output: [B, n_loc]
        """
        # Clamp to avoid negative values before power
        w_pow = (w_shifted.clamp(min=1e-9)) ** gamma  # [B, n_loc]
        return w_pow / (w_pow.sum(dim=-1, keepdim=True) + 1e-9)


# ---------------------------------------------------------------------------
# NTMHead
# ---------------------------------------------------------------------------


class NTMHead(nn.Module):
    """
    A single NTM read or write head.

    The head maps a controller hidden state to addressing parameters,
    performs addressing, and reads from / writes to memory.
    """

    def __init__(
        self,
        d_controller: int,
        word_size: int,
        n_locations: int,
        head_type: str,  # "read" or "write"
    ) -> None:
        super().__init__()

        if head_type not in ("read", "write"):
            raise ValueError(f"head_type must be 'read' or 'write', got '{head_type}'")

        self.head_type = head_type
        self.word_size = word_size
        self.n_locations = n_locations

        # --- Shared addressing parameters ---
        # key        : word_size
        # beta       : 1   (key strength, softplus → > 0)
        # gate       : 1   (interpolation, sigmoid → (0,1))
        # shift_weights: 3 (softmax)
        # gamma      : 1   (sharpening, softplus + 1 → ≥ 1)
        shared_out = word_size + 1 + 1 + 3 + 1  # = word_size + 6

        # --- Write-only parameters ---
        # erase: word_size
        # add  : word_size
        write_out = 2 * word_size if head_type == "write" else 0

        total_out = shared_out + write_out
        self.fc = nn.Linear(d_controller, total_out)

        self.content_addr = ContentAddressing()
        self.location_addr = LocationAddressing()

    # ------------------------------------------------------------------
    def _parse_params(self, controller_state: torch.Tensor) -> dict:
        """Split the linear output into named addressing parameters."""
        W = self.word_size
        raw = self.fc(controller_state)  # [B, total_out]

        offset = 0
        key = raw[:, offset : offset + W]
        offset += W
        beta = raw[:, offset : offset + 1]
        offset += 1
        gate = raw[:, offset : offset + 1]
        offset += 1
        shift = raw[:, offset : offset + 3]
        offset += 3
        gamma = raw[:, offset : offset + 1]
        offset += 1

        params = {
            "key": key,
            "beta": F.softplus(beta),  # > 0
            "gate": torch.sigmoid(gate),  # (0, 1)
            "shift": F.softmax(shift, dim=-1),  # sums to 1
            "gamma": F.softplus(gamma) + 1.0,  # ≥ 1
        }

        if self.head_type == "write":
            erase = raw[:, offset : offset + W]
            offset += W
            add = raw[:, offset : offset + W]
            offset += W
            params["erase"] = torch.sigmoid(erase)  # (0, 1) per element
            params["add"] = torch.tanh(add)  # (-1, 1) per element

        return params

    # ------------------------------------------------------------------
    def forward(
        self,
        controller_state: torch.Tensor,  # [B, d_controller]
        memory: NTMMemory,
        w_prev: torch.Tensor,  # [B, n_locations]
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        Perform one addressing step.

        Returns:
            (read_vec, w_new)  — read head
            (None,     w_new)  — write head  (memory modified in place)
        """
        p = self._parse_params(controller_state)

        # 1. Content addressing
        w_content = ContentAddressing.content_weights(p["key"], memory.memory, p["beta"])

        # 2. Interpolate
        w_g = LocationAddressing.interpolate(w_prev, w_content, p["gate"])

        # 3. Shift
        w_shifted = LocationAddressing.shift(w_g, p["shift"])

        # 4. Sharpen
        w_new = LocationAddressing.sharpen(w_shifted, p["gamma"])

        if self.head_type == "read":
            read_vec = memory.read(w_new)  # [B, word_size]
            return read_vec, w_new
        else:  # write
            memory.write(w_new, p["erase"], p["add"])
            return None, w_new


# ---------------------------------------------------------------------------
# NTMController
# ---------------------------------------------------------------------------


class NTMController(nn.Module):
    """
    LSTM controller that receives the current input concatenated with
    previous read vectors and emits a hidden state.
    """

    def __init__(
        self,
        input_size: int,
        d_controller: int,
        n_reads: int,
        word_size: int,
    ) -> None:
        super().__init__()
        self.d_controller = d_controller
        lstm_input_size = input_size + n_reads * word_size
        self.lstm_cell = nn.LSTMCell(lstm_input_size, d_controller)

        # Will be set during reset()
        self._hidden: torch.Tensor | None = None
        self._cell: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,  # [B, input_size]
        reads: list[torch.Tensor],  # list of [B, word_size]
    ) -> torch.Tensor:  # [B, d_controller]
        """One LSTM step; updates internal hidden/cell state."""
        inp = torch.cat([x] + reads, dim=-1)  # [B, input_size + n_reads*word_size]
        self._hidden, self._cell = self.lstm_cell(inp, (self._hidden, self._cell))
        return self._hidden

    # ------------------------------------------------------------------
    def reset(self, batch_size: int) -> None:
        """Zero out hidden and cell states for a new episode."""
        device = next(self.parameters()).device
        self._hidden = torch.zeros(batch_size, self.d_controller, device=device)
        self._cell = torch.zeros(batch_size, self.d_controller, device=device)


# ---------------------------------------------------------------------------
# NTMModel
# ---------------------------------------------------------------------------


class NTMModel(nn.Module):
    """
    Full Neural Turing Machine.

    Combines an LSTM controller with differentiable external memory
    accessed via content + location addressing.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_controller: int,
        word_size: int,
        n_locations: int,
        n_reads: int = 1,
        n_writes: int = 1,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.d_controller = d_controller
        self.word_size = word_size
        self.n_locations = n_locations
        self.n_reads = n_reads
        self.n_writes = n_writes

        # Sub-modules
        self.controller = NTMController(input_size, d_controller, n_reads, word_size)

        self.read_heads = nn.ModuleList(
            [NTMHead(d_controller, word_size, n_locations, "read") for _ in range(n_reads)]
        )
        self.write_heads = nn.ModuleList(
            [NTMHead(d_controller, word_size, n_locations, "write") for _ in range(n_writes)]
        )

        self.memory = NTMMemory(n_locations, word_size)

        # Output projection: controller state + all read vectors → output
        self.output_layer = nn.Linear(d_controller + n_reads * word_size, output_size)

        # Head attention weights and read vectors (set during reset/forward)
        self._read_weights: list[torch.Tensor] = []
        self._write_weights: list[torch.Tensor] = []
        self._reads: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    def reset(self, batch_size: int) -> None:
        """Reset all stateful components for a new episode."""
        device = next(self.parameters()).device

        self.controller.reset(batch_size)
        self.memory.reset(batch_size)

        # Initialise attention weights as uniform distributions
        uniform = torch.zeros(batch_size, self.n_locations, device=device)
        uniform[:, 0] = 1.0  # start with all attention on location 0

        self._read_weights = [uniform.clone() for _ in range(self.n_reads)]
        self._write_weights = [uniform.clone() for _ in range(self.n_writes)]

        # Initialise reads to zeros
        self._reads = [
            torch.zeros(batch_size, self.word_size, device=device) for _ in range(self.n_reads)
        ]

    # ------------------------------------------------------------------
    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a single time step.

        Args:
            x: [B, input_size]

        Returns:
            output: [B, output_size]
        """
        # Controller step
        h = self.controller(x, self._reads)  # [B, d_controller]

        # Write heads (modify memory before reading)
        for i, head in enumerate(self.write_heads):
            _, w_new = head(h, self.memory, self._write_weights[i])
            self._write_weights[i] = w_new

        # Read heads
        new_reads = []
        for i, head in enumerate(self.read_heads):
            r, w_new = head(h, self.memory, self._read_weights[i])
            self._read_weights[i] = w_new
            new_reads.append(r)
        self._reads = new_reads

        # Output
        out_inp = torch.cat([h] + self._reads, dim=-1)  # [B, d_ctrl + n_reads*W]
        return self.output_layer(out_inp)

    # ------------------------------------------------------------------
    def forward_sequence(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Process a sequence of time steps.

        Args:
            xs: [B, T, input_size]

        Returns:
            outputs: [B, T, output_size]
        """
        B, T, _ = xs.shape
        self.reset(B)
        outputs = []
        for t in range(T):
            out = self.forward_step(xs[:, t, :])
            outputs.append(out)
        return torch.stack(outputs, dim=1)  # [B, T, output_size]

    # ------------------------------------------------------------------
    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper; calls forward_sequence."""
        return self.forward_sequence(xs)


# ---------------------------------------------------------------------------
# NTMConfig
# ---------------------------------------------------------------------------


@dataclass
class NTMConfig:
    """Default hyper-parameters for a small NTM."""

    input_size: int = 8
    output_size: int = 8
    d_controller: int = 32
    word_size: int = 8
    n_locations: int = 16
    n_reads: int = 1
    n_writes: int = 1
