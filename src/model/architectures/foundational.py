"""Foundational architectures: Perceptron, MLP, Backprop, LSTM, GRU, Seq2Seq, Attention.

Papers: Rosenblatt 1958, Rumelhart 1986, Hochreiter 1997, Bahdanau 2014, Sutskever 2014, Cho 2014.
"""

from __future__ import annotations

import math
import random

from .registry import register


class Perceptron:
    """Rosenblatt 1958: single-layer binary linear classifier."""

    def __init__(self, n_inputs: int, lr: float = 0.01) -> None:
        self.weights = [random.gauss(0, 0.1) for _ in range(n_inputs)]
        self.bias = 0.0
        self.lr = lr

    def forward(self, x: list[float]) -> int:
        net = sum(w * xi for w, xi in zip(self.weights, x, strict=True)) + self.bias
        return 1 if net > 0 else 0

    def train(self, x: list[float], target: int) -> None:
        pred = self.forward(x)
        error = target - pred
        for i in range(len(self.weights)):
            self.weights[i] += self.lr * error * x[i]
        self.bias += self.lr * error


register("foundational.perceptron", Perceptron)


class MLP:
    """Multi-Layer Perceptron — universal approximator (Cybenko 1989)."""

    def __init__(self, layer_sizes: list[int], activation: str = "relu") -> None:
        self.weights: list[list[list[float]]] = []
        self.biases: list[list[float]] = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            scale = math.sqrt(2.0 / fan_in)
            self.weights.append(
                [[random.gauss(0, scale) for _ in range(fan_in)] for _ in range(fan_out)]
            )
            self.biases.append([0.0] * fan_out)
        self.activation = activation

    def _act(self, x: float) -> float:
        if self.activation == "relu":
            return max(0.0, x)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + math.exp(-x))
        elif self.activation == "tanh":
            return math.tanh(x)
        return x

    def forward(self, x: list[float]) -> list[float]:
        for w, b in zip(self.weights, self.biases, strict=True):
            x = [
                self._act(sum(w[j][i] * x[i] for i in range(len(x))) + b[j]) for j in range(len(w))
            ]
        return x


register("foundational.mlp", MLP)


class LSTMCell:
    """LSTM cell (Hochreiter & Schmidhuber 1997)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        s = 1.0 / math.sqrt(hidden_size)
        self.Wf = [
            [random.gauss(0, s) for _ in range(input_size + hidden_size)]
            for _ in range(hidden_size)
        ]
        self.Wi = [
            [random.gauss(0, s) for _ in range(input_size + hidden_size)]
            for _ in range(hidden_size)
        ]
        self.Wo = [
            [random.gauss(0, s) for _ in range(input_size + hidden_size)]
            for _ in range(hidden_size)
        ]
        self.Wc = [
            [random.gauss(0, s) for _ in range(input_size + hidden_size)]
            for _ in range(hidden_size)
        ]
        self.bf = [0.0] * hidden_size
        self.bi = [0.0] * hidden_size
        self.bo = [0.0] * hidden_size
        self.bc = [0.0] * hidden_size

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _tanh(x: float) -> float:
        return math.tanh(x)

    def forward(
        self, x: list[float], h: list[float], c: list[float]
    ) -> tuple[list[float], list[float]]:
        x = list(x) + [0.0] * max(0, self.input_size - len(x))
        h = list(h) + [0.0] * max(0, self.hidden_size - len(h))
        concat = x[: self.input_size] + h[: self.hidden_size]
        f = [
            self._sigmoid(sum(self.Wf[j][k] * concat[k] for k in range(len(concat))) + self.bf[j])
            for j in range(self.hidden_size)
        ]
        i = [
            self._sigmoid(sum(self.Wi[j][k] * concat[k] for k in range(len(concat))) + self.bi[j])
            for j in range(self.hidden_size)
        ]
        o = [
            self._sigmoid(sum(self.Wo[j][k] * concat[k] for k in range(len(concat))) + self.bo[j])
            for j in range(self.hidden_size)
        ]
        ct = [
            self._tanh(sum(self.Wc[j][k] * concat[k] for k in range(len(concat))) + self.bc[j])
            for j in range(self.hidden_size)
        ]
        c_next = [f[j] * c[j] + i[j] * ct[j] for j in range(self.hidden_size)]
        h_next = [o[j] * self._tanh(c_next[j]) for j in range(self.hidden_size)]
        return h_next, c_next


register("foundational.lstm", LSTMCell)


class GRUCell:
    """GRU cell (Cho et al. 2014)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        self.hidden_size = hidden_size
        s = 1.0 / math.sqrt(hidden_size)
        self.Wz = [
            [random.gauss(0, s) for _ in range(input_size + hidden_size)]
            for _ in range(hidden_size)
        ]
        self.Wr = [
            [random.gauss(0, s) for _ in range(input_size + hidden_size)]
            for _ in range(hidden_size)
        ]
        self.Wh = [
            [random.gauss(0, s) for _ in range(input_size + hidden_size)]
            for _ in range(hidden_size)
        ]
        self.bz = [0.0] * hidden_size
        self.br = [0.0] * hidden_size
        self.bh = [0.0] * hidden_size

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _tanh(self, x: float) -> float:
        return math.tanh(x)

    def forward(self, x: list[float], h: list[float]) -> list[float]:
        concat = x + h
        z = [
            self._sigmoid(sum(self.Wz[j][k] * concat[k] for k in range(len(concat))) + self.bz[j])
            for j in range(self.hidden_size)
        ]
        r = [
            self._sigmoid(sum(self.Wr[j][k] * concat[k] for k in range(len(concat))) + self.br[j])
            for j in range(self.hidden_size)
        ]
        rh = [r[j] * h[j] for j in range(self.hidden_size)]
        concat2 = x + rh
        hc = [
            self._tanh(sum(self.Wh[j][k] * concat2[k] for k in range(len(concat2))) + self.bh[j])
            for j in range(self.hidden_size)
        ]
        return [(1.0 - z[j]) * h[j] + z[j] * hc[j] for j in range(self.hidden_size)]


register("foundational.gru", GRUCell)


class Seq2Seq:
    """Sequence-to-Sequence model (Sutskever et al. 2014)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        self.encoder = LSTMCell(input_size, hidden_size)
        self.decoder = LSTMCell(output_size, hidden_size)

    def forward(
        self, x_seq: list[list[float]], target_seq: list[list[float]] | None = None
    ) -> list[list[float]]:
        h = [0.0] * self.encoder.hidden_size
        c = [0.0] * self.encoder.hidden_size
        for x in x_seq:
            h, c = self.encoder.forward(x, h, c)
        outputs: list[list[float]] = []
        decoder_input = [0.0] * self.decoder.hidden_size
        for _ in range(len(target_seq) if target_seq else len(x_seq)):
            h, c = self.decoder.forward(decoder_input, h, c)
            outputs.append(h)
            decoder_input = h
        return outputs


register("foundational.seq2seq", Seq2Seq)


class AttentionLayer:
    """Bahdanau Attention (Bahdanau et al. 2014)."""

    def __init__(self, hidden_size: int) -> None:
        self.W = [random.gauss(0, 0.1) for _ in range(hidden_size)]
        self.U = [[random.gauss(0, 0.1) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.v = [random.gauss(0, 0.1) for _ in range(hidden_size)]

    def forward(
        self, query: list[float], keys: list[list[float]], values: list[list[float]]
    ) -> tuple[list[float], list[float]]:
        scores = [
            sum(
                self.v[j]
                * math.tanh(
                    sum(self.U[j][k] * query[k] for k in range(len(query)))
                    + sum(self.W[k] * kvec[k] for k in range(len(kvec)))
                )
                for j in range(len(self.v))
            )
            for kvec in keys
        ]
        exp_scores = [math.exp(s - max(scores)) for s in scores]
        attn = [e / sum(exp_scores) for e in exp_scores]
        context = [
            sum(attn[i] * values[i][j] for i in range(len(values)))
            for j in range(len(values[0]) if values else 0)
        ]
        return context, attn


register("foundational.attention", AttentionLayer)
