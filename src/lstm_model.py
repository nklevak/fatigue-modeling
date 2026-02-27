"""
NumPy-based LSTM for predicting rest_length at each epoch from the full history.
Many-to-many: at each timestep t, predicts rest_length[t] given features [1..t].
No PyTorch required - uses only numpy.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


class RestLSTM:
    """
    Many-to-many LSTM: input (batch, seq_len, n_features) -> output (batch, seq_len, 1).
    At each timestep t, outputs prediction for rest_length at epoch t.
    Pure NumPy implementation.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize LSTM and FC weights (Xavier-like)."""
        scale = 0.1
        self.layers: list[dict] = []
        in_dim = self.n_features
        for _ in range(self.num_layers):
            h, d = self.hidden_size, in_dim
            W = np.random.randn(4 * h, h + d).astype(np.float32) * scale
            b = np.zeros(4 * h, dtype=np.float32)
            self.layers.append({"W": W, "b": b})
            in_dim = h
        self.fc_W = np.random.randn(self.hidden_size, 1).astype(np.float32) * scale
        self.fc_b = np.zeros(1, dtype=np.float32)

    def _lstm_step(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
        W: np.ndarray,
        b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, tuple]:
        """Single LSTM step. Returns (h, c, cache) for backward."""
        h_dim = h_prev.shape[1]
        combined = np.concatenate([h_prev, x], axis=1)
        z = combined @ W.T + b
        i = _sigmoid(z[:, 0 * h_dim : 1 * h_dim])
        f = _sigmoid(z[:, 1 * h_dim : 2 * h_dim])
        g = _tanh(z[:, 2 * h_dim : 3 * h_dim])
        o = _sigmoid(z[:, 3 * h_dim : 4 * h_dim])
        c = f * c_prev + i * g
        h = o * _tanh(c)
        cache = (x, h_prev, c_prev, i, f, g, o, c, combined, h_dim)
        return h, c, cache

    def forward(
        self,
        x: np.ndarray,
        mask: np.ndarray | None = None,
        training: bool = False,
        return_caches: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list]:
        """
        x: (batch, seq_len, n_features)
        mask: (batch, seq_len) True for valid positions
        returns: (batch, seq_len, 1) or (out, caches) if return_caches
        """
        batch, seq_len, _ = x.shape
        dropout_mask = None
        if training and self.dropout > 0 and self.num_layers > 1:
            dropout_mask = (
                np.random.rand(self.num_layers - 1, batch, seq_len, self.hidden_size) > self.dropout
            ) / (1.0 - self.dropout)

        caches: list[list] = []
        layer_out = x
        for layer_idx, layer in enumerate(self.layers):
            h = np.zeros((batch, self.hidden_size), dtype=np.float32)
            c = np.zeros((batch, self.hidden_size), dtype=np.float32)
            seq_out = []
            layer_caches = []
            for t in range(seq_len):
                xt = layer_out[:, t, :]
                h, c, cache = self._lstm_step(xt, h, c, layer["W"], layer["b"])
                if mask is not None:
                    m = mask[:, t : t + 1]
                    h = np.where(m, h, 0.0)
                    c = np.where(m, c, 0.0)
                seq_out.append(h)
                layer_caches.append(cache)
            layer_out = np.stack(seq_out, axis=1)
            caches.append(layer_caches)
            if dropout_mask is not None and layer_idx < self.num_layers - 1:
                layer_out = layer_out * dropout_mask[layer_idx]

        out = layer_out @ self.fc_W + self.fc_b
        if return_caches:
            return out, caches
        return out

    def predict(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Forward pass without dropout."""
        return self.forward(x, mask=mask, training=False)

    def _layer_backward(
        self,
        layer_idx: int,
        dh_upstream: np.ndarray,
        mask: np.ndarray,
        caches: list,
        batch: int,
        seq_len: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """BPTT for one layer. Returns (dW, db, dx) where dx is grad for layer input."""
        layer = self.layers[layer_idx]
        layer_caches = caches[layer_idx]
        h_dim = self.hidden_size
        dW = np.zeros_like(layer["W"])
        db = np.zeros_like(layer["b"])
        dx = np.zeros((batch, seq_len, layer_caches[0][0].shape[1]), dtype=np.float32)
        dc_next = np.zeros((batch, h_dim), dtype=np.float32)

        for t in range(seq_len - 1, -1, -1):
            m = mask[:, t : t + 1]
            dh = np.where(m, dh_upstream[:, t, :], 0.0)
            x_t, h_prev, c_prev, i, f, g, o, c, combined, _ = layer_caches[t]

            # dL/dh = dh, h = o * tanh(c) => dc_from_h = dh * o * (1 - tanh(c)^2)
            dc_in = dh * o * (1 - _tanh(c) ** 2)
            do = dh * _tanh(c) * o * (1 - o)
            dc_in = np.where(m, dc_in + dc_next, 0.0)
            df_gate = dc_in * c_prev * f * (1 - f)
            di_gate = dc_in * g * i * (1 - i)
            dg_gate = dc_in * i * (1 - g * g)
            dc_next = np.where(m, dc_in * f, 0.0)

            dz = np.concatenate([di_gate, df_gate, dg_gate, do], axis=1)
            dW += dz.T @ combined
            db += dz.sum(axis=0)
            d_combined = dz @ layer["W"]
            dh_prev = d_combined[:, :h_dim]
            dx[:, t, :] = d_combined[:, h_dim:]

        return dW, db, dx

    def backward(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        d_out: np.ndarray,
        caches: list,
    ) -> list:
        """
        BPTT. d_out: (batch, seq_len, 1) gradient of loss w.r.t. output.
        Returns list of (dW, db) per layer, then (dfc_W, dfc_b).
        """
        batch, seq_len, _ = x.shape
        dh_upstream = (d_out @ self.fc_W.T)  # (batch, seq_len, 1) -> (batch, seq_len, hidden)
        grads = []

        for layer_idx in range(self.num_layers - 1, -1, -1):
            dW, db, dx = self._layer_backward(
                layer_idx, dh_upstream, mask, caches, batch, seq_len
            )
            grads.insert(0, (dW, db))
            dh_upstream = dx  # upstream for layer below

        # FC gradients: dL/dfc_W = H^T @ d_out, dL/dfc_b = sum(d_out)
        # H = last layer hidden states from caches: h = o * tanh(c)
        last_caches = caches[-1]
        H = np.stack(
            [
                last_caches[t][6] * _tanh(last_caches[t][7])  # o * tanh(c)
                for t in range(seq_len)
            ],
            axis=1,
        )
        dfc_W = H.reshape(-1, self.hidden_size).T @ d_out.reshape(-1, 1)
        dfc_b = np.array([d_out.sum()], dtype=np.float32)
        grads.append((dfc_W, dfc_b))
        return grads


class NumPyAdam:
    """Adam optimizer for NumPy LSTM parameters with optional L2 weight decay."""

    def __init__(
        self,
        model: RestLSTM,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m_W = [np.zeros_like(l["W"]) for l in model.layers]
        self.m_b = [np.zeros_like(l["b"]) for l in model.layers]
        self.m_fc_W = np.zeros_like(model.fc_W)
        self.m_fc_b = np.zeros_like(model.fc_b)
        self.v_W = [np.zeros_like(l["W"]) for l in model.layers]
        self.v_b = [np.zeros_like(l["b"]) for l in model.layers]
        self.v_fc_W = np.zeros_like(model.fc_W)
        self.v_fc_b = np.zeros_like(model.fc_b)

    def step(self, model: RestLSTM, grads: list) -> None:
        self.t += 1
        for i, (dW, db) in enumerate(grads[:-1]):
            if self.weight_decay > 0:
                dW = dW + self.weight_decay * model.layers[i]["W"]
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * dW
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (dW * dW)
            mw = self.m_W[i] / (1 - self.beta1 ** self.t)
            vw = self.v_W[i] / (1 - self.beta2 ** self.t)
            model.layers[i]["W"] -= self.lr * mw / (np.sqrt(vw) + self.eps)

            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db * db)
            mb = self.m_b[i] / (1 - self.beta1 ** self.t)
            vb = self.v_b[i] / (1 - self.beta2 ** self.t)
            model.layers[i]["b"] -= self.lr * mb / (np.sqrt(vb) + self.eps)

        dfc_W, dfc_b = grads[-1]
        if self.weight_decay > 0:
            dfc_W = dfc_W + self.weight_decay * model.fc_W
        self.m_fc_W = self.beta1 * self.m_fc_W + (1 - self.beta1) * dfc_W
        self.v_fc_W = self.beta2 * self.v_fc_W + (1 - self.beta2) * (dfc_W * dfc_W)
        mw = self.m_fc_W / (1 - self.beta1 ** self.t)
        vw = self.v_fc_W / (1 - self.beta2 ** self.t)
        model.fc_W -= self.lr * mw / (np.sqrt(vw) + self.eps)

        self.m_fc_b = self.beta1 * self.m_fc_b + (1 - self.beta1) * dfc_b
        self.v_fc_b = self.beta2 * self.v_fc_b + (1 - self.beta2) * (dfc_b * dfc_b)
        mb = self.m_fc_b / (1 - self.beta1 ** self.t)
        vb = self.v_fc_b / (1 - self.beta2 ** self.t)
        model.fc_b -= self.lr * mb / (np.sqrt(vb) + self.eps)


def get_state_dict(model: RestLSTM) -> dict:
    """Return a copy of model weights for saving."""
    state = {"layers": [], "fc_W": model.fc_W.copy(), "fc_b": model.fc_b.copy()}
    for layer in model.layers:
        state["layers"].append({"W": layer["W"].copy(), "b": layer["b"].copy()})
    return state


def load_state_dict(model: RestLSTM, state: dict) -> None:
    """Load weights from state dict."""
    for i, layer_state in enumerate(state["layers"]):
        model.layers[i]["W"] = layer_state["W"].copy()
        model.layers[i]["b"] = layer_state["b"].copy()
    model.fc_W = state["fc_W"].copy()
    model.fc_b = state["fc_b"].copy()
