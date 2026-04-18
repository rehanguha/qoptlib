"""Quantum-inspired Tunneling optimizer."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np

from qoptlib.opt.base import BaseOptimizer


class QuantumTunneling(BaseOptimizer):
    """Quantum Tunneling optimizer.

    Uses simulated quantum tunneling to escape local minima.

    Args:
        params: List of numpy arrays (parameters).
        lr: Learning rate.
        quantum_strength: Tunneling probability factor.
        beta1, beta2: Exponential decay rates.
        eps: Epsilon for numerical stability.
        weight_decay: L2 weight decay coefficient.
        tunneling_decay: Decay rate for tunneling energy.
        seed: Random seed.
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 0.001,
        quantum_strength: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        tunneling_decay: float = 0.95,
        seed: Optional[int] = None,
    ):
        super().__init__(params, lr, quantum_strength, seed)

        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2: {beta2}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.tunneling_decay = tunneling_decay

        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.tunnel = [np.zeros_like(p) for p in params]

    def _update(self, grads: List[np.ndarray]) -> None:
        t = self.step_count + 1

        for i, (p, g) in enumerate(zip(self.params, grads)):
            if self.weight_decay > 0:
                g = g + self.weight_decay * p

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g

            g_norm = np.linalg.norm(g)
            v_norm = np.linalg.norm(self.v[i])
            barrier = g_norm / (v_norm + self.eps)
            tunnel_prob = np.exp(-barrier / (self.quantum_strength + self.eps))

            direction = self.rng.randn(*p.shape).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + self.eps)

            self.tunnel[i] = (
                self.tunneling_decay * self.tunnel[i]
                + tunnel_prob * direction * g_norm * 0.1
            )

            m_hat = self.m[i] / (1 - self.beta1 ** t)
            v_hat = self.v[i] / (1 - self.beta2 ** t)

            update = m_hat / (np.sqrt(v_hat) + self.eps)
            update = update * (1 - self.quantum_strength) + self.tunnel[i] * self.quantum_strength

            p -= self.lr * update

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["m"] = [m.copy() for m in self.m]
        state["v"] = [v.copy() for v in self.v]
        state["tunnel"] = [t.copy() for t in self.tunnel]
        state["lr"] = self.lr
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.m = state.get("m", [np.zeros_like(p) for p in self.params])
        self.v = state.get("v", [np.zeros_like(p) for p in self.params])
        self.tunnel = state.get("tunnel", [np.zeros_like(p) for p in self.params])
        if "lr" in state:
            self.lr = state["lr"]

    def __repr__(self) -> str:
        return (
            f"QuantumTunneling(lr={self.lr}, quantum_strength={self.quantum_strength}, "
            f"beta1={self.beta1}, beta2={self.beta2}, tunneling_decay={self.tunneling_decay})"
        )
