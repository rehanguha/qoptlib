"""Quantum-inspired Adam optimizer."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from qoptlib.opt.base import BaseOptimizer


class QuantumAdam(BaseOptimizer):
    """Quantum Adam optimizer.

    Args:
        params: List of numpy arrays (parameters).
        lr: Learning rate.
        quantum_strength: Strength of quantum effects (0-1).
        betas: (beta1, beta2) exponential decay rates.
        eps: Epsilon for numerical stability.
        weight_decay: L2 weight decay coefficient.
        amsgrad: Use AMSGrad variant.
        seed: Random seed.
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 0.001,
        quantum_strength: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(params, lr, quantum_strength, seed)

        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps <= 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.max_v = [np.zeros_like(p) for p in params] if amsgrad else None

    def _update(self, grads: List[np.ndarray]) -> None:
        beta1, beta2 = self.betas
        t = self.step_count + 1

        for i, (p, g) in enumerate(zip(self.params, grads)):
            if self.weight_decay > 0:
                g = g + self.weight_decay * p

            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * g * g

            if self.amsgrad:
                self.max_v[i] = np.maximum(self.max_v[i], self.v[i])
                v_hat = self.max_v[i]
            else:
                v_hat = self.v[i]

            m_hat = self.m[i] / (1 - beta1 ** t)
            v_hat = v_hat / (1 - beta2 ** t)

            phase = self.rng.uniform(0, 2 * np.pi, size=p.shape).astype(np.float32)
            quantum_g = g * np.cos(phase)

            update = m_hat / (np.sqrt(v_hat) + self.eps)
            update = update * (1 - self.quantum_strength) + quantum_g * self.quantum_strength

            p -= self.lr * update

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["m"] = [m.copy() for m in self.m]
        state["v"] = [v.copy() for v in self.v]
        state["lr"] = self.lr
        if self.amsgrad and self.max_v is not None:
            state["max_v"] = [v.copy() for v in self.max_v]
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.m = state.get("m", [np.zeros_like(p) for p in self.params])
        self.v = state.get("v", [np.zeros_like(p) for p in self.params])
        if self.amsgrad:
            self.max_v = state.get("max_v", [np.zeros_like(p) for p in self.params])
        if "lr" in state:
            self.lr = state["lr"]

    def __repr__(self) -> str:
        return (
            f"QuantumAdam(lr={self.lr}, quantum_strength={self.quantum_strength}, "
            f"betas={self.betas}, weight_decay={self.weight_decay}, amsgrad={self.amsgrad})"
        )
