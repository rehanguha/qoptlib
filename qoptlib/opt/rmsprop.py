"""Quantum-inspired RMSprop optimizer."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np

from qoptlib.opt.base import BaseOptimizer


class QuantumRMSprop(BaseOptimizer):
    """Quantum RMSprop optimizer.

    Args:
        params: List of numpy arrays (parameters).
        lr: Learning rate.
        quantum_strength: Strength of quantum effects (0-1).
        alpha: Smoothing constant.
        eps: Epsilon for numerical stability.
        momentum: Momentum factor.
        weight_decay: L2 weight decay coefficient.
        centered: Use centered RMSprop.
        seed: Random seed.
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 0.01,
        quantum_strength: float = 0.1,
        alpha: float = 0.99,
        eps: float = 1e-8,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        centered: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(params, lr, quantum_strength, seed)

        if not 0 <= alpha < 1:
            raise ValueError(f"Invalid alpha: {alpha}")
        if momentum < 0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.centered = centered

        self.square_avg = [np.zeros_like(p) for p in params]
        self.mom_buf = [np.zeros_like(p) for p in params] if momentum > 0 else None
        self.grad_avg = [np.zeros_like(p) for p in params] if centered else None

    def _update(self, grads: List[np.ndarray]) -> None:
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if self.weight_decay > 0:
                g = g + self.weight_decay * p

            self.square_avg[i] = self.alpha * self.square_avg[i] + (1 - self.alpha) * g * g

            if self.centered:
                self.grad_avg[i] = self.alpha * self.grad_avg[i] + (1 - self.alpha) * g
                avg = self.square_avg[i] - self.grad_avg[i] ** 2
                avg = np.sqrt(np.maximum(avg, self.eps))
            else:
                avg = np.sqrt(self.square_avg[i] + self.eps)

            grad_mag = np.abs(g).mean()
            if grad_mag < 0.01:
                tunnel = self.rng.randn(*p.shape).astype(np.float32) * self.quantum_strength
                g = g * (1 - self.quantum_strength) + tunnel * 0.1

            if self.momentum > 0:
                self.mom_buf[i] = self.momentum * self.mom_buf[i] + g / avg
                p -= self.lr * self.mom_buf[i]
            else:
                p -= self.lr * g / avg

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["square_avg"] = [s.copy() for s in self.square_avg]
        state["lr"] = self.lr
        if self.momentum > 0 and self.mom_buf is not None:
            state["mom_buf"] = [m.copy() for m in self.mom_buf]
        if self.centered and self.grad_avg is not None:
            state["grad_avg"] = [g.copy() for g in self.grad_avg]
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.square_avg = state.get("square_avg", [np.zeros_like(p) for p in self.params])
        if "mom_buf" in state:
            self.mom_buf = state["mom_buf"]
        if "grad_avg" in state:
            self.grad_avg = state["grad_avg"]
        if "lr" in state:
            self.lr = state["lr"]

    def __repr__(self) -> str:
        return (
            f"QuantumRMSprop(lr={self.lr}, quantum_strength={self.quantum_strength}, "
            f"alpha={self.alpha}, momentum={self.momentum}, weight_decay={self.weight_decay})"
        )
