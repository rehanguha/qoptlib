"""Quantum-inspired SGD optimizer."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np

from qoptlib.opt.base import BaseOptimizer


class QuantumSGD(BaseOptimizer):
    """Quantum SGD optimizer.

    Args:
        params: List of numpy arrays (parameters).
        lr: Learning rate.
        quantum_strength: Strength of quantum effects (0-1).
        momentum: Momentum factor (0-1).
        weight_decay: L2 weight decay coefficient.
        nesterov: Use Nesterov momentum.
        seed: Random seed.
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 0.01,
        quantum_strength: float = 0.1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(params, lr, quantum_strength, seed)

        if momentum < 0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocity = [np.zeros_like(p) for p in params]

    def _update(self, grads: List[np.ndarray]) -> None:
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if self.weight_decay > 0:
                g = g + self.weight_decay * p

            if self.momentum > 0:
                self.velocity[i] = self.momentum * self.velocity[i] + g
                if self.nesterov:
                    g = g + self.momentum * self.velocity[i]
                else:
                    g = self.velocity[i]

            noise = self.rng.randn(*p.shape).astype(np.float32) * self.quantum_strength
            q_state = g + noise * self.quantum_strength * np.abs(g).mean()
            p -= self.lr * q_state

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["velocity"] = [v.copy() for v in self.velocity]
        state["lr"] = self.lr
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.velocity = state.get("velocity", [np.zeros_like(p) for p in self.params])
        if "lr" in state:
            self.lr = state["lr"]

    def __repr__(self) -> str:
        return (
            f"QuantumSGD(lr={self.lr}, quantum_strength={self.quantum_strength}, "
            f"momentum={self.momentum}, weight_decay={self.weight_decay})"
        )
