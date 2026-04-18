"""Base optimizer class for quantopt."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

import numpy as np


class BaseOptimizer(ABC):
    """Abstract base class for quantum-inspired optimizers.

    All optimizers inherit from this class and implement the core update logic.
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 0.001,
        quantum_strength: float = 0.1,
        seed: Optional[int] = None,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= quantum_strength <= 1:
            raise ValueError(f"quantum_strength must be in [0, 1], got {quantum_strength}")

        self.params = list(params)
        self.lr = lr
        self.quantum_strength = quantum_strength
        self.rng = np.random.RandomState(seed)
        self.step_count = 0

    @abstractmethod
    def _update(self, grads: List[np.ndarray]) -> None:
        """Core update logic - implement in subclasses."""
        pass

    def step(self, grad_fn: Optional[Callable[[], List[np.ndarray]]] = None) -> None:
        """Take an optimization step.

        Args:
            grad_fn: Function that returns gradients as list of numpy arrays.
        """
        if grad_fn is None:
            raise ValueError("grad_fn must be provided")

        grads = grad_fn()
        self._update(grads)
        self.step_count += 1

    def zero_grad(self) -> None:
        """No-op for compatibility."""
        pass

    def state_dict(self) -> dict:
        """Get optimizer state as dict."""
        return {
            "step_count": self.step_count,
            "rng_state": self.rng.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Load optimizer state from dict."""
        self.step_count = state.get("step_count", 0)
        if "rng_state" in state:
            self.rng.set_state(state["rng_state"])

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.lr

    def set_lr(self, lr: float) -> None:
        """Set learning rate."""
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        self.lr = lr
