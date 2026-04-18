"""Benchmark test functions for optimizer evaluation."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import torch


class BenchmarkFunction(ABC):
    """Base class for benchmark functions."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    @property
    @abstractmethod
    def bounds(self) -> Tuple[float, float]:
        pass

    @property
    @abstractmethod
    def global_min(self) -> float:
        pass

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        pass

    def to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).float().requires_grad_(True)


class Sphere(BenchmarkFunction):
    """Sphere function: f(x) = sum(x_i^2)

    Simple convex function, good for basic convergence testing.
    """

    @property
    def name(self) -> str:
        return "Sphere"

    @property
    def dim(self) -> int:
        return 10

    @property
    def bounds(self) -> Tuple[float, float]:
        return (-5.12, 5.12)

    @property
    def global_min(self) -> float:
        return 0.0

    def __call__(self, x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2 * x


class Rosenbrock(BenchmarkFunction):
    """Rosenbrock function (banana function).

    Non-convex function with a narrow curved valley. Hard to converge
    to the global minimum.
    """

    def __init__(self, dim: int = 2):
        self._dim = dim

    @property
    def name(self) -> str:
        return "Rosenbrock"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bounds(self) -> Tuple[float, float]:
        return (-5.0, 10.0)

    @property
    def global_min(self) -> float:
        return 0.0

    def __call__(self, x: np.ndarray) -> float:
        val = 0.0
        for i in range(len(x) - 1):
            val += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        return float(val)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(x)
        for i in range(len(x) - 1):
            grad[i] += -400 * x[i] * (x[i + 1] - x[i] ** 2) - 2 * (1 - x[i])
            grad[i + 1] += 200 * (x[i + 1] - x[i] ** 2)
        return grad


class Rastrigin(BenchmarkFunction):
    """Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))

    Highly multimodal with regularly spaced local minima. Very challenging
    for optimizers to find the global minimum.
    """

    def __init__(self, dim: int = 10):
        self._dim = dim

    @property
    def name(self) -> str:
        return "Rastrigin"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bounds(self) -> Tuple[float, float]:
        return (-5.12, 5.12)

    @property
    def global_min(self) -> float:
        return 0.0

    def __call__(self, x: np.ndarray) -> float:
        n = len(x)
        return float(10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)


class Ackley(BenchmarkFunction):
    """Ackley function: highly multimodal with flat outer region.

    Challenging due to the large flat area that provides little gradient
    information to guide the search.
    """

    def __init__(self, dim: int = 10):
        self._dim = dim

    @property
    def name(self) -> str:
        return "Ackley"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bounds(self) -> Tuple[float, float]:
        return (-32.768, 32.768)

    @property
    def global_min(self) -> float:
        return 0.0

    def __call__(self, x: np.ndarray) -> float:
        n = len(x)
        a = 20
        b = 0.2
        c = 2 * np.pi
        sum_sq = np.sum(x ** 2)
        sum_cos = np.sum(np.cos(c * x))
        return float(-a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.exp(1))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        a = 20
        b = 0.2
        c = 2 * np.pi
        sum_sq = np.sum(x ** 2)
        sum_cos = np.sum(np.cos(c * x))

        term1 = a * b * np.exp(-b * np.sqrt(sum_sq / n)) / (2 * n * np.sqrt(sum_sq / n) + 1e-12)
        term2 = c * np.sin(c * x) * np.exp(sum_cos / n) / n

        return 2 * term1 * x + term2


class Schwefel(BenchmarkFunction):
    """Schwefel function: multimodal with many local minima.

    f(x) = 418.9829 * n - sum(x * sin(sqrt(abs(x))))
    Global minimum at x_i = 420.968746 for all i (approx 420.9687461575...)
    """

    def __init__(self, dim: int = 10):
        self._dim = dim

    @property
    def name(self) -> str:
        return "Schwefel"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bounds(self) -> Tuple[float, float]:
        return (-500, 500)

    @property
    def global_min(self) -> float:
        return 0.0  # at x = 420.968746...

    def __call__(self, x: np.ndarray) -> float:
        return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        sqrt_abs_x = np.sqrt(np.abs(x))
        sqrt_abs_x = np.where(sqrt_abs_x < 1e-10, 1e-10, sqrt_abs_x)
        grad = -np.sin(sqrt_abs_x) - 0.5 * sqrt_abs_x * np.cos(sqrt_abs_x) * np.sign(x)
        grad = np.where(np.abs(x) < 1e-10, np.zeros_like(x), grad)
        return grad


class Beale(BenchmarkFunction):
    """Beale function: multimodal with sharp peaks.

    f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + x*y^2)^2 + (2.625 - x + x*y^3)^2
    Global minimum at (3, 0.5) with f = 0.
    """

    @property
    def name(self) -> str:
        return "Beale"

    @property
    def dim(self) -> int:
        return 2

    @property
    def bounds(self) -> Tuple[float, float]:
        return (-4.5, 4.5)

    @property
    def global_min(self) -> float:
        return 0.0  # at (3, 0.5)

    def __call__(self, x: np.ndarray) -> float:
        x1, x2 = float(x[0]), float(x[1])
        return (
            (1.5 - x1 + x1 * x2) ** 2
            + (2.25 - x1 + x1 * x2 ** 2) ** 2
            + (2.625 - x1 + x1 * x2 ** 3) ** 2
        )

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = float(x[0]), float(x[1])
        t1 = 1.5 - x1 + x1 * x2
        t2 = 2.25 - x1 + x1 * x2 ** 2
        t3 = 2.625 - x1 + x1 * x2 ** 3

        dx1 = 2 * t1 * (-1 + x2) + 2 * t2 * (-1 + x2 ** 2) + 2 * t3 * (-1 + x2 ** 3)
        dx2 = 2 * t1 * x1 + 2 * t2 * 2 * x1 * x2 + 2 * t3 * 3 * x1 * x2 ** 2

        return np.array([dx1, dx2], dtype=np.float64)


def get_benchmark_functions() -> Dict[str, BenchmarkFunction]:
    """Return a dictionary of all available benchmark functions."""
    return {
        "sphere": Sphere(),
        "rosenbrock": Rosenbrock(),
        "rastrigin": Rastrigin(),
        "ackley": Ackley(),
        "schwefel": Schwefel(),
        "beale": Beale(),
    }
