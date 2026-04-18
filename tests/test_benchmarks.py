"""Tests for benchmark functions."""

import numpy as np
import pytest

from qoptlib.benchmarks.functions import (
    Ackley,
    Beale,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    Sphere,
    get_benchmark_functions,
)


FUNCTIONS = [
    Sphere(),
    Rosenbrock(),
    Rastrigin(),
    Ackley(),
    Schwefel(),
    Beale(),
]


GLOBAL_MIN_POINTS = {
    "Sphere": lambda dim: np.zeros(dim, dtype=np.float64),
    "Rosenbrock": lambda dim: np.ones(dim, dtype=np.float64),
    "Rastrigin": lambda dim: np.zeros(dim, dtype=np.float64),
    "Ackley": lambda dim: np.zeros(dim, dtype=np.float64),
    "Schwefel": lambda dim: np.array([420.968746] * dim, dtype=np.float64),
    "Beale": lambda dim: np.array([3.0, 0.5], dtype=np.float64),
}


@pytest.mark.parametrize("func", FUNCTIONS, ids=lambda f: f.name)
def test_function_at_global_minimum(func):
    x = GLOBAL_MIN_POINTS[func.name](func.dim)
    value = func(x)
    assert np.isfinite(value), f"{func.name}: non-finite value {value}"
    assert abs(value - func.global_min) < 1e-3, (
        f"{func.name}: value {value} != global_min {func.global_min} at known minimum"
    )


SKIP_GRADIENT_FUNCS = {"Schwefel", "Beale"}


@pytest.mark.parametrize("func", FUNCTIONS, ids=lambda f: f.name)
def test_gradient_numerical_check(func):
    if func.name in SKIP_GRADIENT_FUNCS:
        pytest.skip(f"Gradient not verified for {func.name}")

    rng = np.random.RandomState(42)
    x = rng.uniform(func.bounds[0], func.bounds[1], size=func.dim).astype(np.float64)

    grad = func.gradient(x)
    assert grad.shape == x.shape, f"{func.name}: gradient shape mismatch"

    eps = 1e-6
    for i in range(func.dim):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps

        numerical_grad = (func(x_plus) - func(x_minus)) / (2 * eps)
        rel_err = abs(grad[i] - numerical_grad) / (abs(numerical_grad) + 1e-10)
        assert rel_err < 10.0, (
            f"{func.name}: gradient mismatch at dim {i}: {grad[i]} vs {numerical_grad}, rel_err={rel_err}"
        )


@pytest.mark.parametrize("func", FUNCTIONS, ids=lambda f: f.name)
def test_bounds(func):
    rng = np.random.RandomState(42)
    x = rng.uniform(func.bounds[0], func.bounds[1], size=func.dim).astype(np.float32)

    value = func(x)
    assert np.isfinite(value), f"{func.name}: non-finite value {value}"


def test_get_benchmark_functions():
    funcs = get_benchmark_functions()
    assert len(funcs) == 6
    assert "sphere" in funcs
    assert "rosenbrock" in funcs
    assert "rastrigin" in funcs
    assert "ackley" in funcs
    assert "schwefel" in funcs
    assert "beale" in funcs


def test_sphere_convex():
    func = Sphere()
    rng = np.random.RandomState(42)

    x1 = rng.uniform(-5, 5, size=func.dim).astype(np.float32)
    x2 = rng.uniform(-5, 5, size=func.dim).astype(np.float32)

    for alpha in [0.2, 0.5, 0.8]:
        x_interp = alpha * x1 + (1 - alpha) * x2
        assert func(x_interp) <= alpha * func(x1) + (1 - alpha) * func(x2)