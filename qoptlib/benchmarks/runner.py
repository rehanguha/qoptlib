"""Benchmark runner for comparing optimizers on test functions."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch

from qoptlib.benchmarks.functions import BenchmarkFunction
from qoptlib.opt.base import BaseOptimizer


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    optimizer_name: str
    function_name: str
    final_value: float
    best_value: float
    steps: int
    converged: bool
    elapsed_time: float
    loss_history: List[float] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Runs optimizers on benchmark functions and collects results."""

    def __init__(self, max_steps: int = 1000, tol: float = 1e-6, seed: int = 42):
        self.max_steps = max_steps
        self.tol = tol
        self.seed = seed
        self.results: List[BenchmarkResult] = []

    def run(
        self,
        optimizer_class: Type[QuantumOptimizerBase],
        func: BenchmarkFunction,
        **optimizer_kwargs,
    ) -> BenchmarkResult:
        """Run a single optimizer on a benchmark function.

        Args:
            optimizer_class: The optimizer class to test.
            func: The benchmark function to optimize.
            **optimizer_kwargs: Additional arguments for the optimizer.

        Returns:
            BenchmarkResult with metrics and history.
        """
        rng = np.random.RandomState(self.seed)
        x = rng.uniform(func.bounds[0], func.bounds[1], size=func.dim).astype(np.float32)

        x_tensor = torch.from_numpy(x).requires_grad_(True)
        params = [x_tensor]

        optimizer = optimizer_class(params, **optimizer_kwargs)

        loss_history = []
        best_value = float("inf")
        start_time = time.time()
        converged = False

        for step in range(self.max_steps):
            optimizer.zero_grad()
            loss = func(x_tensor.detach().numpy())
            loss_tensor = torch.tensor(loss, dtype=torch.float32)

            loss_history.append(loss)

            if loss < best_value:
                best_value = loss

            if abs(loss - func.global_min) < self.tol:
                converged = True
                break

            grad_np = func.gradient(x_tensor.detach().numpy())
            x_tensor.grad = torch.from_numpy(grad_np).float()

            optimizer.step()

        elapsed = time.time() - start_time

        result = BenchmarkResult(
            optimizer_name=optimizer_class.__name__,
            function_name=func.name,
            final_value=loss_history[-1] if loss_history else float("inf"),
            best_value=best_value,
            steps=len(loss_history),
            converged=converged,
            elapsed_time=elapsed,
            loss_history=loss_history,
            params=optimizer_kwargs,
        )

        self.results.append(result)
        return result

    def compare(
        self,
        optimizers: List[Type[QuantumOptimizerBase]],
        func: BenchmarkFunction,
        **optimizer_kwargs,
    ) -> List[BenchmarkResult]:
        """Compare multiple optimizers on the same function.

        Args:
            optimizers: List of optimizer classes to compare.
            func: The benchmark function.
            **optimizer_kwargs: Arguments passed to all optimizers.

        Returns:
            List of BenchmarkResults, one per optimizer.
        """
        results = []
        for opt_class in optimizers:
            result = self.run(opt_class, func, **optimizer_kwargs)
            results.append(result)
        return results

    def compare_all(
        self,
        optimizers: List[Type[QuantumOptimizerBase]],
        functions: List[BenchmarkFunction],
        **optimizer_kwargs,
    ) -> List[BenchmarkResult]:
        """Run all optimizers on all functions.

        Args:
            optimizers: List of optimizer classes.
            functions: List of benchmark functions.
            **optimizer_kwargs: Arguments passed to all optimizers.

        Returns:
            List of all BenchmarkResults.
        """
        all_results = []
        for func in functions:
            results = self.compare(optimizers, func, **optimizer_kwargs)
            all_results.extend(results)
        return all_results

    def summary(self) -> str:
        """Generate a text summary of all benchmark results."""
        if not self.results:
            return "No results yet."

        lines = []
        lines.append(f"{'Optimizer':<25} {'Function':<15} {'Best':>12} {'Final':>12} {'Steps':>6} {'Time':>8} {'Conv':>5}")
        lines.append("-" * 85)

        for r in self.results:
            lines.append(
                f"{r.optimizer_name:<25} {r.function_name:<15} "
                f"{r.best_value:>12.6e} {r.final_value:>12.6e} "
                f"{r.steps:>6} {r.elapsed_time:>8.3f} {'Yes' if r.converged else 'No':>5}"
            )

        return "\n".join(lines)
