"""Example: Benchmarking quantum optimizers on classical test functions."""

import numpy as np

from qoptlib import QuantumAdam, QuantumRMSprop, QuantumSGD, QuantumTunneling
from qoptlib.benchmarks.functions import Ackley, Rastrigin, Rosenbrock, Sphere, get_benchmark_functions
from qoptlib.benchmarks.runner import BenchmarkRunner


def main():
    print("=" * 60)
    print("Quantum Optimizers - Benchmark Suite")
    print("=" * 60)

    runner = BenchmarkRunner(max_steps=500, tol=1e-5, seed=42)

    optimizers = [QuantumSGD, QuantumAdam, QuantumRMSprop, QuantumTunneling]
    optimizer_params = {
        "QuantumSGD": {"lr": 0.01, "quantum_strength": 0.1, "momentum": 0.9},
        "QuantumAdam": {"lr": 0.01, "quantum_strength": 0.2},
        "QuantumRMSprop": {"lr": 0.01, "quantum_strength": 0.15},
        "QuantumTunneling": {"lr": 0.01, "quantum_strength": 0.3},
    }

    functions = [Sphere(), Rosenbrock(), Rastrigin(), Ackley()]

    all_results = []

    for func in functions:
        print(f"\n--- {func.name} (dim={func.dim}, global_min={func.global_min}) ---")

        for opt_class in optimizers:
            params = optimizer_params[opt_class.__name__]

            if func.dim != params.get("dim", func.dim):
                pass

            result = runner.run(opt_class, func, **params)
            all_results.append(result)

            status = "CONVERGED" if result.converged else "DID NOT CONVERGE"
            print(
                f"  {opt_class.__name__:<25} best={result.best_value:>12.6e} "
                f"final={result.final_value:>12.6e} steps={result.steps:>4} [{status}]"
            )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(runner.summary())

    try:
        from qoptlib.viz.plotting import plot_comparison_bar, plot_convergence

        for func in functions:
            func_results = [r for r in all_results if r.function_name == func.name]
            if func_results:
                plot_convergence(
                    func_results,
                    save_path=f"convergence_{func.name.lower()}.png",
                    title=f"Convergence on {func.name}",
                )

        plot_comparison_bar(
            all_results,
            metric="best_value",
            save_path="comparison_best.png",
        )
        print("\nPlots saved to convergence_*.png and comparison_best.png")
    except ImportError:
        print("\nInstall matplotlib to generate plots: pip install matplotlib")


if __name__ == "__main__":
    main()
