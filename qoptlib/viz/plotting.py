"""Visualization utilities for optimizer analysis."""

from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from qoptlib.benchmarks.functions import BenchmarkFunction
from qoptlib.benchmarks.runner import BenchmarkResult


def require_matplotlib(func):
    """Decorator to check matplotlib availability."""

    def wrapper(*args, **kwargs):
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for visualization. Install with: pip install matplotlib"
            )
        return func(*args, **kwargs)

    return wrapper


@require_matplotlib
def plot_convergence(
    results: List[BenchmarkResult],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> "matplotlib.figure.Figure":
    """Plot convergence curves for multiple optimizers.

    Args:
        results: List of BenchmarkResult objects.
        save_path: Path to save the figure.
        title: Plot title.
        figsize: Figure size.

    Returns:
        The matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for result in results:
        if result.loss_history:
            ax.plot(
                result.loss_history,
                label=f"{result.optimizer_name} (best={result.best_value:.2e})",
                linewidth=2,
            )

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(title or "Optimizer Convergence Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


@require_matplotlib
def plot_landscape_2d(
    func: BenchmarkFunction,
    resolution: int = 100,
    save_path: Optional[str] = None,
    trajectory: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> "matplotlib.figure.Figure":
    """Plot 2D contour of the loss landscape.

    Args:
        func: A 2D benchmark function.
        resolution: Grid resolution.
        save_path: Path to save the figure.
        trajectory: Array of shape (N, 2) showing optimizer path.
        figsize: Figure size.

    Returns:
        The matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    if func.dim != 2:
        raise ValueError(f"Function must be 2D, got {func.dim}D")

    x = np.linspace(func.bounds[0], func.bounds[1], resolution)
    y = np.linspace(func.bounds[0], func.bounds[1], resolution)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    Z = np.log1p(Z)

    fig, ax = plt.subplots(figsize=figsize)
    contour = ax.contourf(X, Y, Z, levels=50, cmap="viridis")
    ax.contour(X, Y, Z, levels=20, colors="white", alpha=0.3, linewidths=0.5)

    if trajectory is not None:
        ax.plot(trajectory[:, 0], trajectory[:, 1], "r-", linewidth=1, alpha=0.7, label="Path")
        ax.scatter(
            trajectory[0, 0], trajectory[0, 1],
            c="green", s=100, marker="o", label="Start", zorder=5,
        )
        ax.scatter(
            trajectory[-1, 0], trajectory[-1, 1],
            c="red", s=100, marker="x", label="End", zorder=5,
        )

    ax.set_xlabel("x1", fontsize=12)
    ax.set_ylabel("x2", fontsize=12)
    ax.set_title(f"{func.name} Loss Landscape", fontsize=14)
    ax.legend(fontsize=10)
    fig.colorbar(contour, ax=ax, label="log(1 + f(x))")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


@require_matplotlib
def plot_comparison_bar(
    results: List[BenchmarkResult],
    metric: str = "best_value",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> "matplotlib.figure.Figure":
    """Plot bar chart comparing optimizers on a metric.

    Args:
        results: List of BenchmarkResult objects.
        metric: Metric to compare ('best_value', 'final_value', 'elapsed_time', 'steps').
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        The matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    metric_labels = {
        "best_value": "Best Loss",
        "final_value": "Final Loss",
        "elapsed_time": "Time (s)",
        "steps": "Steps",
    }

    if metric not in metric_labels:
        raise ValueError(f"Unknown metric: {metric}. Choose from {list(metric_labels.keys())}")

    optimizers = list(set(r.optimizer_name for r in results))
    functions = list(set(r.function_name for r in results))

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(functions))
    width = 0.8 / len(optimizers)

    for i, opt in enumerate(optimizers):
        values = []
        for func_name in functions:
            matching = [r for r in results if r.optimizer_name == opt and r.function_name == func_name]
            if matching:
                values.append(getattr(matching[0], metric))
            else:
                values.append(float("nan"))

        ax.bar(x + i * width, values, width, label=opt, alpha=0.8)

    ax.set_xlabel("Function", fontsize=12)
    ax.set_ylabel(metric_labels[metric], fontsize=12)
    ax.set_title(f"Optimizer Comparison: {metric_labels[metric]}", fontsize=14)
    ax.set_xticks(x + width * (len(optimizers) - 1) / 2)
    ax.set_xticklabels(functions, rotation=45, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
