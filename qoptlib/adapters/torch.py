"""PyTorch adapter for QOptLib optimizers.

Provides a bridge between PyTorch models and QOptLib optimizers.

Usage:
    >>> from qoptlib.opt import QuantumAdam
    >>> from qoptlib.adapters import TorchAdapter
    >>> 
    >>> model = torch.nn.Linear(10, 1)
    >>> adapter = TorchAdapter(model)
    >>>
    >>> optimizer = QuantumAdam(lr=0.001, quantum_strength=0.2)
    >>> best_weights, best_loss = adapter.optimize(optimizer, loss_fn, dataset)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    raise ImportError("PyTorch is required. Install with: pip install torch")

from qoptlib.opt.base import BaseOptimizer


class TorchAdapter:
    """Adapter for using quantopt optimizers with PyTorch models.

    Converts PyTorch model parameters to/from NumPy arrays,
    enabling gradient-free optimization of neural networks.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to optimize.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.param_shapes: Dict[str, Tuple[int, ...]] = {}
        self.param_names: List[str] = []

        for name, param in model.named_parameters():
            self.param_shapes[name] = param.shape
            self.param_names.append(name)

    def get_weights(self) -> np.ndarray:
        """Get model weights as a flat numpy array."""
        weights = []
        for name in self.param_names:
            w = self.model.state_dict()[name]
            weights.append(w.cpu().numpy().flatten())
        return np.concatenate(weights)

    def set_weights(self, flat_weights: np.ndarray) -> None:
        """Set model parameters from a flat array."""
        state = self.model.state_dict()
        idx = 0
        for name in self.param_names:
            shape = self.param_shapes[name]
            size = int(np.prod(shape))
            param = flat_weights[idx : idx + size].reshape(shape)
            state[name] = torch.from_numpy(param)
            idx += size
        self.model.load_state_dict(state)

    def get_bounds(
        self,
        default_low: float = -5.0,
        default_high: float = 5.0,
    ) -> List[Tuple[float, float]]:
        """Get bounds for all model parameters."""
        bounds = []
        for shape in self.param_shapes.values():
            n_elements = int(np.prod(shape))
            bounds.extend([(default_low, default_high)] * n_elements)
        return bounds

    def get_weights_list(self) -> List[np.ndarray]:
        """Get model weights as list of numpy arrays."""
        return [
            self.model.state_dict()[name].cpu().numpy()
            for name in self.param_names
        ]

    def evaluate(
        self,
        weights: np.ndarray,
        loss_fn: Callable,
        dataset: Optional[Any] = None,
    ) -> float:
        """Evaluate the model with given weights."""
        self.set_weights(weights)

        if dataset is not None:
            total_loss = 0.0
            n_batches = 0
            for batch in dataset:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs = batch
                    targets = None

                if targets is not None:
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs)

                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
                total_loss += loss
                n_batches += 1

            return total_loss / n_batches if n_batches > 0 else float("inf")
        else:
            raise ValueError("dataset is required for evaluation")

    def optimize(
        self,
        optimizer: BaseOptimizer,
        loss_fn: Callable,
        dataset: Any,
        iterations: int = 100,
        verbose: bool = False,
        callback: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, float]:
        """Run optimizer to find best model weights.

        Parameters
        ----------
        optimizer : BaseOptimizer
            A quantopt optimizer (QuantumAdam, QuantumSGD, etc.).
        loss_fn : callable
            Loss function that takes (outputs, targets).
        dataset : DataLoader or iterable
            Training data.
        iterations : int
            Number of optimization iterations.
        verbose : bool
            Print progress.
        callback : callable, optional
            Callback function called with (iteration, best_weights, best_loss).

        Returns
        -------
        best_weights : ndarray
            Best parameter vector found.
        best_loss : float
            Best loss achieved.
        """
        param_shapes = list(self.param_shapes.values())
        n_params = sum(int(np.prod(s)) for s in param_shapes)

        def objective_fn(flat_weights: np.ndarray) -> float:
            return self.evaluate(flat_weights, loss_fn, dataset)

        best_loss = float("inf")
        best_weights = self.get_weights()

        for iteration in range(iterations):
            current_weights = self.get_weights()

            def grad_fn():
                return self._compute_gradients(current_weights, loss_fn, dataset)

            optimizer.step(grad_fn)

            current_weights = self.get_weights()
            current_loss = objective_fn(current_weights)

            if current_loss < best_loss:
                best_loss = current_loss
                best_weights = current_weights.copy()

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: loss = {current_loss:.6f}")

            if callback:
                callback(iteration, best_weights, best_loss)

        self.set_weights(best_weights)
        return best_weights, best_loss

    def _compute_gradients(
        self,
        weights: np.ndarray,
        loss_fn: Callable,
        dataset: Any,
    ) -> List[np.ndarray]:
        """Compute gradients numerically."""
        eps = 1e-5
        grads = []
        param_shapes = list(self.param_shapes.values())

        self.set_weights(weights)
        base_loss = self.evaluate(weights, loss_fn, dataset)

        idx = 0
        for shape in param_shapes:
            grad = np.zeros(shape, dtype=np.float32)
            for i in range(int(np.prod(shape))):
                idx_tuple = np.unravel_index(i, shape)

                weights_plus = weights.copy()
                weights_plus[idx + i] += eps
                loss_plus = self.evaluate(weights_plus, loss_fn, dataset)

                weights_minus = weights.copy()
                weights_minus[idx + i] -= eps
                loss_minus = self.evaluate(weights_minus, loss_fn, dataset)

                grad[idx_tuple] = (loss_plus - loss_minus) / (2 * eps)

            grads.append(grad.flatten())
            idx += int(np.prod(shape))

        return grads

    def get_weight_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about current model weights."""
        stats = {}
        for name, param in self.model.named_parameters():
            w = param.detach().cpu().numpy()
            stats[name] = {
                "mean": float(np.mean(w)),
                "std": float(np.std(w)),
                "min": float(np.min(w)),
                "max": float(np.max(w)),
                "norm": float(np.linalg.norm(w)),
            }
        return stats
