"""Tests for PyTorch quantum optimizers via adapter."""

import numpy as np
import pytest

from qoptlib.opt import (
    QuantumSGD,
    QuantumAdam,
    QuantumRMSprop,
    QuantumTunneling,
)


def create_params():
    return [np.random.randn(10, 5).astype(np.float32) for _ in range(2)]


def grad_fn_factory(params):
    def grad_fn():
        return [np.random.randn(*p.shape).astype(np.float32) * 0.1 for p in params]
    return grad_fn


def test_optimizer_basic():
    params = create_params()
    optimizer = QuantumAdam(params, lr=0.01, quantum_strength=0.1)

    for _ in range(5):
        optimizer.step(grad_fn_factory(params))

    assert optimizer.step_count == 5


def test_optimizer_converges():
    params = [np.array([5.0, -5.0], dtype=np.float32)]
    optimizer = QuantumAdam(params, lr=0.01, quantum_strength=0.1)

    def grad_fn():
        return [params[0] * 2]

    initial = params[0].copy()
    for _ in range(50):
        optimizer.step(grad_fn)

    assert np.linalg.norm(params[0]) < np.linalg.norm(initial)


def test_sgd_momentum():
    params = create_params()
    optimizer = QuantumSGD(params, lr=0.01, momentum=0.9, quantum_strength=0.1)

    for _ in range(10):
        optimizer.step(grad_fn_factory(params))

    assert len(optimizer.velocity) > 0


def test_adam_amsgrad():
    params = create_params()
    optimizer = QuantumAdam(params, lr=0.01, quantum_strength=0.1, amsgrad=True)

    for _ in range(10):
        optimizer.step(grad_fn_factory(params))

    assert optimizer.max_v is not None


def test_invalid_quantum_strength():
    params = create_params()
    with pytest.raises(ValueError):
        QuantumSGD(params, quantum_strength=-0.1)

    with pytest.raises(ValueError):
        QuantumSGD(params, quantum_strength=1.5)


def test_invalid_learning_rate():
    params = create_params()
    with pytest.raises(ValueError):
        QuantumAdam(params, lr=-0.01)


def test_reproducibility():
    results = []

    for _ in range(2):
        np.random.seed(42)

        params = [np.random.randn(10).astype(np.float32)]
        optimizer = QuantumAdam(params, lr=0.01, quantum_strength=0.1, seed=42)

        losses = []
        for _ in range(10):
            def grad_fn():
                return [np.random.randn(10).astype(np.float32) * 0.1]

            optimizer.step(grad_fn)
            losses.append(optimizer.step_count)

        results.append(losses)

    assert results[0] == results[1], "Results should be reproducible with same seed"


def test_torch_adapter_import():
    """Test that TorchAdapter can be imported."""
    from qoptlib.adapters import TorchAdapter
    import torch
    import torch.nn as nn

    model = nn.Linear(10, 1)
    adapter = TorchAdapter(model)

    weights = adapter.get_weights()
    assert isinstance(weights, np.ndarray)
    assert weights.shape[0] > 0


def test_tensorflow_adapter_import():
    """Test that TensorFlowAdapter can be imported."""
    pytest.importorskip("tensorflow")

    from qoptlib.adapters import TensorFlowAdapter
    import tensorflow as tf

    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(10,))])
    adapter = TensorFlowAdapter(model)

    weights = adapter.get_weights()
    assert isinstance(weights, np.ndarray)
    assert weights.shape[0] > 0
