"""Tests for NumPy Quantum Optimizers."""

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


def test_sgd_basic():
    params = create_params()
    optimizer = QuantumSGD(params, lr=0.01, quantum_strength=0.2)

    initial_norm = np.linalg.norm(params[0])

    for _ in range(10):
        optimizer.step(grad_fn_factory(params))

    assert np.linalg.norm(params[0]) != initial_norm


def test_sgd_convergence():
    params = [np.array([5.0, -5.0], dtype=np.float32)]
    optimizer = QuantumSGD(params, lr=0.1, momentum=0.9)

    def grad_fn():
        return [params[0] * 2]

    initial = params[0].copy()
    for _ in range(50):
        optimizer.step(grad_fn)

    assert np.linalg.norm(params[0]) < np.linalg.norm(initial)


def test_sgd_state_dict():
    params = create_params()
    optimizer = QuantumSGD(params, lr=0.01)

    for _ in range(5):
        optimizer.step(grad_fn_factory(params))

    state = optimizer.state_dict()
    assert "step_count" in state
    assert "velocity" in state
    assert state["step_count"] == 5


def test_sgd_load_state_dict():
    params = create_params()
    optimizer = QuantumSGD(params, lr=0.01, momentum=0.9)

    for _ in range(5):
        optimizer.step(grad_fn_factory(params))

    state = optimizer.state_dict()

    params2 = create_params()
    optimizer2 = QuantumSGD(params2, lr=0.01, momentum=0.9)
    optimizer2.load_state_dict(state)

    assert optimizer2.step_count == 5


def test_sgd_weight_decay():
    params = [np.array([1.0, 1.0], dtype=np.float32)]
    optimizer = QuantumSGD(params, lr=0.01, weight_decay=0.1)

    def grad_fn():
        return [np.array([0.0, 0.0], dtype=np.float32)]

    initial = params[0].copy()
    optimizer.step(grad_fn)

    assert np.linalg.norm(params[0]) < np.linalg.norm(initial)


def test_adam_basic():
    params = create_params()
    optimizer = QuantumAdam(params, lr=0.01, quantum_strength=0.2)

    for _ in range(10):
        optimizer.step(grad_fn_factory(params))

    assert optimizer.step_count == 10


def test_adam_amsgrad():
    params = create_params()
    optimizer = QuantumAdam(params, lr=0.01, amsgrad=True)

    for _ in range(10):
        optimizer.step(grad_fn_factory(params))

    assert optimizer.max_v is not None


def test_adam_state_dict():
    params = create_params()
    optimizer = QuantumAdam(params, lr=0.01)

    for _ in range(5):
        optimizer.step(grad_fn_factory(params))

    state = optimizer.state_dict()
    assert "step_count" in state
    assert "m" in state
    assert "v" in state


def test_adam_get_set_lr():
    params = create_params()
    optimizer = QuantumAdam(params, lr=0.01)

    assert optimizer.get_lr() == 0.01
    optimizer.set_lr(0.05)
    assert optimizer.get_lr() == 0.05


def test_rmsprop_basic():
    params = create_params()
    optimizer = QuantumRMSprop(params, lr=0.01, quantum_strength=0.1)

    for _ in range(10):
        optimizer.step(grad_fn_factory(params))

    assert optimizer.step_count == 10


def test_rmsprop_centered():
    params = create_params()
    optimizer = QuantumRMSprop(params, lr=0.01, centered=True)

    for _ in range(10):
        optimizer.step(grad_fn_factory(params))

    assert optimizer.grad_avg is not None


def test_tunneling_basic():
    params = create_params()
    optimizer = QuantumTunneling(params, lr=0.01, quantum_strength=0.2)

    for _ in range(10):
        optimizer.step(grad_fn_factory(params))

    assert optimizer.step_count == 10


def test_tunneling_state_dict():
    params = create_params()
    optimizer = QuantumTunneling(params, lr=0.01)

    for _ in range(5):
        optimizer.step(grad_fn_factory(params))

    state = optimizer.state_dict()
    assert "step_count" in state
    assert "m" in state
    assert "v" in state
    assert "tunnel" in state


def test_invalid_params():
    params = create_params()

    with pytest.raises(ValueError):
        QuantumSGD(params, lr=-0.01)

    with pytest.raises(ValueError):
        QuantumSGD(params, quantum_strength=1.5)

    with pytest.raises(ValueError):
        QuantumAdam(params, lr=-0.01)

    with pytest.raises(ValueError):
        QuantumAdam(params, betas=(1.5, 0.999))


def test_repr():
    params = create_params()

    sgd = QuantumSGD(params, lr=0.01, momentum=0.9)
    assert "QuantumSGD" in repr(sgd)

    adam = QuantumAdam(params, lr=0.001)
    assert "QuantumAdam" in repr(adam)
