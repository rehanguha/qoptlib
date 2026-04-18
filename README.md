# QOptLib: Quantum-Inspired Optimizers

A framework of quantum-inspired classical optimizers for machine learning.

## Installation

```bash
pip install qoptlib
# or for development
pip install -e .
```

---

## Quick Start

### NumPy (Core Optimizers)

```python
import numpy as np
from qoptlib import QuantumAdam

params = [np.random.randn(10, 5).astype(np.float32)]
optimizer = QuantumAdam(params, lr=0.001, quantum_strength=0.2)

def get_grads():
    return [np.random.randn(10, 5).astype(np.float32) * 0.1]

for _ in range(100):
    optimizer.step(get_grads)
```

### PyTorch (via Adapter)

```python
from quantopt.opt import QuantumAdam
from quantopt.adapters import TorchAdapter
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

adapter = TorchAdapter(model)
optimizer = QuantumAdam(lr=0.001, quantum_strength=0.2)

# Run optimization
best_weights, best_loss = adapter.optimize(
    optimizer,
    loss_fn=lambda out, tgt: ((out - tgt) ** 2).mean(),
    dataset=torch.utils.data.TensorDataset(
        torch.randn(100, 10),
        torch.randn(100, 1)
    ),
    iterations=50
)
```

### TensorFlow (via Adapter)

```python
from quantopt.quantopt import QuantumAdam
from quantopt.adapters import TensorFlowAdapter
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])

adapter = TensorFlowAdapter(model)
optimizer = QuantumAdam(lr=0.001, quantum_strength=0.2)

best_weights, best_loss = adapter.optimize(
    optimizer,
    loss_fn=lambda y_true, y_pred: tf.keras.losses.mse(y_true, y_pred),
    dataset=tf.data.Dataset.from_tensor_slices((
        tf.random.normal((100, 10)),
        tf.random.normal((100, 1))
    )).batch(32),
    iterations=50
)
```

---

## Structure

```
quantopt/
├── quantopt/              # CORE: NumPy implementations
│   ├── __init__.py       # Exports: QuantumSGD, QuantumAdam, QuantumRMSprop, QuantumTunneling
│   ├── base.py          # BaseOptimizer
│   ├── sgd.py          # QuantumSGD
│   ├── adam.py          # QuantumAdam
│   ├── rmsprop.py       # QuantumRMSprop
│   └── tunneling.py     # QuantumTunneling
│
├── adapters/            # Framework bridges
│   ├── __init__.py     # Lazy imports
│   ├── torch.py        # TorchAdapter
│   └── tensorflow.py   # TensorFlowAdapter
│
├── benchmarks/         # Test functions
├── examples/         # Usage examples
└── tests/           # Test suite
```

---

## Core Optimizers

| Optimizer | Description |
|-----------|-------------|
| `QuantumSGD` | SGD with quantum noise |
| `QuantumAdam` | Adam with quantum phase |
| `QuantumRMSprop` | RMSprop with tunneling |
| `QuantumTunneling` | Escapes local minima |

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lr` | Learning rate | optimizer-specific |
| `quantum_strength` | Quantum effect (0-1) | 0.1 |
| `momentum` | Momentum factor | 0.0 |
| `weight_decay` | L2 regularization | 0.0 |

---

## Adapters

### TorchAdapter

```python
from quantopt.adapters import TorchAdapter
from quantopt.quantopt import QuantumAdam

adapter = TorchAdapter(model)
optimizer = QuantumAdam(lr=0.001)

best_weights, best_loss = adapter.optimize(
    optimizer,
    loss_fn,
    dataset,
    iterations=100,
    verbose=True
)
```

### TensorFlowAdapter

```python
from quantopt.adapters import TensorFlowAdapter
from quantopt.quantopt import QuantumAdam

adapter = TensorFlowAdapter(model)
optimizer = QuantumAdam(lr=0.001)

best_weights, best_loss = adapter.optimize(
    optimizer,
    loss_fn,
    dataset,
    iterations=100
)
```

---

## API

### Core (NumPy)

```python
from quantopt import QuantumAdam
from quantopt.quantopt import QuantumSGD, QuantumRMSprop, QuantumTunneling

# All have:
opt.step(grad_fn)      # Take step
opt.state_dict()       # Get state
opt.load_state_dict(d) # Load state
opt.get_lr()          # Get LR
opt.set_lr(lr)        # Set LR
```

### Adapters

```python
from quantopt.adapters import TorchAdapter, TensorFlowAdapter

adapter = Adapter(model)
adapter.get_weights()          # Get flat weights
adapter.set_weights(w)         # Set weights
adapter.get_bounds()           # Get bounds
adapter.optimize(optimizer, loss_fn, dataset)
```

---

## Tests

```bash
pytest tests/ -v
```
