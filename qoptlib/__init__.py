"""QOptLib: Quantum-Inspired Classical Optimizers.

A framework of quantum-inspired classical optimizers for machine learning.

Usage:
    # NumPy (core optimizers)
    from qoptlib import QuantumSGD, QuantumAdam, QuantumRMSprop, QuantumTunneling
    
    # PyTorch via adapter
    from qoptlib.adapters import TorchAdapter
    from qoptlib.opt import QuantumAdam
    
    # TensorFlow via adapter
    from qoptlib.adapters import TensorFlowAdapter
    from qoptlib.opt import QuantumAdam
"""

from qoptlib.opt import (
    QuantumSGD,
    QuantumAdam,
    QuantumRMSprop,
    QuantumTunneling,
)


__version__ = "0.1.0"

__all__ = [
    "__version__",
    "QuantumSGD",
    "QuantumAdam",
    "QuantumRMSprop",
    "QuantumTunneling",
]
