"""QOptLib adapters for framework integration."""

__all__ = ["TorchAdapter", "TensorFlowAdapter"]


def __getattr__(name):
    if name == "TorchAdapter":
        from qoptlib.adapters.torch import TorchAdapter
        return TorchAdapter
    elif name == "TensorFlowAdapter":
        from qoptlib.adapters.tensorflow import TensorFlowAdapter
        return TensorFlowAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
