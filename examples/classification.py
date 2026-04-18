"""Example: Using quantum optimizers with PyTorch for a simple classification task."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from qoptlib import QuantumAdam, QuantumSGD, QuantumTunneling


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def generate_synthetic_data(n_samples=1000, input_dim=20, num_classes=4):
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, num_classes, (n_samples,))
    return X, y


def train_model(optimizer_class, optimizer_kwargs, epochs=50):
    torch.manual_seed(42)

    model = SimpleClassifier()
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss()

    X, y = generate_synthetic_data()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  [{optimizer_class.__name__}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return losses


def main():
    print("=" * 60)
    print("Quantum Optimizers - PyTorch Classification Example")
    print("=" * 60)

    configs = [
        (QuantumSGD, {"lr": 0.01, "quantum_strength": 0.1, "momentum": 0.9}),
        (QuantumAdam, {"lr": 0.001, "quantum_strength": 0.2}),
        (QuantumTunneling, {"lr": 0.001, "quantum_strength": 0.3}),
    ]

    all_losses = {}
    for opt_class, kwargs in configs:
        print(f"\nTraining with {opt_class.__name__}...")
        losses = train_model(opt_class, kwargs)
        all_losses[opt_class.__name__] = losses

    print("\n" + "=" * 60)
    print("Final Loss Comparison:")
    print("-" * 60)
    for name, losses in all_losses.items():
        print(f"  {name:<25} {losses[-1]:.6f}")
    print("=" * 60)

    try:
from qoptlib.viz.plotting import plot_convergence
from qoptlib.benchmarks.runner import BenchmarkResult

        results = [
            BenchmarkResult(
                optimizer_name=name,
                function_name="Classification",
                final_value=losses[-1],
                best_value=min(losses),
                steps=len(losses),
                converged=False,
                elapsed_time=0,
                loss_history=losses,
            )
            for name, losses in all_losses.items()
        ]

        fig = plot_convergence(results, save_path="convergence_classification.png")
        print("\nConvergence plot saved to convergence_classification.png")
    except ImportError:
        print("\nInstall matplotlib to generate plots: pip install matplotlib")


if __name__ == "__main__":
    main()
