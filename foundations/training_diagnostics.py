import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []

        with torch.no_grad():
            current = x
            for layer in model:
                current = layer(current)
                if isinstance(layer, nn.Linear):
                    # current shape: (batch_size, features)
                    mean = current.mean().item()
                    std = current.std().item()

                    # A neuron is dead if its output <= 0
                    # for every sample in batch.
                    dead = (current <= 0).all(dim=0)
                    dead_fraction = dead.float().mean().item()
                    stats.append({
                        "mean": round(mean, 4),
                        "std": round(std, 4),
                        "dead_fraction": round(dead_fraction, 4)
                    })
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.

        model.zero_grad()

        # Forward pass
        output = model(x)

        # MSE loss
        loss = nn.MSELoss()(output, y)

        # Backward pass
        loss.backward()

        stats = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                stats.append({
                    "mean": round(grad.mean().item(), 4),
                    "std": round(grad.std().item(), 4),
                    "norm": round(torch.norm(grad).item(), 4)
                })
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)

        # Dead neurons
        if any(stat["dead_fraction"] > 0.5 for stat in activation_stats):
            return "dead_neurons"

        # Exploding gradients
        if any(stat["norm"] > 1000 for stat in gradient_stats):
            return "exploding_gradients"

        # Vanishing gradients in last layer
        if gradient_stats and gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # Check activation standard deviation
        if any(stat["std"] < 0.1 for stat in activation_stats):
            return "vanishing_gradients"

        if any(stat["std"] > 10.0 for stat in activation_stats):
            return "exploding_gradients"

        # Everything looks reasonable
        return "healthy"