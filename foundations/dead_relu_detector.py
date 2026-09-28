import torch
import torch.nn as nn
from typing import List

class Solution:
    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_fractions = []
        with torch.no_grad():
            current = x
            for layer in model:
                current = layer(current)
                if isinstance(layer, nn.ReLU):
                    # Check each neuron across all samples.
                    # A neuron is dead if its output is 0
                    # for every sample in batch.
                    dead = (current == 0).all(dim=0)
                    dead_fraction = dead.float().mean().item()
                    dead_fractions.append(round(dead_fraction, 4))
        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        if not dead_fractions:
            return "healthy"

        # Severe death anywhere
        if any(fraction > 0.5 for fraction in dead_fractions):
            return "use_leaky_relu"

        # Severe death in the first layer
        if dead_fractions[0] > 0.3:
            return "reinitialize"

        # Strictly increasing with depth and significant death in the last layer
        increasing = all(
            dead_fractions[i] < dead_fractions[i + 1]
            for i in range(len(dead_fractions) - 1)
        )

        if increasing and dead_fractions[-1] > 0.1:
            return "reduce_learning_rate"

        # Very few dead neurons
        if max(dead_fractions) < 0.1:
            return "healthy"

        # No critical pattern
        return "healthy"
