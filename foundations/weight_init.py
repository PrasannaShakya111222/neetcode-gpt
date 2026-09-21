import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list

        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        weight = torch.randn(fan_out, fan_in) * std

        return [[round(val, 4) for val in row] for row in weight.tolist()]

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list

        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        weight = torch.randn(fan_out, fan_in) * std

        return [[round(val, 4) for val in row] for row in weight.tolist()]

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.

        torch.manual_seed(0)
        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []
        
        # Generate all weight matrices
        for i in range(num_layers):
            if init_type == "kaiming":
                std = math.sqrt(2.0 / dims[i])
            elif init_type == "xavier":
                std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            elif init_type == "random":
                std = 1.0
            else:
                std = 1.0
            w = torch.randn(dims[i + 1], dims[i]) * std
            weights.append(w)
            
        # Generate input tensor
        x = torch.randn(1, input_dim)
        
        # Perform forward pass and record standard deviations
        stds = []
        for w in weights:
            x = x @ w.T
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))

        return stds