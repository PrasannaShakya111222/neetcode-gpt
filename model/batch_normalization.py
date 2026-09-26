import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists 
        batch_size = len(x)
        num_features = len(x[0])
        
        if training:
            # Compute mean across the batch for each feature
            mean = [sum(x[i][j] for i in range(batch_size)) / batch_size for j in range(num_features)]
            # Compute variance across the batch for each feature
            var = [sum((x[i][j] - mean[j]) ** 2 for i in range(batch_size)) / batch_size for j in range(num_features)]
            
            # Update running statistics using momentum formula
            for j in range(num_features):
                running_mean[j] = (1 - momentum) * running_mean[j] + momentum * mean[j]
                running_var[j] = (1 - momentum) * running_var[j] + momentum * var[j]
            
            use_mean = mean
            use_var = var
        else:
            use_mean = running_mean
            use_var = running_var
            
        # Apply normalization and affine transform
        y = []
        for i in range(batch_size):
            row = []
            for j in range(num_features):
                x_hat = (x[i][j] - use_mean[j]) / ((use_var[j] + eps) ** 0.5)
                val = gamma[j] * x_hat + beta[j]
                row.append(round(val, 4))
            y.append(row)
            
        rounded_running_mean = [round(m, 4) for m in running_mean]
        rounded_running_var = [round(v, 4) for v in running_var]
        
        return y, rounded_running_mean, rounded_running_var