import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        # Convert lists to NumPy arrays
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # Forward pass

        # x -> Linear
        z1 = W1 @ x + b1

        # ReLU
        a1 = np.maximum(0, z1)

        # Hidden -> Output
        z2 = W2 @ a1 + b2

        # Prediction
        y_pred = z2

        # MSE loss
        loss = np.mean((y_pred - y_true) ** 2)

        # Backward pass

        # dL/dy_pred
        dz2 = 2 * (y_pred - y_true) / y_true.size

        # dL/dW2 = dz2 outer a1
        dW2 = np.outer(dz2, a1)

        # dL/db2
        db2 = dz2

        # Gradient flowing back to hidden layer
        da1 = W2.T @ dz2
        
        # ReLU derivative: 1 when z1 > 0, otherwise 0
        relu_mask = (z1 > 0).astype(float)

        # dL/dz1
        dz1 = da1 * relu_mask

        # dL/dW1
        dW1 = np.outer(dz1, x)

        # dL/db1
        db1 = dz1

        # Round everything to 4 decimal places
        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }