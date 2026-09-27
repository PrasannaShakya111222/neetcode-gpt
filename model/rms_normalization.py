import numpy as np
from typing import List

class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list

        # Calculate RMS
        rms = np.sqrt(np.mean(np.array(x) ** 2) + eps)

        # Normalize and scale by gamma
        output = (np.array(x) / rms) * np.array(gamma)

        # Round to 4 decimal places
        return np.round(output, 4).tolist()