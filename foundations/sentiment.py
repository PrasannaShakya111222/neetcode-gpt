import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        super().__init__()
        torch.manual_seed(0)
        self.embedding = nn.Embedding(vocabulary_size, 16)
        self.linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        # (B, T) -> (B, T, 16)
        x = self.embedding(x.long())

        # (B, T, 16) -> (B, 16)
        x = x.mean(dim=1)

        # (B, 16) -> (B, 1)
        x = self.linear(x)

        # Values between 0 and 1
        x = self.sigmoid(x)

        # Round to 4 decimal places
        return torch.round(x * 10000) / 10000