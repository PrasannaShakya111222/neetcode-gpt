import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.attention_dim = attention_dim

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        # embedded: (B, T, embedding_dim)

        # Create Q, K, V
        Q = self.query(embedded)   # (B, T, attention_dim)
        K = self.key(embedded)     # (B, T, attention_dim)
        V = self.value(embedded)   # (B, T, attention_dim)

        # Attention scores
        scores = Q @ K.transpose(-2, -1)

        # Scale by sqrt(d_k)
        scores = scores / (self.attention_dim ** 0.5)

        # Causal mask: prevent looking at future tokens
        T = embedded.shape[1]
        mask = torch.triu(
            torch.ones(T, T, device=embedded.device),
            diagonal=1
        ).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        # Convert scores into attention probabilities
        weights = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        output = weights @ V
        return output.round(decimals=4)