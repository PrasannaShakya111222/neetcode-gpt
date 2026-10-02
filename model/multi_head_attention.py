import torch
import torch.nn as nn
from torchtyping import TensorType

class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        # Create num_heads SingleHeadAttention instances using nn.ModuleList
        # Each head size = attention_dim // num_heads
        # Use: self.SingleHeadAttention(embedding_dim, head_size)
        # After the heads, add an output projection: nn.Linear(attention_dim, attention_dim, bias=False)

        # Calculate size of each individual attention head
        head_size = attention_dim // num_heads
        
        # Create num_heads SingleHeadAttention instances using nn.ModuleList
        self.heads = nn.ModuleList([
            self.SingleHeadAttention(embedding_dim, head_size) for _ in range(num_heads)
        ])
        
        # Add output projection layer after heads
        self.out_proj = nn.Linear(attention_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Run each head on the input, concatenate outputs along dim=2
        # Pass concatenated result through the output projection (W_O)
        # Return result rounded to 4 decimal places
        
        # Run each head on input tensor and collect their outputs
        head_outputs = [head(embedded) for head in self.heads]
        
        # Concatenate all head outputs along dimension 2 (feature/attention dimension)
        concat_outputs = torch.cat(head_outputs, dim=2)
        
        # Pass concatenated result through output projection layer (W_O)
        out = self.out_proj(concat_outputs)
        
        # Return result rounded to 4 decimal places
        return torch.round(out, decimals=4)

    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim: int, attention_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            k = self.key_gen(embedded)
            q = self.query_gen(embedded)
            v = self.value_gen(embedded)

            scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
            context_length, attention_dim = k.shape[1], k.shape[2]
            scores = scores / (attention_dim ** 0.5)

            lower_triangular = torch.tril(torch.ones(context_length, context_length))
            mask = lower_triangular == 0
            scores = scores.masked_fill(mask, float('-inf'))
            scores = nn.functional.softmax(scores, dim = 2)

            return scores @ v
