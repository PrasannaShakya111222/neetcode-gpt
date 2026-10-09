import torch
import torch.nn as nn
from torchtyping import TensorType

# 1. Remember to include an additional LayerNorm after the block sequence and before the final linear layer
# 2. Instantiate in the following order: Word embeddings, position embeddings, transformer blocks, final layer norm, and vocabulary projection.
class GPT(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        # Hint: nn.Sequential() will be useful for the block sequence
        super().__init__()
        torch.manual_seed(0)

        # Word embeddings
        self.token_embedding = nn.Embedding(vocab_size, model_dim)

        # Position embeddings
        self.position_embedding = nn.Embedding(context_length, model_dim)

        # Transformer blocks
        self.blocks = nn.Sequential(
            *[
                self.TransformerBlock(model_dim, num_heads)
                for _ in range(num_blocks)
            ]
        )

        # Final layer normalization
        self.final_norm = nn.LayerNorm(model_dim)

        # Vocabulary projection
        self.lm_head = nn.Linear(model_dim, vocab_size)

    def forward(self, context: TensorType[int]) -> TensorType[float]:
        torch.manual_seed(0)
        # 1. Add token embeddings + position embeddings (use torch.arange for positions)
        # 2. Pass through transformer blocks
        # 3. Apply final LayerNorm, then project to vocab_size
        # 4. Return logits rounded to 4 decimal places (no softmax)
         # Token embeddings
        token_emb = self.token_embedding(context)

        # Position embeddings
        positions = torch.arange(
            context.shape[1],
            device=context.device
        )
        pos_emb = self.position_embedding(positions)

        # Add token and position embeddings
        embedded = token_emb + pos_emb

        # Transformer blocks
        embedded = self.blocks(embedded)

        # Final layer normalization
        embedded = self.final_norm(embedded)

        # Project to vocabulary logits
        logits = self.lm_head(embedded)

        # Round to 4 decimal places
        return torch.round(logits * 10000) / 10000

    # Do NOT modify the code below this line
    class TransformerBlock(nn.Module):

        class MultiHeadedSelfAttention(nn.Module):

            class SingleHeadAttention(nn.Module):
                def __init__(self, model_dim: int, head_size: int):
                    super().__init__()
                    torch.manual_seed(0)
                    self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.value_gen = nn.Linear(model_dim, head_size, bias=False)
                
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
                
            def __init__(self, model_dim: int, num_heads: int):
                super().__init__()
                torch.manual_seed(0)
                self.att_heads = nn.ModuleList()
                for i in range(num_heads):
                    self.att_heads.append(self.SingleHeadAttention(model_dim, model_dim // num_heads))
                self.output_proj = nn.Linear(model_dim, model_dim, bias=False)

            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                head_outputs = []
                for head in self.att_heads:
                    head_outputs.append(head(embedded))
                concatenated = torch.cat(head_outputs, dim = 2)
                return self.output_proj(concatenated)
        
        class VanillaNeuralNetwork(nn.Module):

            def __init__(self, model_dim: int):
                super().__init__()
                torch.manual_seed(0)
                self.up_projection = nn.Linear(model_dim, model_dim * 4)
                self.relu = nn.ReLU()
                self.down_projection = nn.Linear(model_dim * 4, model_dim)
                self.dropout = nn.Dropout(0.2) # using p = 0.2
            
            def forward(self, x: TensorType[float]) -> TensorType[float]:
                torch.manual_seed(0)
                return self.dropout(self.down_projection(self.relu(self.up_projection(x))))

        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)
            self.attention = self.MultiHeadedSelfAttention(model_dim, num_heads)
            self.linear_network = self.VanillaNeuralNetwork(model_dim)
            self.first_norm = nn.LayerNorm(model_dim)
            self.second_norm = nn.LayerNorm(model_dim)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)
            embedded = embedded + self.attention(self.first_norm(embedded)) # skip connection
            embedded = embedded + self.linear_network(self.second_norm(embedded)) # another skip connection
            return embedded
