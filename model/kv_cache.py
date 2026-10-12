import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Append new_k and new_v to the cache along the sequence dimension (dim=1).
        # On the first call, initialize the cache with the given tensors.
        # Return the full (cached) K and V tensors.
        if self.cache_k is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.cat(
                [self.cache_k, new_k], dim=1
            )
            self.cache_v = torch.cat(
                [self.cache_v, new_v], dim=1
            )
        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        # 1. Project x into Q, K, V using the linear layers
        # 2. If kv_cache is None, create a new KVCache
        # 3. Update the cache with the new K and V
        # 4. Compute scaled dot-product attention using Q and the full cached K, V
        # 5. Apply a causal mask offset by the number of previously cached tokens
        # 6. Return (rounded output, kv_cache)
        
        # Project input into Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Create cache if needed
        if kv_cache is None:
            kv_cache = KVCache()

        # Get previous cache length before updating
        past_len = (
            0 if kv_cache.cache_k is None
            else kv_cache.cache_k.size(1)
        )

        # Append new K and V
        k_full, v_full = kv_cache.update(k, v)

        # Scaled dot-product attention scores
        model_dim = q.size(-1)
        scores = torch.matmul(
            q, k_full.transpose(-2, -1)
        ) / (model_dim ** 0.5)

        # Causal mask with cache offset
        query_len = q.size(1)
        key_len = k_full.size(1)

        query_positions = (
            torch.arange(query_len, device=x.device) + past_len
        )
        key_positions = torch.arange(key_len, device=x.device)

        mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)

        scores = scores.masked_fill(
            mask.unsqueeze(0), float("-inf")
        )

        # Softmax and weighted sum of values
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v_full)

        # Return rounded output and cache
        return torch.round(output, decimals=4), kv_cache