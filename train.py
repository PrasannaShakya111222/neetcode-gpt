import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        final_loss = 0.0

        for epoch in range(epochs):
            torch.manual_seed(epoch)
            # Sample random starting positions for sequences
            starts = torch.randint(
                0, len(data) - context_length,
                (batch_size,)
            )
            x = torch.stack([
                data[i:i + context_length] for i in starts
            ])
            y = torch.stack([
                data[i + 1:i + context_length + 1] for i in starts
            ])
            # Forward pass
            logits = model(x)
            # Flatten logits and targets for cross-entropy
            vocab_size = logits.size(-1)
            loss = loss_fn(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
        return round(final_loss, 4)