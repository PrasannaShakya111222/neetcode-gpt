import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)

        # Collect all unique words
        words = set()
        for sentence in positive + negative:
            words.update(sentence.split())

        # Sort lexicographically and assign IDs starting from 1
        vocab = {word: i + 1 for i, word in enumerate(sorted(words))}

        # Encode each sentence
        encoded = []
        for sentence in positive + negative:
            ids = [vocab[word] for word in sentence.split()]
            encoded.append(torch.tensor(ids, dtype=torch.float))

        # Pad all sentences to same length
        # Positive sentences remain first, followed by negative
        return nn.utils.rnn.pad_sequence(
            encoded,
            batch_first=True,
            padding_value=0
        )