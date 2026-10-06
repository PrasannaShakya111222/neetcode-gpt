from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        result = []
        # Try longer tokens first.
        sorted_lengths = sorted(set(len(k) for k in vocab.keys()), reverse=True)
        for number in numbers:
            text = str(number)
            tokenized = []
            i = 0
            while i < len(text):
                match = None
                for length in sorted_lengths:
                    if i + length <= len(text):
                        token = text[i:i+length]
                        if token in vocab:
                            match = token
                            break
                if match is None:
                    raise ValueError(
                        f"Cannot tokenize '{text}': no vocabulary token "
                        f"matches at position {i}."
                    )
                tokenized.append(match)
                i += len(match)
            result.append(tokenized)
        return result

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        sorted_lengths = sorted(set(len(k) for k in vocab.keys()), reverse=True)
        count = 0
        i = 0
        while i < len(text):
            match = None
            for length in sorted_lengths:
                if i + length <= len(text):
                    token = text[i:i+length]
                    if token in vocab:
                        match = token
                        break
            if match is None:
                raise ValueError(
                    f"Cannot tokenize text at position {i}: "
                    f"'{text[i:]}'"
                )
            count += 1
            i += len(match)
        return count

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words = text.split()
        if not words:
            return 0.0
        token_count = self.count_tokens(text, vocab)
        return round(token_count / len(words), 4)