"""
Minimal character-level tokenizer.

In real LLMs (GPT, LLaMA), subword tokenizers like BPE are used.
Here we use character-level tokenization for simplicity — each
character is one token.
"""


class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0

    def fit(self, text: str):
        """Build vocabulary from text."""
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        """Convert text to list of token IDs."""
        return [self.char_to_id[ch] for ch in text if ch in self.char_to_id]

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back to text."""
        return "".join(self.id_to_char.get(i, "?") for i in ids)
