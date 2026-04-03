import tiktoken


class BPETokenizer:
    """BPE Tokenizer wrapper using tiktoken.

    Uses the cl100k_base encoding (GPT-4 tokenizer) which is
    a BPE tokenizer similar in spirit to what LLaMA 3 uses.

    Args:
        encoding_name: tiktoken encoding name (default: cl100k_base)
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab
        self.bos_id = self.enc.encode("<|endoftext|>", allowed_special="all")[0]
        self.eos_id = self.bos_id

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.enc.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to text."""
        return self.enc.decode(tokens)

    def __len__(self) -> int:
        return self.vocab_size
