from dataclasses import dataclass

import torch
import torch.nn as nn

from llama3.norm import RMSNorm
from llama3.rope import precompute_freqs_cis
from llama3.transformer_block import TransformerBlock


@dataclass
class LLaMAConfig:
    """Configuration for the LLaMA model.

    Args:
        vocab_size: Vocabulary size
        dim: Model embedding dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        n_kv_heads: Number of key-value heads (for GQA)
        head_dim: Dimension per attention head
        hidden_dim: FeedForward intermediate dimension
        max_seq_len: Maximum sequence length
        rope_theta: Base value for RoPE frequency computation
    """

    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 64
    hidden_dim: int = 1024
    max_seq_len: int = 512
    rope_theta: float = 500000.0


class LLaMA(nn.Module):
    """LLaMA language model.

    Architecture:
        Token Embedding -> Transformer Blocks -> RMSNorm -> LM Head

    Uses:
        - Grouped-Query Attention (GQA)
        - Rotary Position Embeddings (RoPE)
        - SwiGLU activation in FeedForward
        - Pre-Normalization with RMSNorm
        - Causal attention mask

    Args:
        config: LLaMAConfig instance
    """

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=config.dim,
                    num_heads=config.n_heads,
                    num_kv_heads=config.n_kv_heads,
                    head_dim=config.head_dim,
                    hidden_dim=config.hidden_dim,
                )
                for _ in range(config.n_layers)
            ]
        )

        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.head_dim, config.max_seq_len, config.rope_theta),
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: Input token IDs of shape (batch, seq_len)

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        bsz, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)

        mask = self._make_causal_mask(seq_len, tokens.device)
        freqs_cis = self.freqs_cis[:seq_len]

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        h = self.norm(h)
        return self.output(h)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a causal (triangular) attention mask.

        Args:
            seq_len: Sequence length
            device: Device to place the mask on

        Returns:
            Causal mask of shape (1, 1, seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.view(1, 1, seq_len, seq_len)

    @torch.inference_mode()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generate text autoregressively.

        Args:
            prompt: Input token IDs of shape (batch, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering (None = disabled)

        Returns:
            Generated token IDs including the prompt
        """
        for _ in range(max_new_tokens):
            seq_len = prompt.size(1)
            if seq_len > self.config.max_seq_len:
                prompt = prompt[:, -self.config.max_seq_len :]
                seq_len = self.config.max_seq_len

            logits = self.forward(prompt)[:, -1, :]

            if temperature > 0:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            prompt = torch.cat([prompt, next_token], dim=1)

        return prompt
