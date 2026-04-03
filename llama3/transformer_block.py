import torch
import torch.nn as nn

from llama3.attention import GroupedQueryAttention
from llama3.feedforward import SwiGLU
from llama3.norm import RMSNorm


class TransformerBlock(nn.Module):
    """Single LLaMA transformer block with pre-normalization.

    Structure:
        x -> RMSNorm -> Attention -> +x -> RMSNorm -> FFN -> +x

    Uses Pre-Norm (normalization before the sub-layer) for training stability.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads
        head_dim: Dimension per head
        hidden_dim: FeedForward intermediate dimension
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.attention = GroupedQueryAttention(dim, num_heads, num_kv_heads, head_dim)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.ffn(self.ffn_norm(h))
        return out
