import torch
import torch.nn as nn

from llama3.rope import apply_rotary_emb


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention as used in LLaMA 3.

    Multiple query heads share the same key/value projections,
    reducing memory usage while maintaining modeling performance.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads (<= num_heads)
        head_dim: Dimension per head (defaults to dim // num_heads)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.num_kv_groups = num_kv_heads
        self.group_size = num_heads // num_kv_heads

        inner_dim = self.head_dim * num_heads
        kv_inner_dim = self.head_dim * num_kv_heads

        self.W_query = nn.Linear(dim, inner_dim, bias=False)
        self.W_key = nn.Linear(dim, kv_inner_dim, bias=False)
        self.W_value = nn.Linear(dim, kv_inner_dim, bias=False)
        self.W_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        queries = self.W_query(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        keys = self.W_key(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        values = self.W_value(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        queries, keys = apply_rotary_emb(queries, keys, freqs_cis)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        scale = 1.0 / (self.head_dim**0.5)
        attn_scores = queries @ keys.transpose(2, 3) * scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores.float(), dim=-1).type_as(queries)
        attn_output = (attn_weights @ values).transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, seq_len, -1)

        return self.W_out(attn_output)
