import torch
import torch.nn.functional as F
from rope import apply_rotary_emb, precompute_freqs_cis
from torch import nn


class GroupedQuerySlidingWindowAttentionSDPA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        window_size: int,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        use_rope: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.use_rope = use_rope

        assert num_heads % num_kv_heads == 0
        assert embed_dim % num_heads == 0
        self.group_size = num_heads // num_kv_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.head_dim * num_kv_heads)
        self.v_proj = nn.Linear(embed_dim, self.head_dim * num_kv_heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len, theta=rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        causal = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        window = torch.ones(max_seq_len, max_seq_len, dtype=torch.bool).triu(
            diagonal=-window_size
        )
        self.register_buffer("attn_mask", causal & window, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, E = hidden_states.shape
        assert S <= self.max_seq_len, f"seq_len {S} > max_seq_len {self.max_seq_len}"

        # (B, S, heads, head_dim) — layout expected by apply_rotary_emb
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)

        # RoPE on Q and K (frequencies broadcast over the head dim)
        if self.use_rope:
            q, k = apply_rotary_emb(q, k, self.freqs_cis)

        # (B, heads, S, head_dim) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Fused attention: bool mask (True = attend), GQA broadcast natively —
        # K/V are never expanded in memory.
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self.attn_mask[:S, :S],
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=True,
        )

        output = output.transpose(1, 2).contiguous().view(B, S, E)
        return self.out_proj(output)


if __name__ == "__main__":
    torch.manual_seed(0)

    fused = GroupedQuerySlidingWindowAttentionSDPA(
        embed_dim=512, num_heads=8, num_kv_heads=2, window_size=4
    )
    x = torch.randn(1, 10, 512)
    print("Fused (RoPE) output shape:", fused(x).shape)

    from gqa_swa import GroupedQuerySlidingWindowAttention

    ref = GroupedQuerySlidingWindowAttention(
        embed_dim=512, num_heads=8, num_kv_heads=2, window_size=4
    )
    fused_no_rope = GroupedQuerySlidingWindowAttentionSDPA(
        embed_dim=512, num_heads=8, num_kv_heads=2, window_size=4, use_rope=False
    )
    fused_no_rope.load_state_dict(ref.state_dict(), strict=False)
    fused_no_rope.eval()
    ref.eval()

    with torch.no_grad():
        out_ref = ref(x)
        out_fused = fused_no_rope(x)
    max_diff = (out_ref - out_fused).abs().max().item()
    print(f"Max |fused - reference| (no RoPE): {max_diff:.2e}")
    assert max_diff < 1e-5, "fused implementation diverges from reference!"
    print("Equivalence check passed ✅")
