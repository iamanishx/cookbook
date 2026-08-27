import torch
import torch.nn as nn
import math

class GroupedQuerySlidingWindowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads, window_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        
        assert num_heads % num_kv_heads == 0
        self.group_size = num_heads // num_kv_heads
        
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.head_dim * num_kv_heads)
        self.v_proj = nn.Linear(embed_dim, self.head_dim * num_kv_heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_states):
        B, S, E = hidden_states.shape
        
  
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)


        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        mask = torch.ones(S, S, device=hidden_states.device, dtype=torch.bool)
        mask = torch.tril(mask) 
        
        window_mask = torch.ones(S, S, device=hidden_states.device, dtype=torch.bool)
        window_mask = window_mask.triu(diagonal=-self.window_size)
        mask = mask & window_mask
        
        attn_mask = torch.zeros((S, S), device=hidden_states.device)
        attn_mask.masked_fill_(~mask, float('-inf'))
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
        
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        output = output.transpose(1, 2).contiguous().view(B, S, E)
        return self.out_proj(output)


gqa_swa = GroupedQuerySlidingWindowAttention(embed_dim=512, num_heads=8, num_kv_heads=2, window_size=4)
sample_input = torch.randn(1, 10, 512)
output = gqa_swa(sample_input)
print("Output shape:", output.shape)
