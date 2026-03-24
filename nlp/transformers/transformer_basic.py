"""
Transformer Architectures in PyTorch
=====================================
Two implementations side-by-side:
  1. Encoder-Decoder Transformer  (T5 / original "Attention is All You Need" style)
  2. Decoder-Only Transformer     (GPT style)

Run:
  python main.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# SHARED BUILDING BLOCKS
# ─────────────────────────────────────────────


class PositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding from "Attention Is All You Need".

    Adds a unique position signal to every token embedding so the model
    knows WHERE each token sits in the sequence.

    Shape: (seq_len, d_model)  -> added to token embeddings of same shape.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build the (max_len, d_model) sinusoidal table once
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims  → sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd  dims  → cos

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)  # not a trained param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention.

    Used in THREE places:
      Encoder  → self-attention        (bidirectional, no mask)
      Decoder  → masked self-attention (causal mask, can't see future)
      Decoder  → cross-attention       (queries from decoder, keys/values from encoder)

    In a decoder-only model there is NO cross-attention.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # dimension per head

        # Single linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,  # (B, T_q, d_model)
        key: torch.Tensor,  # (B, T_k, d_model)
        value: torch.Tensor,  # (B, T_k, d_model)
        mask: torch.Tensor = None,  # additive mask: -inf blocks attention
    ) -> torch.Tensor:
        B, T_q, _ = query.shape
        T_k = key.size(1)

        # 1. Project + split into heads
        Q = self.W_q(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        # shapes: (B, num_heads, T, head_dim)

        # 2. Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T_q, T_k)

        if mask is not None:
            scores = scores + mask  # mask contains 0 or -inf

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 3. Weighted sum of values
        out = torch.matmul(attn_weights, V)  # (B, H, T_q, head_dim)

        # 4. Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Same in encoder, decoder, and decoder-only models.
    Two linear layers with a ReLU in between.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# 1. ENCODER-DECODER TRANSFORMER
# ─────────────────────────────────────────────
#
# Architecture:
#
#   [src tokens]                [tgt tokens]
#       │                           │
#   Embedding + PE             Embedding + PE
#       │                           │
#  ┌────────────┐             ┌─────────────────────────┐
#  │ Encoder    │             │ Decoder                 │
#  │            │             │                         │
#  │ Self-Attn  │             │ Masked Self-Attn        │
#  │ (bidir)    │ ─────────►  │ Cross-Attn (← encoder) │
#  │ FFN        │             │ FFN                     │
#  │  × N       │             │  × N                    │
#  └────────────┘             └─────────────────────────┘
#                                          │
#                                     Linear + Softmax
#                                          │
#                                    [output tokens]
#


class EncoderLayer(nn.Module):
    """
    One encoder layer:
      - Bidirectional self-attention  (all tokens attend to all tokens)
      - Feed-forward network
      - Layer norm + residual connections
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention: query = key = value = x  (same sequence)
        # No causal mask → bidirectional → every token sees every other token
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))  # Add & Norm

        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))  # Add & Norm
        return x


class DecoderLayer(nn.Module):
    """
    One decoder layer (encoder-decoder style):
      1. Masked self-attention   → can only attend to PAST tokens (causal mask)
      2. Cross-attention         → attends to ENCODER OUTPUT (the "recipe")
      3. Feed-forward network
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # decoder input   (B, T_tgt, d_model)
        enc_output: torch.Tensor,  # encoder output  (B, T_src, d_model)
        tgt_mask: torch.Tensor = None,  # causal mask for decoder self-attn
        src_mask: torch.Tensor = None,  # padding mask for encoder output
    ) -> torch.Tensor:
        # Step 1: masked self-attention
        # query = key = value = x  but with causal mask
        # → each decoder token only sees itself and earlier decoder tokens
        attn1 = self.masked_self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn1))

        # Step 2: cross-attention  ← THIS IS WHAT USES THE ENCODER "RECIPE"
        # query  = decoder hidden states  (what we're generating)
        # key    = encoder output         (what we're reading from)
        # value  = encoder output
        attn2 = self.cross_attn(x, enc_output, enc_output, mask=src_mask)
        x = self.norm2(x + self.dropout(attn2))

        # Step 3: feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class EncoderDecoderTransformer(nn.Module):
    """
    Full encoder-decoder transformer (T5 / original paper style).

    src_vocab_size : vocabulary size for input  (e.g. source language)
    tgt_vocab_size : vocabulary size for output (e.g. target language)
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Separate embeddings for source and target
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        # Same positional encoding for both
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # Stack of N encoder layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Stack of N decoder layers
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.norm_enc = nn.LayerNorm(d_model)
        self.norm_dec = nn.LayerNorm(d_model)

        # Final projection: hidden state → vocab logits
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        Upper-triangular matrix of -inf.
        Forces decoder to only attend to current and past positions.
        """
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def encode(self, src: torch.Tensor, src_mask=None) -> torch.Tensor:
        """Run the encoder and return its hidden states."""
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.norm_enc(x)  # (B, T_src, d_model)

    def decode(
        self, tgt: torch.Tensor, enc_output: torch.Tensor, src_mask=None
    ) -> torch.Tensor:
        """Run the decoder given encoder output."""
        B, T_tgt = tgt.shape
        tgt_mask = self._causal_mask(T_tgt, tgt.device)  # (T_tgt, T_tgt)

        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            x = layer(x, enc_output, tgt_mask=tgt_mask, src_mask=src_mask)
        return self.norm_dec(x)  # (B, T_tgt, d_model)

    def forward(
        self,
        src: torch.Tensor,  # (B, T_src)  source token ids
        tgt: torch.Tensor,  # (B, T_tgt)  target token ids
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        enc_output = self.encode(src, src_mask)  # (B, T_src, d_model)
        dec_output = self.decode(tgt, enc_output, src_mask)  # (B, T_tgt, d_model)
        logits = self.output_projection(dec_output)  # (B, T_tgt, vocab_size)
        return logits


# ─────────────────────────────────────────────
# 2. DECODER-ONLY TRANSFORMER  (GPT style)
# ─────────────────────────────────────────────
#
# Architecture:
#
#   [all tokens: prompt + generated so far]
#               │
#          Embedding + PE
#               │
#   ┌──────────────────────────┐
#   │ Decoder-Only Block       │
#   │                          │
#   │ Causal Self-Attn         │  ← NO cross-attention
#   │ (each token sees past)   │     NO separate encoder
#   │ FFN                      │
#   │  × N                     │
#   └──────────────────────────┘
#               │
#          Linear + Softmax
#               │
#         [next token logits]
#


class GPTBlock(nn.Module):
    """
    One GPT-style decoder block.

    Differences from the encoder-decoder DecoderLayer:
      - NO cross-attention  (no encoder to attend to)
      - Only causal self-attention + FFN
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        # GPT uses pre-norm (LayerNorm before sublayer), original paper used post-norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # Pre-norm variant (more stable for large models)
        # Causal mask ensures token i can only attend to tokens 0..i
        attn_out = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x), mask=causal_mask
        )
        x = x + self.dropout(attn_out)  # residual

        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)  # residual
        return x


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only transformer (GPT style).

    One vocabulary, one embedding, one stack of GPT blocks.
    Everything — prompt + answer — lives in the same sequence.

    Training objective: predict next token at every position.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Single token embedding (no src/tgt split — it's all one sequence)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # Stack of GPT blocks (decoder-only, no cross-attention)
        self.blocks = nn.ModuleList(
            [GPTBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(d_model)

        # Project hidden state → vocab logits
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights: embedding matrix == output projection matrix
        # (common trick in GPT — halves parameters, often improves perplexity)
        self.output_projection.weight = self.token_embed.weight

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular -inf mask: token i cannot see token j > i."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T)  token ids  (prompt + any already-generated tokens)
        returns logits: (B, T, vocab_size)
        logits[:, t, :] = distribution over next token after position t
        """
        B, T = x.shape
        causal_mask = self._causal_mask(T, x.device)  # (T, T)

        h = self.pos_enc(
            self.token_embed(x) * math.sqrt(self.d_model)
        )  # (B, T, d_model)

        for block in self.blocks:
            h = block(h, causal_mask)

        h = self.norm(h)
        logits = self.output_projection(h)  # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self, prompt: torch.Tensor, max_new_tokens: int = 20, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Greedy / temperature-sampled autoregressive generation.

        prompt : (B, T_prompt)  already-tokenised prompt

        At each step:
          1. Forward pass on current sequence
          2. Take logits of LAST position only  ← that's the next-token distribution
          3. Sample / argmax
          4. Append to sequence
          5. Repeat
        """
        self.eval()
        seq = prompt.clone()

        for _ in range(max_new_tokens):
            logits = self.forward(seq)  # (B, seq_len, vocab_size)
            next_logits = logits[:, -1, :] / temperature  # only last position
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            seq = torch.cat([seq, next_token], dim=1)  # extend sequence

        return seq


# ─────────────────────────────────────────────
# DEMO: instantiate both and do a forward pass
# ─────────────────────────────────────────────


def demo():
    torch.manual_seed(42)
    B = 2  # batch size

    print("=" * 60)
    print("1. Encoder-Decoder Transformer (T5 style)")
    print("=" * 60)

    enc_dec = EncoderDecoderTransformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
    )

    src = torch.randint(0, 1000, (B, 10))  # source sequence  (e.g. English)
    tgt = torch.randint(
        0, 1000, (B, 7)
    )  # target sequence  (e.g. French, teacher-forced)

    logits = enc_dec(src, tgt)
    print(f"  src shape   : {src.shape}")
    print(f"  tgt shape   : {tgt.shape}")
    print(f"  output shape: {logits.shape}")  # (B, T_tgt, vocab)
    print(f"  params      : {sum(p.numel() for p in enc_dec.parameters()):,}")

    print()
    print("=" * 60)
    print("2. Decoder-Only Transformer (GPT style)")
    print("=" * 60)

    gpt = DecoderOnlyTransformer(
        vocab_size=1000,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
    )

    tokens = torch.randint(0, 1000, (B, 10))  # prompt tokens

    logits = gpt(tokens)
    print(f"  input shape : {tokens.shape}")
    print(f"  output shape: {logits.shape}")  # (B, T, vocab)
    print(f"  params      : {sum(p.numel() for p in gpt.parameters()):,}")

    # Generation demo
    prompt = torch.randint(0, 1000, (1, 5))
    generated = gpt.generate(prompt, max_new_tokens=10)
    print(f"  prompt len  : {prompt.shape[1]}  →  generated len: {generated.shape[1]}")

    print()
    print("KEY DIFFERENCE SUMMARY")
    print("-" * 40)
    print("Encoder-Decoder:")
    print("  encode(src) → enc_output  (the 'recipe')")
    print("  decode(tgt, enc_output)   (uses the recipe via cross-attention)")
    print()
    print("Decoder-Only:")
    print("  forward(prompt + generated_so_far)")
    print("  no cross-attention, no separate encoder")
    print("  everything is one unified sequence")


if __name__ == "__main__":
    demo()
