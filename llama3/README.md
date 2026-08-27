# LLaMA3 From Scratch

A clean, modular implementation of the LLaMA architecture in pure PyTorch. No third-party LLM libraries — just the building blocks that make modern language models work.

This project was inspired by Sebastian Raschka's [Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) article, which breaks down what makes models like LLaMA 3, Qwen3, Gemma 3, and others tick.

## What's Inside

The architecture is split into small, focused modules — each one does a single job and does it well:

```
llama3/
├── llama3/
│   ├── norm.py              # RMSNorm layer
│   ├── rope.py              # Rotary Position Embeddings
│   ├── attention.py         # Grouped-Query Attention
│   ├── feedforward.py       # SwiGLU FeedForward network
│   ├── transformer_block.py # One transformer block (attn + ffn)
│   └── model.py             # Full LLaMA model + generation
├── train.py                 # Training script
├── generate.py              # Text generation script
└── pyproject.toml
```

## The Architecture, Piece by Piece

### 1. RMSNorm (`norm.py`)

LLaMA doesn't use the standard LayerNorm you'd find in the original transformer paper. Instead, it uses **RMSNorm** (Root Mean Square Normalization), which is simpler and faster.

The key difference: LayerNorm subtracts the mean and divides by the standard deviation. RMSNorm skips the mean subtraction entirely — it just divides by the root mean square. Fewer operations, same stability.

```
RMSNorm(x) = x / RMS(x) * weight
where RMS(x) = sqrt(mean(x²) + eps)
```

### 2. RoPE (`rope.py`)

Position information is critical in transformers because self-attention doesn't inherently know about token order. Early models like GPT used absolute positional embeddings (just adding a learned vector per position). LLaMA uses **Rotary Position Embeddings (RoPE)** instead.

RoPE works by rotating the query and key vectors in the complex plane. The rotation angle depends on the token's position. This gives the model a natural sense of relative distance — tokens that are closer together have more similar rotations.

The math is elegant: instead of adding position info, you multiply by a complex rotation. This preserves the dot-product structure of attention while encoding position.

### 3. Grouped-Query Attention (`attention.py`)

The original transformer used Multi-Head Attention (MHA), where every attention head has its own query, key, and value projections. This is expensive in terms of memory, especially the KV cache during generation.

LLaMA 3 uses **Grouped-Query Attention (GQA)**, which is a middle ground:
- **MHA**: Each head has its own Q, K, V (most expressive, most expensive)
- **MQA**: All heads share one K, V (cheapest, least expressive)
- **GQA**: Groups of heads share K, V (sweet spot)

In our implementation, `num_heads` query heads share `num_kv_heads` key-value pairs. For example, with 8 query heads and 2 KV heads, every 4 query heads share the same key and value projections.

After computing attention scores, we apply a causal mask so each token can only see itself and previous tokens — this is what makes the model autoregressive.

### 4. SwiGLU FeedForward (`feedforward.py`)

The original transformer used a simple feed-forward network: `ReLU(xW1) * W2`. LLaMA replaces this with **SwiGLU**, which is a gated architecture:

```
SwiGLU(x) = Swish(xW_gate) * (xW_up) * W_down
where Swish(x) = x * sigmoid(x)
```

This has three linear layers instead of two, but it's more parameter-efficient. The gating mechanism (Swish) lets the model dynamically control how much information flows through. Think of it like a learnable "attention" within the feed-forward layer itself.

### 5. Transformer Block (`transformer_block.py`)

Each transformer block follows the **Pre-Norm** pattern:

```
x -> RMSNorm -> Attention -> +x (residual) -> RMSNorm -> SwiGLU -> +x (residual)
```

The normalization happens *before* the sub-layers (attention and feed-forward), not after. This was shown to give better gradient flow at initialization and makes training more stable. The residual connections (skip connections) let gradients flow directly through the network, which is crucial for deep models.

### 6. Full Model (`model.py`)

The complete model stacks everything together:

```
Token Embedding -> [Transformer Block × N] -> RMSNorm -> LM Head (output projection)
```

There are no separate positional embedding layers — RoPE handles position encoding inside the attention mechanism. The output is a linear projection from the model dimension to the vocabulary size, giving us logits for the next token prediction.

## Quick Start

### Install

```bash
cd llama3
uv sync
```

### Train

```bash
uv run python -m llama3.train --epochs 50 --batch_size 32 --seq_len 64
```

This trains on a built-in sample dataset (simple English sentences). The default config is small (~500K parameters) so it runs on CPU in a few minutes.

### Generate

```bash
uv run python -m llama3.generate --prompt "the " --max_new_tokens 100
```

### Customize

Both scripts accept command-line arguments:

```bash
# Training options
uv run python -m llama3.train --epochs 100 --batch_size 64 --lr 5e-4 --device cuda

# Generation options
uv run python -m llama3.generate --prompt "the cat " --temperature 0.7 --top_k 20
```

## Configuration

The model is controlled by `LLaMAConfig` in `model.py`. The default small config:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | 32000 | Vocabulary size |
| `dim` | 128 | Model embedding dimension |
| `n_layers` | 4 | Number of transformer blocks |
| `n_heads` | 4 | Number of attention heads |
| `n_kv_heads` | 2 | KV heads for GQA (ratio 2:1) |
| `head_dim` | 32 | Dimension per head |
| `hidden_dim` | 256 | SwiGLU intermediate dimension |
| `max_seq_len` | 512 | Maximum sequence length |

Scale these up for a bigger model. The LLaMA 3 8B config, for reference, uses dim=4096, n_layers=32, n_heads=32, n_kv_heads=8.

## Design Decisions

**Why character-level training?** This is a learning project. Character-level models are simpler to understand because you don't need a tokenizer library. The tradeoff is that the model has to learn everything from scratch — including what words are.

**Why GQA instead of MHA?** GQA reduces memory usage during generation (smaller KV cache) with minimal impact on quality. It's the standard in modern LLMs.

**Why Pre-Norm?** Pre-Normalization gives better gradient flow at initialization. Post-Norm (used in the original transformer) requires careful learning rate warmup. Pre-Norm is more forgiving.

**Why SwiGLU?** It's more parameter-efficient than ReLU-based FFNs. The gating mechanism lets the model be more expressive with the same number of parameters.

## What's Missing (On Purpose)

This is a minimal, educational implementation. Production LLMs add many more things:

- **KV caching** for efficient generation
- **FlashAttention** for faster training
- **Mixed precision** training (bf16/fp16)
- **Distributed training** across multiple GPUs
- **Learning rate scheduling** (cosine decay with warmup)
- **Gradient clipping** for stability
- **Weight tying** (some models tie embedding and output weights)
- **Mixture of Experts** (MoE) for scaling
- **Sliding window attention** for long contexts

These are great next steps if you want to extend the project.

## References

- [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) by Sebastian Raschka
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
