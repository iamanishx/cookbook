import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor for rotary position embeddings.

    Args:
        dim: Head dimension (must be even)
        end: Maximum sequence length
        theta: Base value for the frequency computation

    Returns:
        Complex tensor of shape (end, dim//2) containing the frequency components
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        xq: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        xk: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        freqs_cis: Precomputed frequency tensor

    Returns:
        Tuple of (rotated_query, rotated_key)
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = freqs_cis[: xq_.shape[1], :]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
