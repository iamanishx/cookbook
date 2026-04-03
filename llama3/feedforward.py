import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU FeedForward network as used in LLaMA.

    Uses the Swish-Gated Linear Unit activation:
    SwiGLU(x) = Swish(xW_gate) * (xW_up) * W_down

    This replaces the standard FFN (xW1 * ReLU * W2) and has been
    shown to be more parameter-efficient.

    Args:
        dim: Input/output dimension
        hidden_dim: Intermediate dimension (typically 2-4x dim)
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.W_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.W_up = nn.Linear(dim, hidden_dim, bias=False)
        self.W_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))
