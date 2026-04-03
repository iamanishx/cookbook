import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Simplified version of LayerNorm that removes the mean-centering
    and only scales by the root mean square of the inputs.

    Args:
        dim: Feature dimension
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).sqrt()
        x = x * (norm + self.eps).type_as(x)
        return self.weight * x
