import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.register_parameter("weight", nn.Parameter(torch.ones(dim)))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x * self.weight.type_as(x) * rms.type_as(x)


class LayerNorm(nn.Module):
    def __init__(self, features: int):
        super().__init__()

        self.ln = nn.LayerNorm(features)

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype

        x = self.ln(x)
        return x.type(dtype)
