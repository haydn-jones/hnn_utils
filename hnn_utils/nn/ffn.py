import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FFNSwiGLU(nn.Module):
    """
    GLU Variants Improve Transformer
    https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        dim: int,
        dim_feedforward: int,
        out_dim: int | None = None,
        use_bias: bool = True,
    ):
        super().__init__()

        out_dim = out_dim or dim

        self.ff1 = nn.Linear(dim, dim_feedforward * 2, bias=use_bias)
        self.ff2 = nn.Linear(dim_feedforward, out_dim, bias=use_bias)

    def forward(self, x: Tensor) -> Tensor:
        y, gate = self.ff1(x).chunk(2, dim=-1)
        x = y * F.silu(gate)
        return self.ff2(x)


class FFNN(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_feedforward: int,
        out_dim: int | None = None,
        use_bias: bool = True,
    ):
        super().__init__()

        out_dim = out_dim or dim

        self.block = nn.Sequential(
            nn.Linear(dim, dim_feedforward, bias=use_bias),
            nn.SiLU(),
            nn.Linear(dim_feedforward, out_dim, bias=use_bias),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
