import torch
import torch.nn as nn
from torch import Tensor
import math


class ALiBi(nn.Module):
    """
    Attention with Linear Biases. Added to attention logits before softmax.
    https://arxiv.org/abs/2108.12409
    """

    def __init__(self, heads: int):
        super().__init__()

        self.heads = heads

        self.register_buffer("slopes", self.get_slopes(heads).reshape(heads, 1, 1))

    def get_slopes(self, heads: int) -> Tensor:
        def calculate_slopes(num_elements: int) -> list:
            base_ratio = 2 ** (-(2 ** -(math.log2(num_elements) - 3)))
            return [base_ratio * (base_ratio**i) for i in range(num_elements)]

        power_of_2 = 1 << (heads.bit_length() - 1)
        slopes = calculate_slopes(power_of_2)

        if heads != power_of_2:
            extra_slopes = calculate_slopes(2 * power_of_2)
            slopes += extra_slopes[0::2][: heads - power_of_2]

        return torch.tensor(slopes)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
    ) -> Tensor:
        SL = q.shape[-2]
        TL = k.shape[-2]

        dtype = q.dtype

        i = torch.arange(SL, dtype=dtype, device=q.device).reshape(1, -1, 1)
        j = torch.arange(TL, dtype=dtype, device=q.device).reshape(1, 1, -1)
        bias = -torch.abs(j - i)

        bias = bias * self.slopes.type_as(bias)
        return bias
