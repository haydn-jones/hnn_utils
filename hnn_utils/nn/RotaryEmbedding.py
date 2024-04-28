# Much of this is based on lucidrains implementation of rotary embeddings
# https://github.com/lucidrains/rotary-embedding-torch
# and https://github.com/microsoft/torchscale/blob/f69a87f4911d4bbdf5b851fb41400423bad6d36f/torchscale/component/xpos_relative_position.py#L38

import torch
from torch import nn


class XPOS(nn.Module):
    def __init__(self, head_dim: int, scale_base: int = 512):
        super().__init__()

        self.head_dim = head_dim
        self.scale_base = scale_base

        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq.unsqueeze(0))

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        SL = q.shape[-2]

        seq = torch.arange(SL, device=q.device, dtype=q.dtype).unsqueeze(-1)

        power = (seq - SL // 2) / self.scale_base
        scale = self.scale.pow(power).type_as(q)

        freqs = seq @ self.inv_freq
        sin = freqs.sin()
        cos = freqs.cos()

        q = self.apply_rotary_pos_emb(q, sin, cos, scale)
        k = self.apply_rotary_pos_emb(k, sin, cos, (1.0 / scale).type_as(q))
        return q, k

    def apply_rotary_pos_emb(self, x, sin, cos, scale):
        sin = torch.repeat_interleave(sin * scale, 2, dim=-1)
        cos = torch.repeat_interleave(cos * scale, 2, dim=-1)
        return (x * cos) + (self.rotate_every_two(x) * sin)

    def rotate_every_two(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x = torch.stack((-x2, x1), dim=-1)
        return x.flatten(-2)
