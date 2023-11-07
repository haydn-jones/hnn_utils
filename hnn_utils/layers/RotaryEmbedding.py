# Much of this is based on lucidrains implementation of rotary embeddings
# https://github.com/lucidrains/rotary-embedding-torch

# Removed many of the configurable options to simplify the code
# and allow it to work with torch.compile
# Uses xpos

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        theta=10000,
        scale_base=512,
    ):
        super().__init__()

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        self.register_buffer("freqs", freqs)

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer("scale", scale)

        self.scale_base = scale_base

    def forward(self, q, k):
        device, dtype, seq_len = q.device, q.dtype, q.shape[-2]

        seq = torch.arange(seq_len, device=device, dtype=dtype)

        freqs = seq.unsqueeze(1) * self.freqs.unsqueeze(0)
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)

        power = (seq - seq_len // 2) / self.scale_base
        scale = self.scale.pow(power.unsqueeze(-1))
        scale = torch.cat((scale, scale), dim=-1)

        fc = freqs.cos()
        fs = freqs.sin()

        rotated_q = self.apply_rotary_emb(fc, fs, q, scale=scale)
        rotated_k = self.apply_rotary_emb(fc, fs, k, scale=scale.reciprocal())

        return rotated_q, rotated_k

    def apply_rotary_emb(self, freqs_cos, freqs_sin, t, scale):
        t = (t * freqs_cos * scale) + (self.rotate_half(t) * freqs_sin * scale)
        return t

    def rotate_half(self, x):
        d = x.size(-1) // 2
        x = x.view(*x.shape[:-1], d, 2)

        x1, x2 = x.unbind(dim=-1)

        x = torch.stack((-x2, x1), dim=-1)
        x = x.view(*x.shape[:-2], -1)
        return x
