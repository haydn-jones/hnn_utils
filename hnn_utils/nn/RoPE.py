from typing import Tuple

import torch
from einops import rearrange
from torch import Tensor


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    SL, D = xq.shape[-2:]

    freqs_cis = precompute_freqs_cis(
        dim=D,
        sl=SL,
        device=xq.device,
    ).reshape(1, 1, SL, D // 2)

    xq_ = torch.view_as_complex(rearrange(xq.float(), "b h sl (d n) -> b h sl d n", n=2))
    xk_ = torch.view_as_complex(rearrange(xk.float(), "b h sl (d n) -> b h sl d n", n=2))

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def precompute_freqs_cis(
    dim: int,
    sl: int,
    device: torch.device,
    theta: float = 10000.0,
) -> Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(sl, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis
