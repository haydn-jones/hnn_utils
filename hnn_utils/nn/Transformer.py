from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hnn_utils.nn.ALiBi import ALiBi


class TransformerEncoder(nn.Module):
    """
    PyTorch API compatible (more or less) encoder for Transformers, with support for rotary embeddings.

    Args:
        encoder_layers: encoder layers to use
    """

    def __init__(self, *encoder_layers: nn.Module) -> None:
        super().__init__()

        self.layers = nn.ModuleList(list(*encoder_layers))

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        output = src

        for layer in self.layers:
            output = layer(output, mask, src_padding_mask, is_causal=is_causal)

        return output


class TransformerDecoder(nn.Module):
    """
    PyTorch API compatible (more or less) decoder for Transformers, with support for rotary embeddings.

    Args:
        decoder_layers: decoder layers to use
    """

    def __init__(
        self,
        *decoder_layers: nn.Module,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(list(*decoder_layers))

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask,
                memory_mask,
                tgt_padding_mask,
                memory_padding_mask,
                tgt_is_causal,
                memory_is_causal,
            )

        return output


class TransformerEncoderLayer(nn.Module):
    """
    PyTorch API compatible (more or less) encoder layer for Transformers, with support for rotary embeddings.

    Note: Some of the defaults do not match the PyTorch implementation as they are used in my own experiments.

    Args:
        d_model: dimension of the embeddings
        nhead: number of attention heads
        dim_feedforward: dimension of the feedforward network
        dropout: dropout value
        activation: activation function in feedforward (nn.Module)
        norm_first: whether to apply layer norm before or after blocks
        self_rotary: use rotary embeddings in self-attention
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        self_alibi: bool = True,
    ) -> None:
        super().__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, use_alibi=self_alibi)

        self.ff_block = FFNSwiGLU(d_model, dim_feedforward)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.apply(_reset_parameters)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_padding_mask, is_causal=is_causal)
        x = x + self.ff_block(self.norm2(x))

        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, padding_mask=padding_mask, is_causal=is_causal)
        return self.dropout(x)


class TransformerDecoderLayer(nn.Module):
    """
    PyTorch API compatible (more or less) decoder layer for Transformers, with support for rotary embeddings.

    Note: Some of the defaults do not match the PyTorch implementation, or may be 'strange' (i.e. memory_rotary=False)
    as they are used in my own experiments.

    Args:
        d_model: dimension of the embeddings
        nhead: number of attention heads
        dim_feedforward: dimension of the feedforward network
        dropout: dropout value
        activation: activation function in feedforward (nn.Module)
        norm_first: whether to apply layer norm before or after blocks
        self_rotary: use rotary embeddings in self-attention
        memory_rotary: use rotary embeddings in cross-attention
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        self_alibi: bool = False,
        cross_alibi: bool = False,
    ):
        super().__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, use_alibi=self_alibi)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, use_alibi=cross_alibi)

        self.ff_block = FFNSwiGLU(d_model, dim_feedforward)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.apply(_reset_parameters)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt

        x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_padding_mask, is_causal=tgt_is_causal)
        x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_padding_mask, memory_is_causal)
        x = x + self.ff_block(self.norm3(x))
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, padding_mask=padding_mask, is_causal=is_causal)
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.cross_attn(x, mem, mem, attn_mask=attn_mask, padding_mask=padding_mask, is_causal=is_causal)
        return self.dropout2(x)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        heads: int,
        dropout: float = 0.0,
        use_alibi: bool = False,
    ):
        super().__init__()

        self.embed_size = embed_dim
        self.num_heads = heads
        self.head_dim = embed_dim // heads
        self.dropout = dropout

        assert self.head_dim * heads == embed_dim, "Embedding size needs to be divisible by heads"

        self.q_proj = nn.Linear(self.embed_size, self.embed_size)
        self.k_proj = nn.Linear(self.embed_size, self.embed_size)
        self.v_proj = nn.Linear(self.embed_size, self.embed_size)

        self.out_proj = nn.Linear(heads * self.head_dim, embed_dim)

        if use_alibi:
            self.alibi = ALiBi(heads)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        N = query.shape[0]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(N, -1, self.num_heads, self.head_dim).swapaxes(1, 2)
        k = k.view(N, -1, self.num_heads, self.head_dim).swapaxes(1, 2)
        v = v.view(N, -1, self.num_heads, self.head_dim).swapaxes(1, 2)

        mask = combine_masks(attn_mask, padding_mask, self.num_heads, query.dtype)
        if hasattr(self, "alibi"):
            bias = self.alibi(q, k)
            mask = mask + bias if mask is not None else bias

        dropout = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, mask, is_causal=is_causal, dropout_p=dropout)

        attn = attn.swapaxes(1, 2).reshape(N, -1, self.embed_size)
        return self.out_proj(attn)


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


class FFNSwiGLU(nn.Module):
    """
    GLU Variants Improve Transformer
    https://arxiv.org/abs/2002.05202
    """

    def __init__(self, dim, dim_feedforward):
        super().__init__()

        self.w = nn.Linear(dim, dim_feedforward, bias=False)
        self.v = nn.Linear(dim, dim_feedforward, bias=False)
        self.w2 = nn.Linear(dim_feedforward, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = F.silu(self.w(x)) * self.v(x)
        return self.w2(x)


def combine_masks(
    attn_mask: Optional[Tensor],
    pad_mask: Optional[Tensor],
    heads: int = 1,
    dtype: Optional[torch.dtype] = None,
) -> Optional[Tensor]:
    """
    Combines the masks for attention and padding.
    """

    dtype = dtype or torch.get_default_dtype()

    def floatify(x):
        if x is None:
            return None
        if x.dtype == torch.bool:
            return torch.where(x, -torch.inf, 0.0)
        return x

    mask = floatify(attn_mask)
    if pad_mask is not None:
        BSZ, SL = pad_mask.shape
        key_mask = pad_mask.view(BSZ, 1, 1, SL).expand(-1, heads, -1, -1)
        key_mask = floatify(key_mask)
        mask = mask + key_mask if mask is not None else key_mask

    return mask


def _reset_parameters(mod: nn.Module) -> None:
    if isinstance(mod, nn.Linear):
        nn.init.xavier_uniform_(mod.weight)
        if mod.bias is not None:
            nn.init.zeros_(mod.bias)


def causal_mask(embed: Tensor) -> Tensor:
    """
    Creates a causal mask for self-attention.
    """
    mask = torch.full((embed.shape[1], embed.shape[1]), -torch.inf, device=embed.device, dtype=embed.dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask
