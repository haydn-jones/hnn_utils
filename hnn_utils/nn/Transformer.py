from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hnn_utils.nn.RotaryEmbedding import XPOS


class RotaryEncoder(nn.Module):
    """
    PyTorch API compatible (more or less) encoder for Transformers, with support for rotary embeddings.

    Args:
        encoder_layer: encoder layer to use
        num_layers: number of encoder layers
        fix_init: Torch clones the encoder layer causing all layers to have identical initialization. This corrects that.
    """

    def __init__(
        self,
        encoder_layers: List[nn.Module],
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(encoder_layers)

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        output = src

        for layer in self.layers:
            output = layer(output, mask, src_key_padding_mask, is_causal=is_causal)

        return output


class RotaryDecoder(nn.Module):
    """
    PyTorch API compatible (more or less) decoder for Transformers, with support for rotary embeddings.

    Args:
        decoder_layer: decoder layer to use
        num_layers: number of decoder layers
        fix_init: Torch clones the encoder layer causing all layers to have identical initialization. This corrects that.
    """

    def __init__(
        self,
        decoder_layers: List[nn.Module],
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(decoder_layers)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
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
                tgt_key_padding_mask,
                memory_key_padding_mask,
                tgt_is_causal,
                memory_is_causal,
            )

        return output


class RotaryEncoderLayer(nn.Module):
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
        self_rotary: bool = True,
    ) -> None:
        super().__init__()

        self.self_attn = RotaryMultiheadAttention(
            d_model, nhead, dropout=dropout, use_rotory_emb=self_rotary
        )

        self.ff_block = FFNSwiGLU(d_model, dim_feedforward)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        x = src

        x = x + self._sa_block(
            self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
        )
        x = x + self.ff_block(self.norm2(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )
        return self.dropout(x)


class RotaryDecoderLayer(nn.Module):
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
        self_rotary: bool = True,
        memory_rotary: bool = False,
    ):
        super().__init__()

        self.self_attn = RotaryMultiheadAttention(
            d_model, nhead, dropout=dropout, use_rotory_emb=self_rotary
        )
        self.cross_attn = RotaryMultiheadAttention(
            d_model, nhead, dropout=dropout, use_rotory_emb=memory_rotary
        )

        self.ff_block = FFNSwiGLU(d_model, dim_feedforward)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt

        x = x + self._sa_block(
            self.norm1(x), tgt_mask, tgt_key_padding_mask, is_causal=tgt_is_causal
        )
        x = x + self._mha_block(
            self.norm2(x),
            memory,
            memory_mask,
            memory_key_padding_mask,
            memory_is_causal,
        )
        x = x + self.ff_block(self.norm3(x))
        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.cross_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )
        return self.dropout2(x)


class RotaryMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        heads: int,
        dropout: float = 0.0,
        use_rotory_emb: bool = True,
    ):
        super().__init__()

        self.embed_size = embed_dim
        self.num_heads = heads
        self.head_dim = embed_dim // heads
        self.dropout = dropout

        assert (
            self.head_dim * heads == embed_dim
        ), "Embedding size needs to be divisible by heads"

        self.q_proj = nn.Linear(self.embed_size, self.embed_size)
        self.k_proj = nn.Linear(self.embed_size, self.embed_size)
        self.v_proj = nn.Linear(self.embed_size, self.embed_size)
        self.out_proj = nn.Linear(heads * self.head_dim, embed_dim)

        if use_rotory_emb:
            self.rotary_emb = XPOS(self.head_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        N = query.shape[0]

        mask = combine_masks(query, key, self.num_heads, attn_mask, key_padding_mask)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(self, "rotary_emb"):
            q, k = self.rotary_emb(q, k)

        dropout = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(
            q, k, v, mask, is_causal=is_causal, dropout_p=dropout
        )
        attn = attn.transpose(1, 2).reshape(N, -1, self.embed_size)
        return self.out_proj(attn)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.register_parameter("weight", nn.Parameter(torch.ones(dim)))

    def forward(self, x):
        rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x * self.weight.type_as(x) * rms.type_as(x)


class FFNSwiGLU(nn.Module):
    def __init__(self, dim, dim_feedforward):
        super().__init__()

        self.w = nn.Linear(dim, dim_feedforward, bias=False)
        self.v = nn.Linear(dim, dim_feedforward, bias=False)
        self.w2 = nn.Linear(dim_feedforward, dim, bias=False)

    def forward(self, x):
        x = F.silu(self.w(x)) * self.v(x)
        return self.w2(x)


def combine_masks(
    q: Tensor,
    k: Tensor,
    heads: int,
    attn_mask: Optional[Tensor],
    key_padding_mask: Optional[Tensor],
) -> Optional[Tensor]:
    """
    Combines the masks for attention and key padding.
    """
    N, L = q.shape[0], q.shape[-2]
    S = k.shape[-2]

    key_padding_mask = F._canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=F._none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=q.dtype,
    )

    attn_mask = F._canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=q.dtype,
        check_other=False,
    )

    mask = None
    if attn_mask is not None:
        mask = attn_mask
    if key_padding_mask is not None:
        key_mask = key_padding_mask.view(N, 1, 1, key_padding_mask.shape[-1]).expand(
            N, heads, L, S
        )
        mask = mask + key_mask if mask is not None else key_mask

    return mask
