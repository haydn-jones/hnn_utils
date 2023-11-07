import copy
from math import log
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hnn_utils.layers.RotaryEmbedding import RotaryEmbedding


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
        encoder_layer: nn.Module,
        num_layers: int,
        fix_init: bool = True,
    ) -> None:
        super().__init__()

        self.layers = _get_clones(encoder_layer, num_layers)

        if fix_init:
            self.reset_parameters()

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

    @torch.no_grad()
    def reset_parameters(self):
        for mod in self.layers.modules():
            # Linear and LayerNorm mods have this
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()

        # Reset MHA
        for layer in self.layers:
            layer.self_attn.reset_parameters()


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
        decoder_layer: nn.Module,
        num_layers: int,
        fix_init: bool = True,
    ) -> None:
        super().__init__()

        self.layers = _get_clones(decoder_layer, num_layers)

        if fix_init:
            self.reset_parameters()

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

    @torch.no_grad()
    def reset_parameters(self):
        for mod in self.layers.modules():
            # Linear and LayerNorm mods have this
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()

        # Reset MHA
        for layer in self.layers:
            layer.self_attn.reset_parameters()
            layer.cross_attn.reset_parameters()


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
        activation: nn.Module = nn.GELU(),
        norm_first: bool = False,
        self_rotary: bool = True,
    ) -> None:
        super().__init__()

        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, use_rotory_emb=self_rotary
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = activation

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        x = src

        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

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

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


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
        activation: nn.Module = nn.GELU(),
        norm_first: bool = False,
        self_rotary: bool = True,
        memory_rotary: bool = False,
    ):
        super().__init__()

        self.norm_first = norm_first

        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, use_rotory_emb=self_rotary
        )
        self.cross_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, use_rotory_emb=memory_rotary
        )

        self.activation = activation

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

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

        if self.norm_first:
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
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x, tgt_mask, tgt_key_padding_mask, is_causal=tgt_is_causal
                )
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x))

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

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        heads: int,
        dropout: float = 0.0,
        use_rotory_emb: bool = False,
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
            self.rotary_emb = RotaryEmbedding(self.head_dim)

        self.reset_parameters()

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

        mask = combine_masks(query, self.num_heads, attn_mask, key_padding_mask)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if hasattr(self, "rotary_emb"):
            q, k = self.rotary_emb(q, k)

        dropout = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(
            q, k, v, mask, is_causal=is_causal, dropout_p=dropout
        )
        attn = attn.permute(0, 2, 1, 3).reshape(N, -1, self.embed_size)
        return self.out_proj(attn)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)

        nn.init.constant_(self.q_proj.bias, 0.0)
        nn.init.constant_(self.k_proj.bias, 0.0)
        nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)


class SinePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5_000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


def combine_masks(
    q: Tensor,
    heads: int,
    attn_mask: Optional[Tensor],
    key_padding_mask: Optional[Tensor],
) -> Optional[Tensor]:
    """
    Combines the masks for attention and key padding.
    """
    N, sl = q.shape[0], q.shape[1]

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
        key_mask = key_padding_mask.view(N, 1, 1, sl).expand(N, heads, sl, sl)
        mask = mask + key_mask if mask is not None else key_mask

    return mask


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
