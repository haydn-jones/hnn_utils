from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from hnn_utils.nn.ALiBi import ALiBi
from hnn_utils.nn.normalization import LayerNorm
from hnn_utils.nn.ffn import FFNSwiGLU

try:
    from flash_attn import flash_attn_kvpacked_func, flash_attn_qkvpacked_func

    FLASH_AVAILABLE = True
except ImportError as e:
    print("Flash not available, using PyTorch implementation: ", e)
    FLASH_AVAILABLE = False


class TransformerEncoder(nn.Module):
    """
    Initializes a TransformerEncoder instance.

    Args:
        *encoder_layers (nn.Module): Variable number of encoder layers.

    """

    def __init__(self, *encoder_layers: nn.Module) -> None:
        super().__init__()

        self.layers = nn.ModuleList(list(*encoder_layers))

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_pad_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        output = src

        for layer in self.layers:
            output = layer(output, mask, src_pad_mask, is_causal=is_causal)

        return output


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder module.

    Args:
        *decoder_layers: Variable number of decoder layers.

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
        mem: Tensor,
        tgt_mask: Optional[Tensor] = None,
        mem_mask: Optional[Tensor] = None,
        tgt_pad_mask: Optional[Tensor] = None,
        mem_pad_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        mem_is_causal: bool = False,
    ) -> Tensor:
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                mem,
                tgt_mask,
                mem_mask,
                tgt_pad_mask,
                mem_pad_mask,
                tgt_is_causal,
                mem_is_causal,
            )

        return output


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer.

    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        dim_feedforward (int, optional): The dimension of the feedforward network model. Default is 2048.
        dropout (float, optional): The dropout value. Default is 0.1.
        self_alibi (bool, optional): Whether to use self-alibi attention. Default is True.
        norm_cls (nn.Module, optional): The normalization layer class. Default is LayerNorm.
        ffn_cls (nn.Module, optional): The feedforward network class. Default is FFNSwiGLU.
        norm_first (bool, optional): Whether to apply normalization before the self-attention block. Default is True.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        self_alibi: bool = True,
        norm_cls: nn.Module = LayerNorm,
        ffn_cls: nn.Module = FFNSwiGLU,
        norm_first: bool = True,
    ) -> None:
        super().__init__()

        self.self_attn = SelfAttention(
            d_model, nhead, dropout=dropout, use_alibi=self_alibi
        )

        self.ff_block = ffn_cls(d_model, dim_feedforward)

        self.norm1 = norm_cls(d_model)
        self.norm2 = norm_cls(d_model)

        self.dropout = nn.Dropout(dropout)

        self.norm_first = norm_first

        self.apply(_reset_parameters)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_pad_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_pad_mask, is_causal=is_causal)  # fmt: skip
            x = x + self.ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_pad_mask, is_causal=is_causal))  # fmt: skip
            x = self.norm2(x + self.ff_block(x))

        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        pad_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x, attn_mask=attn_mask, pad_mask=pad_mask, is_causal=is_causal
        )
        return self.dropout(x)


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer.

    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        dim_feedforward (int, optional): The dimension of the feedforward network model. Default is 2048.
        dropout (float, optional): The dropout value. Default is 0.1.
        self_alibi (bool, optional): Whether to use self-alibi attention. Default is True.
        cross_alibi (bool, optional): Whether to use cross-alibi attention. Default is False.
        ffn_cls (nn.Module, optional): The feedforward network class. Default is FFNSwiGLU.
        norm_cls (nn.Module, optional): The normalization class. Default is LayerNorm.
        norm_first (bool, optional): Whether to apply normalization before each sub-layer. Default is False.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        self_alibi: bool = True,
        cross_alibi: bool = False,
        ffn_cls: nn.Module = FFNSwiGLU,
        norm_cls: nn.Module = LayerNorm,
        norm_first: bool = False,
    ):
        super().__init__()

        self.self_attn = SelfAttention(
            d_model, nhead, dropout=dropout, use_alibi=self_alibi
        )
        self.cross_attn = CrossAttention(
            d_model, nhead, dropout=dropout, use_alibi=cross_alibi
        )

        self.ff_block = ffn_cls(d_model, dim_feedforward)

        self.norm1 = norm_cls(d_model)
        self.norm2 = norm_cls(d_model)
        self.norm3 = norm_cls(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm_first = norm_first

        self.apply(_reset_parameters)

    def forward(
        self,
        tgt: Tensor,
        mem: Tensor,
        tgt_mask: Optional[Tensor] = None,
        mem_mask: Optional[Tensor] = None,
        tgt_pad_mask: Optional[Tensor] = None,
        mem_pad_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        mem_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_pad_mask, is_causal=tgt_is_causal)  # fmt: skip
            x = x + self._mha_block(self.norm2(x), mem, mem_mask, mem_pad_mask, mem_is_causal)  # fmt: skip
            x = x + self.ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_pad_mask, is_causal=tgt_is_causal))  # fmt: skip
            x = self.norm2(x + self._mha_block(x, mem, mem_mask, mem_pad_mask, mem_is_causal))  # fmt: skip
            x = self.norm3(x + self.ff_block(x))
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        pad_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(x, attn_mask=attn_mask, pad_mask=pad_mask, is_causal=is_causal)  # fmt: skip
        return self.dropout1(x)

    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        pad_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.cross_attn(x, mem, attn_mask=attn_mask, pad_mask=pad_mask, is_causal=is_causal)  # fmt: skip
        return self.dropout2(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        heads: int,
        dropout: float = 0.0,
        use_alibi: bool = True,
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
        self.kv_proj = nn.Linear(self.embed_size, self.embed_size * 2)

        self.out_proj = nn.Linear(heads * self.head_dim, embed_dim)

        if use_alibi:
            self.alibi = ALiBi(heads)

    def forward(
        self,
        query: Tensor,
        kv: Tensor,
        pad_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        if can_flash(query, pad_mask, attn_mask):
            return self.flash_forward(query, kv, is_causal)

        q = self.q_proj(query)
        k, v = self.kv_proj(kv).chunk(2, dim=-1)

        q = rearrange(q, "... n (h d) -> ... h n d", h=self.num_heads)
        k = rearrange(k, "... n (h d) -> ... h n d", h=self.num_heads)
        v = rearrange(v, "... n (h d) -> ... h n d", h=self.num_heads)

        mask = combine_masks(attn_mask, pad_mask, self.num_heads, query.dtype)

        if hasattr(self, "alibi"):
            bias = self.alibi(q, k)
            mask = mask + bias if mask is not None else bias

        dropout = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(
            q, k, v, mask, is_causal=is_causal, dropout_p=dropout
        )

        attn = rearrange(attn, "... h n d -> ... n (h d)")
        return self.out_proj(attn)

    def flash_forward(self, query: Tensor, kv: Tensor, is_causal: bool) -> Tensor:
        q = self.q_proj(query)
        kv = self.kv_proj(kv)

        q = rearrange(q, "... (h d) -> ... h d", h=self.num_heads)
        kv = rearrange(kv, "... (n h d) -> ... n h d", h=self.num_heads, n=2)

        slopes = self.alibi.slopes.float().squeeze() if hasattr(self, "alibi") else None

        dtype = query.dtype

        attn = flash_attn_kvpacked_func(
            q=q.bfloat16(),
            kv=kv.bfloat16(),
            causal=is_causal,
            alibi_slopes=slopes,
        ).type(dtype)

        attn = rearrange(attn, "... h d -> ... (h d)")
        return self.out_proj(attn)


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        heads: int,
        dropout: float = 0.0,
        use_alibi: bool = True,
    ):
        super().__init__()

        self.embed_size = embed_dim
        self.num_heads = heads
        self.head_dim = embed_dim // heads
        self.dropout = dropout

        assert (
            self.head_dim * heads == embed_dim
        ), "Embedding size needs to be divisible by heads"

        self.qkv_proj = nn.Linear(self.embed_size, self.embed_size * 3)

        self.out_proj = nn.Linear(heads * self.head_dim, embed_dim)

        if use_alibi:
            self.alibi = ALiBi(heads)

    def forward(
        self,
        x: Tensor,
        pad_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        if can_flash(x, pad_mask, attn_mask):
            return self.flash_forward(x, is_causal)

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        q = rearrange(q, "... n (h d) -> ... h n d", h=self.num_heads)
        k = rearrange(k, "... n (h d) -> ... h n d", h=self.num_heads)
        v = rearrange(v, "... n (h d) -> ... h n d", h=self.num_heads)

        mask = combine_masks(attn_mask, pad_mask, self.num_heads, q.dtype)

        if hasattr(self, "alibi"):
            bias = self.alibi(q, k)
            mask = mask + bias if mask is not None else bias

        dropout = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(
            q, k, v, mask, is_causal=is_causal, dropout_p=dropout
        )

        attn = rearrange(attn, "... h n d -> ... n (h d)")
        return self.out_proj(attn)

    def flash_forward(self, x: Tensor, is_causal: bool) -> Tensor:
        qkv = self.qkv_proj(x)
        qkv = rearrange(qkv, "... (n h d) -> ... n h d", h=self.num_heads, n=3)

        slopes = None
        if hasattr(self, "alibi"):
            slopes = self.alibi.slopes.float().squeeze()

        dtype = x.dtype

        attn = flash_attn_qkvpacked_func(
            qkv=qkv.bfloat16(), causal=is_causal, alibi_slopes=slopes
        ).type(dtype)

        attn = rearrange(attn, "... h d -> ... (h d)")
        return self.out_proj(attn)


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
            return torch.zeros_like(x, dtype=dtype).masked_fill_(x, -torch.inf)
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
    mask = torch.full((embed.shape[1], embed.shape[1]), -torch.inf, device=embed.device, dtype=embed.dtype)  # fmt: skip
    mask = torch.triu(mask, diagonal=1)
    return mask


def can_flash(
    x: Tensor,
    pad_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
):
    return (
        pad_mask is None
        and attn_mask is None
        and FLASH_AVAILABLE
        and x.device.type == "gpu"
    )
