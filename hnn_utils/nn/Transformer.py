import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from hnn_utils.nn.ffn import FFNSwiGLU
from hnn_utils.nn.normalization import LayerNorm
from hnn_utils.nn.RoPE import apply_rotary_emb


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
        mask: Tensor | None = None,
        src_pad_mask: Tensor | None = None,
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
        tgt_mask: Tensor | None = None,
        mem_mask: Tensor | None = None,
        tgt_pad_mask: Tensor | None = None,
        mem_pad_mask: Tensor | None = None,
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
        norm_cls: type[nn.Module] = LayerNorm,
        ffn_cls: type[nn.Module] = FFNSwiGLU,
        norm_first: bool = True,
    ) -> None:
        super().__init__()

        self.self_attn = SelfAttention(d_model, nhead, dropout=dropout)

        self.ff_block = ffn_cls(d_model, dim_feedforward)

        self.norm1 = norm_cls(d_model)
        self.norm2 = norm_cls(d_model)

        self.dropout = nn.Dropout(dropout)

        self.norm_first = norm_first

        self.apply(_reset_parameters)

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_pad_mask: Tensor | None = None,
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
        attn_mask: Tensor | None,
        pad_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(x, attn_mask=attn_mask, pad_mask=pad_mask, is_causal=is_causal)
        return self.dropout(x)


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer.

    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        dim_feedforward (int, optional): The dimension of the feedforward network model. Default is 2048.
        dropout (float, optional): The dropout value. Default is 0.1.
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
        ffn_cls: type[nn.Module] = FFNSwiGLU,
        norm_cls: type[nn.Module] = LayerNorm,
        norm_first: bool = False,
    ):
        super().__init__()

        self.self_attn = SelfAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = CrossAttention(d_model, nhead, dropout=dropout)

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
        tgt_mask: Tensor | None = None,
        mem_mask: Tensor | None = None,
        tgt_pad_mask: Tensor | None = None,
        mem_pad_mask: Tensor | None = None,
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
        attn_mask: Tensor | None,
        pad_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(x, attn_mask=attn_mask, pad_mask=pad_mask, is_causal=is_causal)  # fmt: skip
        return self.dropout1(x)

    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Tensor | None,
        pad_mask: Tensor | None,
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
    ):
        super().__init__()

        self.embed_size = embed_dim
        self.num_heads = heads
        self.head_dim = embed_dim // heads
        self.dropout = dropout

        assert self.head_dim * heads == embed_dim, "Embedding size needs to be divisible by heads"

        self.q_proj = nn.Linear(self.embed_size, self.embed_size)
        self.kv_proj = nn.Linear(self.embed_size, self.embed_size * 2)

        self.out_proj = nn.Linear(heads * self.head_dim, embed_dim)

    def forward(
        self,
        query: Tensor,
        kv: Tensor,
        pad_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        assert not is_causal or (
            pad_mask is None and attn_mask is None
        ), "is_causal not supported with padding or attention masks."

        q = self.q_proj(query)
        k, v = self.kv_proj(kv).chunk(2, dim=-1)

        q = rearrange(q, "... n (h d) -> ... h n d", h=self.num_heads)
        k = rearrange(k, "... n (h d) -> ... h n d", h=self.num_heads)
        v = rearrange(v, "... n (h d) -> ... h n d", h=self.num_heads)

        mask = combine_masks(attn_mask, pad_mask, self.num_heads, query.dtype)

        dropout = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=is_causal,
            dropout_p=dropout,
        )

        attn = rearrange(attn, "... h n d -> ... n (h d)")
        return self.out_proj(attn)


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_size = embed_dim
        self.num_heads = heads
        self.head_dim = embed_dim // heads
        self.dropout = dropout

        assert self.head_dim * heads == embed_dim, "Embedding size needs to be divisible by heads"

        self.qkv_proj = nn.Linear(self.embed_size, self.embed_size * 3)

        self.out_proj = nn.Linear(heads * self.head_dim, embed_dim)

    def forward(
        self,
        x: Tensor,
        pad_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        assert not is_causal or (
            pad_mask is None and attn_mask is None
        ), "is_causal not supported with padding or attention masks."

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        q = rearrange(q, "... n (h d) -> ... h n d", h=self.num_heads)
        k = rearrange(k, "... n (h d) -> ... h n d", h=self.num_heads)
        v = rearrange(v, "... n (h d) -> ... h n d", h=self.num_heads)

        mask = combine_masks(attn_mask, pad_mask, self.num_heads, q.dtype)

        q, k = apply_rotary_emb(q, k)

        if is_causal:
            cm = causal_mask(x)
            mask = mask + cm if mask is not None else cm

        dropout = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, mask, dropout_p=dropout)

        attn = rearrange(attn, "... h n d -> ... n (h d)")
        return self.out_proj(attn)


def combine_masks(
    attn_mask: Tensor | None,
    pad_mask: Tensor | None,
    heads: int = 1,
    dtype: torch.dtype | None = None,
) -> Tensor | None:
    """
    Combines the masks for attention and padding.
    """

    dtype = dtype or torch.get_default_dtype()

    def floatify(x: None | Tensor) -> Tensor | None:
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
    Creates a causal mask for self-attention. Expects sequence length to be the 2nd to last dimension.
    """
    # Sequence length is 2nd to last dimension
    SL = embed.shape[-2]
    mask = torch.full((SL, SL), -torch.inf, device=embed.device, dtype=embed.dtype)  # fmt: skip
    mask = torch.triu(mask, diagonal=1)
    return mask
