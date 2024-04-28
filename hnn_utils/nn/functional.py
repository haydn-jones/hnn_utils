import torch
from torch import Tensor


def exp_lin(p, min_sigma: float = 0.0, max_sigma: float = 1.0):
    """
    More stable parameterization of standard deviation, constrained to be in [min_sigma, max_sigma]
    See https://arxiv.org/abs/2106.13739 eqn. 18 for details
    """
    pplus = (-p.clamp_min(0.0)).exp()
    pneg = p.clamp_max(0.0).exp()

    sigma = torch.where(
        p <= 0,
        pneg,
        (2.0 - pplus),
    )

    sigma = 0.5 * sigma * (max_sigma - min_sigma) + min_sigma
    return sigma.type_as(p)


def gaussian_kl(
    mup: Tensor,
    sigmap: Tensor,
    muq: Tensor,
    sigmaq: Tensor,
):
    """
    KL divergence between two diagonal Gaussians KL(p || q)
    """
    return (
        0.5 * (sigmap.square() + (mup - muq).square()) * sigmaq.reciprocal().square()
        + sigmaq.log()
        - sigmap.log()
        - 0.5
    )


def gaussian_kl_standard_normal(
    mup: Tensor,
    sigmap: Tensor,
) -> Tensor:
    """
    KL(p || N(0, 1))
    """
    var = sigmap.square()
    return 0.5 * (var + mup.square() - 1.0 - var.log())


def cross_entropy(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """
    Cross entropy loss, with optional ignore mask.
    Shapes:
        logits: [..., num_classes]
        labels: [...]
        ignore: [...]
    """
    with torch.no_grad():
        logits_max = torch.amax(logits, dim=-1, keepdim=True)

    logits = logits - logits_max
    label_logits = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    log_normalizers = logits.logsumexp(dim=-1)
    ce = log_normalizers - label_logits

    ce = torch.where(labels == ignore_index, 0.0, ce)
    return ce


def causal_mask(embed: Tensor) -> Tensor:
    """
    Creates a causal mask for self-attention.
    """
    mask = torch.full(
        (embed.shape[1], embed.shape[1]),
        -torch.inf,
        device=embed.device,
        dtype=embed.dtype,
    )
    mask = torch.triu(mask, diagonal=1)
    return mask
