from typing import Optional

import torch
from torch import Tensor


def exp_lin(p: Tensor, min_sigma: float = 0.0, max_sigma: float = 1.0) -> Tensor:
    """
    More stable parameterization of standard deviation, constrained to be in [min_sigma, max_sigma]
    See https://arxiv.org/abs/2106.13739 eqn. 18 for details
    """
    pplus = (-p.clamp_min(0.0)).exp()
    pneg = p.clamp_max(0.0).exp()

    sigma = torch.where(
        p <= 0,
        0.5 * pneg,
        0.5 * (2.0 - pplus),
    )
    sigma = sigma * (max_sigma - min_sigma) + min_sigma
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
    var_ratio = (sigmap / sigmaq).square()
    t1 = ((mup - muq) / sigmaq).square()
    return 0.5 * (var_ratio + t1 - 1.0 - var_ratio.log())


def cross_entropy(
    logits: Tensor,
    labels: Tensor,
    ignore: Optional[Tensor] = None,
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

    if ignore is not None:
        ce = torch.where(
            ignore,
            0.0,
            ce,
        )

    return ce
