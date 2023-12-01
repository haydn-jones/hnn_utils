import torch
from torch import Tensor
import math


def exp_lin(p: Tensor, min_sigma: float = 0.0, max_sigma: float = 1.0):
    """
    More stable parameterization of standard deviation, constrained to be in [min_sigma, max_sigma]
    See https://arxiv.org/abs/2106.13739 eqn. 18 for details
    """
    pplus = p.clamp_min(0.0)
    pneg = p.clamp_max(0.0)
    mask = p <= 0

    sigma = pneg.exp() * mask + (pplus + 1.0) * (~mask)
    logsigma = pneg * mask + (pplus + 1.0).log() * (~mask)

    sigma = sigma.clamp(min=min_sigma, max=max_sigma)
    logsigma = logsigma.clamp(
        min=math.log(min_sigma) if min_sigma > 0.0 else -torch.inf,
        max=math.log(max_sigma),
    )

    return (
        sigma.type_as(p),
        logsigma.type_as(p),
    )


def gaussian_kl(
    mu1: Tensor,
    logsigma1: Tensor,
    mu2: Tensor,
    logsigma2: Tensor,
):
    """
    More numerically stable diagonal Gaussian KL (dont want to exp then log)
    Make sure to pass in log(sigma), not sigma
    """
    return (
        0.5
        * ((2.0 * logsigma1).exp() + (mu1 - mu2).square())
        * (-2.0 * logsigma2).exp()
        + logsigma2
        - logsigma1
        - 0.5
    )
