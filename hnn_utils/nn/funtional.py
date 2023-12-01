import torch
from torch import Tensor
import math


def upbounded_omega(p: Tensor, min_sigma: float = 0.01, max_sigma: float = 1.0):
    """
    More stable parameterization of variance, in addition the variance is capped at omega (default 1).
    See https://arxiv.org/abs/2106.13739 eqn. 12 for details
    """
    mask = p <= 0

    ppos = p.clamp_min(0.0).mul(-1.0).exp()
    pneg = p.clamp_max(0.0).exp()

    sigma = torch.where(
        mask,
        (max_sigma / 2) * pneg,
        (max_sigma / 2) * (2.0 - ppos),
    )

    lo = math.log(max_sigma / 2.0)
    logsigma = torch.where(
        mask,
        p + lo,
        (2.0 - ppos).log() + lo,
    )
    return sigma.clamp_min(min_sigma).type_as(p), logsigma.clamp_min(
        math.log(min_sigma)
    ).type_as(p)


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
