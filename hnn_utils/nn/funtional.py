import torch
from torch import Tensor
import math


def upbounded_omega(p: Tensor, omega: float = 1.0):
    """
    More stable parameterization of variance, in addition the variance is capped at omega (default 1).
    See https://arxiv.org/abs/2106.13739 eqn. 12 for details
    """
    mask = p <= 0

    ppos = p.clamp(0.0).mul(-1.0).exp()
    pneg = p.clamp_max(0.0).exp()

    sigma = torch.where(
        mask,
        (omega / 2) * pneg,
        (omega / 2) * (2.0 - ppos),
    )

    eps = 1e-5
    lo = math.log(omega / 2.0)
    logsigma = torch.where(
        mask,
        p + lo,
        (2.0 - ppos).clamp_min(eps).log() + lo,
    )
    return sigma.type_as(p), logsigma.type_as(p)


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
