import torch
from torch import Tensor
import math


def upbounded_omega(p: Tensor):
    """
    More stable parameterization of variance, in addition the variance is capped at 1.
    See https://arxiv.org/abs/2106.13739 eqn. 12 for details
    """
    mask = p <= 0

    pexp = p.exp()
    npexp = (-p).exp()

    sigma = torch.where(
        mask,
        0.5 * pexp,
        0.5 * (2.0 - npexp),
    )

    lo = math.log(0.5)
    # Clamp to avoid log(0), 1e-8 is not representable in fp16 (it is in bfloat16 though), so we use 1e-5 to be safe
    # Even though the NaNs/nonfinites are masked out, they still can cause issues with gradient propagation
    logsigma = torch.where(
        mask,
        p + lo,
        (2.0 - npexp).clamp_min(1e-5).log() + lo,
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
