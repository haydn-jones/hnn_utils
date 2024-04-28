# Much of this is inspired by lucidrains ema-pytorch
# https://github.com/lucidrains/ema-pytorch

from enum import Enum, auto
from typing import Any, Dict, List, Optional

import lightning as L
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch.optim import Optimizer


class UpdateAfter(Enum):
    STEP = auto()
    EPOCH = auto()


class EMACallback(L.Callback):
    """
    Lightning callback to compute a moving average of a models parameters.
    You can easily load the ema

    Attributes:
        tau (float): The decay factor for EMA, should be between 0 and 1. Defaults to 0.9999.
        update_every_n_steps (int): Update the EMA weights every n steps. Defaults to 1 (every step).
        start_on_step (int): Start updating the EMA weights on this step. If None, start immediately.
        start_on_epoch (int): Start updating the EMA weights on this epoch. If None, start immediately.
        excluded_parameters (list of str): List of model parameter names to exclude from EMA.
        use_for_val (bool): If True, use the EMA weights for validation. Defaults to False.
    """

    def __init__(
        self,
        tau: float = 0.9999,
        update_every_n_steps: int = 1,
        start_on_step: Optional[int] = None,
        start_on_epoch: Optional[int] = None,
        excluded_parameters: Optional[List[str]] = None,
        use_for_val: bool = False,
    ):
        super().__init__()

        self.tau = tau

        self.every_n_steps = update_every_n_steps

        if start_on_step is not None and start_on_epoch is not None:
            raise ValueError(
                "Only one of `update_after_n_steps` and `update_after_n_epochs` can be set."
            )

        if start_on_step is not None:
            self.update_start = UpdateAfter.STEP
            self.update_time = start_on_step
        elif start_on_epoch is not None:
            self.update_start = UpdateAfter.EPOCH
            self.update_time = start_on_epoch
        else:
            self.update_start = UpdateAfter.STEP
            self.update_time = 0

        self.excluded_parameters = (
            set(excluded_parameters) if excluded_parameters else set()
        )

        self.ema_weights = None

        # We track the step instead of using the trainer's global_step because
        # multiple optimizers can increment the global step multiple times per batch
        self.step = 0

        # Caching the original state dict for validation
        self._cached_sd = None
        self.use_for_val = use_for_val

        # If we restore from a checkpoint, we need to know if we've transferred the weights
        # to the device yet or not
        self.on_device = False

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer
    ) -> None:
        if not self.should_update(trainer):
            return

        self.step += 1

        self._init(trainer, pl_module)

        for name, param in pl_module.named_parameters():
            if name in self.excluded_parameters:
                continue

            # If its not a float just copy it over
            if not param.is_floating_point() and not param.is_complex():
                self.ema_weights[name].copy_(param.data)
                continue

            self.ema_weights[name].lerp_(param.data, 1.0 - self.tau)

    def _init(self, trainer: Trainer, pl_module: LightningModule):
        if self.ema_weights is None:
            self.ema_weights = sd_copy(pl_module.state_dict())
            self.step = 0
            self.on_device = True
        elif not self.on_device:
            self.ema_weights = {
                k: v.to(pl_module.device) for k, v in self.ema_weights.items()
            }
            self.on_device = True

    def should_update(self, trainer: Trainer):
        if self.step % self.every_n_steps != 0:
            return False

        if self.update_start == UpdateAfter.STEP:
            return self.step >= self.update_time
        elif self.update_start == UpdateAfter.EPOCH:
            return trainer.current_epoch >= self.update_time

    @staticmethod
    def load_ema_weights(pl_module: LightningModule, ckpt_path: str, **kwargs) -> None:
        """Load the EMA weights from a checkpoint"""

        sd = torch.load(ckpt_path)
        ema_weights = sd["callbacks"]["EMACallback"]["state_dict"]
        pl_module.load_state_dict(ema_weights, **kwargs)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule):
        """Swap the weights for validation"""
        if self.use_for_val and self.ema_weights is not None:
            self._cached_sd = sd_copy(pl_module.state_dict())
            pl_module.load_state_dict(self.ema_weights)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        """Swap the weights back after validation"""
        if self.use_for_val and self._cached_sd is not None:
            pl_module.load_state_dict(self._cached_sd)
            self._cached_sd = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.strategy, (FSDPStrategy, DeepSpeedStrategy, DDPStrategy)):
            raise MisconfigurationException(
                "I haven't tested this with FSDP, DeepSpeed, or DDP. Don't use it."
            )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "state_dict": self.ema_weights,
            "step": self.step,
            "update_start": self.update_start,
            "update_time": self.update_time,
            "excluded_parameters": self.excluded_parameters,
            "use_for_val": self.use_for_val,
            "tau": self.tau,
            "every_n_steps": self.every_n_steps,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.ema_weights = state_dict["state_dict"]
        self.step = state_dict["step"]
        self.update_start = state_dict["update_start"]
        self.update_time = state_dict["update_time"]
        self.excluded_parameters = state_dict["excluded_parameters"]
        self.use_for_val = state_dict["use_for_val"]
        self.tau = state_dict["tau"]
        self.every_n_steps = state_dict["every_n_steps"]


def sd_copy(state_dict):
    """Probably faster than deepcopy idk"""
    return {k: v.clone().detach() for k, v in state_dict.items()}
