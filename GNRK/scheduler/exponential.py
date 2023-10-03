from __future__ import annotations

import torch.nn as nn
import torch.optim as optim

from .parameter import SchedulerParameter
from .step import StepScheduler


class ExponentialScheduler(StepScheduler):
    """Extended version of pytorch schedular: ExponentialLR

    ----------------------------------------------------------------------------
    learning rate(lr) = lr_max at the start
    For each epoch, lr decreased as lr_max_mult
    ----------------------------------------------------------------------------
    Note: lr cannot be smaller than lr_min

    lr_min: Minimum learning rate. Extracted from initial learning rate of optimizer
    lr_max: Maximum learning rate
    lr_max_mult: Decreasing rate of lr_max per each cycle
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        lr_max: float,
        lr_max_mult: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(
            optimizer,
            lr_max=lr_max,
            period=1,
            lr_max_mult=lr_max_mult,
            period_mult=1,
            last_epoch=last_epoch,
        )

    @classmethod
    def from_hp(
        cls,
        hp: SchedulerParameter,
        optimizer: optim.Optimizer | None = None,
        last_epoch: int = -1,
    ) -> ExponentialScheduler:
        if optimizer is None:
            optimizer = optim.SGD(nn.Linear(1, 1).parameters(), lr=hp.lr)

        return cls(optimizer, hp.lr_max, hp.lr_max_mult, last_epoch)
