from __future__ import annotations

import math

import torch.nn as nn
import torch.optim as optim

from .base import BaseScheduler
from .parameter import SchedulerParameter


class StepScheduler(BaseScheduler):
    """Extended version of pytorch schedular: StepLR

    ----------------------------------------------------------------------------
    learning rate(lr) is constant as lr_max for each cycle
    ----------------------------------------------------------------------------
    Note: lr cannot be smaller than lr_min

    lr_min: Minimum learning rate. Extracted from initial learning rate of optimizer
    lr_max: Maximum learning rate
    period: Number of epochs for each cycle
    lr_max_mult: Decreasing rate of lr_max per each cycle
    period_mult: Increasing rate of period per each cycle
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        lr_max: float,
        period: int,
        lr_max_mult: float = 1.0,
        period_mult: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        # Check input
        assert period > 0, f"At step schedular, {period=}"
        assert lr_max > 0, f"At step schedular, {lr_max=}"
        assert period_mult >= 1, f"At step schedular, {period_mult=}"
        assert lr_max_mult > 0, f"At step schedular, {lr_max_mult=}"

        # Constant properties of schedular
        self._lr_max_mult = float(lr_max_mult)
        self._period_mult = float(period_mult)

        # Properties varying per cycle
        self._cycle_start = 0  # Epoch when current cycle is started
        self._period = period  # Period of current cycle
        self._lr_max = lr_max  # max learning rate used for current cycle

        # Current location of schedular since cycle start
        # Since super().__init__ calls self.step once, it should be -1 at first
        self.__t = -1

        super().__init__(optimizer, last_epoch)

    @classmethod
    def from_hp(
        cls,
        hp: SchedulerParameter,
        optimizer: optim.Optimizer | None = None,
        last_epoch: int = -1,
    ) -> StepScheduler:
        if optimizer is None:
            optimizer = optim.SGD(nn.Linear(1, 1).parameters(), lr=hp.lr)
        return cls(
            optimizer,
            hp.lr_max,
            hp.period,
            hp.lr_max_mult,
            hp.period_mult,
            last_epoch,
        )

    def step(self, epoch: int | float | None = None):
        """
        epoch: epoch + iteration/(tot iteration per epoch)
        If Nothing is given, increase last_epoch
        """
        if epoch is None:
            # Input epoch is None: step up 1 epoch
            self.__t += 1
            self.last_epoch += 1
        else:
            # Input epoch is given
            self.__t = epoch - self._cycle_start
            self.last_epoch = math.floor(epoch)

        # If current location is bigger than period of cycle, update cycle
        if self.__t >= self._period:
            self._update_cycle()

        # Set learning rate of optimizer
        for param_group, lr in zip(self.optimizer.param_groups, self._get_lr()):
            param_group["lr"] = lr

    def _get_lr(self) -> list[float]:
        return [max(self._lr_max, base_lr) for base_lr in self.base_lrs]

    def _update_cycle(self) -> None:
        # Start of current cycle is now increased by period of cycle
        self._cycle_start += self._period

        # Reset current location of new cycle
        self.__t -= self._period

        # Period of cycle is updated by multiplier
        self._period = int(self._period * self._period_mult)

        # Maximum learning rate is updated by multipler
        self._lr_max *= self._lr_max_mult
