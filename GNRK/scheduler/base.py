from __future__ import annotations

from abc import ABC, abstractmethod

import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from .parameter import SchedulerParameter


class BaseScheduler(_LRScheduler, ABC):
    # Variables defined at _LRScheduler
    optimizer: optim.Optimizer
    last_epoch: int
    base_lrs: list[float]

    def __init__(self, optimizer: optim.Optimizer, last_epoch: int) -> None:
        super().__init__(optimizer, last_epoch)

    @classmethod
    @abstractmethod
    def from_hp(
        cls,
        hp: SchedulerParameter,
        optimizer: optim.Optimizer | None = None,
        last_epoch: int = -1,
    ) -> BaseScheduler:
        ...

    @abstractmethod
    def step(self, epoch: int | float | None = None) -> None:
        ...

    @abstractmethod
    def _get_lr(self) -> list[float]:
        ...

    def resume(self, epoch: int) -> None:
        """load scheduler based on input epoch"""
        if epoch > 0:
            for _ in range(epoch):
                self.step()

        # Set learning rate of optimizer
        for param_group, lr in zip(self.optimizer.param_groups, self._get_lr()):
            param_group["lr"] = lr

    @classmethod
    def get_lr(
        cls, hp: SchedulerParameter, max_epoch: int, num_batch: int = 0
    ) -> tuple[list[float], list[float]]:
        scheduler = cls.from_hp(hp)
        optimizer = scheduler.optimizer

        epochs: list[float] = []
        lrs: list[float] = []

        if num_batch == 0:
            for epoch in range(max_epoch):
                epochs.append(epoch)
                lrs.append(optimizer.param_groups[0]["lr"])
                scheduler.step()
            return epochs, lrs

        for epoch in range(max_epoch):
            for batch in range(num_batch):
                epochs.append(epoch + batch / num_batch)
                lrs.append(optimizer.param_groups[0]["lr"])
                scheduler.step(epoch + batch / num_batch)
        return epochs, lrs

    @staticmethod
    def get_updated_epochs(hp: SchedulerParameter, max_epoch: int) -> list[int]:
        updated_epoch = [0]
        idx = 0
        while updated_epoch[-1] <= max_epoch:
            period = int(hp.period * hp.period_mult**idx)
            updated_epoch.append(updated_epoch[-1] + period + hp.warmup)
            idx += 1
        return updated_epoch[1:-1]
