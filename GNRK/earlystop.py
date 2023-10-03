from __future__ import annotations

import sys
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class EarlystopParameter:
    patience: int | None
    delta: float


class Earlystop:
    """Early stops training if validation loss doesn't improve 'delta' after 'patience' epochs"""

    def __init__(
        self,
        patience: int | None = 10,
        loss_delta: float = 0.0,
        descending: bool = True,
        best_val: torch.Tensor | float | None = None,
    ) -> None:
        """
        patience: How many epochs to wait after validation loss is improved.
        loss_delta: Minimum change of validation loss to regard as improved
        descending: If true, lower validation is better. Otherwise, higher is better
        best_val: initial best validation score
        """
        # * Properties of early stop instance
        self.patience = sys.maxsize if patience is None else patience
        self.loss_delta = loss_delta
        self.descending = 1 if descending else -1

        # * Current state of early stop
        if best_val is None:
            best_val = float("inf") if descending else -float("inf")
        self.best_val = float(best_val)
        self.counter = 0  # Early stop counter
        self.abort = False  # If true, stop training
        self.is_best = False  # If current state is best

    @classmethod
    def from_hp(
        cls,
        hp: EarlystopParameter,
        descending: bool = True,
        best_val: torch.Tensor | float | None = None,
    ) -> Earlystop:
        return cls(hp.patience, hp.delta, descending, best_val)

    def __call__(self, val: float | torch.Tensor) -> None:
        """
        Do early stop update about input value
        self.is_best will be updated
        self.abort will be updated
        Args
            val: current validation
        """
        # * Change validation in descending order
        descending_val = self.descending * float(val)
        descending_best_val = self.descending * self.best_val

        # * Check if this validation is best until now
        self.is_best = descending_val < (descending_best_val - self.loss_delta)
        self.best_val = self.descending * min(descending_val, descending_best_val)

        # * If best, reset counter to zero. Otherwise, increase 1
        self.counter = (self.counter + 1) * (not self.is_best)

        # * Check whether to abort
        self.abort = self.counter >= self.patience

    def resume(
        self, counter: int | torch.Tensor, best_val: float | torch.Tensor
    ) -> None:
        """
        Resume from previous train
        counter: Last early stop counter
        best_val: best validation value
        """
        # * Cutoff history and update best epoch, best_val
        self.best_val = float(best_val)
        self.counter = int(counter)
        self.is_best = self.counter == 0
        self.abort = False
