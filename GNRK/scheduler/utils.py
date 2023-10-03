import torch.nn as nn
import torch.optim as optim

from .base import BaseScheduler
from .cosine import CosineScheduler
from .exponential import ExponentialScheduler
from .parameter import SchedulerParameter
from .step import StepScheduler


def get_scheduler(
    hp: SchedulerParameter, optimizer: optim.Optimizer | None = None
) -> BaseScheduler:
    if optimizer is None:
        # If optimizer is not given, create dummy optimizer
        optimizer = optim.SGD(nn.Linear(1, 1).parameters(), lr=hp.lr)

    match hp.name:
        case "cosine":
            return CosineScheduler.from_hp(hp, optimizer)
        case "step":
            return StepScheduler.from_hp(hp, optimizer)
        case "exponential":
            return ExponentialScheduler.from_hp(hp, optimizer)
        case _:
            raise ValueError(f"Invalid scheduler name: {hp.name=}")
