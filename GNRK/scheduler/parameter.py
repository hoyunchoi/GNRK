import warnings
from dataclasses import dataclass
from typing import Literal

SCHEDULER = Literal["cosine", "step", "exponential"]


@dataclass(slots=True)
class SchedulerParameter:
    name: SCHEDULER
    lr: float
    lr_max: float
    period: int
    warmup: int
    lr_max_mult: float
    period_mult: float

    def __post_init__(self) -> None:
        assert self.lr <= self.lr_max

        # scheduler dependent variables
        if self.name == "step":
            if self.warmup != 0:
                warnings.warn(
                    "Step scheduler does not support warmup. warmup is ignored."
                )
                self.warmup = 0

        elif self.name == "exponential":
            if self.warmup != 0:
                warnings.warn(
                    "Exponential scheduler does not support warmup. warmup is ignored."
                )
                self.warmup = 0

            if self.period != 1:
                warnings.warn(
                    "Exponential scheduler does not have period. period is ignored."
                )
                self.period = 1
            if self.period_mult != 1.0:
                warnings.warn(
                    "Exponential scheduler does not have period. period_mult is"
                    " ignored."
                )
                self.period_mult = 1.0
