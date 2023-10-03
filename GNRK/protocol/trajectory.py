from typing import Protocol

import torch


class IsDivergingProtocol(Protocol):
    tolerance: float
    def __call__(self, trajectory: torch.Tensor) -> torch.BoolTensor:
        ...