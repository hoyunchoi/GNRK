import string
from typing import Any

import numpy as np
import torch
import torch.optim as optim


def dummy_print(_: str) -> None:
    return


class DummyGradScaler:
    def __init__(self) -> None:
        return

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer: optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, _: dict[str, torch.Tensor]) -> None:
        return


class DummySampler:
    def set_epoch(self, _: int) -> None:
        return


class DummyWandbRun:
    def log(self, _: dict[str, Any]) -> None:
        return

    @property
    def summary(self) -> dict[str, Any]:
        return {}

    def finish(self) -> None:
        return

    @property
    def name(self) -> str:
        return "".join(
            np.random.choice(list(string.ascii_lowercase + string.digits), 8)
        )
