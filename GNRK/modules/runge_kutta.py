from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from GNRK.protocol import ApproximatorProtocol

BUTCHER_TABLEAU: dict[str, list[torch.Tensor]] = {
    "RK1": [torch.tensor([]), torch.tensor([1.0])],
    "RK2": [torch.tensor([]), torch.tensor([1.0]), torch.tensor([1.0, 1.0])],
    "RK4": [
        torch.tensor([]),
        torch.tensor([0.5]),
        torch.tensor([0.0, 0.5]),
        torch.tensor([0.0, 0.0, 1.0]),
        torch.tensor([1.0, 2.0, 2.0, 1.0]),
    ],
}


class RungeKutta(nn.Module):
    def __init__(
        self,
        approximator: ApproximatorProtocol,
        butcher_tableau: str | list[torch.Tensor],
    ) -> None:
        """
        approximator: Differential approximator to be solved
        butcher_tableau: coefficients for Runge-Kutta method.
                         Could be selected from preset
        """
        super().__init__()
        self.approximator = approximator
        if isinstance(butcher_tableau, str):
            self.butcher_tableau = BUTCHER_TABLEAU[butcher_tableau.upper()]
        else:
            self.butcher_tableau = butcher_tableau

    @property
    def order(self) -> int:
        return len(self.butcher_tableau) - 1  # The last is for aggregation

    def to(self, device: torch.device, *args, **kwargs) -> RungeKutta:
        self.butcher_tableau = [
            coefficient.to(device) for coefficient in self.butcher_tableau
        ]
        return super().to(device=device, *args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: torch.LongTensor,
        dt: torch.Tensor,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
        glob_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (BN, state_feature), node states, will be updated
        edge_index: (2, BE), edge index of graph
        node_attr: (BN, node_feature), node features
        batch: (BN, ), index of batch where each node belongs
        dt: (BN, 1), dt for each data

        node_attr: (BN, node_feature)
        edge_attr: (BE, edge_feature)
        glob_attr: (B, glob_feature)
        """
        approx: Callable[[torch.Tensor], torch.Tensor] = partial(
            self.approximator,
            edge_index=edge_index,
            batch=batch,
            node_attr=node_attr,
            edge_attr=edge_attr,
            glob_attr=glob_attr,
        )

        states: list[torch.Tensor] = []
        for coefficients in self.butcher_tableau[:-1]:
            intermediate_state = self._weighted_sum(coefficients, states)
            states.append(approx(x + intermediate_state * dt))

        coefficients = self.butcher_tableau[-1]
        return torch.divide(
            self._weighted_sum(coefficients, states), coefficients.sum()
        )

    @staticmethod
    def _weighted_sum(
        weights: torch.Tensor, tensors: list[torch.Tensor]
    ) -> torch.Tensor:
        """Sum all tensors with weight"""
        if weights.numel() == 0:
            return torch.tensor(0.0)

        return (weights.reshape(-1, 1, 1) * torch.stack(tensors, dim=0)).sum(dim=0)
