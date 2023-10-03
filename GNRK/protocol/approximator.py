from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import torch
from torch.nn.modules.module import _IncompatibleKeys

from GNRK.hyperparameter import ApproximatorParameter

if TYPE_CHECKING:
    from GNRK.modules.mlp import ACTIVATION


class ApproximatorProtocol(Protocol):
    def __init__(
        self,
        state_embedding_dims: list[int],
        node_embedding_dims: list[int],
        edge_embedding_dims: list[int],
        glob_embedding_dims: list[int],
        edge_hidden_dim: int,
        node_hidden_dim: int,
        bn_momentum: float,
        activation: ACTIVATION,
        dropout: float,
    ) -> None:
        ...

    @classmethod
    def from_hp(
        cls,
        hp: ApproximatorParameter,
    ) -> ApproximatorProtocol:
        ...

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: torch.LongTensor,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
        glob_attr: torch.Tensor,
    ) -> torch.Tensor:
        ...

    # ------------------- torch.nn.Module --------------------
    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: torch.LongTensor,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
        glob_attr: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> _IncompatibleKeys:
        ...
