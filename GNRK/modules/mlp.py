from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch_geometric.nn as gnn

ACTIVATION = Literal["relu", "elu", "selu", "gelu", "tanh", "sigmoid", "silu"]


def get_activation(activation: ACTIVATION) -> nn.Module:
    match activation:
        case "relu":
            return nn.ReLU()
        case "elu":
            return nn.ELU()
        case "selu":
            return nn.SELU()
        case "gelu":
            return nn.GELU()
        case "tanh":
            return nn.Tanh()
        case "sigmoid":
            return nn.Sigmoid()
        case "silu":
            return nn.SiLU()
        case _:
            raise ValueError(f"Unknown activation function: {activation}")


def get_batch_norm_layer(in_channels: int, bn_momentum: float = -1.0) -> nn.Module:
    """
    Return batch normalization with given momentum
    if given momentum is -1.0, return identity layer
    """
    if bn_momentum < 0.0:
        return nn.Identity()
    else:
        return gnn.BatchNorm(in_channels, momentum=bn_momentum)


def get_dropout_layer(dropout: float = 0.0) -> nn.Module:
    """
    Return dropout layer with given dropout rate
    if given dropout rate is 0.0, return identity layer
    """
    if dropout == 0.0:
        return nn.Identity()
    else:
        return nn.Dropout(dropout)


def get_slp(
    in_dim: int,
    out_dim: int,
    bn_momentum: float = 1.0,
    activation: ACTIVATION = "gelu",
    dropout: float = 0.0,
    last: bool = False,
) -> list[nn.Module]:
    """
    Create single layer perceptron (SLP)
    If last is True, only linear layer is used
    """
    if last:
        return [nn.Linear(in_dim, out_dim)]

    # If bn_momentum is negative, batch normalization is not used and therefore use bias term
    use_bias = bn_momentum < 0.0
    return [
        nn.Linear(in_dim, out_dim, bias=use_bias),
        get_batch_norm_layer(out_dim, bn_momentum),
        get_activation(activation),
        get_dropout_layer(dropout),
    ]


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        bn_momentum: float,
        activation: ACTIVATION,
        dropout: float,
        last: bool = False,
    ) -> None:
        """
        Create MLP module of shape in_dim -> hidden -> ... -> hidden -> out_dim
        If last is True, do not use activation to the final output
        """
        assert depth >= 2, "Depth must be greater than or equal to 2"
        super().__init__()

        # Initial layer
        modules = get_slp(in_dim, hidden_dim, bn_momentum, activation, dropout)

        # hidden layers
        for _ in range(depth - 2):
            modules.extend(
                get_slp(hidden_dim, hidden_dim, bn_momentum, activation, dropout)
            )

        # Last layer
        if last:
            modules.extend(get_slp(hidden_dim, out_dim, last=True))
        else:
            modules.extend(
                get_slp(hidden_dim, out_dim, bn_momentum, activation, dropout)
            )

        self.mlp = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
