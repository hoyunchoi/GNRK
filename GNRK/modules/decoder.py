from __future__ import annotations

import torch
import torch.nn as nn

from .mlp import ACTIVATION, MLP


class Decoder(nn.Module):
    def __init__(
        self,
        node_embedding_dim: int,
        node_feature_dim: int,
        bn_momentum: float,
        activation: ACTIVATION,
        dropout: float,
    ) -> None:
        """
        node_embedding_dim: dimension for encoded node features
        node_feature_dim: dimension for decoded node features
        bn_momentum: Refer get_batch_norm_layer
        scaler: inverse transform the decoded value
        """
        super().__init__()
        self.node_decoder = MLP(
            in_dim=node_embedding_dim,
            hidden_dim=node_embedding_dim,
            out_dim=node_feature_dim,
            depth=2,
            bn_momentum=bn_momentum,
            activation=activation,
            dropout=dropout,
            last=True,
        )

    def forward(self, node_attr: torch.Tensor) -> torch.Tensor:
        """node_attr: [BN, node_emb], embedding of nodes"""
        return self.node_decoder(node_attr)

