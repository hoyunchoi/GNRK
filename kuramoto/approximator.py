from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_sum

from GNRK.hyperparameter import ApproximatorParameter
from GNRK.modules import ACTIVATION, MLP, Decoder, Encoder
from GNRK.protocol import ApproximatorProtocol


class KuramotoApproximator(nn.Module):
    def __init__(
        self,
        state_embedding_dims: list[int],  # 1 (theta)
        node_embedding_dims: list[int],  # 1 (natural angular velocity)
        edge_embedding_dims: list[int],  # 1 (coupling constant)
        glob_embedding_dims: list[int],  # Empty
        edge_hidden_dim: int,
        node_hidden_dim: int,
        bn_momentum: float,
        activation: ACTIVATION,
        dropout: float,
    ) -> None:
        super().__init__()
        state_dim = len(state_embedding_dims)
        node_emb_dim = 2 * sum(state_embedding_dims) + sum(node_embedding_dims)
        edge_emb_dim = sum(edge_embedding_dims)

        self.state_encoder = Encoder(
            node_embedding_dims=2 * state_embedding_dims,  # theta -> (sin, cos)
            bn_momentum=bn_momentum,
            activation=activation,
            dropout=dropout,
        )
        self.encoder = Encoder(
            node_embedding_dims,
            edge_embedding_dims,
            glob_embedding_dims,
            bn_momentum,
            activation,
            dropout,
        )
        self.phi_e = MLP(
            in_dim=2 * node_emb_dim + edge_emb_dim,
            hidden_dim=edge_hidden_dim,
            out_dim=edge_emb_dim,
            depth=2,
            bn_momentum=bn_momentum,
            activation=activation,
            dropout=dropout,
        )
        self.phi_v = MLP(
            in_dim=edge_emb_dim,
            hidden_dim=node_hidden_dim,
            out_dim=node_emb_dim,
            depth=2,
            bn_momentum=bn_momentum,
            activation=activation,
            dropout=dropout,
        )

        self.state_decoder = Decoder(
            node_emb_dim, state_dim, bn_momentum, activation, dropout
        )

    @classmethod
    def from_hp(cls, hp: ApproximatorParameter) -> ApproximatorProtocol:
        return cls(
            hp.state_embedding,
            hp.node_embedding,
            hp.edge_embedding,
            hp.glob_embedding,
            hp.edge_hidden,
            hp.node_hidden,
            hp.bn_momentum,
            hp.activation,
            hp.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: torch.LongTensor,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
        glob_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (BN, 1), state_dim=len(state_embedding_dims), State of each node
        edge_index: (2, BE),
        batch: (BN, ), index of batch where each node belongs

        node_attr: (N, 1), node_attr_dim=len(node_embedding_dims), Parameters of each node
        edge_attr: (BE, 1), edge_attr_dim=len(edge_embedding_dims), Parameters of each edge
        glob_attr: (B, 0), glob_attr_dim=len(glob_embedding_dims), Parameters of each glob
        """
        row, col = edge_index

        # Encoding: (BN, 2 * state_emb + node_emb), (BE, edge_emb), _
        state_emb, *_ = self.state_encoder(torch.cat((x.sin(), x.cos()), dim=-1))
        node_emb, edge_emb, _ = self.encoder(node_attr, edge_attr, glob_attr)
        node_emb = torch.cat((state_emb, node_emb), dim=-1)

        # Per-edge update: (BE, 2 * (2 * state_emb + node_emb) + edge_emb) -> (BE, edge_emb)
        edge_prime = self.phi_e(
            torch.cat((node_emb[row], node_emb[col], edge_emb), dim=-1)
        )

        # Sum aggregation: (BE, edge_emb) -> (BN, edge_emb)
        edge_bar = scatter_sum(edge_prime, col, dim=0, dim_size=len(batch))

        #  Per-node update: (BN, edge_emb) -> (BN, node_emb)
        w = self.phi_v(edge_bar)

        # Decoding: (BN, node_emb) -> (BN, state_dim)
        return self.state_decoder(w)
