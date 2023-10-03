from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_sum

from GNRK.hyperparameter import ApproximatorParameter
from GNRK.modules import ACTIVATION, MLP, Decoder, DuplicatedEncoder
from GNRK.protocol import ApproximatorProtocol


def clean(x: torch.Tensor) -> torch.Tensor:
    """Inf, NaN to zero"""
    return x.nan_to_num(0.0, posinf=0.0, neginf=0.0)


class BurgersApproximator(nn.Module):
    def __init__(
        self,
        state_embedding_dims: list[int],  # 2 (u, v)
        node_embedding_dims: list[int],  # Empty
        edge_embedding_dims: list[int],  # 2 (x-spacing, y-spacing)
        glob_embedding_dims: list[int],  # 1 (nu)
        edge_hidden_dim: int,
        node_hidden_dim: int,
        bn_momentum: float,
        activation: ACTIVATION,
        dropout: float,
    ) -> None:
        super().__init__()
        state_dim = len(state_embedding_dims)
        node_emb_dim = state_embedding_dims[0]
        edge_emb_dim1 = edge_embedding_dims[0]  # spacing_emb
        edge_emb_dim2 = edge_embedding_dims[1]  # inv_spacing_emb
        glob_emb_dim = glob_embedding_dims[0]

        self.encoder1 = DuplicatedEncoder(
            node_emb_dim,
            edge_emb_dim1,
            glob_emb_dim,
            bn_momentum,
            activation,
            dropout,
        )
        self.encoder2 = DuplicatedEncoder(
            edge_embedding_dim=edge_emb_dim2,
            bn_momentum=bn_momentum,
            activation=activation,
            dropout=dropout,
        )
        self.phi_e1 = MLP(
            in_dim=4 * node_emb_dim + 2 * edge_emb_dim1,
            hidden_dim=edge_hidden_dim,
            out_dim=2 * edge_emb_dim1,
            depth=2,
            bn_momentum=bn_momentum,
            activation=activation,
            dropout=dropout,
        )
        self.phi_v1 = MLP(
            in_dim=4 * edge_emb_dim1 + 2 * edge_emb_dim2,
            hidden_dim=node_hidden_dim,
            out_dim=2 * node_emb_dim,
            depth=2,
            bn_momentum=bn_momentum,
            activation=activation,
            dropout=dropout,
        )
        self.phi_e2 = MLP(
            in_dim=4 * edge_emb_dim1,
            hidden_dim=edge_hidden_dim,
            out_dim=2 * edge_emb_dim2,
            depth=2,
            bn_momentum=bn_momentum,
            activation=activation,
            dropout=dropout,
        )
        self.phi_v2 = MLP(
            in_dim=6 * edge_emb_dim2,
            hidden_dim=node_hidden_dim,
            out_dim=2 * node_emb_dim,
            depth=2,
            bn_momentum=bn_momentum,
            activation=activation,
            dropout=dropout,
        )

        self.extended_decoder = MLP(
            in_dim=6 * node_emb_dim + 2 * glob_emb_dim,
            hidden_dim=node_hidden_dim,
            out_dim=2 * node_emb_dim,
            depth=2,
            bn_momentum=bn_momentum,
            activation=activation,
            dropout=dropout,
        )
        self.state_decoder = Decoder(
            2 * node_emb_dim, state_dim, bn_momentum, activation, dropout
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
        node_attr: torch.Tensor,  # not used
        edge_attr: torch.Tensor,
        glob_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [BN, 2], State of each node
        edge_index: [2, BE],
        batch: [BN, ], index of batch where each node belongs

        node_attr: Not used
        edge_attr: [BE, 2], spacing of edges
        glob_attr: [B, 1], nu
        """
        row, col = edge_index
        spacing = edge_attr  # [BE, 2]
        glob_attr = glob_attr.repeat(1, 2)  # [B, 2]
        inv_spacing = clean(1.0 / spacing)  # [BE, 2]
        inv_double_spacing = clean(  # [BN, 2]
            1.0 / (scatter_sum(spacing, row, dim=0) + scatter_sum(spacing, col, dim=0))
        )

        # Encoding: [BN, 2*node_emb], [BE, 2*edge_emb1], [B, 2*glob_emb] -> Enc
        node_emb, spacing_emb, glob_emb = self.encoder1(x, spacing, glob_attr)
        # [BE, 2*edge_emb2], [BN, 2*edge_emb2]
        _, inv_spacing_emb, _ = self.encoder2(edge_attr=inv_spacing)
        _, inv_double_spacing_emb, _ = self.encoder2(edge_attr=inv_double_spacing)

        # Per-edge update 1: [BE, 4*node_emb + 2*edge_emb1] -> [BE, 2*edge_emb1]
        edge_prime1 = self.phi_e1(
            torch.cat((node_emb[row], node_emb[col], inv_spacing_emb), dim=-1)
        )

        # Sum aggregation 1: [BE, 2*edge_emb1] -> [BN, 2*edge_emb1] x 2
        edge_bar1_plus = scatter_sum(edge_prime1, row, dim=0)
        edge_bar1_minus = scatter_sum(edge_prime1, col, dim=0)

        # Per-node update 1: [BN, 4*edge_emb1 + 2*edge_emb2] -> [BN, 2*node_emb]
        w1 = self.phi_v1(
            torch.cat((edge_bar1_plus, edge_bar1_minus, inv_double_spacing_emb), dim=-1)
        )

        # Per-edge update 2: [BE, 4*edge_emb1] -> [BE, 2 * edge_emb2] x 2
        edge_prime2_plus = self.phi_e2(
            torch.cat((spacing_emb, edge_bar1_plus[col]), dim=-1)
        )
        edge_prime2_minus = self.phi_e2(
            torch.cat((spacing_emb, edge_bar1_minus[row]), dim=-1)
        )

        # Sum aggregation 2: [BE, 2*edge_emb2] -> [BN, 2*edge_emb2] x 2
        edge_bar2_plus = scatter_sum(edge_prime2_plus, col, dim=0)
        edge_bar2_minus = scatter_sum(edge_prime2_minus, row, dim=0)

        # Per-node update 2: [BN, 6*edge_emb2] -> [BN, 2*node_emb]
        w2 = self.phi_v2(
            torch.cat((edge_bar2_plus, edge_bar2_minus, inv_double_spacing_emb), dim=-1)
        )

        # Decoding1: [BN, 6*node_emb + glob_emb] -> [BN, 2*node_emb]
        result = self.extended_decoder(
            torch.cat((node_emb, w2, glob_emb[batch], w1), dim=-1)
        )
        # Decoding2: [BN, node_emb] -> [BN, 2]
        return self.state_decoder(result)
