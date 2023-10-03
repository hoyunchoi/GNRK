from typing import TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

import graph.utils as gUtils


class SimulationData(TypedDict):
    """
    N: number of nodes
    E: number of edges
    S: number of steps

    network_type: RR, ER, BA, Nx_Ny
    edge_index: [E, 2], directed
    node_attr: [N, node_dim]
    edge_attr: [E, edge_dim]
    glob_attr: [1, glob_dim]
    dts: [S, 1]
    trajectories: [S+1, N, state_dim], including initial condition
    """

    network_type: str
    edge_index: npt.NDArray[np.int64]
    node_attr: npt.NDArray[np.float32]
    edge_attr: npt.NDArray[np.float32]
    glob_attr: npt.NDArray[np.float32]
    dts: npt.NDArray[np.float32]
    trajectories: npt.NDArray[np.float32]


def to_df(data: list[SimulationData], directed: bool = False) -> pd.DataFrame:
    """
    Convert input data that fit to torch_geometric format
    Args
    data: data to be converted into dataframe
    directed: If True, used direct graph. Otherwise, use undirected graph
    """
    df = pd.DataFrame.from_records(data)

    df.node_attr = df.node_attr.map(torch.from_numpy)  # [N, node_dim]
    df.glob_attr = df.glob_attr.map(torch.from_numpy)  # [1, glob_dim]
    df.dts = df.dts.map(torch.from_numpy)  # [S, 1]
    df.trajectories = df.trajectories.map(torch.from_numpy)  # [S+1, N, state_dim]

    if directed:
        # [2, E] / [E, edge_dim]
        df.edge_index = df.edge_index.map(lambda arr: torch.from_numpy(arr.T))
        df.edge_attr = df.edge_attr.map(torch.from_numpy)
    else:
        # [2, 2E] / [2E, edge_dim]
        df.edge_index = df.edge_index.map(
            lambda arr: torch.from_numpy(gUtils.directed2undirected(arr))
        )
        df.edge_attr = df.edge_attr.map(
            lambda arr: torch.from_numpy(gUtils.repeat_weight(arr))
        )
    return df
