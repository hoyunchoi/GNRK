import itertools
from typing import cast

import networkx as nx
import numpy as np
import numpy.typing as npt
import torch

arr = npt.NDArray[np.float32]


def get_1d_square_grid(Nx: int, periodic: bool) -> nx.DiGraph:
    grid = nx.DiGraph()

    if periodic:
        for node in range(Nx):
            grid.add_edge(node, (node + 1) % Nx, axis="x")
    else:
        for node in range(Nx - 1):
            grid.add_edge(node, node + 1, axis="x")

    # Remove self-loop
    grid.remove_edges_from(nx.function.selfloop_edges(grid))
    return grid


def get_2d_square_grid(Nx: int, Ny: int, periodic: bool) -> nx.DiGraph:
    grid = nx.DiGraph()

    if periodic:
        for x, y in itertools.product(range(Nx), range(Ny)):
            grid.add_edge((x, y), ((x + 1) % Nx, y), axis="x")
            grid.add_edge((x, y), (x, (y + 1) % Ny), axis="y")
    else:
        for x, y in itertools.product(range(Nx - 1), range(Ny - 1)):
            grid.add_edge((x, y), (x + 1, y), axis="x")
            grid.add_edge((x, y), (x, y + 1), axis="y")
        for x in range(Nx - 1):
            grid.add_edge((x, Ny - 1), (x + 1, Ny - 1), axis="x")
        for y in range(Ny - 1):
            grid.add_edge((Nx - 1, y), (Nx - 1, y + 1), axis="y")

    # Relabel 2D position to corresponding indices
    pos2idx = {
        (x, y): i for i, (y, x) in enumerate(itertools.product(range(Ny), range(Nx)))
    }
    grid = nx.relabel.relabel_nodes(grid, pos2idx)

    # Sort node idx
    graph = nx.DiGraph()
    graph.add_nodes_from(range(grid.number_of_nodes()))
    graph.add_edges_from(grid.edges)

    # Remove self-loop
    graph.remove_edges_from(nx.function.selfloop_edges(graph))
    return graph


def get_3d_square_grid(Nx: int, Ny: int, Nz: int, periodic: bool) -> nx.DiGraph:
    grid = nx.DiGraph()
    if periodic:
        for x, y, z in itertools.product(range(Nx), range(Ny), range(Nz)):
            grid.add_edge((x, y, z), ((x + 1) % Nx, y, z), axis="x")
            grid.add_edge((x, y, z), (x, (y + 1) % Ny, z), axis="y")
            grid.add_edge((x, y, z), (x, y, (z + 1) % Nz), axis="z")
    else:
        raise NotImplementedError("Non-periodic 3D grid is not implemented yet")

    # Relabel 2D position to corresponding indices
    pos2idx = {
        (x, y, z): i
        for i, (z, y, x) in enumerate(
            itertools.product(range(Nz), range(Ny), range(Nx))
        )
    }
    grid = nx.relabel.relabel_nodes(grid, pos2idx)

    # Sort node idx
    graph = nx.DiGraph()
    graph.add_nodes_from(range(grid.number_of_nodes()))
    graph.add_edges_from(grid.edges)

    # Remove self-loop
    graph.remove_edges_from(nx.function.selfloop_edges(graph))
    return graph


def get_square_grid(*dims: int, periodic: bool = True) -> nx.DiGraph:
    """Get graph representation of square mesh with periodic boundary condition"""
    if len(dims) == 1:
        return get_1d_square_grid(*dims, periodic=periodic)
    elif len(dims) == 2:
        return get_2d_square_grid(*dims, periodic=periodic)
    elif len(dims) == 3:
        return get_3d_square_grid(*dims, periodic=periodic)
    raise ValueError


def to_periodic_field(field: arr) -> arr:
    """
    Make input field to be periodic: [Ny, Nx, 2] -> [Ny+1, Nx+1, 2]
    """
    pad = [(0, 1), (0, 1), (0, 0)]
    periodic_field = np.pad(field, pad)

    periodic_field[-1, :-1] = field[0, :]
    periodic_field[:-1, -1] = field[:, 0]
    periodic_field[-1, -1] = field[0, 0]

    return periodic_field


def grid2node(node_attr: arr) -> arr:
    """
    Grid point to node point: [Ny, Nx, 2] -> [Nx*Ny, 2]
    Inverse of node2grid
    """
    return node_attr.reshape(-1, 2)


def node2grid(node_attr: torch.Tensor | arr, Nx: int, Ny: int) -> arr:
    """
    Node point to grid point: [Nx*Ny, 2] -> [Ny, Nx, 2]
    Inverse of grid2node
    """
    if isinstance(node_attr, torch.Tensor):
        node_attr = cast(arr, node_attr.cpu().numpy())

    # [Nx*Ny, 2] -> [Ny, Nx, 2]
    return node_attr.reshape(Ny, Nx, 2)


def dxdy2pos(dx: arr, dy: arr) -> arr:
    """
    From the spacing of each axis, generate positions of all grid points
    [Nx, ], [Ny, ] -> [Ny+1, Nx+1, 2]
    """
    return np.stack(
        np.meshgrid(np.insert(np.cumsum(dx), 0, 0.0), np.insert(np.cumsum(dy), 0, 0.0)),
        axis=-1,
    )


def dxdy2edge(dx: arr, dy: arr) -> arr:
    """
    From the spacing of each axis, generate edge attributes
    [Nx, ], [Ny, ] -> [2*Nx*Ny, 2] = [E, 2]
    Inverse of edge2dxdy

    Return
    edge_attr[..., 0]: nonzero for x-axis \\
    edge_attr[..., 1]: nonzero for y-axis \\
    """
    Nx, Ny = len(dx), len(dy)

    # [E, 2]
    edge_attr = np.zeros((2 * Nx * Ny, 2), dtype=np.float32)
    edge_attr[::2, 0] = np.tile(dx, Ny)
    edge_attr[1::2, 1] = np.repeat(dy, Nx)
    return edge_attr


def edge2dxdy(edge_attr: torch.Tensor | arr, Nx: int, Ny: int) -> tuple[arr, arr]:
    """
    From the edge attributes, generate spacing of each axis \\
    Inverse of dxdy2edge

    [E, 2] = [2*Nx*Ny, 2] -> [Nx, ], [Ny, ] \\
    """
    if isinstance(edge_attr, torch.Tensor):
        edge_attr = cast(arr, edge_attr.cpu().numpy())
    assert edge_attr.shape[0] == 2 * Nx * Ny

    return edge_attr[: 2 * Nx : 2, 0], edge_attr[1 :: 2 * Nx, 1]
