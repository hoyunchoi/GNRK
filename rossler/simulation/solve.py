import functools
from typing import Callable, Literal

import networkx as nx
import numpy as np
import numpy.typing as npt
from numba import njit

import graph.utils as gUtils

arr = npt.NDArray[np.float32]
HALF = np.array(0.5, dtype=np.float32)
DOUBLE = np.array(2.0, dtype=np.float32)


"""
Solve coupled rossler oscillator following the equation

dx_i/dt = -y_i-z_i
dy_i/dt = x_i + ay_i + sum_j A_ij (y_j-y_i)
        = x_i + ay_i + [Ay]_i - deg(i)*y_i
dz_i/dt = b + z_i(x_i-c)

compiled with jit
"""


@njit(fastmath=True)
def get_velocity(weighted_adjacency_matrix: arr, params: arr, position: arr):
    """
    weighted adjacency matrix: (N, N)
    params: (3, ), three parameters a, b, c
    position: (3, N)

    return: (3, N) delta position x, y, z
    """
    a, b, c = params

    velocity = np.zeros_like(position)
    velocity[0] = -position[1] - position[2]
    velocity[1] = (
        position[0]
        + a * position[1]
        + weighted_adjacency_matrix @ position[1]
        - weighted_adjacency_matrix.sum(axis=0) * position[1]
    )
    velocity[2] = b + position[2] * (position[0] - c)

    return velocity


@njit
def rk1(weighted_adjacency_matrix: arr, params: arr, position: arr, dt: arr) -> arr:
    """
    weighted adjacency matrix: (N, N)
    params: (3, ), three parameters a, b, c
    position: (3, N)
    dt: (1, )

    return: (3, N) next position
    """
    velocity = get_velocity(weighted_adjacency_matrix, params, position)
    return position + dt * velocity


@njit(fastmath=True)
def rk2(weighted_adjacency_matrix: arr, params: arr, position: arr, dt: arr) -> arr:
    """
    weighted adjacency matrix: (N, N)
    params: (3, ), three parameters a, b, c
    position: (3, N)
    dt: (1, )

    return: (3, N) next position
    """
    velocity1 = get_velocity(weighted_adjacency_matrix, params, position)

    temp_pos = position + dt * velocity1
    velocity2 = get_velocity(weighted_adjacency_matrix, params, temp_pos)

    velocity = HALF * (velocity1 + velocity2)
    return position + dt * velocity


@njit(fastmath=True)
def rk4(weighted_adjacency_matrix: arr, params: arr, position: arr, dt: arr) -> arr:
    """
    weighted adjacency matrix: (N, N)
    params: (3, ), three parameters a, b, c
    position: (3, N)
    dt: (1, )

    return: (3, N) next position
    """
    velocity1 = get_velocity(weighted_adjacency_matrix, params, position)

    temp_pos = position + HALF * dt * velocity1
    velocity2 = get_velocity(weighted_adjacency_matrix, params, temp_pos)

    temp_pos = position + HALF * dt * velocity2
    velocity3 = get_velocity(weighted_adjacency_matrix, params, temp_pos)

    temp_pos = position + dt * velocity3
    velocity4 = get_velocity(weighted_adjacency_matrix, params, temp_pos)

    velocity = (
        velocity1 + DOUBLE * velocity2 + DOUBLE * velocity3 + velocity4
    ) / np.array(6.0, dtype=np.float32)
    return position + dt * velocity


def solve(
    solver_name: Literal["rk1", "rk2", "rk4"],
    graph: nx.Graph,
    weights: arr,
    position: arr,
    dts: arr,
    params: tuple[float, float, float],
) -> arr:
    """Solve coupled rossler attractor equation
    dx_i/dt = -y_i-z_i
    dy_i/dt = x_i + ay_i + sum_j A_ij (y_j-y_i)
    dz_i/dt = b + z_i(x_i-c)

    solver_name: How to solve. ex) "rk4_gpu_scatter"
    graph: underlying graph
    weights: (E, 1) coupling strength of edges
    position: (3, N), Initial position of each node
    dts: (S, 1), dt for each time step
    params: three parameters a, b, c

    Return: [S+1, N, 3], trajectory
    """
    weighted_adjacency_matrix = gUtils.get_weighted_adjacency_matrix(graph, weights)

    rk: Callable[[arr, arr], arr]
    if solver_name == "rk1":
        rk = functools.partial(
            rk1, weighted_adjacency_matrix, np.array(params, dtype=np.float32)
        )
    elif solver_name == "rk2":
        rk = functools.partial(
            rk2, weighted_adjacency_matrix, np.array(params, dtype=np.float32)
        )
    else:
        rk = functools.partial(
            rk4, weighted_adjacency_matrix, np.array(params, dtype=np.float32)
        )

    trajectory = np.stack([np.empty_like(position)] * (len(dts) + 1))
    trajectory[0] = position

    for step, dt in enumerate(dts):
        position = rk(position, dt)
        trajectory[step + 1] = position

    return trajectory.transpose(0, 2, 1)
