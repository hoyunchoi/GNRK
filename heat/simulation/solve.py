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


@njit(fastmath=True)
def get_dissipation(weighted_laplacian_matrix: arr, temperature: arr) -> arr:
    """
    weighted_laplacian_matrix: [N, N]
    temperature: [N, 1]

    Return: [N, 1], delta temperature
    """
    return -np.dot(weighted_laplacian_matrix, temperature)


@njit(fastmath=True)
def rk1(weighted_laplacian_matrix: arr, temperature: arr, dt: arr) -> arr:
    """
    weighted_laplacian_matrix: [N, N]
    temperature: [N, 1]
    dt: [1, ]

    Return: [N, 1], next temperature
    """
    delta_temp = get_dissipation(weighted_laplacian_matrix, temperature)
    return temperature + dt * delta_temp


@njit(fastmath=True)
def rk2(weighted_laplacian_matrix: arr, temperature: arr, dt: arr) -> arr:
    """
    weighted_laplacian_matrix: [N, N]
    temperature: [N, 1]
    dt: [1, ]

    Return: [N, 1], next temperature
    """
    delta_temp1 = get_dissipation(weighted_laplacian_matrix, temperature)

    tmp_temp = temperature + dt * delta_temp1
    delta_temp2 = get_dissipation(weighted_laplacian_matrix, tmp_temp)

    delta_temp = HALF * (delta_temp1 + delta_temp2)
    return temperature + dt * delta_temp


@njit(fastmath=True)
def rk4(weighted_laplacian_matrix: arr, temperature: arr, dt: arr) -> arr:
    """
    weighted_laplacian_matrix: [N, N]
    temperature: [N, 1]
    dt: [1, ]

    Return: [N, 1], next temperature
    """
    delta_temp1 = get_dissipation(weighted_laplacian_matrix, temperature)

    tmp_temp = temperature + HALF * dt * delta_temp1
    delta_temp2 = get_dissipation(weighted_laplacian_matrix, tmp_temp)

    tmp_temp = temperature + HALF * dt * delta_temp2
    delta_temp3 = get_dissipation(weighted_laplacian_matrix, tmp_temp)

    tmp_temp = temperature + dt * delta_temp3
    delta_temp4 = get_dissipation(weighted_laplacian_matrix, tmp_temp)

    delta_temp = (
        delta_temp1 + DOUBLE * delta_temp2 + DOUBLE * delta_temp3 + delta_temp4
    ) / np.array(6.0, dtype=np.float32)
    return temperature + dt * delta_temp


def solve(
    solver_name: Literal["rk1", "rk2", "rk4"],
    graph: nx.Graph,
    weights: arr,
    temperature: arr,
    dts: arr,
) -> arr:
    """
    Solve heat equation with given runge-kutta solver

    solver_name: How to solve
    graph: underlying graph
    weights: [E, 1], dissipation rate of edges
    temperature: [N, 1], initial temperature of each node
    dts: [S, 1], dt for each time step

    return: [S+1, N, 1]
    """
    weighted_laplacian_matrix = gUtils.get_weighted_laplacian_matrix(graph, weights)

    rk: Callable[[arr, arr], arr]
    if solver_name == "rk1":
        rk = functools.partial(rk1, weighted_laplacian_matrix)
    elif solver_name == "rk2":
        rk = functools.partial(rk2, weighted_laplacian_matrix)
    else:
        rk = functools.partial(rk4, weighted_laplacian_matrix)

    temperatures = np.stack([np.zeros_like(temperature)] * (len(dts) + 1))
    temperatures[0] = temperature

    for step, dt in enumerate(dts):
        temperature = rk(temperature, dt)
        temperatures[step + 1] = temperature

    return temperatures
