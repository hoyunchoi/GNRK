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
def get_dphase(weighted_adjacency_matrix: arr, omega: arr, phase: arr) -> arr:
    """
    weighted adjacency matrix: [N, N]
    omega: [N, 1], omega of each node
    phase: [N, 1], phase of each node

    Return: [N, 1], delta phase
    """
    sin_phase, cos_phase = np.sin(phase), np.cos(phase)

    return (
        omega
        + cos_phase * np.dot(weighted_adjacency_matrix, sin_phase)
        - sin_phase * np.dot(weighted_adjacency_matrix, cos_phase)
    )


@njit(fastmath=True)
def rk1(weighted_adjacency_matrix: arr, omega: arr, phase: arr, dt: arr) -> arr:
    """
    weighted adjacency matrix: [N, N]
    omega: [N, 1], omega of each node
    phase: [N, 1], phase of each node
    dt: [1, ]

    Return: [N, 1], next phase
    """
    dphase = get_dphase(weighted_adjacency_matrix, omega, phase)
    return phase + dt * dphase


@njit(fastmath=True)
def rk2(weighted_adjacency_matrix: arr, omega: arr, phase: arr, dt: arr) -> arr:
    """
    weighted adjacency matrix: [N, N]
    omega: [N, 1], omega of each node
    phase: [N, 1], phase of each node
    dt: [1, ]

    Return: [N, 1], next phase
    """
    dphase1 = get_dphase(weighted_adjacency_matrix, omega, phase)

    temp_phase = phase + dt * dphase1
    dphase2 = get_dphase(weighted_adjacency_matrix, omega, temp_phase)

    dphase = HALF * (dphase1 + dphase2)
    return phase + dt * dphase


@njit(fastmath=True)
def rk4(weighted_adjacency_matrix: arr, omega: arr, phase: arr, dt: arr) -> arr:
    """
    weighted adjacency matrix: [N, N]
    omega: [N, 1], omega of each node
    phase: [N, 1], phase of each node
    dt: [1, ]

    Return: [N, 1], next phase
    """
    dphase1 = get_dphase(weighted_adjacency_matrix, omega, phase)

    temp_phase = phase + HALF * dt * dphase1
    dphase2 = get_dphase(weighted_adjacency_matrix, omega, temp_phase)

    temp_phase = phase + HALF * dt * dphase2
    dphase3 = get_dphase(weighted_adjacency_matrix, omega, temp_phase)

    temp_phase = phase + dt * dphase3
    dphase4 = get_dphase(weighted_adjacency_matrix, omega, temp_phase)

    dphase = (dphase1 + DOUBLE * dphase2 + DOUBLE * dphase3 + dphase4) / np.array(
        6.0, dtype=np.float32
    )
    return phase + dt * dphase


def solve(
    solver_name: Literal["rk1", "rk2", "rk4"],
    graph: nx.Graph,
    weights: arr,
    phase: arr,
    dts: arr,
    omega: arr,
) -> arr:
    """
    Solve kuramoto equation
    dtheta_i/dt = omega_i + sum_j K_ij A_ij sin (theta_j-theta_i)

    solver_name: How to solve.
    graph: underlying graph
    weights: [E, 1], coupling strength of each edges on graph
    initial phase: [N, 1], phase of each node
    omega: [N, 1], natural frequency of each node
    dts: [S, 1], dt for each time step

    Return: [S+1, N, 1], phases
    """
    weighted_adjacency_matrix = gUtils.get_weighted_adjacency_matrix(graph, weights)

    rk: Callable[[arr, arr], arr]
    if solver_name == "rk1":
        rk = functools.partial(rk1, weighted_adjacency_matrix, omega)
    elif solver_name == "rk2":
        rk = functools.partial(rk2, weighted_adjacency_matrix, omega)
    else:
        rk = functools.partial(rk4, weighted_adjacency_matrix, omega)

    phases = np.stack([np.zeros_like(phase)] * (len(dts) + 1))
    phases[0] = phase

    for step, dt in enumerate(dts):
        phase = rk(phase, dt)
        phases[step + 1] = phase

    return phases
