import argparse
from typing import Literal

import numpy as np
import numpy.typing as npt

arr = npt.NDArray[np.float32]


def get_args(options: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name of the dataset")

    # * Graph parameters
    parser.add_argument(
        "--network_type",
        default=["rr"],
        nargs="+",
        choices=["er", "ba", "rr"],
        help="Type of networks. If given 2 or more values, choose among them.",
    )
    parser.add_argument(
        "-N",
        "--num_nodes",
        type=int,
        nargs="+",
        default=[100],
        help=(
            "Number of nodes at a graph. If given single value, it is constant. If"
            " given 2 values, uniform random between two values (inclusive). If given 3"
            " or more values, choose among them."
        ),
    )
    parser.add_argument(
        "-M",
        "--mean_degree",
        type=float,
        nargs="+",
        default=[4.0],
        help=(
            "Mean degree of a graph. If given single value, it is constant. If given 2"
            " values, uniform random between two values (inclusive). If given 3 or more"
            " values, choose among them."
        ),
    )

    # * Initial condition
    parser.add_argument(
        "--hot_ratio",
        type=float,
        nargs="+",
        default=[0.5],
        help=(
            "Number of hot points at the initial condition. If the given singl value,"
            " constant over all samples.  If given 2 values, uniform random between two"
            " values (inclusive). If given 3 or more values, choose among them."
        ),
    )

    # * Parameters for heat equation
    parser.add_argument(
        "-D",
        "--dissipation",
        type=float,
        nargs="+",
        default=[0.5],
        help=(
            "dissipation rate. of each edges. If single argument is given, it is"
            " constant for every edge. If two arguments are given, it is determined to"
            " be random value between two numbers (inclusive). If given 3 or more"
            " values, choose among them."
        ),
    )

    # * Simulation parameters
    parser.add_argument(
        "--solver",
        choices=["rk1", "rk2", "rk4"],
        help="Which Runge-Kutta method to solve equation",
    )
    parser.add_argument(
        "--max_time", type=float, default=2.0, help="Maximum time for simulation."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help=(
            "Number of time steps per simulation. dt is determined by the average of"
            " max_time / steps."
        ),
    )
    parser.add_argument(
        "--dt_delta",
        type=float,
        default=0.0,
        help="Delta percentile for dt. See argument.divide_randomly",
    )

    # * Ensembles
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If given, seed the random engine for reproducibility",
    )

    # Parse the arguments and return
    args = parser.parse_args() if options is None else parser.parse_args(options)
    return args


def divide_randomly(
    total: float, size: int, delta_percentile: float, rng: np.random.Generator
) -> arr:
    """
    Find array of floats with sum of total whose distribution is uniform

    Args
    total: sum(return) = total
    size: number of random numbers
    delta_percentile: return value lies between [total / size - delta, total / size + delta]
                      where delta = avg * delta_percentile

    Return
    floats: [size, ] where sum(floats) = total
    """
    assert 0.0 <= delta_percentile <= 1.0

    # Distribution setting
    avg = total / size
    delta = avg * delta_percentile
    low_range, high_range = avg - delta, avg + delta
    if total < size * low_range or total > size * high_range:
        raise ValueError("No sulution exists with given parameters.")

    # Find random floats with sum of total
    numbers = np.zeros(size, dtype=np.float32)
    remaining = total
    for i in range(size):
        # Maximum/minimum range of current number
        high = min(high_range, remaining - (size - i - 1) * low_range)
        low = max(low_range, remaining - (size - i - 1) * high_range)

        # Randomly select number
        value = rng.uniform(min(low, high), max(low, high))
        numbers[i] = value
        remaining -= value

    rng.shuffle(numbers)
    return numbers


def get_network_type(
    args_network_types: list[str], rng: np.random.Generator
) -> Literal["ba", "er", "rr"]:
    """
    Ranomly choose network type

    Args
    args_network_types: network type will be randomly selected between the values

    Return: network type
    """
    return rng.choice(args_network_types)


def get_num_nodes(args_num_nodes: list[int], rng: np.random.Generator) -> int:
    """
    Randomly choose number of nodes

    Args
    args_num_nodes: num_nodes will be randomly selected between the values

    Return: number of nodes
    """
    if len(args_num_nodes) <= 2:
        return rng.integers(min(args_num_nodes), max(args_num_nodes), endpoint=True)
    else:
        return rng.choice(args_num_nodes)


def get_mean_degree(args_mean_degree: list[float], rng: np.random.Generator) -> float:
    """
    Randomly choose mean degree of a graph

    Args
    args_mean_degree: mean_degree will be randomly selected between the values

    Return: mean degree
    """

    if len(args_mean_degree) <= 2:
        return rng.uniform(min(args_mean_degree), max(args_mean_degree))
    else:
        return rng.choice(args_mean_degree)


def get_dt(
    max_time: float, steps: int, delta_percentile: float, rng: np.random.Generator
) -> arr:
    """
    Randomly create time spacing of given number of steps for given max_time

    Args
    max_time: maximum time will be randomly selected between the values
    steps: Number of time points
    delta_percentile: see divide_randomly
    clip: see divide_randomly

    Return
    dt: [S, 1]
    """
    # Sample dt
    return divide_randomly(max_time, steps, delta_percentile, rng)[:, None]


def get_dissipation(
    num_edges: int, args_dissipation: list[float], rng: np.random.Generator
) -> arr:
    """
    Randomly create disspation rate for each edges

    Args
    num_edges: Number of edges
    args_dissipation: dissipation rate will be randomly selected between the values

    Return: [E, 1], dissipation rate of each edges
    """
    return rng.uniform(
        min(args_dissipation), max(args_dissipation), size=(num_edges, 1)
    ).astype(np.float32)


def get_initial_condition(
    num_nodes: int, args_hot_ratio: list[float], rng: np.random.Generator
) -> arr:
    """
    Randomly create hot-spots to nodes

    Args
    num_nodes: Number of nodes
    args_hot_ratio: Number of hot points / Number of nodes

    Return: [N, 1]. initial temperature values between [0.0, 1.0]
    """
    # Choose hot ratio
    if len(args_hot_ratio) <= 2:
        hot_ratio = rng.uniform(min(args_hot_ratio), max(args_hot_ratio))
    else:
        hot_ratio = rng.choice(args_hot_ratio)
    assert 0.0 < hot_ratio < 1.0

    # Choose hot nodes
    hot_spots = rng.choice(num_nodes, size=int(num_nodes * hot_ratio), replace=False)

    # Assign hot temperature
    temperature = np.zeros((num_nodes, 1), dtype=np.float32)
    temperature[hot_spots, 0] = 1.0
    return temperature
