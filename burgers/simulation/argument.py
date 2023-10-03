import argparse
from typing import Callable

import numpy as np
import numpy.typing as npt

arr = npt.NDArray[np.float32]


def get_args(options: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name of the dataset")

    # * Grid parameters
    parser.add_argument(
        "--Nx",
        type=int,
        nargs="+",
        default=[100],
        help=(
            "Number of grid points at x-axis. If given single value, it is constant. If"
            " given 2 values, uniform random between two values (inclusive). If given 3"
            " or more values, choose among them."
        ),
    )
    parser.add_argument(
        "--Ny",
        type=int,
        nargs="+",
        default=[100],
        help=(
            "Number of grid points at y-axis. If given sigle value, it is constant. If"
            " given 2 values, uniform random between two values (inclusive). If given 3"
            " or more values, choose among them."
        ),
    )
    parser.add_argument(
        "--spacing_delta",
        type=float,
        default=0.0,
        help=(
            "Delta percentile for grid spacings of both x, y axis. See"
            " argument.divide_randomly"
        ),
    )

    # * Initial condition
    parser.add_argument(
        "--phase",
        type=float,
        default=[0.0],
        nargs="+",
        help=(
            "Phase of initial field. If the given single value, constant over all"
            " samples. If given 2 values, uniform random between two values and (-pi,"
            " pi) (inclusive). If given 3 or more values, choose among them."
        ),
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=[0.5],
        nargs="+",
        help=(
            "Offset of initial field. If the given single value, constant over"
            " all samples. If given 2 values, uniform random between two values"
            " (inclusive). If given 3 or more values, choose among them."
        ),
    )

    # * Parameters for burgers equation
    parser.add_argument(
        "--nu",
        type=float,
        nargs="+",
        default=[0.01],
        help=(
            "dissipation rate. If single value is given, it is constant over every"
            " edges. If two values are given, uniform random between two values"
            " (inclusive). If 3 or more values are given, choose among them."
        ),
    )

    # * Simulation parameters
    parser.add_argument(
        "--solver",
        choices=["rk1", "rk2", "rk4"],
        default="rk4",
        help="Which Runge-Kutta method to solve equation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
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

    # * constant over samples
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples",
    )
    parser.add_argument(
        "--const_ic",
        action="store_true",
        help="If this flag is on, initial condition is constant over samples",
    )
    parser.add_argument(
        "--const_coeff",
        action="store_true",
        help="If this flag is on, coefficient (nu) is constant over samples",
    )
    parser.add_argument(
        "--const_graph",
        action="store_true",
        help="If this flag is on, graph is constant over samples",
    )
    parser.add_argument(
        "--const_dt",
        action="store_true",
        help="If this flag is on, dt is constant over samples",
    )

    # * Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If given, seed the random engine for reproducibility",
    )
    parser.add_argument(
        "--seed_ic",
        type=int,
        default=None,
        help=(
            "If given, create new random engine only for initial condition. If not"
            " given use default random engine"
        ),
    )

    # Parse the arguments and return
    if options is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args=options)


def divide_randomly(
    total: float,
    size: int,
    delta_percentile: float,
    rng: np.random.Generator,
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


def get_NxNy(
    args_Nx: list[int], args_Ny: list[int], rng: np.random.Generator
) -> tuple[int, int]:
    """
    Randomly choose number of grid points Nx, Ny of each axis x,y

    Args
    args_Nx: Nx will be randomly selected between the values
    args_Ny: Ny will be randomly selected between the values

    Return
    Nx, Ny: Number of grid point of each axis x, y
    """
    if len(args_Nx) <= 2:
        Nx = rng.integers(min(args_Nx), max(args_Nx), endpoint=True)
    else:
        Nx = rng.choice(args_Nx)

    if len(args_Ny) <= 2:
        Ny = rng.integers(min(args_Ny), max(args_Ny), endpoint=True)
    else:
        Ny = rng.choice(args_Ny)
    return Nx, Ny


def get_dxdy(
    LxLy: tuple[float, float],
    NxNy: tuple[int, int],
    delta_percentile: float,
    rng: np.random.Generator,
) -> tuple[arr, arr]:
    """
    Randomly create grid spacing of given number of Nx,Ny inside domain of length LxLy

    Args
    LxLy: Length of each axis x, y
    NxNy: Number of grid points of each axis x,y
    delta_percentile: see divide_randomly

    Return
    dx, dy: [Nx, ], [Ny, ], Spacing of each axis
    """
    Lx, Ly = LxLy
    Nx, Ny = NxNy

    # Sample dx, dy
    dx = divide_randomly(Lx, Nx, delta_percentile, rng)
    dy = divide_randomly(Ly, Ny, delta_percentile, rng)

    return dx, dy


def get_dt(
    max_time: float,
    steps: int,
    delta_percentile: float,
    rng: np.random.Generator,
) -> arr:
    """
    Randomly create time spacing of given number of steps for given max_time

    Args
    max_time: maximum time will be randomly selected between the values
    steps: Number of time points
    delta_percentile: see divide_randomly

    Return
    dt: [S, 1]
    """
    # Make negative value to None
    # Sample dt
    return divide_randomly(max_time, steps, delta_percentile, rng)[:, None]


def get_nu(args_nu: list[float], rng: np.random.Generator) -> float:
    """
    Randomly select nu in the range of args_nu, distribution: uniform / log-uniform

    Args
    args_nu: nu will be randomly selected between values

    Return
    nu_x, nu_y
    e.g., ndim=1 -> (nu, 0.0), ndim=2 -> [nu, nu]
    """
    if len(args_nu) <= 2:
        nu = rng.uniform(min(args_nu), max(args_nu))
    else:
        nu = rng.choice(args_nu)
    return nu


def get_initial_condition(
    LxLy: tuple[float, float],
    args_phase: list[float],
    args_offset: list[float],
    rng: np.random.Generator,
) -> Callable[[arr], arr]:
    """
    Create 2D periodic sin assignment function

    Args
    LxLy: Length of each axis x, y
    args_phase: Initial phase of each field (u, v) and axis (x, y) will be randomly selected
    args_offset: Initial offset of each field (u, v) and axis (x, y) will be randomly selected
    args_asymmetry: Initial asymmetry of each field (u, v) and axis (x, y) will be randomly selected

    Return
    function with
        Args: position [Ny, Nx, 2]
        Return: initial condition [Ny, Nx, 2], u,v of each grid point
    """
    Lx, Ly = LxLy

    # Phase of each field (u, v) and axis (x, y)
    args_phase = np.clip(np.array(args_phase), -np.pi, np.pi, dtype=np.float32)
    if len(args_phase) <= 2:
        phase = rng.uniform(min(args_phase), max(args_phase), size=(4,))
    else:
        phase = rng.choice(args_phase, size=(4,))
    phase = phase.astype(np.float32)

    # Offset of each field (u, v) and axis (x, y)
    if len(args_offset) <= 2:
        offset = rng.uniform(min(args_offset), max(args_offset), size=(4,))
    else:
        offset = rng.choice(args_offset, size=(4,))
    offset = offset.astype(np.float32)

    def asymmetric_sin_2d(position: arr) -> arr:
        def sin(x: arr, period: float, phase: float) -> arr:
            return np.sin(2.0 * np.pi / period * x - phase)

        x, y = position[..., 0], position[..., 1]
        # Offset term for asymmetricity
        asymmetry_u = np.exp(-((x - offset[0]) ** 2 + (y - offset[1]) ** 2))
        asymmetry_v = np.exp(-((x - offset[2]) ** 2 + (y - offset[3]) ** 2))

        # 2D sin function with asymmetry term
        initial_u = sin(x, Lx, phase[0]) * sin(y, Ly, phase[1]) * asymmetry_u
        initial_v = sin(x, Lx, phase[2]) * sin(y, Ly, phase[3]) * asymmetry_v

        # Normalization
        initial_u /= np.abs(initial_u).max()
        initial_v /= np.abs(initial_v).max()

        return np.stack((initial_u, initial_v), axis=-1)

    return asymmetric_sin_2d
