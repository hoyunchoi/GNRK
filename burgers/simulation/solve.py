import functools
from typing import Callable, Literal

import numpy as np
import numpy.typing as npt

from graph.grid import grid2node

arr = npt.NDArray[np.float32]


def burgers_2d(spacing: arr, nu: float, field: arr) -> arr:
    """
    spacing: [Ny, Nx, 2] distance between grid points
    nu: [2, ] diffusion coefficient
    field: [Ny, Nx, 2], xy-field of each grid point

    Return: [Ny, Nx, 2], delta field
    """
    u, v = field[..., 0], field[..., 1]
    dx, dy = spacing[..., 0], spacing[..., 1]
    dx_minus, dy_minus = np.roll(dx, 1, axis=1), np.roll(dy, 1, axis=0)
    dx_double, dy_double = dx + dx_minus, dy + dy_minus

    # Half derivatives, handle like uniform grid
    u_x_plus = (np.roll(u, -1, axis=1) - u) / dx  # (u(i+1) - u(i)) / dx(i)
    u_x_minus = np.roll(u_x_plus, 1, axis=1)  # (u(i) - u(i-1)) / dx(i-1)
    u_y_plus = (np.roll(u, -1, axis=0) - u) / dy
    u_y_minus = np.roll(u_y_plus, 1, axis=0)
    v_x_plus = (np.roll(v, -1, axis=1) - v) / dx
    v_x_minus = np.roll(v_x_plus, 1, axis=1)
    v_y_plus = (np.roll(v, -1, axis=0) - v) / dy
    v_y_minus = np.roll(v_y_plus, 1, axis=0)

    # Derivatives: Average of half derivatives
    u_x = (dx_minus * u_x_plus + dx * u_x_minus) / dx_double
    u_y = (dy_minus * u_y_plus + dy * u_y_minus) / dy_double
    v_x = (dx_minus * v_x_plus + dx * v_x_minus) / dx_double
    v_y = (dy_minus * v_y_plus + dy * v_y_minus) / dy_double

    # Second derivatives: differentiation over half derivatives
    u_xx = 2.0 * (u_x_plus - u_x_minus) / dx_double
    u_yy = 2.0 * (u_y_plus - u_y_minus) / dy_double
    v_xx = 2.0 * (v_x_plus - v_x_minus) / dx_double
    v_yy = 2.0 * (v_y_plus - v_y_minus) / dy_double

    return np.stack(
        (
            -u * u_x - v * u_y + nu * (u_xx + u_yy),
            -u * v_x - v * v_y + nu * (v_xx + v_yy),
        ),
        axis=-1,
        out=np.empty_like(field),
    )


def rk1(spacing: arr, nu: float, field: arr, dt: float) -> arr:
    """
    spacing: [Ny, Nx, 2] distance between grid points
    nu: [2, ] diffusion coefficient
    field: [Ny, Nx, 2], xy-field of each grid point
    dt

    Return: [Ny, Nx, 2] next field
    """
    delta_field = burgers_2d(spacing, nu, field)
    return field + dt * delta_field


def rk2(spacing: arr, nu: float, field: arr, dt: float) -> arr:
    """
    spacing: [Ny, Nx, 2] distance between grid points
    nu: [2, ] diffusion coefficient
    field: [Ny, Nx, 2], xy-field of each grid point
    dt

    Return: [Ny, Nx, 2] next field
    """
    delta_field1 = burgers_2d(spacing, nu, field)

    temp_field = field + dt * delta_field1
    delta_field2 = burgers_2d(spacing, nu, temp_field)

    delta_field = 0.5 * (delta_field1 + delta_field2)
    return field + dt * delta_field


def rk4(spacing: arr, nu: float, field: arr, dt: float) -> arr:
    """
    spacing: [Ny, Nx, 2] distance between grid points
    nu: [2, ] diffusion coefficient
    field: [Ny, Nx, 2], xy-field of each grid point
    dt

    Return: [Ny, Nx, 2] next field
    """
    delta_field1 = burgers_2d(spacing, nu, field)

    temp_field = field + 0.5 * dt * delta_field1
    delta_field2 = burgers_2d(spacing, nu, temp_field)

    temp_field = field + 0.5 * dt * delta_field2
    delta_field3 = burgers_2d(spacing, nu, temp_field)

    temp_field = field + dt * delta_field3
    delta_field4 = burgers_2d(spacing, nu, temp_field)

    delta_field = (
        delta_field1 + 2.0 * delta_field2 + 2.0 * delta_field3 + delta_field4
    ) / 6.0
    return field + dt * delta_field


def solve(
    solver_name: Literal["rk1", "rk2", "rk4"],
    dxdy: tuple[arr, arr],
    nu: float,
    field: arr,
    dts: arr,
) -> arr:
    """
    solver_name: How to solve \\
    dxdy: [Nx, ], [Ny, ] distance between grid points \\
    nu: [2, ] diffusion coefficient \\
    initial_field: [Ny, Nx, 2] \\
    dts: [S, 1]

    Return
    trajectory: [S+1, N, 2] = [S+1, Nx*Ny, 2]
    """
    spacing = np.stack(np.meshgrid(*dxdy), axis=-1)  # [Ny, Nx, 2]

    rk: Callable[[arr, float], arr]
    if "rk1" == solver_name:
        rk = functools.partial(rk1, spacing, nu)
    elif "rk2" == solver_name:
        rk = functools.partial(rk2, spacing, nu)
    else:
        rk = functools.partial(rk4, spacing, nu)

    trajectory = np.stack([np.empty_like(field).reshape(-1, 2)] * (len(dts) + 1))
    trajectory[0] = grid2node(field)
    for step, dt in enumerate(dts):
        try:
            field = rk(field, dt)
            trajectory[step + 1] = grid2node(field)
        except FloatingPointError:
            # When field diverge: stop iteration and return nan
            return np.array([np.nan])

    return trajectory
