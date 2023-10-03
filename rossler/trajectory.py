from typing import cast

import numpy as np
import numpy.typing as npt
import torch

arr = npt.NDArray[np.float32]
TOLERANCE = 100.0


class IsDiverging:
    tolerance = TOLERANCE

    def __call__(self, trajectory: torch.Tensor) -> torch.BoolTensor:
        return cast(
            torch.BoolTensor,
            trajectory.isnan().any() + trajectory.abs().max() >= self.tolerance,
        )


class IsDivergingPrecise:
    tolerance = TOLERANCE

    def __call__(self, trajectory: torch.Tensor) -> torch.BoolTensor:
        nan_indices = trajectory.isnan().nonzero()
        if nan_indices.numel():
            print(f"Nan detected at {nan_indices.cpu().squeeze()}", end=" ")
            return cast(torch.BoolTensor, torch.tensor(True, dtype=torch.bool))

        diverge_indices = (trajectory.abs() >= self.tolerance).nonzero()
        if diverge_indices.numel():
            print(f"Diverging detected at {diverge_indices.cpu().squeeze()}", end=" ")
            return cast(torch.BoolTensor, torch.tensor(True, dtype=torch.bool))

        return cast(torch.BoolTensor, torch.tensor(False, dtype=torch.bool))


def compare_trajectory(
    trajectory1: arr | torch.Tensor, trajectory2: arr | torch.Tensor, log: bool = True
) -> arr:
    """
    trajectory1: [S+1, N, 3]
    trajectory2: [S+1, N, 3]

    return: [S+1, 3], MAE of each time step, coordinate. averaged over nodes
    """
    if isinstance(trajectory1, torch.Tensor):
        trajectory1 = trajectory1.cpu().numpy()
    if isinstance(trajectory2, torch.Tensor):
        trajectory2 = trajectory2.cpu().numpy()

    abs_error = np.abs(trajectory1 - trajectory2)
    if log:
        max_err_idx = np.unravel_index(np.argmax(abs_error), abs_error.shape)
        step, node, coordinate = max_err_idx
        xyz = ["x", "y", "z"]
        print(
            f"MAE: {abs_error.mean():.4f}, Maximum err: {abs_error[max_err_idx]:.4f} at"
            f" {step=}, {node=}, coordinate={xyz[coordinate]}"
        )
    return np.mean(abs_error, axis=1)
