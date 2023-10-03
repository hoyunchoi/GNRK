from typing import cast

import numpy as np
import numpy.typing as npt
import torch

TOLERANCE = 20.0


class IsDiverging:
    tolerance = TOLERANCE

    def __call__(self, trajectory: torch.Tensor) -> torch.BoolTensor:
        return cast(
            torch.BoolTensor,
            trajectory.isnan().any() + trajectory.abs().max() > self.tolerance,
        )


class IsDivergingPrecise:
    tolerance = TOLERANCE

    def __call__(self, trajectory: torch.Tensor) -> torch.BoolTensor:
        nan_indices = trajectory.isnan().nonzero()
        if nan_indices.numel():
            print(f"Nan detected at {[idx.tolist() for idx in nan_indices]}")
            return cast(torch.BoolTensor, torch.tensor(True, dtype=torch.bool))

        diverging_indices = (trajectory.abs() > self.tolerance).nonzero()
        if diverging_indices.numel():
            print(
                f"Diverging detected at {[idx.tolist() for idx in diverging_indices]}"
            )
            return cast(torch.BoolTensor, torch.tensor(True, dtype=torch.bool))
        return cast(torch.BoolTensor, torch.tensor(False, dtype=torch.bool))


def compare_trajectory(
    trajectory1: npt.NDArray[np.float32] | torch.Tensor,
    trajectory2: npt.NDArray[np.float32] | torch.Tensor,
    log: bool = True,
) -> npt.NDArray[np.float32]:
    """
    trajectory1: [S+1, N, 2]
    trajectory2: [S+1, N, 2]

    return: [S+1, 2], MAE of each time step, flow. averaged over nodes
    """
    if isinstance(trajectory1, torch.Tensor):
        trajectory1 = trajectory1.cpu().numpy()
    if isinstance(trajectory2, torch.Tensor):
        trajectory2 = trajectory2.cpu().numpy()

    abs_error = np.abs(trajectory1 - trajectory2)
    if log:
        max_err_idx = np.unravel_index(np.argmax(abs_error), abs_error.shape)
        step, node, axis = max_err_idx
        uv = ["u", "v"]
        print(
            f"MAE: {abs_error.mean():.4f}, Maximum err: {abs_error[max_err_idx]:.4f} at"
            f" {step=}, {node=}, field={uv[axis]}"
        )
    return np.mean(abs_error, axis=1)
