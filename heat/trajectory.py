from typing import cast

import numpy as np
import numpy.typing as npt
import torch

arr = npt.NDArray[np.float32]
TOLERANCE = 1.0


class IsDiverging:
    tolerance = TOLERANCE

    def __call__(self, trajectory: torch.Tensor) -> torch.BoolTensor:
        return cast(
            torch.BoolTensor,
            trajectory.isnan().any() + (trajectory - 0.5).abs().max() > self.tolerance,
        )


class IsDivergingPrecise:
    tolerance = TOLERANCE

    def __call__(self, trajectory: torch.Tensor) -> torch.BoolTensor:
        nan_indices = trajectory.isnan().nonzero()
        if nan_indices.numel():
            print(f"Nan detected at node {[node.item() for node, _ in nan_indices]}")
            return cast(torch.BoolTensor, torch.tensor(True, dtype=torch.bool))

        diverge_indices = ((trajectory - 0.5).abs() >= self.tolerance).nonzero()
        if diverge_indices.numel():
            print(
                f"Diverging detected at {[node.item() for node, _ in diverge_indices]}"
            )
            return cast(torch.BoolTensor, torch.tensor(True, dtype=torch.bool))

        return cast(torch.BoolTensor, torch.tensor(False, dtype=torch.bool))


def compare_trajectory(
    trajectory1: arr | torch.Tensor, trajectory2: arr | torch.Tensor, log: bool = True
) -> arr:
    """
    trajectory1: [S+1, N, 1]
    trajectory2: [S+1, N, 1]

    return: [S, ], MAE of each time step, averaged over nodes
    """
    if isinstance(trajectory1, torch.Tensor):
        trajectory1 = trajectory1.cpu().numpy()
    if isinstance(trajectory2, torch.Tensor):
        trajectory2 = trajectory2.cpu().numpy()

    abs_err = np.abs(trajectory2 - trajectory1).squeeze()
    if log:
        max_err_idx = np.unravel_index(np.argmax(abs_err), abs_err.shape)
        step, node = max_err_idx
        print(
            f"Maximum err: {abs_err[max_err_idx]:.4e} at {step=}, {node=}"
            f", MAE {abs_err.mean():.4e}"
        )
    return np.mean(abs_err, axis=1)
