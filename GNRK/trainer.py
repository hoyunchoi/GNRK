from typing import Type, cast

import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from .dummy import DummyGradScaler
from .modules.runge_kutta import RungeKutta
from .protocol import IsDivergingProtocol


def amp_dtype(device: torch.device) -> Type:
    return torch.float16 if device.type == "cuda" else torch.bfloat16


def train(
    model: RungeKutta,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    use_amp: bool,
    grad_scaler: GradScaler | DummyGradScaler,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.train()
    tot_loss = torch.tensor(0.0, device=device)
    tot_mae = torch.tensor(0.0, device=device)

    for batch_data in data_loader:
        optimizer.zero_grad(set_to_none=True)
        batch_data = batch_data.to(device, non_blocking=True)

        # Input with random noise
        with autocast(device.type, dtype=amp_dtype(device), enabled=use_amp):
            delta_x = model(
                x=batch_data.x,
                edge_index=batch_data.edge_index,  # (2, BE)
                batch=batch_data.batch,  # (BN,)
                dt=batch_data.dt,  # (BN, 1)
                node_attr=batch_data.node_attr,  # (BN, node_feature)
                edge_attr=batch_data.edge_attr,  # (BE, edge_feature)
                glob_attr=batch_data.glob_attr,  # (B, glob_feature)
            )
            loss = F.mse_loss(delta_x, batch_data.y)  # (BN, state_feature)
            with torch.no_grad():
                mae = F.l1_loss(delta_x, batch_data.y)
                tot_loss += loss
                tot_mae += mae

        cast(torch.Tensor, grad_scaler.scale(loss)).backward()
        grad_scaler.step(optimizer)  # Replacing optimizer.step()
        grad_scaler.update()

    return tot_loss / len(data_loader), tot_mae / len(data_loader)


@torch.no_grad()
def validate(
    model: RungeKutta,
    data_loader: DataLoader,
    use_amp: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    tot_loss = torch.tensor(0.0, device=device)
    tot_mae = torch.tensor(0.0, device=device)

    for batch_data in data_loader:
        batch_data = batch_data.to(device, non_blocking=True)

        with autocast(device.type, dtype=amp_dtype(device), enabled=use_amp):
            delta_x = model(
                x=batch_data.x,  # (BN, state_feature)
                edge_index=batch_data.edge_index,  # (2, BE)
                batch=batch_data.batch,  # (BN,)
                dt=batch_data.dt,  # (BN, 1)
                node_attr=batch_data.node_attr,  # (BN, node_feature)
                edge_attr=batch_data.edge_attr,  # (BE, edge_feature)
                glob_attr=batch_data.glob_attr,  # (B, glob_feature)
            )
            tot_loss += F.mse_loss(delta_x, batch_data.y)
            tot_mae += F.l1_loss(delta_x, batch_data.y)
    return tot_loss / len(data_loader), tot_mae / len(data_loader)


@torch.no_grad()
def validate_rollout(
    model: RungeKutta,
    data_loader: DataLoader,
    use_amp: bool,
    device: torch.device,
    is_diverging: IsDivergingProtocol,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    tot_loss = torch.tensor(0.0, device=device)
    tot_mae = torch.tensor(0.0, device=device)

    for batch_data in data_loader:
        batch_data = batch_data.to(device, non_blocking=True)

        # Default prediction: all element 1e2 (diverge)
        pred_delta_x = is_diverging.tolerance * torch.ones_like(batch_data.y)  # (W, BN, state_feature)

        x: torch.Tensor = batch_data.x  # (BN, state_feature)
        with autocast(device.type, dtype=amp_dtype(device), enabled=use_amp):
            for step, dt in enumerate(batch_data.dt):  # (W, BN, 1)
                delta_x = model(
                    x=x,  # (BN, state_feature)
                    edge_index=batch_data.edge_index,  # (2, BE)
                    batch=batch_data.batch,  # (BN,)
                    dt=dt,  # (BN, 1)
                    node_attr=batch_data.node_attr,  # (BN, node_feature)
                    edge_attr=batch_data.edge_attr,  # (BE, edge_feature)
                    glob_attr=batch_data.glob_attr,  # (B, glob_feature)
                )
                x += delta_x * dt

                # Check if diverge: above tol
                if is_diverging(x):
                    break
                pred_delta_x[step] = delta_x

            tot_loss += F.mse_loss(pred_delta_x, batch_data.y)
            tot_mae += F.l1_loss(pred_delta_x, batch_data.y)

    return tot_loss / len(data_loader), tot_mae / len(data_loader)


@torch.no_grad()
def rollout(
    model: RungeKutta,
    series: pd.Series,
    use_amp: bool,
    device: torch.device,
    is_diverging: IsDivergingProtocol,
) -> torch.Tensor:
    """
    series.edge_index: [2, 2E], udirected edge list
    series.dts: (S, 1)
    series.trajectories: (S+1, N, state_dim)
    series.node_attr: (N, node_dim)
    series.edge_attr: (2E, edge_dim)
    series.glob_attr: (1, glob_dim)

    use_amp: If true, use amp(automatic mixed precision)
    device: where model is located
    is_diverging: Check if rollout prediction is diverging or not

    Return
    predicted trajectory of shape (S+1, N, state_dim)
    """
    model.eval()

    num_nodes = series.trajectories.shape[1]
    dts = torch.unsqueeze(series.dts, -1).to(device)  # (S, 1 ,1)
    edge_index = series.edge_index.to(device=device)  # (2, 2E)
    batch = torch.zeros(num_nodes, dtype=torch.int64, device=device)  # (N, )
    node_attr = series.node_attr.to(device)  # (N, node_dim)
    edge_attr = series.edge_attr.to(device)  # (2E, edge_dim)
    glob_attr = series.glob_attr.to(device)  # (1, glob_dim)

    # Prediction
    pred_trajectory = is_diverging.tolerance * torch.ones_like(  # (S+1, N, state_dim)
        series.trajectories, device=device
    )
    x = series.trajectories[0].to(device=device)  # (N, state_dim)
    pred_trajectory[0] = x
    for step, dt in enumerate(dts):
        with autocast(device.type, dtype=amp_dtype(device), enabled=use_amp):
            delta_x = model(x, edge_index, batch, dt, node_attr, edge_attr, glob_attr)
        x += delta_x * dt

        # Check if diverge
        if is_diverging(x):
            print(f"at {step=}")
            break

        # Store to predicted trajectory
        pred_trajectory[step + 1] = x

    return pred_trajectory
