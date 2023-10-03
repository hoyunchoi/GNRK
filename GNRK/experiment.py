import copy
import sys
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from tqdm.auto import tqdm, trange
from wandb.sdk.wandb_run import Run as WandbRun

import wandb

from . import trainer
from .dataset import Dataset, get_data_loader
from .dummy import DummyGradScaler, DummyWandbRun, dummy_print
from .earlystop import Earlystop
from .hyperparameter import HyperParameter
from .modules.runge_kutta import RungeKutta
from .modules.utils import count_trainable_param
from .path import RESULT_DIR
from .protocol import ApproximatorProtocol
from .scheduler import get_scheduler


def config_wandb(use_wandb: bool, hp: dict[str, Any]) -> WandbRun | DummyWandbRun:
    if use_wandb:
        wandb_run = cast(
            WandbRun,
            wandb.init(project=hp["wandb_project"], config=hp),
        )
        wandb_run.name = wandb_run.id
    else:
        wandb_run = DummyWandbRun()
    return wandb_run


def run(
    rank: int,
    hp: HyperParameter,
    approximator: ApproximatorProtocol,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    save: bool = True,
) -> None:
    # Multiprocessing configuration
    world_size = len(hp.device)
    is_ddp = world_size > 1
    if is_ddp:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    is_prime = rank == 0

    # Device configuration
    device = torch.device(hp.device[rank])
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Data configuration
    train_dataset = Dataset(train_df, window=1)
    val_dataset = Dataset(val_df, window=1)
    rollout_dataset = Dataset(val_df, window=-1)
    train_loader = get_data_loader(
        train_dataset,
        is_ddp=is_ddp,
        device=device,
        pin_memory=True,
        shuffle=True,
        batch_size=hp.batch_size,
    )
    val_loader = get_data_loader(
        val_dataset,
        is_ddp=is_ddp,
        device=device,
        shuffle=False,
        batch_size=hp.batch_size,
        pin_memory=True,
    )
    rollout_loader = get_data_loader(
        rollout_dataset,
        is_ddp=is_ddp,
        device=device,
        shuffle=False,
        batch_size=hp.rollout_batch_size,
        pin_memory=True,
    )

    # Model configuration
    model = RungeKutta(approximator, hp.rk).to(device)
    if is_ddp:
        model = cast(
            RungeKutta,
            DDP(model, static_graph=True, gradient_as_bucket_view=True),  # type: ignore
        )

    # Optimizer, scheduler, gradient scaler, earlystop configuration
    optimizer = optim.AdamW(
        model.parameters(), lr=hp.scheduler.lr, weight_decay=hp.weight_decay
    )
    scheduler = get_scheduler(hp.scheduler, optimizer)
    grad_scaler = (
        GradScaler() if hp.amp and device.type == "cuda" else DummyGradScaler()
    )
    early_stop = Earlystop.from_hp(hp.earlystop)

    # Empty variables to store train trajectory
    train_maes: list[torch.Tensor] = []
    val_maes: list[torch.Tensor] = []
    rollout_maes: list[torch.Tensor] = []
    if is_prime:
        best_model_state_dict = copy.deepcopy(model.state_dict())
    else:
        best_model_state_dict = {}

    # Wandb configuration
    wandb_run = config_wandb(is_prime and hp.wandb, hp.dict)
    exp_id = f"{hp.equation}_{wandb_run.name}"

    # tqdm configuration
    if is_prime:
        print_fn = tqdm.write if hp.tqdm else print
    else:
        print_fn = dummy_print
    epoch_range = (
        trange(hp.epochs, file=sys.stdout) if hp.tqdm and is_prime else range(hp.epochs)
    )

    # Ready to train
    print_fn(f"Start running experiment {exp_id}")
    print_fn(f"Number of trainable parameters : {count_trainable_param(model)}")
    print_fn(f"Number of train data: {len(train_dataset)}")
    print_fn(f"Number of validation data: {len(val_dataset)}")
    print_fn(f"Number of rollout data: {len(rollout_dataset)}")

    match hp.equation:
        case "burgers":
            from burgers.trajectory import IsDiverging
        case "heat":
            from heat.trajectory import IsDiverging
        case "kuramoto":
            from kuramoto.trajectory import IsDiverging
        case "rossler":
            from rossler.trajectory import IsDiverging
        case _:
            raise NotImplementedError(f"No such equation {hp.equation}")

    # ---------------------------------- Start training --------------------------------
    for epoch in epoch_range:
        train_dataset.sampler.set_epoch(epoch)
        train_loss, train_mae = trainer.train(
            model,
            train_loader,
            optimizer,
            hp.amp,
            grad_scaler,
            device,
        )
        scheduler.step()

        val_dataset.sampler.set_epoch(epoch)
        val_loss, val_mae = trainer.validate(model, val_loader, hp.amp, device)

        rollout_dataset.sampler.set_epoch(epoch)
        rollout_loss, rollout_mae = trainer.validate_rollout(
            model, rollout_loader, hp.amp, device, is_diverging=IsDiverging()
        )

        # Collect results from all processes
        if is_ddp:
            dist.reduce(train_loss, dst=0, op=dist.ReduceOp.AVG)  # type:ignore
            dist.reduce(train_mae, dst=0, op=dist.ReduceOp.AVG)  # type:ignore
            dist.reduce(val_loss, dst=0, op=dist.ReduceOp.AVG)  # type:ignore
            dist.reduce(val_mae, dst=0, op=dist.ReduceOp.AVG)  # type:ignore
            dist.reduce(rollout_loss, dst=0, op=dist.ReduceOp.AVG)  # type:ignore
            dist.all_reduce(rollout_mae, op=dist.ReduceOp.AVG)  # type:ignore

        # Log loss, mae
        wandb_run.log(
            {
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "rollout_loss": rollout_loss,
                "rollout_mae": rollout_mae,
            }
        )
        train_maes.append(train_mae)  # Only used for prime processs
        val_maes.append(val_mae)  # Only used for prime processs
        rollout_maes.append(rollout_mae)  # Only used for prime processs

        # Early stop
        early_stop(rollout_mae)
        if early_stop.is_best:
            print_fn(f"Best model at {epoch=}")
            if is_prime:
                best_model_state_dict = copy.deepcopy(model.state_dict())

        if early_stop.abort:
            print_fn(f"Early stopping at {epoch=}")
            break

    # ---------------------------------- Finish training --------------------------------
    if is_ddp:
        dist.destroy_process_group()

    if not (save and is_prime):
        # Do nothing if not saving or this is not the prime process
        wandb_run.finish()
        return

    result_dir = RESULT_DIR / exp_id
    result_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameter
    hp.to_yaml(result_dir / "hyperparameter.yaml")

    # State dictionaries and data scalers
    torch.save(best_model_state_dict, result_dir / "best.pth")

    # Loss history
    train_mae = np.array([mae.item() for mae in train_maes])
    val_mae = np.array([mae.item() for mae in val_maes])
    rollout_mae = np.array([mae.item() for mae in rollout_maes])

    # Wandb summary
    best_epoch = np.argmin(rollout_mae).item()
    wandb_run.summary.update(
        {
            "early_stop": early_stop.abort,
            "best_epoch": best_epoch,
            "best_val_mae": val_maes[best_epoch],
            "best_rollout_mae": rollout_maes[best_epoch],
        }
    )
    wandb_run.finish()
