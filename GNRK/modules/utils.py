import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from GNRK.dummy import DummyGradScaler
from GNRK.hyperparameter import HyperParameter


def count_trainable_param(model: nn.Module) -> int:
    """Return number of trainable parameters of model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prune_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    consume_prefix_in_state_dict_if_present(state_dict, "module.")
    consume_prefix_in_state_dict_if_present(state_dict, "approximator.")
    return state_dict


def load_optimizer(
    hp: HyperParameter, model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> optim.Optimizer:
    optimizer = optim.AdamW(
        model.parameters(), lr=hp.scheduler.lr, weight_decay=hp.weight_decay
    )
    optimizer.load_state_dict(state_dict)
    return optimizer


def load_grad_scaler(
    use_amp: bool, device: torch.device, state_dict: dict[str, torch.Tensor]
) -> GradScaler | DummyGradScaler:
    grad_scaler = (
        GradScaler() if use_amp and device.type == "cuda" else DummyGradScaler()
    )
    grad_scaler.load_state_dict(state_dict)
    return grad_scaler
