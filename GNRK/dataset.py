from typing import cast

import pandas as pd
import torch
import torch.utils.data as tData
import torch_geometric.data as gData

from .dummy import DummySampler


class Dataset(tData.Dataset):
    def __init__(self, df: pd.DataFrame, window: int = 1) -> None:
        """
        df: Dataframe to extract gData from
        window: How many time step will model predict

        N: number of nodes
        E: number of edges
        S: number of steps
        W: window
        """
        assert window != 0, f"Window should be nonzero"
        self.sampler: tData.DistributedSampler | DummySampler = DummySampler()

        # Number of total steps each sample has, considering time window
        steps_per_sample = [len(trajectory) for trajectory in df.trajectories]
        if window < 0:
            assert all_equal(steps_per_sample), (
                "If you want to do full rollout with batch, "
                "number of steps of each samples should be equal"
            )
            window += steps_per_sample[0]
        steps_per_sample = [num_step - window for num_step in steps_per_sample]

        # store sampled data
        self.data: list[gData.Data] = []
        for num_step, (_, series) in zip(steps_per_sample, df.iterrows()):
            self.data.extend(
                [
                    gData.Data(
                        x=series.trajectories[step],  # (N, state_dim)
                        edge_index=series.edge_index,  # (2, 2E)
                        dt=series.dts[step : step + window].unsqueeze(0),  # (1, W, 1)
                        node_attr=series.node_attr,  # (N, node_dim)
                        edge_attr=series.edge_attr,  # (2E, edge_dim)
                        glob_attr=series.glob_attr,  # (1, glob_dim)
                        y=torch.divide(  # (N, W, state_dim)
                            series.trajectories[step + 1 : step + window + 1]
                            - series.trajectories[step : step + window],
                            series.dts[step : step + window, None],
                        ).transpose(0, 1),
                    )
                    for step in range(num_step)
                ]
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> gData.Data:
        return self.data[index]


def all_equal(numbers: list[int]) -> bool:
    return all(numbers[0] == x for x in numbers)


def collate_fn(data: list[gData.data.BaseData]):
    batch_data = gData.Batch.from_data_list(data)

    # (BN, W, state_feature) -> (W, BN, stae_feature) -> (BN, state_feature) if W=1
    batch_data.y = torch.transpose(batch_data.y, 0, 1).squeeze(0)

    # (B, W, 1) -> (BN, W, 1) -> (W, BN, 1) -> (BN, 1) if W = 1
    batch_data.dt = batch_data.dt[batch_data.batch]
    batch_data.dt = torch.transpose(batch_data.dt, 0, 1).squeeze(0)
    return batch_data


def get_data_loader(
    dataset: Dataset, is_ddp: bool, device: torch.device | str = "", **kwargs
) -> tData.DataLoader:
    device = str(device) if kwargs.get("pin_memory", False) else ""
    shuffle = kwargs.pop("shuffle")

    if is_ddp:
        dataset.sampler = tData.DistributedSampler(
            cast(tData.Dataset, dataset), shuffle=shuffle
        )
        return tData.DataLoader(
            cast(tData.Dataset, dataset),
            sampler=dataset.sampler,
            pin_memory_device=device,
            collate_fn=collate_fn,
            **kwargs,
        )
    else:
        return tData.DataLoader(
            cast(tData.Dataset, dataset),
            shuffle=shuffle,
            pin_memory_device=device,
            collate_fn=collate_fn,
            **kwargs,
        )
