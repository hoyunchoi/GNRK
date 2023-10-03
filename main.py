import os
import time

import pandas as pd
import torch
import torch.multiprocessing as mp

from burgers.approximator import BurgersApproximator
from GNRK.experiment import run
from GNRK.hyperparameter import get_hp
from GNRK.path import DATA_DIR
from heat.approximator import HeatApproximator
from kuramoto.approximator import KuramotoApproximator
from rossler.approximator import RosslerApproximator


def main() -> None:
    hp = get_hp()

    # ---------------- Read data ----------------
    start = time.perf_counter()
    train_df = pd.read_pickle(DATA_DIR / f"{hp.dataset}_train.pkl")
    val_df = pd.read_pickle(DATA_DIR / f"{hp.dataset}_val.pkl")
    print(f"Reading data took {time.perf_counter()-start} seconds")

    # --------- Create governing equation approximator ----------
    match hp.equation:
        case "burgers":
            approximator = BurgersApproximator.from_hp(hp.approximator)
        case "heat":
            approximator = HeatApproximator.from_hp(hp.approximator)
        case "kuramoto":
            approximator = KuramotoApproximator.from_hp(hp.approximator)
        case "rossler":
            approximator = RosslerApproximator.from_hp(hp.approximator)
        case _:
            raise NotImplementedError(f"No such equation {hp.equation}")

    # ----------------- Training -----------------
    start = time.perf_counter()
    save = True

    if len(hp.device) == 1:
        run(0, hp, approximator, train_df, val_df, save)
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = f"{hp.port}"

        mp.spawn(  # type:ignore
            run,
            args=(hp, approximator, train_df, val_df, save),
            nprocs=len(hp.device),
            join=True,
        )

    print(f"Training took {time.perf_counter()-start} seconds")


if __name__ == "__main__":
    main()
