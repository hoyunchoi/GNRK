import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parents[1]))

from GNRK.path import DATA_DIR
from GNRK.simulation_data import SimulationData, to_df
from graph import get_ba, get_er, get_rr
from graph.utils import get_edge_list
from rossler.simulation import argument, solve
from rossler.trajectory import IsDivergingPrecise


def main() -> None:
    args = argument.get_args()
    rng = np.random.default_rng(args.seed)
    is_diverging = IsDivergingPrecise()

    # * Start simulation
    data: list[SimulationData] = []
    num_diverging = 0
    while len(data) < args.num_samples:
        # Graph setting
        network_type = argument.get_network_type(args.network_type, rng)
        num_nodes = argument.get_num_nodes(args.num_nodes, rng)
        mean_degree = argument.get_mean_degree(args.mean_degree, rng)

        if network_type == "er":
            graph = get_er(num_nodes, mean_degree, rng=rng)
        elif network_type == "ba":
            graph = get_ba(num_nodes, mean_degree, rng=rng)
        else:
            graph = get_rr(num_nodes, mean_degree, rng=rng)

        # Since only gcc is selected, the graph can have smaller num_nodes
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        edge_list = get_edge_list(graph)

        # dt setting
        try:
            dts = argument.get_dt(args.max_time, args.steps, args.dt_delta, rng)
        except ValueError:
            print("Could not find proper dt")
            continue

        # constant coefficients (a,b,c, coupling constant) setting
        params = argument.get_params(args.a, args.b, args.c, rng)
        coupling = argument.get_coupling(num_edges, args.coupling, rng)

        # Initial condition setting
        initial_position = argument.get_initial_condition(num_nodes, rng)

        # * Solve rossler equation
        positions = solve.solve(
            args.solver, graph, coupling, initial_position, dts, params
        )

        # * Check divergence of the trajectory
        if is_diverging(torch.from_numpy(positions)):
            # Divergence detected: drop the data
            num_diverging += 1
            print(f"{len(data)=}, {num_diverging=}")
            continue

        # * Store the result
        data.append(
            {
                "network_type": network_type,
                "edge_index": edge_list,  # [E, 2]
                "node_attr": np.zeros((num_nodes, 0), dtype=np.float32),  # [N, 0]
                "edge_attr": coupling,  # [E, 1]
                "glob_attr": np.array(params, dtype=np.float32)[None, :],  # [1, 3]
                "dts": dts,  # [S, 1]
                "trajectories": positions,  # [S+1, N, 3]
            }
        )

    df = to_df(data)
    df.to_pickle(DATA_DIR / f"{args.name}.pkl")


if __name__ == "__main__":
    main()
