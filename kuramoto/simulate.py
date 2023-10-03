import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parents[1]))

from GNRK.path import DATA_DIR
from GNRK.simulation_data import SimulationData, to_df
from graph import get_ba, get_er, get_rr
from graph.utils import get_edge_list
from kuramoto.simulation import argument, solve
from kuramoto.trajectory import IsDivergingPrecise


def main() -> None:
    args = argument.get_args()
    rng = np.random.default_rng(args.seed)
    is_diverging = IsDivergingPrecise()

    # * Start simulation
    num_diverging = 0
    data: list[SimulationData] = []
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
            dts = argument.get_dt(
                args.max_time, args.steps, args.dt_delta, rng
            )
        except ValueError:
            print("Could not find proper dt")
            continue

        # constant coefficients (omega, coupling) setting
        omega = argument.get_omega(num_nodes, args.omega, rng)
        coupling = argument.get_coupling(num_edges, args.coupling, rng)

        # Initial condition setting
        initial_phase = argument.get_initial_condition(num_nodes, rng)

        # * Solve kuramoto equation
        phases = solve.solve(args.solver, graph, coupling, initial_phase, dts, omega)

        # * Check divergence of the trajectory
        if is_diverging(torch.from_numpy(phases)):
            # Divergence detected: drop the data
            num_diverging += 1
            print(f"{len(data)=}, {num_diverging=}")
            continue

        # * Store the result
        data.append(
            {
                "network_type": network_type,
                "edge_index": edge_list,  # [E, 2]
                "node_attr": omega,  # [N, 1]
                "edge_attr": coupling,  # [E, 1]
                "glob_attr": np.zeros((1, 0), dtype=np.float32),  # [1, 0]
                "dts": dts,  # [S, 1]
                "trajectories": phases,  # [S+1, N, 1]
            }
        )

    df = to_df(data)
    df.to_pickle(DATA_DIR / f"{args.name}.pkl")


if __name__ == "__main__":
    main()
