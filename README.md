# GNRK: Graph Neural Runge-Kutta method for solving partial differential equations
The repository includes simulation code and GNRK implementation presented in paper: [GNRK: Graph Neural Runge-Kutta method for solving partial differential equations](https://arxiv.org/abs/2310.00618).

## Requirements
- pytorch
- pytorch_geometric
- wandb (optional, comment out if not used)

or, use environment.yaml to create a conda environment
``` bash
conda env create -f environment.yaml -n <my_env>
```

## Preparing dataset
In the paper, we covered two types of systems based on their spatial domain: Euclidean and Graph spatial domain.


#### Euclidean spatial domain
We consider the 2-dimensional Burgers' equation, represented b y the following coupled partial differential equation.
$$
\frac{\partial u}{\partial t} = -u \frac{\partial u}{\partial x} - v\frac{\partial u}{\partial y} + \nu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right), \quad \frac{\partial v}{\partial t} = -u \frac{\partial v}{\partial x} - v\frac{\partial v}{\partial y} + \nu \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right).
$$

Specifically, the system is defined using the following default settings:
- Spatial domain: $0 \le x, y \le 1$ with perodic boundary condition.
- Time domain: $0 \le t \le 1$.
- Initial condition: $A \sin(2\pi x- \phi_x) \sin(2\pi y - \phi_y) \exp(-(x-x_0)^2-(y-y_0)^2)$. $A$ serves as the normalization constant, ensuring a peak value of 1. Additionally, $\phi_x$ and $\phi_y$ represent the initial phases along each axis, and $x_0$ and $y_0$ denote the offsets accounting for any asymmetry.
- PDE coefficient: $\nu = 0.01$.
- Spatial discretization: Uniform square grid with spacing = $1.0 / 100 = 0.01$.
- Temporal discretization: $dt = 1.0 / 1000 = 10^{-3}$.
- Use 4th order Runge-Kutta method to simulate.

We provide the simulation code for 2D Burgers' equation in `burgers/simulate.py`.
Three commands is required to create three diffrent files containing train, validation, and test dataset.
The resulting dataset will be stored in `GNRK/data/burgers_<dataset_name>.pkl`.
For detailed documention of the simulation, please run `python burgers/simulate.py --help`.

1. Dataset I : Different initial condition over samples
``` bash
python burgers/simulate.py --name=dataset1_train --num_samples=20 --phase -3.15 3.15 --offset 0.0 1.0 --const_coeff --const_graph --const_dt
python burgers/simulate.py --name=dataset1_val --num_samples=10 --phase -3.15 3.15 --offset 0.0 1.0 --const_coeff --const_graph --const_dt
python burgers/simulate.py --name=dataset1_test --num_samples=50 --phase -3.15 3.15 --offset 0.0 1.0 --const_coeff --const_graph --const_dt
```

2. Dataset II : $\nu$ choosen randomly from $[0.005, 0.02]$, different over samples
```bash
python burgers/simulate.py --name=dataset2_train --nu 0.005 0.02 --num_samples=20 --seed_ic=0 --const_ic --const_graph --const_dt
python burgers/simulate.py --name=dataset2_val --nu 0.005 0.02 --num_samples=10 --seed_ic=0 --const_ic --const_graph --const_dt
python burgers/simulate.py --name=dataset2_test --nu 0.005 0.02 --num_samples=50 --seed_ic=0 --const_ic --const_graph --const_dt
```

3. Dataset III : Nonuniform 2D square grid with different sizes $N_x, N_y \in [50, 150]$. The spacings between grid points deviates by $\pm 10\%$ from that of the corresponding uniform grids of the same size.
```bash
python burgers/simulate.py --name=dataset3_train --Nx 50 150 --Ny 50 150 --spacing_delta 0.1 --num_samples=20 --seed_ic=0 --const_ic --const_coeff --const_dt
python burgers/simulate.py --name=dataset3_val --Nx 50 150 --Ny 50 150 --spacing_delta 0.1 --num_samples=10 --seed_ic=0 --const_ic --const_coeff --const_dt
python burgers/simulate.py --name=dataset3_test --Nx 50 150 --Ny 50 150 --spacing_delta 0.1 --num_samples=50 --seed_ic=0 --const_ic --const_coeff --const_dt
```

4. Dataset IV : Nonuniform temporal discretization $\Delta t$ of 1000 time steps. The difference in $\Delta t$ flucutates by $\pm 10\%$ compared to the uniform case. For training and validation dataset, we use 1st order Runge-Kutta method and 4th order Runge-Kutta method for test dataset.
```bash
python burgers/simulate.py --name=dataset4_train --solver=rk1 --dt_delta=0.1 --num_samples=20 --seed_ic=0 --const_ic --const_coeff --const_graph
python burgers/simulate.py --name=dataset4_val --solver=rk1 --dt_delta=0.1 --num_samples=10 --seed_ic=0 --const_ic --const_coeff --const_graph
python burgers/simulate.py --name=dataset4_test --solver=rk4 --dt_delta=0.1 --num_samples=50 --seed_ic=0 --const_ic --const_coeff --const_graph
```

#### Graph spatial domain
The simulation code for three different coupled system defined in graph spatial domain.

1. Heat system
$$
\frac{dT_i}{dt} = \sum_{j \in \mathcal{N}(i)} D_{ij} (T_j - T_i)
$$
- Time domain $\mathcal{T} \equiv 0 \le t \le 2.0$
- Initial condition: Random assignment of each node to either a hot state ($T_i=1$) and cold state ($T_i=0$) are uniformly choosen in $[0.1, 0.9]$, with ratio of these two states also being randomly determined. They are different over all samples.
- Graph: BA/ER/RR with num_nodes $N \in [50, 150]$, mean_degree $\in [2, 6]$, different over all samples
- Dissipation rate $D_{ij}$: random values between $[0.1, 1.0]$, different over all edges and samples
- Nonuniform $\Delta t$, which differs $\pm 10\%$ of uniform $\Delta t = 2.0 / 100 = 0.02$, different over all samples
- 1st order Runge-Kutta for train/validation, and 4th order Runge-Kutta for test dataset

For detailed documention of simulation, please run `python heat/simulate.py --help`.
``` bash
python heat/simulate.py --name=heat_train --network_type er ba rr --num_nodes 50 150 --mean_degree 2.0 6.0 --hot_ratio 0.1 0.9 --dissipation 0.1 1.0 --solver=rk1 --dt_delta=0.1 --num_samples=200
python heat/simulate.py --name=heat_val --network_type er ba rr --num_nodes 50 150 --mean_degree 2.0 6.0 --hot_ratio 0.1 0.9 --dissipation 0.1 1.0 --solver=rk1 --dt_delta=0.1 --num_samples=20
python heat/simulate.py --name=heat_test --network_type er ba rr --num_nodes 50 150 --mean_degree 2.0 6.0 --hot_ratio 0.1 0.9 --dissipation 0.1 1.0 --solver=rk4 --dt_delta=0.1 --num_samples=40
```


2. Kuramoto system
$$
\frac{d \theta_i}{dt} = \omega_i + \sum_{j \in \mathcal{N}(i)} K_{ij} \sin (\theta_j - \theta_i)
$$
- Time domain $\mathcal{T} \equiv 0 \le t \le 10.0$
- Initial condition: Random phase in $\theta_i \in (-\pi, \pi]$, different over samples.
- Graph: BA/ER/RR with num_nodes $N \in [50, 150]$, mean_degree $\in [2, 6]$, different over all samples
- Natural angular velocity $\omega_i \sim N(0, 1)$, different over all nodes and samples
- Coupling constant $K_{ij} \in [0.1, 0.5]$, different over all edges and samples
- Nonuniform $\Delta t$, which differs $\pm 10\%$ of uniform $\Delta t = 10.0 / 500 = 0.02$, different over all samples
- 1st order Runge-Kutta for train/validation, and 4th order Runge-Kutta for test dataset

For detailed documention of simulation, please run `python kuramoto/simulate.py --help`.
``` bash
python kuramoto/simulate.py --name=kuramoto_train --network_type er ba rr --num_nodes 50 150 --mean_degree 4.0 6.0 --coupling 0.1 0.5 --solver=rk1 --dt_delta=0.1 --num_samples=200
python kuramoto/simulate.py --name=kuramoto_val --network_type er ba rr --num_nodes 50 150 --mean_degree 4.0 6.0 --coupling 0.1 0.5 --solver=rk1 --dt_delta=0.1 --num_samples=20
python kuramoto/simulate.py --name=kuramoto_test --network_type er ba rr --num_nodes 50 150 --mean_degree 4.0 6.0 --coupling 0.1 0.5 --solver=rk4 --dt_delta=0.1 --num_samples=40
```

3. Coupled Rössler system
$$
\frac{dx_i}{dt} = -y_i -z_i, \quad \frac{dy_i}{dt} = x_i + ay_i + \sum_{j \in \mathcal{N}(i)} K_{ij} (y_j - y_i), \quad \frac{dz_i}{dt} = b + z_i (x_i - c)
$$
- Time domain $\mathcal{T} \equiv 0 \le t \le 40.0$
- Initial condition: Randomly selecting $x_i,y_i \in [-4, 4], z_i \in [0, 6]$ for each node, different over nodes and samples
- Graph: BA/ER/RR with num_nodes $N \in [50, 150]$, mean_degree $\in [2, 6]$, different over all samples
- coefficients $a, b \in [0,1, 0.3], c \in [5.0, 7.0]$, different over all samples
- Coupling constant $K_{ij} \in [0.02, 0.04]$, different over all edges and samples
- Nonuniform $\Delta t$, which differs $\pm 10\%$ of uniform $\Delta t = 40.0 / 2000 = 0.02$, different over all samples
- 1st order Runge-Kutta for train/validation, and 4th order Runge-Kutta for test dataset

For detailed documention of simulation, please run `python rossler/simulate.py --help`.
``` bash
python rossler/simulate.py --name=rossler_train --network_type er ba rr --num_nodes 50 150 --mean_degree 2.0 6.0 --a 0.1 0.3 --b 0.1 0.3 --c 5.0 7.0 --coupling 0.02 0.04 --solver=rk1 --dt_delta=0.1 --num_samples=200
python rossler/simulate.py --name=rossler_val --network_type er ba rr --num_nodes 50 150 --mean_degree 2.0 6.0 --a 0.1 0.3 --b 0.1 0.3 --c 5.0 7.0 --coupling 0.02 0.04 --solver=rk1 --dt_delta=0.1 --num_samples=20
python rossler/simulate.py --name=rossler_test --network_type er ba rr --num_nodes 50 150 --mean_degree 2.0 6.0 --a 0.1 0.3 --b 0.1 0.3 --c 5.0 7.0 --coupling 0.02 0.04 --solver=rk4 --dt_delta=0.1 --num_samples=40
```


## GNRK implementation

#### Training predefined equations
Run `main.ipynb` or `main.py` with proper hyperparameters.
For example, to train GNRK in 2D Burgers' equation presented in the paper, run the following command.
``` bash
python main.py --equation=burgers --dataset=burgers_dataset1 --rk=RK4 --approximator_state_embedding 8 --approximator_edge_embedding 8 --approximator_glob_embedding 8 --approximator_edge_hidden=32 --approximator_node_hidden=32 --scheduler_name=step --scheduler_lr=0.0001 --scheduler_lr_max=0.004 --scheduler_lr_max_mult=0.5 --scheduler_period=20 --scheduler_period_mult=1.5 --device 0 1 2 3 --epochs=413 --batch_size=64 --tqdm --amp
```
For detailed documention of training options, please run `python main.py --help`.
The train result will be stored in `GNRK/result/<experiment_name>` directory along with its' hyperparameter.
The `<experiment_name>` is randomly assigned as 8-digit combination of English characters and numbers.

#### Applying to other systems
To adapt GNRK to other systems, one need to make `approximator.py` and `trajectory.py` for the system.
- `approximator.py` : Approximator class for the governing equation. This should follow the ApproximatorProtocol defined in `GNRK/protocol/approximator.py`. One may use the existing NN modules defined in `GNRK/modules`
- `trajectory.py` : This file should contain an IsDiverging class that returns a Boolean tensor that checks whether the trajectory of the system is diverging. Divergence can be detected in a various ways, and one can choose the appropriate method for your target system. However, its form must follow IsDivergingProtocol as defined in `GNRK/protocol/trajectory.py`. This divergence check allows GNRK to ignore subsequent states when the model predicts a divergent state while training.

Additionally, one may need to register the equation in `GNRK/experiment.py` and `main.py`