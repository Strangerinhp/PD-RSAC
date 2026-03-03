# PD-RSAC: Robust Semi-MDP Planning for EV Fleet Optimization

**Primal-Dual Robust SAC via Wasserstein-1 DRO and MILP-Constrained Actor-Critic**

A GPU-native reinforcement learning framework for city-scale electric vehicle (EV) fleet dispatch, repositioning, and charging optimization. Built on Soft Actor-Critic (SAC) with Graph Convolutional Networks (GCN), Semi-MDP duration discounting, and Wasserstein Distributionally Robust Optimization (WDRO).

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Configuration](#configuration)
- [Data](#data)
- [Algorithms](#algorithms)
- [Evaluation](#evaluation)
- [Documentation](#documentation)

---

## Overview

This project addresses the problem of **optimally controlling a large EV ride-hailing fleet** (1000+ vehicles) across an urban hexagonal grid (1300+ zones). At each 5-minute time step, every vehicle must decide:

| Action | Description |
|--------|-------------|
| **IDLE** | Wait at current location |
| **SERVE** | Pick up and deliver a passenger |
| **CHARGE** | Drive to a charging station |
| **REPOSITION** | Move to a high-demand area |

The objective is to **maximize net fleet revenue** (trip fares - driving costs - electricity costs) while respecting hard constraints (battery limits, charger capacity, feeder power limits) under demand uncertainty.

### Problem Challenges

- **Scale**: 1000 vehicles x 1300 hex zones x 2000 trips per step
- **Hard constraints**: Battery SoC safety, charging station port limits, feeder power caps
- **Uncertainty**: Demand patterns shift across hours, days, and seasons
- **Variable durations**: A trip from Manhattan to Brooklyn takes longer than a local ride
- **Spatial correlation**: Demand shifts in adjacent zones are correlated

---

## Key Features

- **GCN-based Actor-Critic**: Spatial reasoning over hexagonal grid topology via Graph Convolutional Networks
- **Semi-MDP**: Correct temporal discounting for variable-duration actions (gamma^delta)
- **WDRO**: Wasserstein-1 distributionally robust optimization for robustness against demand distribution shifts
- **GPU-native**: Entire pipeline (environment, replay buffer, networks, training) runs on GPU
- **Prioritized Experience Replay**: Importance-weighted sampling for sample efficiency
- **Curriculum Learning**: Gradually tightens pickup distance (10km -> 0km) via cosine schedule
- **Temperature Annealing**: Explore broadly early, exploit later
- **Mixed Precision Training**: AMP for 2x memory savings and faster training
- **Real Data**: Supports NYC Taxi & Limousine Commission trip data (13.7M+ trips)
- **Modular Design**: Clean separation of environment, features, networks, and training

---

## Architecture

```
                         +------------------+
                         |   HexGrid (H3)   |
                         | 1300 hex zones    |
                         +--------+---------+
                                  |
                    +-------------+-------------+
                    |                           |
           +--------v--------+       +---------v---------+
           | GPUEnvironmentV2|       |  Adjacency Matrix |
           | - TripManager   |       |  (for GCN)        |
           | - ActionProcessor       +---------+---------+
           | - RewardComputer|                 |
           +--------+--------+                 |
                    |                           |
              state dict                        |
           (vehicle, hex, context)              |
                    |                           |
           +--------v--------+                  |
           |  Feature Builder |                 |
           |  V:16D H:5D C:9D|                 |
           +--------+--------+                  |
                    |                           |
           +--------v-----------+    +----------v----------+
           |    GCN Actor       |    |    GCN Twin Critic  |
           | - GCN Encoder      |    | - GCN Encoder       |
           | - Action Type Head |    | - Q-Value Head (x2) |
           | - Reposition Head  |    | - Per-vehicle Q     |
           | - Trip Select Head |    +----------+----------+
           | - Charge Power Head|               |
           +--------+----------+               |
                    |                           |
              actions + log_probs          Q1, Q2 values
                    |                           |
           +--------v---------------------------v----------+
           |          Enhanced SAC Trainer                  |
           | - Semi-MDP duration discounting (gamma^delta) |
           | - WDRO robust targets (Wasserstein ball)      |
           | - Primal-dual lambda update                   |
           | - Temperature annealing                       |
           | - Mixed precision (AMP)                       |
           +-----------------------------------------------+
```

### Data Flow

```
Environment.step() -> State Dict -> Feature Builder -> GCN Actor -> Actions
                                                    -> GCN Critic -> Q-values
                                  -> Replay Buffer (GPU-resident, PER)
                                  -> Trainer.train_step() -> Gradient Updates
```

---

## Project Structure

```
Wasserstein/
├── gpu_core/                          # Main framework
│   ├── config/                        # Configuration system
│   │   ├── base.py                    # Dataclass configs (Environment, Training, Reward, ...)
│   │   ├── loader.py                  # YAML config loader
│   │   └── defaults.yaml              # Default hyperparameters
│   │
│   ├── networks/                      # Neural network architectures
│   │   ├── sac.py                     # SACAgent - orchestrates Actor/Critic/Entropy
│   │   ├── gcn.py                     # GCNEncoder - graph convolutional backbone
│   │   ├── gcn_actor.py               # GCNActor - per-vehicle policy with 4 action heads
│   │   ├── gcn_critic.py              # GCNTwinCritic - spatial twin Q-networks
│   │   └── critic.py                  # Legacy flat MLP critic
│   │
│   ├── simulator/                     # GPU-accelerated environment
│   │   ├── environment_v2.py          # GPUEnvironmentV2 (modular, recommended)
│   │   ├── environment.py             # GPUEnvironment (legacy)
│   │   ├── trip_manager.py            # Trip lifecycle management
│   │   ├── action_processor.py        # Action execution engine
│   │   ├── reward.py                  # Reward computation (Eq. 8)
│   │   ├── dynamics.py                # Energy & time dynamics
│   │   └── orchestrator.py            # Step orchestration
│   │
│   ├── training/                      # Training algorithms
│   │   ├── enhanced_trainer.py        # EnhancedSACTrainer (Semi-MDP + WDRO)
│   │   ├── trainer.py                 # Base SACTrainer
│   │   ├── semi_mdp.py               # Semi-MDP duration discounting (Eq. 12-13)
│   │   ├── wdro.py                    # WDRO robust optimization (Section 4)
│   │   ├── curriculum.py              # Curriculum learning scheduler
│   │   ├── episode_collector.py       # Episode data collection
│   │   ├── enhanced_collector.py      # Enhanced collector with duration tracking
│   │   ├── smart_assignment.py        # Smart vehicle-trip matching (~95% MILP)
│   │   └── distributed.py            # Multi-GPU distributed training
│   │
│   ├── features/                      # Feature engineering
│   │   ├── builder.py                 # FeatureBuilder (vehicle 16D, hex 5D, context 9D)
│   │   ├── replay_buffer.py           # GPU-resident PER replay buffer
│   │   └── structured_state_builder.py
│   │
│   ├── spatial/                       # Spatial operations
│   │   ├── grid.py                    # HexGrid (H3 hexagonal grid)
│   │   ├── distance.py                # Haversine distance matrix
│   │   ├── neighbors.py               # K-ring hex neighbor lookup
│   │   └── assignment.py              # Hungarian/Greedy/Auction matching
│   │
│   ├── state/                         # Tensor state representations
│   │   ├── fleet.py                   # TensorFleetState (positions, SoC, status)
│   │   ├── trips.py                   # TensorTripState (pickup, dropoff, revenue)
│   │   ├── stations.py                # TensorStationState (ports, power)
│   │   └── batched.py                 # BatchedEpisodeState
│   │
│   ├── data/                          # Data loading
│   │   └── real_trip_loader.py        # NYC taxi data loader
│   │
│   ├── scripts/                       # Entry points
│   │   ├── train.py                   # Main training script (CLI)
│   │   ├── evaluate.py                # Model evaluation
│   │   ├── benchmark.py               # Performance benchmarking
│   │   ├── visualize.py               # Result visualization
│   │   └── config.yaml                # Production config
│   │
│   ├── tests/                         # Test suite
│   │   ├── test_environment.py
│   │   ├── test_networks.py
│   │   ├── test_training.py
│   │   └── ...
│   │
│   └── utils/                         # Utilities
│       ├── profiler.py
│       └── visualizer.py
│
├── data/                              # Trip data
│   ├── nyc_full/                      # NYC taxi dataset (13.7M trips)
│   └── nyc_real_res9/                 # H3 resolution 9 variant
│
├── documentation/                     # Detailed docs (11 markdown files)
│   ├── main.md                        # Documentation index
│   ├── system_overview.md
│   ├── model_architecture.md
│   ├── training_process.md
│   └── ...
│
├── scripts/                           # Data preprocessing
│   └── preprocess_full_data.py
│
├── paper.md                           # Research paper references (BibTeX)
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.7+ (GPU required)
- 8GB+ GPU VRAM (16GB+ recommended for 1000 vehicles)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd ChargineOptimizationWasserstein/Wasserstein

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or
.\venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >= 1.13.0 | Deep learning framework |
| torch-geometric | >= 2.3.0 | Graph neural networks |
| numpy | >= 1.21.0 | Numerical computing |
| pandas | >= 1.3.0 | Data processing |
| scipy | >= 1.7.0 | Scientific computing |
| h3 | >= 3.7.0 | Hexagonal grid system |
| networkx | >= 2.6.0 | Graph operations |
| pyyaml | >= 6.0 | Configuration parsing |
| matplotlib | >= 3.4.0 | Visualization |
| gurobipy | >= 10.0 | MILP solver (optional, requires license) |

---

## Quick Start

### Minimal Training (synthetic data)

```bash
cd Wasserstein

python gpu_core/scripts/train.py \
    --env-v2 \
    --num-vehicles 100 \
    --num-hexes 200 \
    --episodes 500 \
    --batch-size 64 \
    --gpus 0
```

### Full Training (1000 vehicles, all features enabled)

```bash
python gpu_core/scripts/train.py \
    --env-v2 \
    --num-vehicles 1000 \
    --num-hexes 1300 \
    --episodes 10000 \
    --episode-duration-hours 10.0 \
    --batch-size 128 \
    --warmup-steps 10000 \
    --smart-assignment \
    --semi-mdp \
    --wdro \
    --temperature-annealing \
    --initial-temperature 1.0 \
    --final-temperature 0.1 \
    --gpus 0 \
    --tensorboard \
    --tensorboard-dir runs/full_training \
    --checkpoint-dir checkpoints/full_training \
    --save-interval 100 \
    --log-interval 10 \
    --eval-interval 50 \
    --eval-episodes 5
```

### Background Training (with nohup)

```bash
nohup python gpu_core/scripts/train.py \
    --env-v2 \
    --num-vehicles 1000 \
    --num-hexes 1300 \
    --episodes 10000 \
    --episode-duration-hours 10.0 \
    --batch-size 128 \
    --warmup-steps 10000 \
    --smart-assignment \
    --semi-mdp \
    --wdro \
    --temperature-annealing \
    --gpus 0 \
    --tensorboard \
    --tensorboard-dir runs/full_training \
    --checkpoint-dir checkpoints/full_training \
    > training.log 2>&1 &

# Monitor training
tail -f training.log
```

### With Real NYC Data

```bash
python gpu_core/scripts/train.py \
    --env-v2 \
    --real-data data/yellow_tripdata_2009-01.parquet \
    --trip-sample 0.3 \
    --num-vehicles 1000 \
    --episodes 5000 \
    --semi-mdp \
    --wdro \
    --gpus 0
```

---

## Training

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--env-v2` | False | Use modular GPUEnvironmentV2 (recommended) |
| `--num-vehicles` | 1000 | Number of vehicles in fleet |
| `--num-hexes` | 1300 | Number of hexagonal grid zones |
| `--episodes` | 5000 | Total training episodes |
| `--episode-duration-hours` | 10.0 | Simulated hours per episode |
| `--batch-size` | 256 | Training batch size |
| `--warmup-steps` | 10000 | Random exploration steps before training |
| `--gpus` | auto | Comma-separated GPU IDs (e.g., "0,1") |
| `--semi-mdp` | True | Enable Semi-MDP duration discounting |
| `--wdro` | True | Enable WDRO robust training |
| `--wdro-rho` | 0.3 | WDRO ambiguity radius |
| `--temperature-annealing` | True | Temperature decay for exploration |
| `--initial-temperature` | 1.0 | Starting softmax temperature |
| `--final-temperature` | 0.1 | Final softmax temperature |
| `--smart-assignment` | True | Smart vehicle-trip matching |
| `--mixed-precision` | False | Enable AMP training |
| `--distributed` | False | Enable multi-GPU DDP |
| `--tensorboard` | False | Enable TensorBoard logging |
| `--checkpoint-dir` | checkpoints | Checkpoint save directory |
| `--save-interval` | 100 | Save checkpoint every N episodes |
| `--log-interval` | 10 | Log metrics every N episodes |
| `--eval-interval` | 50 | Evaluate every N episodes |
| `--resume` | None | Resume from checkpoint path |

### Monitoring

```bash
# TensorBoard
tensorboard --logdir runs/

# Logged metrics:
# - Episode/reward, Episode/avg_reward_100
# - Episode/trips_served, Episode/avg_soc
# - Loss/actor_loss, Loss/critic_loss, Loss/avg_alpha
# - Actions/idle, Actions/serve, Actions/charge, Actions/reposition
# - Curriculum/pickup_distance, Curriculum/progress
# - Speed/steps_per_sec, Speed/updates_per_sec
```

---

## Configuration

### Hierarchy (lowest to highest priority)

1. `gpu_core/config/base.py` - Dataclass defaults
2. `gpu_core/config/defaults.yaml` - YAML defaults
3. `gpu_core/scripts/config.yaml` - Production config
4. CLI arguments - Highest priority overrides

### Key Hyperparameters

#### SAC Learning
| Parameter | Value | Description |
|-----------|-------|-------------|
| gamma | 0.99 | Discount factor |
| tau | 0.005 | Polyak soft update rate |
| lr_actor | 3e-4 | Actor learning rate |
| lr_critic | 3e-4 | Critic learning rate |
| lr_alpha | 3e-4 | Entropy coefficient learning rate |
| auto_alpha | True | Automatic entropy tuning |
| initial_alpha | 0.2 | Starting entropy weight |
| target_entropy | -0.98 * ln(4) | Target entropy ratio |

#### Network Architecture
| Parameter | Value | Description |
|-----------|-------|-------------|
| actor_hidden_dims | [128, 128] | Actor MLP layers |
| critic_hidden_dims | [256, 256] | Critic MLP layers |
| gcn_hidden_dim | 128 | GCN hidden dimension |
| gcn_layers | 2 | Number of GCN layers |
| vehicle_feature_dim | 16 | Per-vehicle features |
| hex_feature_dim | 5 | Per-hex features |
| context_dim | 9 | Global context features |

#### WDRO
| Parameter | Value | Description |
|-----------|-------|-------------|
| wdro_rho | 0.3 | Ambiguity radius |
| wdro_rho_target | 0.2 | Target risk budget |
| wdro_lambda_lr | 0.01 | Dual variable learning rate |
| wdro_inner_steps | 3 | Inner optimization steps |

#### Replay Buffer
| Parameter | Value | Description |
|-----------|-------|-------------|
| capacity | 30,000 (auto) | Auto-adjusted by fleet size & GPU memory |
| prioritized | True | Prioritized Experience Replay |
| alpha | 0.6 | PER priority exponent |
| beta | 0.4 -> 1.0 | Importance sampling annealing |

#### Reward Function
| Parameter | Value | Description |
|-----------|-------|-------------|
| driving_cost_per_km | 0.30 | Vehicle operating cost |
| electricity_cost_per_kwh | 0.18 | Charging cost |
| serve_bonus | 1.0 | Bonus per trip served |
| drop_penalty_per_order | 0.5 | Penalty per dropped request |
| wait_penalty_per_step | 0.02 | Customer wait penalty |
| reposition_success_bonus | 0.5 | Bonus for repositioning to demand |

### YAML Config Example

```yaml
# gpu_core/scripts/config.yaml
environment:
  num_vehicles: 1000
  num_hexes: 1300
  num_stations: 150

training:
  total_episodes: 10000
  batch_size: 128
  gamma: 0.99
  tau: 0.005
  learning_rate:
    actor: 0.0003
    critic: 0.0003

reward:
  driving_cost_per_km: 0.30
  electricity_cost_per_kwh: 0.18
  serve_bonus: 1.0
```

---

## Data

### Synthetic Data (default)

When no real data is provided, the system generates a synthetic hexagonal grid covering ~20km^2 of NYC with random trip patterns. Useful for testing and development.

### NYC Taxi Data

The system supports real trip data from the NYC Taxi & Limousine Commission:

```bash
# Preprocess raw data
python scripts/preprocess_full_data.py

# Train with real data
python gpu_core/scripts/train.py \
    --real-data data/yellow_tripdata_2009-01.parquet \
    --trip-sample 0.3 \
    --target-h3-resolution 8
```

**Data format**: Parquet files with columns for pickup/dropoff hex IDs, timestamps, fares.

**Included data** (`data/nyc_full/`):
- `hexagons.txt` - H3 hex grid IDs and coordinates
- `metadata.txt` - Dataset statistics (13.7M trips)

---

## Algorithms

### 1. Soft Actor-Critic (SAC)

Off-policy RL with maximum entropy regularization:

```
J(pi) = E[ sum_t gamma^t ( r_t + alpha * H(pi(.|s_t)) ) ]
```

- Twin Q-networks to reduce overestimation: Q = min(Q1, Q2)
- Automatic entropy tuning via dual gradient descent on alpha
- Per-vehicle stochastic policy with Categorical sampling

### 2. Graph Convolutional Network (GCN)

Spatial reasoning over the hexagonal grid (Paper Eq. 15):

```
H^(l+1) = sigma( D^{-1/2} A D^{-1/2} H^(l) W^(l) )
```

- 2-layer GCN with symmetric normalization
- Input: per-hex features (demand, supply, infrastructure)
- Output: contextual embeddings capturing neighborhood information
- Used in both Actor and Critic for consistent spatial reasoning

### 3. Semi-MDP Duration Discounting (Paper Eq. 12-13)

Actions have variable durations:

```
delta_serve = T(pickup) + T(trip)    # Travel time
delta_reposition = T(travel)          # Repositioning time
delta_charge = 1                      # One step

y_t = r_t + gamma^delta * V(s_{t+delta})
```

### 4. WDRO - Wasserstein Distributionally Robust Optimization (Paper Section 4)

Robust value estimation against demand distribution shifts:

```
max_pi  inf_{P: W_1(P, P_hat) <= rho}  E_P[R(xi; pi)]
```

- Graph-aligned Mahalanobis distance: `d_Q(xi, xi') = ||Q^{1/2}(xi - xi')||_2`
- Inner optimization via projected gradient ascent (K steps)
- Primal-dual lambda update for automatic risk budget calibration

### 5. Curriculum Learning

Gradually increases task difficulty:

```
pickup_distance: 10km -> 0km (cosine schedule over 80% of training)
```

Early episodes allow long pickups (easy), later episodes require nearby matches (realistic).

---

## Evaluation

### Run Evaluation

```bash
python gpu_core/scripts/evaluate.py \
    --checkpoint checkpoints/full_training/best.pt \
    --episodes 100 \
    --gpus 0
```

### Benchmarking

```bash
python gpu_core/scripts/benchmark.py
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| Net Revenue | Total trip fares - driving costs - electricity costs |
| Service Rate | Percentage of trip requests served |
| Average SOC | Fleet-wide average state of charge |
| Trips Served | Absolute number of trips completed |
| Action Distribution | % of IDLE / SERVE / CHARGE / REPOSITION |
| Steps/Second | Training throughput |

---

## Documentation

Detailed documentation is available in the `documentation/` directory:

| File | Content |
|------|---------|
| [main.md](documentation/main.md) | Documentation index |
| [system_overview.md](documentation/system_overview.md) | High-level architecture |
| [model_architecture.md](documentation/model_architecture.md) | SAC + GCN network details |
| [simulator.md](documentation/simulator.md) | Environment and physics |
| [training_process.md](documentation/training_process.md) | Training loop, curriculum, PER |
| [data_pipeline.md](documentation/data_pipeline.md) | Data loading and preprocessing |
| [configuration.md](documentation/configuration.md) | All configuration parameters |
| [features_and_options.md](documentation/features_and_options.md) | Feature vectors and action space |
| [replay_and_losses.md](documentation/replay_and_losses.md) | Replay buffer and loss functions |
| [constraints_and_metrics.md](documentation/constraints_and_metrics.md) | Physical constraints and metrics |
| [command.md](documentation/command.md) | Command-line usage guide |

### Troubleshooting

See [gpu_core/COMMON_ERRORS.md](gpu_core/COMMON_ERRORS.md) for solutions to common issues.

---

## Tests

```bash
# Run all tests
python -m pytest gpu_core/tests/ -v

# Run specific test module
python -m pytest gpu_core/tests/test_networks.py -v
python -m pytest gpu_core/tests/test_environment.py -v
python -m pytest gpu_core/tests/test_training.py -v
```

---

## GPU Memory Guide

| Fleet Size | Recommended Batch Size | Replay Buffer | GPU VRAM |
|------------|----------------------|---------------|----------|
| 100 vehicles | 256 | 100,000 | 4 GB |
| 500 vehicles | 128 | 30,000 | 8 GB |
| 1000 vehicles | 128 | 30,000 | 12-16 GB |
| 2000+ vehicles | 64 | 10,000 | 24+ GB |

The system auto-adjusts replay buffer capacity based on available GPU memory.

---

## References

Key papers that this work builds upon:

- Haarnoja et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor.* ICML.
- Kipf & Welling (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
- Qin et al. (2020). *Ride-hailing order dispatching at DiDi via reinforcement learning.* INFORMS.
- Liu et al. (2022). *Deep dispatching: A deep RL approach for vehicle dispatching.* Transportation Research Part E.
- Esfahani & Kuhn (2018). *Data-driven distributionally robust optimization using the Wasserstein metric.* Mathematical Programming.
- Miao et al. (2015). *Taxi dispatch with real-time sensing data: A receding horizon control approach.* IEEE TASE.

---

## License

This project is for academic research purposes.
