# PD-RSAC: Primal-Dual Robust Soft Actor-Critic

## PD-RSAC

**PD-RSAC** stands for **Primal-Dual Robust Soft Actor-Critic**. It is a vehicle-level EV fleet control model that combines **Semi-MDP duration-aware backups**, **Wasserstein-1 DRO** for robustness to demand shift, and **MILP feasible-action projection** so final actions satisfy SoC, charger, and feeder constraints. In this repository, PD-RSAC is the `full` preset: the vehicle-level SAC with WDRO, semi-MDP and the MILP projection enabled.

## Prepare data (once)

```bash
# 1. preprocess each raw monthly file into per-day partitions (H3 r7, exclusion reasons kept)
python -m gpu_core.data.preprocess --input <RAW_DIR>/yellow_tripdata_2009-01.parquet --output-root ../data/processed
python -m gpu_core.data.preprocess --input <RAW_DIR>/yellow_tripdata_2009-12.parquet --output-root ../data/processed
# 2. freeze the service domain, stations and initial-position weights from the training split
python -m gpu_core.spatial.manifest --processed-root ../data/processed --datasets yellow_tripdata_2009-01     --train-start 2009-01-02 --train-end 2009-01-22 --out ../data/manifests/nyc_r7_v1.json
```

Paths, sampling and the train / val / test / xmas / ctrl splits live in `gpu_core/scripts/base.yaml` (`data`).

---

## Run PD-RSAC training

```bash
python gpu_core/scripts/train.py --preset full --episodes 500 --tensorboard
```

A preset fixes the family YAML and every variant switch (MILP, WDRO, scenario metric, semi-MDP, population
critic) and writes to `checkpoint/<preset>` and `logs/<preset>`:

| Preset | Family | MILP | WDRO | Metric | gamma | Role |
|---|---|---|---|---|---|---|
| `full` | wdro | yes | yes | graph | gamma^Delta | PD-RSAC |
| `no_milp` | wdro | - | yes | graph | gamma^Delta | ablation |
| `no_graph` | wdro | yes | yes | identity | gamma^Delta | ablation |
| `diag_only` | wdro | yes | yes | diag(w) | gamma^Delta | ablation |
| `iac` | wdro | yes | yes | graph | gamma^Delta | ablation: critic without m_i |
| `sac_milp` | sac | yes | - | - | gamma | ablation "w/o WDRO" |
| `sac` | sac | - | - | - | gamma | baseline |

Baselines: `mappo`, `mappo_milp`, `maddpg`, `maddpg_milp` (same encoder, actor network, candidates,
executor / projection, batch, learning rates, gamma and episodes as `sac`; fixed gamma), and `greedy`
(not trained; run it with `run_protocol.py --preset greedy`). 

Trains on the `train` split and selects checkpoints on `--eval-split` (default `val`, 22-24 Jan, one
continuous 864-epoch episode) through the same pipeline, MILP included. With the MILP, every episode appends
the projection statistics (outcome codes, effective fallback rate, timings) to
`<log_dir>/projection_history.csv`. Add `--resume <CHECKPOINT_PATH>` to continue a run; `--synthetic` runs a tiny
smoke test without data.

---

## Evaluate a PD-RSAC checkpoint

```bash
python gpu_core/scripts/evaluate.py --checkpoint <CHECKPOINT_PATH> --split test --projection-csv <CSV>
python gpu_core/scripts/evaluate.py --checkpoint <CHECKPOINT_PATH> --split xmas     
```

One deterministic episode spans the whole split; the configuration, MILP included, is the one embedded in
the checkpoint (`--milp` / `--no-milp` override it and the result says so). Results report profit, revenue,
costs, penalties, service, feeder peak / violations / share of epochs at the limit, port use per station,
denied bookings, D11 projection statistics, per-epoch timings and a per-day table.

## Baselines

```bash
python gpu_core/scripts/train.py --preset mappo_milp          # or mappo, maddpg, maddpg_milp, sac, sac_milp
python gpu_core/scripts/run_protocol.py --preset greedy --out-dir results/greedy
python gpu_core/scripts/heuristic_matching.py --split xmas --output results/greedy_xmas.json
```

## Test protocol

```bash
python gpu_core/scripts/run_protocol.py --checkpoint checkpoint/full/best.pt --label full --out-dir results/full
python gpu_core/scripts/run_protocol.py --compare results/full results/sac --out results/table.csv
```

Runs the test week (25-31 Jan, 2016 epochs, per-day table), `xmas` and `ctrl` (288 epochs each) and reports
`xmas - ctrl`. At 1000 vehicles with the MILP an epoch takes about 0.8 s on CPU (projection about 0.7 s).

---

## Run power-grid evaluation

```bash
python gpu_core/scripts/powergrid_evaluate.py --checkpoint <CHECKPOINT_PATH> --split xmas --out-csv <OUTPUT_CSV> --out-json <OUTPUT_JSON>
```

---

## Configuration

Each family YAML (`config_sac.yaml`, `config_wdro.yaml`, `config_mappo.yaml`, `config_maddpg.yaml`) starts with
`base: base.yaml`. The base owns `environment, episode, data, reward, physics, vehicle, station, milp`; a family file may
not redefine them, and unknown keys raise an error. CLI flags override the YAML only when given. Each run writes
`resolved_config.yaml` and `manifest.json` (git commit, data hash, spatial hash, library versions, CLI args) into
its log directory, and the same record is stored in every checkpoint.

---

## Requirements / conditions
- Set `GRB_LICENSE_FILE` to your license file if it is not at the default location.
- Install Python dependencies from `requirements.txt`
- `--distributed` is refused: the old multi-GPU path never synchronised gradients
- GPU training is the intended setup - 17GB VRAM needed to run all settings
- `--milp` requires **Gurobi** with a full license (`GRB_LICENSE_FILE`)
