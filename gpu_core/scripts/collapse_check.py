#!/usr/bin/env python3
"""Policy-collapse check on a scaled-down real environment (workstream E diagnostics).

The full experiment is 1000 vehicles on a 10 % trip sample with 35 stations x 5 ports and a
7000 kW feeder. This script keeps every ratio per vehicle: N vehicles see a (10 % x N / 1000)
sample, round(175 x N / 1000) single-port stations (the first ones of the cover-then-fill order)
and a feeder of 7 x N kW. It trains the vehicle-level agent and prints, per episode, what the
deciding vehicles intend and execute, how often they choose SERVE when an order is reachable,
whether drained vehicles charge, and how many end the episode stranded.

Columns: ``service_pct`` = orders accepted / orders that appeared (%); ``i_*`` = share of the
deciding vehicles whose *intention* (the actor's sample) was serve / reposition / charge;
``x_*`` = share whose *executed* action was serve (got an order), reposition, stay (incl. serve
intentions left without an order), charge (new port) or continue (kept charging).

    python gpu_core/scripts/collapse_check.py --vehicles 100 --episodes 40            # raw executor
    python gpu_core/scripts/collapse_check.py --vehicles 100 --episodes 40 --milp     # projection
"""

import argparse
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.config import ConfigLoader  # noqa: E402
from gpu_core.data.factory import build_env, load_manifest  # noqa: E402
from gpu_core.learning.factory import make_agent  # noqa: E402
from gpu_core.learning.trainer import Trainer  # noqa: E402


def scaled_manifest(manifest, stations: int, ports: int):
    m = dataclasses.replace(manifest, station_cells=list(manifest.station_cells[:stations]), ports_per_station=ports)
    m.spatial_hash, m.infra_hash = m.compute_hashes()
    return m


def scaled_config(config, vehicles: int, full_vehicles: int = 1000, full_sample: float = 0.1):
    f = vehicles / full_vehicles
    config.environment.num_vehicles = vehicles
    config.data.sample_ratio = full_sample * f
    total_ports = config.environment.num_stations * config.station.num_ports
    config.environment.num_stations = max(1, round(total_ports * f))
    config.station.num_ports = 1
    config.station.feeder_kw = config.station.feeder_kw * f
    return config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="gpu_core/scripts/config_wdro.yaml")
    ap.add_argument("--vehicles", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--batch", type=int, default=None, help="default: 2048 x vehicles / 1000")
    ap.add_argument("--warmup-episodes", type=int, default=2)
    ap.add_argument("--milp", action="store_true")
    ap.add_argument("--wdro", action="store_true", help="enable the robust target after wdro.warmup_episodes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates-per-step", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=5, help="deterministic evaluation episode every N episodes (0 = off)")
    ap.add_argument("--out", default=None, help="CSV of per-episode diagnostics")
    ap.add_argument("--save", default=None, help="checkpoint path written at the end")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    cfg = ConfigLoader.from_yaml(a.config)
    full_ports = cfg.environment.num_stations * cfg.station.num_ports
    cfg = scaled_config(cfg, a.vehicles)
    cfg.training.batch_size = a.batch or max(64, round(2048 * a.vehicles / 1000))
    cfg.training.warmup_steps = a.warmup_episodes * cfg.episode.steps_per_episode
    cfg.wdro.enabled = a.wdro
    cfg.milp.enabled = a.milp
    if a.updates_per_step:
        cfg.agent.updates_per_step = a.updates_per_step
    manifest = scaled_manifest(load_manifest(cfg), cfg.environment.num_stations, 1)
    env, train_data, _ = build_env(cfg, torch.device("cpu"), split="train", manifest=manifest, verbose=False)
    projection = None
    if a.milp:
        from gpu_core.simulator.projection import make_projection
        projection = make_projection(env)
    if cfg.training.algo in ("mappo", "ppo"):
        cfg.training.warmup_steps = 0          # on-policy: no uniform warm-up
    agent = make_agent(env, cfg, "cpu", projected=a.milp)
    trainer = Trainer(cfg, env, agent, projection=projection, dataset=train_data, seed=a.seed)
    print(f"[collapse] {a.vehicles} vehicles, sample {cfg.data.sample_ratio:.3%}, {cfg.environment.num_stations} x 1 ports "
          f"(full run {full_ports}), feeder {cfg.station.feeder_kw:.0f} kW, {train_data.total_trips:,} train trips, "
          f"batch {cfg.training.batch_size}, {'MILP' if a.milp else 'raw executor'}, WDRO {'on' if a.wdro else 'off'}, "
          f"learner {agent.algo}")
    rows = []
    for ep in range(a.episodes):
        r = trainer.train_episode(a.episodes)
        u = r.update_stats
        row = dict(episode=ep + 1, warm=int(trainer.warmup_left > 0 or ep < a.warmup_episodes), profit=round(r.profit, 1),
                   served=r.served, loaded=r.loaded, dropped=r.dropped, service_pct=round(100.0 * r.served / max(r.loaded, 1), 1),
                   i_serve=r.intention_share.get("serve", 0), i_repos=r.intention_share.get("reposition", 0),
                   i_charge=r.intention_share.get("charge", 0),
                   x_serve=r.executed_share.get("serve", 0), x_repos=r.executed_share.get("reposition", 0),
                   x_stay=r.executed_share.get("stay", 0), x_charge=r.executed_share.get("charge", 0),
                   x_continue=r.executed_share.get("continue", 0),
                   serve_available=r.serve_available_share, serve_when_available=r.serve_when_available,
                   low_energy_stay=r.low_energy_idle_share, energy_end=r.mean_energy_end, stuck_end=r.stuck_vehicles_end,
                   pi_hat_match=r.pi_hat_match, h_mode=u.get("h_mode", np.nan), pi_serve=u.get("pi_serve", np.nan),
                   alpha_mode=u.get("alpha_mode", np.nan), loss_q=u.get("loss_q", np.nan), q_mean=u.get("q_mean", np.nan),
                   tau=round(agent.tau, 3), seconds=round(r.seconds, 1))
        rows.append(row)
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        if a.eval_every and (ep + 1) % a.eval_every == 0:
            ev = trainer.evaluate(env, projection, episodes=1, seed=99_000 + ep)[0]
            print(f"EVAL episode={ep + 1} profit={ev.profit:.1f} served={ev.served} loaded={ev.loaded} "
                  f"service_pct={100.0 * ev.served / max(ev.loaded, 1):.1f} "
                  f"i_serve={ev.intention_share.get('serve', 0):.3f} i_charge={ev.intention_share.get('charge', 0):.3f} "
                  f"x_serve={ev.executed_share.get('serve', 0):.3f} x_stay={ev.executed_share.get('stay', 0):.3f} "
                  f"x_charge={ev.executed_share.get('charge', 0):.3f} x_continue={ev.executed_share.get('continue', 0):.3f} "
                  f"serve_when_available={ev.serve_when_available:.3f} low_energy_stay={ev.low_energy_idle_share:.3f} "
                  f"energy_end={ev.mean_energy_end:.3f} stuck_end={ev.stuck_vehicles_end}", flush=True)
    if a.save:
        trainer.save(a.save)
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
