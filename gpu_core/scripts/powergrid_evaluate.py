#!/usr/bin/env python3
"""
Charging power drawn from the grid over one split or calendar day (paper Fig. 6; F5, F48).

    python gpu_core/scripts/powergrid_evaluate.py --checkpoint checkpoint/full/best.pt --split xmas \
        --out-csv powergrid.csv --out-json powergrid.json
    python gpu_core/scripts/powergrid_evaluate.py --preset greedy --day 2009-01-27 --out-csv greedy.csv

Any vehicle-level checkpoint (PD-RSAC, SAC, MAPPO, MADDPG) runs with its own pipeline (MILP as
trained; ``--milp`` / ``--no-milp`` override); ``--preset greedy`` runs the heuristic. The power
of an epoch is the energy actually drawn during it divided by the epoch length, at the epoch's
wall-clock time; the summary adds the feeder limit, violations and the share of epochs at the limit.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.data.factory import add_data_args, apply_data_args, build_run, config_for_evaluation, split_hours  # noqa: E402

FIELDS = ["label", "day", "step", "date", "timestamp_hhmm", "minute_of_day", "total_charging_power_kw",
          "num_vehicles_charging", "mean_power_per_vehicle_kw", "feeder_violation", "ports_held"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None, help="vehicle-level checkpoint")
    p.add_argument("--preset", default=None, help="greedy: the heuristic instead of a checkpoint")
    p.add_argument("--config", default=None, help="YAML instead of the config embedded in the checkpoint")
    p.add_argument("--label", default=None)
    p.add_argument("--milp", dest="milp", action="store_true", default=None, help="force the MILP projection")
    p.add_argument("--no-milp", dest="milp", action="store_false", help="force the raw executor")
    p.add_argument("--day", default=None, help="single calendar day YYYY-MM-DD of a configured dataset; overrides --split")
    add_data_args(p, default_split="xmas")
    p.add_argument("--num-vehicles", type=int, default=None, help="with --preset greedy")
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--out-json", default=None)
    return p.parse_args(argv)


def timeseries(rows: List[Dict], day: str) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        n = r["vehicles_charging"]
        out.append({"label": r["label"], "day": day, "step": r["step"], "date": r["date"], "timestamp_hhmm": r["time"],
                    "minute_of_day": int((r["clock_s"] % 86400) // 60), "total_charging_power_kw": r["feeder_load_kw"],
                    "num_vehicles_charging": n, "mean_power_per_vehicle_kw": r["feeder_load_kw"] / n if n else 0.0,
                    "feeder_violation": r["feeder_violation"], "ports_held": r["ports_held"]})
    return out


def main(argv=None) -> Dict:
    args = parse_args(argv)
    from gpu_core.evaluation.protocol import evaluate_agent_on_split, evaluate_policy_on_split
    if bool(args.checkpoint) == bool(args.preset):
        raise SystemExit("give exactly one of --checkpoint or --preset greedy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = config_for_evaluation(checkpoint, args.config)
        if args.milp is not None:
            config.milp.enabled = bool(args.milp)
        label = args.label or Path(args.checkpoint).stem
    else:
        from gpu_core.config.presets import resolve_preset
        if args.preset != "greedy":
            raise SystemExit("only --preset greedy runs without a checkpoint")
        config = resolve_preset("greedy")
        if args.num_vehicles is not None:
            config.environment.num_vehicles = args.num_vehicles
        label = args.label or "greedy"
    apply_data_args(config, args)
    split = args.split
    if args.day:
        import pandas as pd
        split = f"day:{args.day}"
        config.data.splits[split] = [args.day, (pd.Timestamp(args.day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")]
    config.episode.duration_hours = 24.0 if args.synthetic else split_hours(config, split)
    run = build_run(config, device, split=split, synthetic=args.synthetic)
    if checkpoint is not None:
        from gpu_core.learning.evaluation import load_vehicle_agent
        agent = load_vehicle_agent(checkpoint, config, run.env, device, projected=run.projection is not None)
        res, rec = evaluate_agent_on_split(run, agent, split=split, label=label, seed=args.seed, max_steps=args.max_steps,
                                           progress=True)
    else:
        from gpu_core.baselines.greedy import GreedyPolicy
        res, rec = evaluate_policy_on_split(run, GreedyPolicy(run.env), split=split, label=label, seed=args.seed,
                                            max_steps=args.max_steps, progress=True)
    rows = timeseries(rec.rows, args.day or split)
    g = res["grid"]
    summary = {"label": label, "day": args.day or split, "milp_enabled": res["milp"], "num_steps": res["steps"],
               "step_duration_minutes": config.episode.step_duration_minutes,
               "average_charging_power_kw": g["feeder_mean_kw"], "peak_charging_power_kw": g["feeder_peak_kw"],
               "peak_time": g["feeder_peak_time"], "feeder_limit_kw": g["feeder_limit_kw"],
               "feeder_violation_steps": g["feeder_violation_steps"], "feeder_tight_share": g["feeder_tight_share"],
               "total_charged_energy_kwh": g["energy_charged_kwh"], "total_bought_energy_kwh": g["energy_bought_kwh"]}
    print(f"[Power grid | {label} | {summary['day']}] {summary['num_steps']} epochs, mean {summary['average_charging_power_kw']:.0f} kW, "
          f"peak {summary['peak_charging_power_kw']:.0f} kW at {summary['peak_time']} (limit {summary['feeder_limit_kw']:.0f} kW), "
          f"violations {summary['feeder_violation_steps']}, stored {summary['total_charged_energy_kwh']:.0f} kWh")
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "timeseries": rows}, f, indent=2, default=float)
    return summary


if __name__ == "__main__":
    main()
