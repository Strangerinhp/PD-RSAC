"""Greedy heuristic of paper VIII-B (workstream G, F54).

"At each decision epoch, vehicles are assigned to trips by maximizing immediate profit under a
maximum pickup distance constraint. Vehicles with state-of-charge below 20% are prioritized for
charging. An iterative greedy matching strategy is applied to maximize the number of served
trips while satisfying energy and assignment constraints."

Per decision epoch, over the deciding vehicles:

1. charging first (SoC < ``low``): a vehicle already at a charger keeps charging; an idle one
   books the nearest of its 5 candidate stations with a free port that it can reach (E >= 0 on
   arrival, Q4), lowest energy first. A low vehicle that finds no port joins the matching;
2. iterative greedy matching: every (vehicle, order) pair with the pickup in the vehicle's 1-ring
   (the shared pickup radius, Q3a) and E - eta (d_pickup + d_trip) >= E_min (Eq. 4) gets its
   immediate profit fare - c_drv (d_pickup + d_trip) (Eq. 8; electricity is paid when charging);
   the best remaining pair is taken until no feasible pair with profit > ``min_profit`` is left;
3. everything else: a charger below ``charge_target`` keeps charging, otherwise unplugs; an idle
   vehicle stays (the paper describes no rebalancing).

The chosen (vehicle, trip, station, power) go to the simulator's exact path, so the pairs the
heuristic matched are the pairs executed (no re-matching); every action is feasible by
construction (``infeasible_executed`` stays 0). Power is P_max^s, limited by battery headroom.
"""

from typing import Dict

import numpy as np
import torch

from ..simulator.executor import CHARGE, NOOP, REPOSITION, SERVE
from ..state import VehicleStatus


class GreedyPolicy:

    def __init__(self, env, low: float = None, charge_target: float = 0.8, min_profit: float = 0.0):
        self.env = env
        self.low = float(env.config.vehicle.low_energy_fraction if low is None else low)
        self.charge_target = float(charge_target)
        self.min_profit = float(min_profit)
        self.c_drv = float(env.config.reward.driving_cost_per_km)
        self.dist = env._distances.cpu().numpy().astype(np.float64)
        self.ring1 = env._ring1.cpu().numpy()
        self.near = np.asarray(env.candidates.near_stations)
        self.station_cells = env._station_cells.cpu().numpy()
        self.ports = int(env.config.station.num_ports)

    def act(self, env=None) -> Dict[str, torch.Tensor]:
        env = env or self.env
        ph, fl, ts = env.physics, env.fleet_state, env.trip_state
        V = env.num_vehicles
        dec = fl.get_available_mask().cpu().numpy()
        E = fl.socs.cpu().numpy().astype(np.float64)
        pos = fl.positions.cpu().numpy()
        status = fl.status.cpu().numpy()
        own = fl.charging_station.cpu().numpy()
        cap = ph.battery_kwh
        mode = np.full(V, NOOP, np.int64)
        target = pos.copy()
        station = np.full(V, -1, np.int64)
        power = np.ones(V, np.float32)
        trip = np.full(V, -1, np.int64)
        decided = np.zeros(V, bool)

        def p_frac(energy):
            return float(ph.max_power(torch.tensor(energy, dtype=torch.float32))) / ph.p_max_kw

        charging = dec & (status == VehicleStatus.CHARGING)
        low = dec & (E < self.low * cap)
        # 1) charging priority
        for v in np.nonzero(low & charging)[0]:
            mode[v], station[v], power[v], decided[v] = CHARGE, own[v], p_frac(E[v]), True
        holders = fl.holds_port_mask().cpu().numpy()
        free = self.ports - np.bincount(own[holders], minlength=len(self.station_cells))
        idle_low = np.nonzero(low & ~charging)[0]
        for v in idle_low[np.argsort(E[idle_low], kind="stable")]:
            for s in self.near[pos[v]]:
                if s < 0 or free[s] <= 0:
                    continue
                e_arr = E[v] - ph.energy_per_km * self.dist[pos[v], self.station_cells[s]]
                if e_arr < -1e-6 or p_frac(max(e_arr, 0.0)) * ph.p_max_kw < ph.p_min_kw:
                    continue
                free[s] -= 1
                mode[v], station[v], power[v], decided[v] = CHARGE, s, p_frac(max(e_arr, 0.0)), True
                break

        # 2) iterative greedy matching on immediate profit
        open_slots = ts.get_unassigned_mask().nonzero(as_tuple=True)[0].cpu().numpy()
        cand = np.nonzero(dec & ~decided)[0]
        if len(open_slots) and len(cand):
            pick = ts.pickup_hex.cpu().numpy()[open_slots]
            km_trip = ts.distance_km.cpu().numpy().astype(np.float64)[open_slots]
            fare = ts.fare.cpu().numpy().astype(np.float64)[open_slots]
            tid = ts.trip_ids.cpu().numpy()[open_slots]
            d_pick = self.dist[pos[cand][:, None], pick[None, :]]                       # [n_v, n_o]
            km = d_pick + km_trip[None, :]
            ok = self.ring1[pos[cand][:, None], pick[None, :]]
            ok &= E[cand][:, None] - ph.energy_per_km * km >= ph.reserve_kwh - 1e-6
            profit = fare[None, :] - self.c_drv * km
            ok &= profit > self.min_profit
            vi, oi = np.nonzero(ok)
            if len(vi):
                # best profit first; ties: shorter pickup, then earlier order, then vehicle index
                order = np.lexsort((cand[vi], tid[oi], d_pick[vi, oi], -profit[vi, oi]))
                v_used, o_used = set(), set()
                for j in order:
                    v, o = int(cand[vi[j]]), int(oi[j])
                    if v in v_used or o in o_used:
                        continue
                    v_used.add(v)
                    o_used.add(o)
                    mode[v], trip[v], decided[v] = SERVE, int(open_slots[o]), True

        # 3) the rest: chargers continue below the target or unplug; idle vehicles stay
        for v in np.nonzero(dec & ~decided)[0]:
            if charging[v] and E[v] < self.charge_target * cap and p_frac(E[v]) > 0:
                mode[v], station[v], power[v] = CHARGE, own[v], p_frac(E[v])
            elif charging[v]:
                mode[v], target[v] = REPOSITION, pos[v]          # explicit stay: unplug
            else:
                mode[v] = NOOP
        return {"mode": torch.as_tensor(mode), "target": torch.as_tensor(target), "station": torch.as_tensor(station),
                "power": torch.as_tensor(power), "trip": torch.as_tensor(trip)}

    def step(self, env=None):
        """Act and execute on the exact path; returns (intent, step outputs)."""
        env = env or self.env
        a = self.act(env)
        res = env.step(a["mode"], a["target"], vehicle_charge_power=a["power"], milp_serve_trip_ids=a["trip"],
                       charge_station=a["station"])
        return a["mode"], res
