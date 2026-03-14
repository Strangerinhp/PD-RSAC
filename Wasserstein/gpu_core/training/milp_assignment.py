import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class AssignmentResult:
    action_types: torch.Tensor
    serve_targets: torch.Tensor
    charge_targets: torch.Tensor
    reposition_targets: torch.Tensor
    assignment_quality: float


class MILPAssignment:
    """
    MILP feasible-action projection per IEEE_T_IV.pdf Section IV (Eq. 12-16).

    Solves:
        max  Σ_{i,o} R_t(o)·x_{io}
           - Σ_i c_drv·d_{i,t}
           - Σ_{i,s} p^elec·p_{i,s}·Δt
           - μ·‖v − ã_t‖₁                      (intention proximity, Eq. 12)

    subject to:
        Eq. 13  — one action per vehicle; each trip served at most once
        Eq. 14  — port capacity, feeder limit, per-port power bounds
        Eq. 15  — minimum charging power when z_{i,s}=1
        Eq. 16  — energy feasibility (SoC bounds after the action)

    If no feasible solution is found before τ_max, Algorithm 1 (greedy with
    SoC-ascending priority) is applied as the fallback.
    """

    def __init__(
        self,
        num_vehicles: int,
        num_hexes: int,
        num_stations: int = 50,
        device: str = "cuda",
        # Physical parameters — paper Section VIII.A
        e_max_kwh: float = 100.0,           # Battery capacity (paper: 100 kWh)
        e_min_ratio: float = 0.20,           # Safety floor as fraction of e_max
        eta_c: float = 0.95,                # Charging efficiency
        eta_drv: float = 0.20,              # Driving energy kWh/km
        p_max_s: float = 50.0,              # Max charging power per port (kW)
        p_min: float = 1.0,                 # Min charging power if z_{i,s}=1 (Eq. 15)
        p_max_feed: float = 2000.0,         # Feeder limit kW (Eq. 14)
        port_capacity: int = 10,            # Ports per station (Eq. 14)
        delta_t: float = 1.0,              # Time-step duration (hours)
        c_drv: float = 0.30,               # Driving cost $/km (paper: $0.30/km)
        p_elec: float = 0.18,              # Electricity price $/kWh
        mu: float = 1.0,                   # Intention-proximity weight μ (Eq. 12)
        tau_max: float = 10.0,             # Branch-and-bound time limit τ_max (s)
        max_reposition_neighbors: int = 6,  # |N(h_{i,t})| candidates per vehicle
        max_pickup_distance: float = 5.0,       # Hard pickup radius (km) — must match env
        station_positions: Optional[np.ndarray] = None,  # Real station hex IDs
    ):
        self.num_vehicles = num_vehicles
        self.num_hexes = num_hexes
        self.num_stations = num_stations
        self.device = torch.device(device)

        self.e_max_kwh = e_max_kwh
        self.e_min_kwh = e_max_kwh * e_min_ratio
        self.eta_c = eta_c
        self.eta_drv = eta_drv
        self.p_max_s = p_max_s
        self.p_min = p_min
        self.p_max_feed = p_max_feed
        self.port_capacity = port_capacity
        self.delta_t = delta_t
        self.c_drv = c_drv
        self.p_elec = p_elec
        self.mu = mu
        self.tau_max = tau_max
        self.max_reposition_neighbors = max_reposition_neighbors
        self.max_pickup_distance = max_pickup_distance

        # Station positions: real data preferred, evenly-spaced indices as fallback
        if station_positions is not None:
            self.station_positions = np.asarray(station_positions, dtype=int)
        else:
            self.station_positions = np.linspace(
                0, num_hexes - 1, num_stations
            ).astype(int)

        self.station_capacity = np.full(num_stations, float(port_capacity))
        self._first_assign_call = True
        print(f"[MILP] MILPAssignment initialized (V={num_vehicles}, S={num_stations}, tau_max={tau_max}s)")

    # ── Public API ────────────────────────────────────────────────────────────

    def assign(
        self,
        vehicle_positions: torch.Tensor,
        vehicle_socs: torch.Tensor,        # SoC in kWh (range 0-100)
        vehicle_status: torch.Tensor,
        trip_pickups: torch.Tensor,
        trip_dropoffs: torch.Tensor,
        trip_fares: torch.Tensor,
        distance_matrix: torch.Tensor,
        policy_probs: torch.Tensor,         # [V, 3]: SERVE|CHARGE|REPOS (IDLE removed)
        current_step: int = 0,
        episode_steps: int = 120,
    ) -> AssignmentResult:
        """
        Solve the MILP projection (Eq. 12-16).
        Falls back to Algorithm 1 (greedy) if MILP times out or finds no solution.
        """
        V = len(vehicle_positions)
        T = len(trip_pickups)
        S = self.num_stations

        # ── Convert to CPU NumPy ──────────────────────────────────────────────
        v_pos     = vehicle_positions.cpu().numpy().astype(int)
        v_soc_kwh = vehicle_socs.cpu().numpy() / 100.0 * self.e_max_kwh
        v_probs   = policy_probs.cpu().numpy()
        v_status  = vehicle_status.cpu().numpy().astype(int)
        t_pick    = trip_pickups.cpu().numpy().astype(int)  if T > 0 else np.array([], dtype=int)
        t_drop    = trip_dropoffs.cpu().numpy().astype(int) if T > 0 else np.array([], dtype=int)
        t_fares   = trip_fares.cpu().numpy()                if T > 0 else np.array([])
        dist_mat  = distance_matrix.cpu().numpy()
        s_pos     = self.station_positions
        repo_cands = self._get_repo_candidates(v_pos, dist_mat, t_pick if T > 0 else None)
        K = repo_cands.shape[1]

        _log_step = (current_step % 20 == 0)

        if self._first_assign_call:
            print(f"[MILP] assign() first call — V={V}, T={T}")
            if T > 0:
                print(f"[MILP]   fares:    min={t_fares.min():.2f}  max={t_fares.max():.2f}  mean={t_fares.mean():.2f}")
            print(f"[MILP]   SoC(kWh): min={v_soc_kwh.min():.1f}  max={v_soc_kwh.max():.1f}")
            print(f"[MILP]   params:   delta_t={self.delta_t:.4f}h  mu={self.mu}  p_elec={self.p_elec}")
            print(f"[MILP]   max charge cost/step = {self.p_elec * self.delta_t * self.p_max_s:.3f} $/vehicle")
            sample_net_rv = (t_fares.mean() - self.c_drv * 3.0) if T > 0 else 0.0
            print(f"[MILP]   sample serve net_rv (3km total) = {sample_net_rv:.3f} $")
            print(f"[MILP]   intention range: mu*(2*1-1)={self.mu:.1f}  mu*(2*0-1)={-self.mu:.1f}")
            self._first_assign_call = False

        def _greedy_np():
            res = self._greedy_fallback(
                v_pos, v_soc_kwh, v_probs, t_pick, t_drop, t_fares, dist_mat, s_pos, repo_cands
            )
            return (
                res.action_types.cpu().numpy(), res.serve_targets.cpu().numpy(),
                res.charge_targets.cpu().numpy(), res.reposition_targets.cpu().numpy(), 0.0,
            )

        # ── Greedy warm start ─────────────────────────────────────────────────
        ws         = self._greedy_fallback(
            v_pos, v_soc_kwh, v_probs, t_pick, t_drop, t_fares, dist_mat, s_pos, repo_cands
        )
        ws_actions = ws.action_types.cpu().numpy()
        ws_serve   = ws.serve_targets.cpu().numpy()
        ws_charge  = ws.charge_targets.cpu().numpy()
        ws_repos   = ws.reposition_targets.cpu().numpy()

        # ── Build Gurobi model ────────────────────────────────────────────────
        gp_env = gp.Env(empty=True)
        gp_env.setParam("OutputFlag", 0)
        gp_env.start()
        model = gp.Model("MILP_Fleet", env=gp_env)
        model.setParam("TimeLimit", self.tau_max)
        model.setParam("Heuristics", 0.3)
        model.setParam("NoRelHeurTime", self.tau_max * 0.4)

        x = {}; y = {}; z = {}; p = {}  # serve, repos, charge, priority vars (idle removed)

        if T > 0:
            for i in range(V):
                for j in range(T):
                    if float(dist_mat[v_pos[i], t_pick[j]]) <= self.max_pickup_distance:
                        x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        for i in range(V):
            for k in range(K):
                y[i, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}")

        for i in range(V):
            for s in range(S):
                z[i, s] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{s}")
                p[i, s] = model.addVar(lb=0.0, ub=self.p_max_s, name=f"p_{i}_{s}") # No idle variable - IDLE action removed from policy

        n_reachable_pairs = len(x)  # valid (i,j) within max_pickup_distance
        model.update()

        # ── Warm start hints ──────────────────────────────────────────────────
        for i in range(V):
            a = int(ws_actions[i])
            # idle[i].Start = 1 if a == 0 else 0 # IDLE action removed

            if T > 0:
                j_warm = int(ws_serve[i]) if a == 0 else -1  # SERVE=0 in new scheme
                for j in range(T):
                    if (i, j) in x:
                        x[i, j].Start = 1 if j == j_warm else 0

            s_warm = int(ws_charge[i]) if a == 1 else -1  # CHARGE=1 in new scheme
            for s in range(S):
                z[i, s].Start = 1 if s == s_warm else 0
                p[i, s].Start = float(self.p_max_s) if s == s_warm else 0.0

            g_warm = int(ws_repos[i]) if a == 2 else -1  # REPOSITION=2 in new scheme
            for k in range(K):
                y[i, k].Start = 1 if (a == 3 and int(repo_cands[i, k]) == g_warm) else 0

        # ── Objective — Eq. 12 ───────────────────────────────────────────────
        obj = gp.LinExpr()

        if T > 0:
            for i in range(V):
                for j in range(T):
                    if (i, j) not in x:
                        continue
                    d_pk   = float(dist_mat[v_pos[i], t_pick[j]])
                    d_tr   = float(dist_mat[t_pick[j], t_drop[j]])
                    net_rv = float(t_fares[j]) - self.c_drv * (d_pk + d_tr)
                    obj   += net_rv * x[i, j]

        for i in range(V):
            for k in range(K):
                d_rb = float(dist_mat[v_pos[i], int(repo_cands[i, k])])
                obj -= self.c_drv * d_rb * y[i, k]

        for i in range(V):
            for s in range(S):
                obj -= self.p_elec * self.delta_t * p[i, s]

        if T > 0:
            for i in range(V):
                # SERVE=0, CHARGE=1, REPOS=2 in new scheme
                a_serve = float(v_probs[i, 0])
                for j in range(T):
                    if (i, j) in x:
                        obj += self.mu * (2.0 * a_serve - 1.0) * x[i, j]

        for i in range(V):
            # SERVE=0, CHARGE=1, REPOS=2 in new scheme
            a_repos = float(v_probs[i, 2])
            for k in range(K):
                obj += self.mu * (2.0 * a_repos - 1.0) * y[i, k]

        for i in range(V):
            # SERVE=0, CHARGE=1, REPOS=2 in new scheme
            a_chg = float(v_probs[i, 1])
            for s in range(S):
                obj += self.mu * (2.0 * a_chg - 1.0) * z[i, s]

        # for i in range(V): # IDLE action removed
        #     a_idle = float(v_probs[i, 0])
        #     obj += self.mu * (2.0 * a_idle - 1.0) * idle[i]

        model.setObjective(obj, GRB.MAXIMIZE)

        # ── Constraints ───────────────────────────────────────────────────────
        for i in range(V):
            serve_sum = gp.quicksum(x[i, j] for j in range(T) if (i, j) in x) if T > 0 else gp.LinExpr()
            repos_sum = gp.quicksum(y[i, k] for k in range(K))
            chg_sum   = gp.quicksum(z[i, s] for s in range(S))
            model.addConstr(serve_sum + repos_sum + chg_sum == 1, name=f"one_action_{i}")

        if T > 0:
            for j in range(T):
                model.addConstr(
                    gp.quicksum(x[i, j] for i in range(V) if (i, j) in x) <= 1, name=f"trip_once_{j}"
                )

        for i in range(V):
            pass  # No idle constraint — IDLE removed from policy

        for s in range(S):
            model.addConstr(
                gp.quicksum(z[i, s] for i in range(V)) <= self.station_capacity[s],
                name=f"port_cap_{s}",
            )

        model.addConstr(
            gp.quicksum(p[i, s] for i in range(V) for s in range(S)) <= self.p_max_feed,
            name="feeder_limit",
        )

        for i in range(V):
            for s in range(S):
                model.addConstr(p[i, s] <= self.p_max_s * z[i, s], name=f"p_ub_{i}_{s}")
                model.addConstr(p[i, s] >= self.p_min  * z[i, s], name=f"p_lb_{i}_{s}")

        for i in range(V):
            e_next = gp.LinExpr(v_soc_kwh[i])
            if T > 0:
                for j in range(T):
                    if (i, j) not in x:
                        continue
                    d_pk = float(dist_mat[v_pos[i], t_pick[j]])
                    d_tr = float(dist_mat[t_pick[j], t_drop[j]])
                    e_next -= self.eta_drv * (d_pk + d_tr) * x[i, j]
            for k in range(K):
                d_rb = float(dist_mat[v_pos[i], int(repo_cands[i, k])])
                e_next -= self.eta_drv * d_rb * y[i, k]
            for s in range(S):
                e_next += self.eta_c * self.delta_t * p[i, s]
            model.addConstr(e_next >= self.e_min_kwh, name=f"soc_min_{i}")
            model.addConstr(e_next <= self.e_max_kwh, name=f"soc_max_{i}")

        model.optimize()

        if model.SolCount == 0:
            acts, srv, chg, rep, qual = _greedy_np()
        else:
            acts = np.zeros(V, dtype=int)
            srv  = np.full(V, -1, dtype=int)
            chg  = np.full(V, -1, dtype=int)
            rep  = np.zeros(V, dtype=int)

            for i in range(V):
                assigned = False
                if T > 0:
                    for j in range(T):
                        if (i, j) in x and x[i, j].X > 0.5:
                            acts[i] = 0; srv[i] = j; assigned = True; break  # SERVE=0
                if assigned:
                    continue
                for s in range(S):
                    if z[i, s].X > 0.5:
                        acts[i] = 1; chg[i] = s; assigned = True; break  # CHARGE=1
                if assigned:
                    continue
                for k in range(K):
                    if y[i, k].X > 0.5:
                        acts[i] = 2; rep[i] = int(repo_cands[i, k]); assigned = True; break  # REPOS=2
                if not assigned:
                    acts[i] = 2; rep[i] = int(repo_cands[i, 0])  # REPOS=2 fallback

            if model.Status == GRB.OPTIMAL:
                qual = 1.0
            else:
                denom = max(abs(float(model.ObjBound)), 1e-9)
                qual  = min(abs(float(model.ObjVal)) / denom, 1.0)

            if int((acts == -1).sum()) > 0:  # -1 means unassigned (should not happen)
                acts, srv, chg, rep, qual = _greedy_np()

        if _log_step:
            n_srv = int((acts == 0).sum())  # SERVE=0
            n_chg = int((acts == 1).sum())  # CHARGE=1
            n_rep = int((acts == 2).sum())  # REPOS=2
            print(f"[MILP step {current_step}] V_avail={V}, T={T}, reach={n_reachable_pairs} "
                  f"→ serve={n_srv} charge={n_chg} repos={n_rep}")

        return AssignmentResult(
            action_types=torch.tensor(acts, dtype=torch.long, device=self.device),
            serve_targets=torch.tensor(srv,  dtype=torch.long, device=self.device),
            charge_targets=torch.tensor(chg, dtype=torch.long, device=self.device),
            reposition_targets=torch.tensor(rep, dtype=torch.long, device=self.device),
            assignment_quality=qual,
        )

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _get_repo_candidates(
        self, v_pos: np.ndarray, dist_mat: np.ndarray,
        t_pick: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Top-K repo candidates per vehicle, demand-aware when trips are present.

        Targets the K nearest trip pickup hexes so that repositioning moves
        vehicles toward actual demand rather than arbitrary adjacent cells.
        Falls back to nearest spatial hexes when fewer than K trips exist.
        """
        K = min(self.max_reposition_neighbors, self.num_hexes - 1)
        candidates = np.zeros((len(v_pos), K), dtype=int)

        demand_hexes = np.unique(t_pick) if (t_pick is not None and len(t_pick) > 0) else None

        for i, pos in enumerate(v_pos):
            if demand_hexes is not None and len(demand_hexes) > 0:
                # Remove current hex from demand candidates
                dh = demand_hexes[demand_hexes != pos]
                if len(dh) == 0:
                    dh = demand_hexes  # edge case: all trips at current pos

                d_dem = dist_mat[pos][dh]
                n_dem = min(K, len(dh))
                order_dem = np.argpartition(d_dem, n_dem - 1)[:n_dem] if n_dem > 1 else np.array([0])
                chosen = dh[order_dem[np.argsort(d_dem[order_dem])]]

                if len(chosen) < K:
                    # Pad with nearest spatial hexes not already in chosen
                    d_sp = dist_mat[pos].copy()
                    d_sp[pos] = np.inf
                    for h in chosen:
                        d_sp[h] = np.inf
                    n_pad = K - len(chosen)
                    pad_idx = np.argpartition(d_sp, n_pad)[:n_pad]
                    pad = pad_idx[np.argsort(d_sp[pad_idx])]
                    chosen = np.concatenate([chosen, pad])
            else:
                # No trips: pure spatial proximity
                d = dist_mat[pos].copy()
                d[pos] = np.inf
                idx = np.argpartition(d, K)[:K]
                chosen = idx[np.argsort(d[idx])]

            candidates[i] = chosen[:K]
        return candidates

    def _greedy_fallback(
        self,
        v_pos: np.ndarray,
        v_soc_kwh: np.ndarray,
        v_probs: np.ndarray,
        t_pick: np.ndarray,
        t_drop: np.ndarray,
        t_fares: np.ndarray,
        dist_mat: np.ndarray,
        s_pos: np.ndarray,
        repo_cands: np.ndarray,
    ) -> AssignmentResult:
        """
        Algorithm 1 — Greedy feasible-action projection (paper Section IV.C).

        Vehicles processed in ascending SoC order.  Per vehicle:
          1. Assign highest-net-revenue feasible trip (SoC guard via Eq. 16).
          2. Else assign nearest charger with available port & feeder headroom.
          3. Else assign nearest reposition target (ignoring SoC if needed).
        IDLE is never assigned — it is not a valid policy action (removed).
        """
        V = len(v_pos)
        T = len(t_pick)
        S = self.num_stations
        K = repo_cands.shape[1]

        action_types   = np.zeros(V, dtype=int)
        serve_targets  = np.full(V, -1, dtype=int)
        charge_targets = np.full(V, -1, dtype=int)
        repos_targets  = np.zeros(V, dtype=int)

        trip_taken       = np.zeros(T, dtype=bool)
        port_remaining   = self.station_capacity.copy()
        feeder_remaining = self.p_max_feed

        order = np.argsort(v_soc_kwh)         # ascending SoC

        for i in order:
            # ── 1. Serve ─────────────────────────────────────────────────────
            best_trip   = -1
            best_profit = -np.inf
            for j in range(T):
                if trip_taken[j]:
                    continue
                d_pk = dist_mat[v_pos[i], t_pick[j]]
                if d_pk > self.max_pickup_distance:
                    continue                        # pickup distance guard
                d_tr = dist_mat[t_pick[j], t_drop[j]]
                e_needed = self.eta_drv * (d_pk + d_tr)
                if v_soc_kwh[i] - e_needed < self.e_min_kwh:
                    continue                        # SoC guard
                profit = float(t_fares[j]) - self.c_drv * (d_pk + d_tr)
                if profit > best_profit:
                    best_profit = profit
                    best_trip   = j

            if best_trip >= 0:
                action_types[i]       = 0  # SERVE=0
                serve_targets[i]      = best_trip
                trip_taken[best_trip] = True
                continue

            # ── 2. Charge ─────────────────────────────────────────────────────
            best_station = -1
            best_dist    = np.inf
            for s in range(S):
                if port_remaining[s] < 1 or feeder_remaining < self.p_min:
                    continue
                d_s = dist_mat[v_pos[i], s_pos[s]]
                if d_s < best_dist:
                    best_dist    = d_s
                    best_station = s

            if best_station >= 0:
                action_types[i]              = 1  # CHARGE=1
                charge_targets[i]            = best_station
                port_remaining[best_station] -= 1
                feeder_remaining             -= min(self.p_max_s, feeder_remaining)
                continue

            # ── 3. Reposition or Idle ─────────────────────────────────────────
            best_repo      = -1
            best_repo_dist = np.inf
            for k in range(K):
                g    = int(repo_cands[i, k])
                d_rb = dist_mat[v_pos[i], g]
                if v_soc_kwh[i] - self.eta_drv * d_rb >= self.e_min_kwh \
                        and d_rb < best_repo_dist:
                    best_repo_dist = d_rb
                    best_repo      = g

            if best_repo >= 0:
                action_types[i]  = 2  # REPOS=2
                repos_targets[i] = best_repo
            else:
                # IDLE is removed — force nearest repo candidate ignoring SoC
                action_types[i]  = 2  # REPOS=2
                repos_targets[i] = int(repo_cands[i, 0])

        return AssignmentResult(
            action_types=torch.tensor(action_types,    dtype=torch.long, device=self.device),
            serve_targets=torch.tensor(serve_targets,  dtype=torch.long, device=self.device),
            charge_targets=torch.tensor(charge_targets, dtype=torch.long, device=self.device),
            reposition_targets=torch.tensor(repos_targets, dtype=torch.long, device=self.device),
            assignment_quality=0.0,
        )
