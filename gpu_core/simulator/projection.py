"""Feasible-action projection Pi_feas: rolling MILP (Eq. 12-16) with Algorithm 1 fallback (workstream D).

Decision variables, for every vehicle i that decides at this epoch (shared candidates, D6):

- x_io  serve order o whose pickup lies in i's 1-ring and keeps E >= E_min (Eq. 16);
- y_ig  reposition to the actor's target g (if energy-feasible) or stay where i is (D2, D7);
- z_is  book a free port at one of the 5 nearest stations reachable with E >= 0, power p_is;
- c_i   (vehicles already CHARGING) continue charging at the held port, power q_i (D12).

    max  sum R(o) x_io - c_drv * km - p_elec * dt * (p + q) - mu * ||v - a~||_1          (12)
    s.t. one action per vehicle; each order at most once                                    (13)
         new bookings + continuing chargers <= ports - ports held by non-deciding vehicles  (14, D3)
         sum q_i <= P_feed - power of vehicles committed to charge this epoch               (14, D3, D12)
         p_min z <= p <= min(P_max, battery headroom) z (same for q)                         (14, 15, 16)

Energy bounds (Eq. 16) are enforced by only creating variables that satisfy them: every action
is the vehicle's only action, so its post-action energy is known when the variable is created.
New bookings draw power only after the drive, so the feeder row covers the vehicles charging
during this epoch (D12). The L1 term compares one-hot mode / target / station choices with
the intention (mode probabilities, target, station) and the power as a fraction of P_max; it is
linearised with an auxiliary variable per power variable, so the model is a pure MILP (D1).

The solver runs under ``tau_max`` seconds, warm-started from Algorithm 1 (D8). Each call
records one outcome code (D11): OPTIMAL, TL_IMPROVED, TL_WARMSTART_ONLY, FALLBACK_NO_INCUMBENT,
FALLBACK_ERROR or EMPTY; Algorithm 1's solution is returned whenever there is no incumbent.
"""

import csv
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import torch

from ..state import VehicleStatus
from .physics import EPS

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - checked when the projection is built
    gp = None

SERVE, REPOS, STAY, CHARGE, CONTINUE = range(5)
CODES = ("OPTIMAL", "TL_IMPROVED", "TL_WARMSTART_ONLY", "FALLBACK_NO_INCUMBENT", "FALLBACK_ERROR", "EMPTY")


@dataclass
class Intention:
    """Per-vehicle intention a~ ([V] tensors; rows of vehicles that do not decide are ignored)."""
    mode_probs: torch.Tensor                 # [V, 3] P(serve), P(charge), P(reposition)
    repos_target: torch.Tensor               # [V] target cell (own cell = stay)
    power_frac: torch.Tensor                 # [V] preferred power / P_max
    station: Optional[torch.Tensor] = None   # [V] preferred station, -1 = none


@dataclass
class ProjectedAction:
    """Executable action for the simulator's exact path ([V]; mode -1 for non-deciding vehicles)."""
    mode: torch.Tensor          # 0 serve, 1 charge (book or continue), 2 reposition (own cell = stay)
    trip: torch.Tensor          # trip slot for serve, else -1
    target: torch.Tensor        # reposition target / own cell for stay / station cell
    station: torch.Tensor       # booked or continued station, else -1
    power_frac: torch.Tensor    # p / P_max for charge
    code: str
    stats: Dict = field(default_factory=dict)

    def step_kwargs(self) -> Dict:
        return dict(vehicle_charge_power=self.power_frac, milp_serve_trip_ids=self.trip, charge_station=self.station)


@dataclass
class _Problem:
    dec: np.ndarray            # global vehicle index of each deciding vehicle
    cells: np.ndarray
    energy: np.ndarray
    probs: np.ndarray          # [n, 3]
    g_hat: np.ndarray
    s_hat: np.ndarray
    p_tilde: np.ndarray
    slots: np.ndarray          # trip slot of each open order
    b_veh: np.ndarray          # binaries: local vehicle
    b_kind: np.ndarray
    b_ref: np.ndarray          # order (local) / target cell / station
    b_coef: np.ndarray         # objective without power terms
    b_reward: np.ndarray       # immediate reward (Algorithm 1 serve ranking, station distance for charge)
    b_station: np.ndarray      # station row (-1 none)
    p_bin: np.ndarray          # binary of each power variable
    p_ub: np.ndarray
    p_feeder: np.ndarray       # power drawn during this epoch (continuing chargers)
    cap: np.ndarray            # free ports per station for this epoch
    feeder_rhs: float


class ProjectionLog:
    """Per-call records and per-episode / per-test-set summaries (D11)."""

    def __init__(self):
        self.rows: List[Dict] = []

    def add(self, stats: Dict) -> None:
        self.rows.append(stats)

    def reset(self) -> None:
        self.rows = []

    def summary(self, rows: Optional[List[Dict]] = None) -> Dict:
        rows = self.rows if rows is None else rows
        n = len(rows)
        counts = {c: sum(r["code"] == c for r in rows) for c in CODES}
        out = {"calls": n, **{f"n_{c}": counts[c] for c in CODES}}
        out["effective_fallback_rate"] = ((counts["TL_WARMSTART_ONLY"] + counts["FALLBACK_NO_INCUMBENT"]
                                           + counts["FALLBACK_ERROR"]) / n) if n else 0.0
        solved = [r for r in rows if r["code"] != "EMPTY"]
        for key in ("t_build", "t_model", "t_solve", "t_total", "n_vehicles", "n_binary", "mode_changed",
                    "target_changed", "station_changed", "power_dev"):
            vals = [r[key] for r in solved if r.get(key) is not None]
            out[f"mean_{key}"] = float(np.mean(vals)) if vals else 0.0
        gaps = [r["mip_gap"] for r in solved if r.get("mip_gap") is not None]
        out["mean_mip_gap"] = float(np.mean(gaps)) if gaps else 0.0
        out["max_t_total"] = max((r["t_total"] for r in solved), default=0.0)
        return out

    def write_csv(self, path: str) -> None:
        if not self.rows:
            return
        keys = sorted({k for r in self.rows for k in r})
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)


class MILPProjection:

    def __init__(self, env, milp_config):
        if gp is None:
            raise ImportError("the MILP projection needs gurobipy; install it or run a variant without --milp (D9)")
        self.env = env
        self.ph = env.physics
        self.cand = env.candidates
        self.dist = self.cand.dist
        self.station_cells = self.cand.station_cells
        self.num_stations = len(self.station_cells)
        self.ports = int(env.config.station.num_ports)
        self.c_drv = float(env.config.reward.driving_cost_per_km)
        self.p_elec = float(env.config.reward.electricity_cost_per_kwh)
        self.mu = float(milp_config.mu)
        self.tau_max = float(milp_config.tau_max)
        self.warm_start = bool(milp_config.warm_start)
        self.threads = int(milp_config.threads)
        self.strict = bool(milp_config.strict)
        # D7: with only {actor target, stay} per vehicle the MILP must never prefer a far target
        # for its own sake; this holds when the farthest candidate costs less than 2 mu.
        if not self.cand.max_repos_km < 2.0 * self.mu / self.c_drv:
            raise ValueError(f"max reposition distance {self.cand.max_repos_km:.1f} km >= 2 mu / c_drv = "
                             f"{2 * self.mu / self.c_drv:.1f} km; raise milp.mu")
        self._gp_env = gp.Env(empty=True)
        self._gp_env.setParam("OutputFlag", 0)
        self._gp_env.start()
        self.log = ProjectionLog()

    # ------------------------------------------------------------------ problem
    def _problem(self, it: Intention) -> _Problem:
        env, ph = self.env, self.ph
        fleet, trips = env.fleet_state, env.trip_state
        avail = fleet.get_available_mask()
        dec = avail.nonzero(as_tuple=True)[0].cpu().numpy()
        cells = fleet.positions.cpu().numpy()[dec]
        energy = fleet.socs.cpu().numpy().astype(np.float64)[dec]
        status = fleet.status.cpu().numpy()[dec]
        own = fleet.charging_station.cpu().numpy()[dec]
        probs = it.mode_probs.detach().float().cpu().numpy()[dec].clip(min=0.0)
        probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-9)
        g_hat = it.repos_target.detach().long().cpu().numpy()[dec]
        p_tilde = it.power_frac.detach().float().cpu().numpy()[dec].clip(0.0, 1.0)
        s_hat = (it.station.detach().long().cpu().numpy()[dec] if it.station is not None
                 else np.full(len(dec), -1, dtype=np.int64))
        mu, H = self.mu, self.dist.shape[0]

        open_slots = trips.get_unassigned_mask().nonzero(as_tuple=True)[0].cpu().numpy()
        pick = trips.pickup_hex.cpu().numpy()[open_slots]
        tkm = trips.distance_km.cpu().numpy().astype(np.float64)[open_slots]
        fare = trips.fare.cpu().numpy().astype(np.float64)[open_slots]

        veh, kind, ref, coef, rew, st = [], [], [], [], [], []
        p_bin, p_ub, p_feed = [], [], []

        def add(v, k, r, c, w, s):
            veh.append(v); kind.append(k); ref.append(r); coef.append(c); rew.append(w); st.append(s)

        # serve
        vi, oi, km = self.cand.serve_pairs(cells, energy, pick, tkm)
        immediate = fare[oi] - self.c_drv * km
        for a, b_, c, w in zip(vi, oi, immediate - mu * 2.0 * (1.0 - probs[vi, 0]), immediate):
            add(int(a), SERVE, int(b_), float(c), float(w), -1)
        # reposition to the intended target, and stay
        for i in range(len(dec)):
            g, h = int(g_hat[i]), int(cells[i])
            pr = probs[i, 2]
            if 0 <= g < H and g != h:
                d = self.dist[h, g]
                if energy[i] - ph.energy_per_km * d >= ph.reserve_kwh - EPS:
                    add(i, REPOS, g, -self.c_drv * d - mu * 2.0 * (1.0 - pr), -self.c_drv * d, -1)
            add(i, STAY, h, -mu * (2.0 * (1.0 - pr) + (2.0 if g != h else 0.0)), 0.0, -1)
        # free ports for this epoch: ports minus those held by vehicles that do not decide
        holders = fleet.holds_port_mask() & ~avail
        held = torch.bincount(fleet.charging_station[holders], minlength=self.num_stations).cpu().numpy()
        cap = (self.ports - held).astype(np.float64)
        # charging: continue at the held port, or book one of the nearest stations
        stations = self.cand.station_candidates(cells)
        for i in range(len(dec)):
            pc = probs[i, 1]
            charging = status[i] == VehicleStatus.CHARGING
            if charging:
                ub = float(ph.max_power(torch.tensor(energy[i])))
                if ub >= ph.p_min_kw - 1e-9:
                    s = int(own[i])
                    pen = 2.0 * (1.0 - pc) + (2.0 if 0 <= s_hat[i] != s else 0.0)
                    add(i, CONTINUE, s, -mu * pen, 0.0, s)
                    p_bin.append(len(veh) - 1); p_ub.append(ub); p_feed.append(True)
            for s in stations[i]:
                s = int(s)
                if charging and s == own[i]:
                    continue
                if cap[s] < 1:
                    continue
                d = self.dist[cells[i], self.station_cells[s]]
                e_arr = energy[i] - ph.energy_per_km * d
                if e_arr < -EPS:
                    continue
                ub = float(ph.max_power(torch.tensor(max(e_arr, 0.0))))
                if ub < ph.p_min_kw - 1e-9:
                    continue
                pen = 2.0 * (1.0 - pc) + (2.0 if 0 <= s_hat[i] != s else 0.0)
                add(i, CHARGE, s, -self.c_drv * d - mu * pen, -d, s)
                p_bin.append(len(veh) - 1); p_ub.append(ub); p_feed.append(False)
        # feeder: vehicles committed to charge this epoch but not deciding (arrivals, D3)
        committed = (fleet.status == VehicleStatus.CHARGING) & (fleet.busy_until >= 0) & ~avail
        fixed = float(torch.minimum(fleet.charge_power[committed], ph.max_power(fleet.socs[committed])).sum()) \
            if committed.any() else 0.0
        return _Problem(
            dec=dec, cells=cells, energy=energy, probs=probs, g_hat=g_hat, s_hat=s_hat, p_tilde=p_tilde,
            slots=open_slots, b_veh=np.array(veh, dtype=np.int64), b_kind=np.array(kind, dtype=np.int64),
            b_ref=np.array(ref, dtype=np.int64), b_coef=np.array(coef), b_reward=np.array(rew),
            b_station=np.array(st, dtype=np.int64), p_bin=np.array(p_bin, dtype=np.int64),
            p_ub=np.array(p_ub), p_feeder=np.array(p_feed, dtype=bool), cap=cap,
            feeder_rhs=max(0.0, ph.feeder_kw - fixed))

    # ------------------------------------------------------------------ Algorithm 1
    def _fallback(self, P: _Problem):
        """Algorithm 1: vehicles by increasing energy; serve the feasible order with the highest
        immediate reward, else charge (continue at the held port while the feeder allows p_min, or
        book the nearest candidate station with a free port), else reposition to the intended
        target if energy-feasible, else stay. Returns (chosen binary per vehicle, power per power var)."""
        n = len(P.dec)
        by_v: List[List[int]] = [[] for _ in range(n)]
        for b, v in enumerate(P.b_veh):
            by_v[v].append(b)
        pvar = {int(b): k for k, b in enumerate(P.p_bin)}
        taken = np.zeros(len(P.slots), dtype=bool)
        cap = P.cap.copy()
        f_rem = P.feeder_rhs
        chosen = np.full(n, -1, dtype=np.int64)
        power = np.zeros(len(P.p_bin))
        for i in np.argsort(P.energy, kind="stable"):
            bs = by_v[i]
            serve = [b for b in bs if P.b_kind[b] == SERVE and not taken[P.b_ref[b]]]
            if serve:
                b = max(serve, key=lambda b: (P.b_reward[b], -P.b_ref[b]))
                chosen[i] = b
                taken[P.b_ref[b]] = True
                continue
            pick = None
            cont = [b for b in bs if P.b_kind[b] == CONTINUE]
            if cont and f_rem >= self.ph.p_min_kw - 1e-9 and cap[P.b_station[cont[0]]] >= 1:
                b = cont[0]
                p = min(P.p_ub[pvar[b]], f_rem)
                f_rem -= p
                pick = (b, p)
            else:
                new = [b for b in bs if P.b_kind[b] == CHARGE and cap[P.b_station[b]] >= 1]
                if new:
                    b = max(new, key=lambda b: (P.b_reward[b], -P.b_station[b]))     # nearest station
                    pick = (b, P.p_ub[pvar[b]])
            if pick is not None:
                b, p = pick
                chosen[i] = b
                power[pvar[b]] = p
                cap[P.b_station[b]] -= 1
                continue
            rep = [b for b in bs if P.b_kind[b] == REPOS]
            chosen[i] = rep[0] if rep else next(b for b in bs if P.b_kind[b] == STAY)
        return chosen, power

    def _objective(self, P: _Problem, chosen, power) -> float:
        on = np.isin(P.p_bin, chosen)
        dev = np.abs(power / self.ph.p_max_kw - P.p_tilde[P.b_veh[P.p_bin]] * on)
        return float(P.b_coef[chosen].sum() - self.p_elec * self.ph.step_hours * power.sum() - self.mu * dev.sum())

    # ------------------------------------------------------------------ MILP
    def build_model(self, P: _Problem, ws_chosen=None, ws_power=None):
        """Gurobi model of Eq. (12)-(16) built with the matrix API; returns (model, b, p, u)."""
        nB, nP, n = len(P.b_veh), len(P.p_bin), len(P.dec)
        m = gp.Model("pi_feas", env=self._gp_env)
        m.Params.TimeLimit = self.tau_max
        m.Params.Seed = 0
        if self.threads:
            m.Params.Threads = self.threads
        b = m.addMVar(nB, vtype=GRB.BINARY)
        p = m.addMVar(nP, lb=0.0, ub=P.p_ub if nP else 0.0)
        u = m.addMVar(nP, lb=0.0)
        m.setObjective(P.b_coef @ b - self.p_elec * self.ph.step_hours * p.sum() - self.mu * u.sum(), GRB.MAXIMIZE)
        ones = np.ones(nB)
        cols = np.arange(nB)
        m.addConstr(sp.csr_matrix((ones, (P.b_veh, cols)), shape=(n, nB)) @ b == 1)          # (13) one action
        srv = P.b_kind == SERVE
        if srv.any():
            m.addConstr(sp.csr_matrix((ones[srv], (P.b_ref[srv], cols[srv])), shape=(len(P.slots), nB)) @ b <= 1)
        stn = P.b_station >= 0
        if stn.any():
            A = sp.csr_matrix((ones[stn], (P.b_station[stn], cols[stn])), shape=(self.num_stations, nB))
            m.addConstr(A @ b <= P.cap)                                                      # (14) ports
        if nP:
            rows = np.arange(nP)
            if P.p_feeder.any():
                m.addConstr(P.p_feeder.astype(float) @ p <= P.feeder_rhs)                   # (14) feeder
            m.addConstr(p - sp.csr_matrix((P.p_ub, (rows, P.p_bin)), shape=(nP, nB)) @ b <= 0)            # (14)
            m.addConstr(p - sp.csr_matrix((np.full(nP, self.ph.p_min_kw), (rows, P.p_bin)), shape=(nP, nB)) @ b >= 0)  # (15)
            T = sp.csr_matrix((P.p_tilde[P.b_veh[P.p_bin]], (rows, P.p_bin)), shape=(nP, nB))
            m.addConstr(u - p / self.ph.p_max_kw + T @ b >= 0)                              # |p/P - p~ z|
            m.addConstr(u + p / self.ph.p_max_kw - T @ b >= 0)
        if self.warm_start and ws_chosen is not None:
            start = np.zeros(nB)
            start[ws_chosen] = 1.0
            b.Start = start
            if nP:
                p.Start = ws_power
                on = np.isin(P.p_bin, ws_chosen)
                u.Start = np.abs(ws_power / self.ph.p_max_kw - P.p_tilde[P.b_veh[P.p_bin]] * on)
        return m, b, p, u

    def _solve(self, P: _Problem, ws_chosen, ws_power):
        nP, n = len(P.p_bin), len(P.dec)
        t0 = time.perf_counter()
        m, b, p, u = self.build_model(P, ws_chosen, ws_power)
        t_model = time.perf_counter() - t0
        m.optimize()
        status, nsol = m.Status, m.SolCount
        out = None
        if nsol > 0:
            bx = b.X > 0.5
            chosen = np.full(n, -1, dtype=np.int64)
            chosen[P.b_veh[bx]] = np.nonzero(bx)[0]
            if (np.bincount(P.b_veh[bx], minlength=n) == 1).all():
                out = (chosen, (p.X if nP else np.zeros(0)) * np.isin(P.p_bin, chosen), float(m.ObjVal))
        gap = float(m.MIPGap) if nsol > 0 and np.isfinite(m.MIPGap) else None
        runtime = float(m.Runtime)
        m.dispose()
        return status, out, gap, runtime, t_model

    # ------------------------------------------------------------------ public
    def project(self, it: Intention) -> ProjectedAction:
        t0 = time.perf_counter()
        V, dev = self.env.num_vehicles, self.env.device
        P = self._problem(it)
        mode = torch.full((V,), -1, dtype=torch.long)
        trip = torch.full((V,), -1, dtype=torch.long)
        target = torch.full((V,), -1, dtype=torch.long)
        station = torch.full((V,), -1, dtype=torch.long)
        power = torch.zeros(V)
        stats = {"step": self.env.current_step, "n_vehicles": len(P.dec), "n_orders": len(P.slots),
                 "n_binary": len(P.b_veh), "n_continuous": 2 * len(P.p_bin), "mip_gap": None,
                 "obj": None, "ws_obj": None, "t_build": 0.0, "t_model": 0.0, "t_solve": 0.0}
        if len(P.dec) == 0:
            stats.update(code="EMPTY", t_total=time.perf_counter() - t0)
            self.log.add(stats)
            return ProjectedAction(mode.to(dev), trip.to(dev), target.to(dev), station.to(dev), power.to(dev), "EMPTY", stats)

        ws_chosen, ws_power = self._fallback(P)
        ws_obj = self._objective(P, ws_chosen, ws_power)
        stats["ws_obj"] = ws_obj
        stats["t_build"] = time.perf_counter() - t0
        try:
            status, out, gap, runtime, t_model = self._solve(P, ws_chosen, ws_power)
            stats["t_solve"], stats["mip_gap"], stats["t_model"] = runtime, gap, t_model
            if out is None:
                code, chosen, pw = "FALLBACK_NO_INCUMBENT", ws_chosen, ws_power
            else:
                chosen, pw, obj = out
                stats["obj"] = obj
                if status == GRB.OPTIMAL:
                    code = "OPTIMAL"
                elif self.warm_start and obj <= ws_obj + 1e-6 * max(1.0, abs(ws_obj)):
                    code = "TL_WARMSTART_ONLY"
                else:
                    code = "TL_IMPROVED"
        except gp.GurobiError as exc:
            if "size-limited license" in str(exc):
                # A restricted license would silently turn every call into Algorithm 1: fail closed.
                raise RuntimeError(f"Gurobi license is size-limited ({exc}); set GRB_LICENSE_FILE to the full license") from exc
            code, chosen, pw = "FALLBACK_ERROR", ws_chosen, ws_power
            stats["error"] = str(exc)
        if code.startswith("FALLBACK"):
            stats["obj"] = ws_obj
        stats["code"] = code

        # assemble the executable action
        pk = {int(b): k for k, b in enumerate(P.p_bin)}
        kinds = P.b_kind[chosen]
        refs = P.b_ref[chosen]
        dec = torch.from_numpy(P.dec)
        m_ = np.select([kinds == SERVE, np.isin(kinds, (CHARGE, CONTINUE))], [0, 1], 2)
        mode[dec] = torch.from_numpy(m_.astype(np.int64))
        srv = kinds == SERVE
        trip[dec[srv]] = torch.from_numpy(P.slots[refs[srv]])
        rep = np.isin(kinds, (REPOS, STAY))
        target[dec[rep]] = torch.from_numpy(refs[rep])
        chg = np.isin(kinds, (CHARGE, CONTINUE))
        station[dec[chg]] = torch.from_numpy(refs[chg])
        target[dec[chg]] = torch.from_numpy(self.station_cells[refs[chg]])
        pw_v = np.array([pw[pk[int(b)]] for b in chosen[chg]]) if chg.any() else np.zeros(0)
        power[dec[chg]] = torch.from_numpy(pw_v / self.ph.p_max_kw).float()

        # how far the projection moved from the intention (D11)
        intended = P.probs.argmax(axis=1)
        stats["mode_changed"] = float((intended != m_).mean())
        want_rep = (intended == 2) & rep
        stats["target_changed"] = float((refs[want_rep] != P.g_hat[want_rep]).mean()) if want_rep.any() else 0.0
        want_st = chg & (P.s_hat >= 0)
        stats["station_changed"] = float((refs[want_st] != P.s_hat[want_st]).mean()) if want_st.any() else 0.0
        stats["power_dev"] = float(np.abs(pw_v / self.ph.p_max_kw - P.p_tilde[chg]).mean()) if chg.any() else 0.0
        stats["t_total"] = time.perf_counter() - t0
        self.log.add(stats)
        return ProjectedAction(mode.to(dev), trip.to(dev), target.to(dev), station.to(dev), power.to(dev), code, stats)


def make_projection(env) -> MILPProjection:
    """Pi_feas for this environment: stations, ports, physics, prices and candidate sets come from
    the env, the MILP settings from ``config.milp`` (one source; C4, D6, F40). Raises ImportError
    without gurobipy (D9)."""
    return MILPProjection(env, env.config.milp)


def check_executed(info, projection: MILPProjection) -> None:
    """Strict mode: every projected action must have been executed exactly (C7)."""
    c = info.executed.counts
    if projection.strict and (c["infeasible"] or c["power_clipped"]):
        raise RuntimeError(f"projected action not executable at step {info.step}: "
                           f"{c['infeasible']} infeasible, {c['power_clipped']} power clipped")
