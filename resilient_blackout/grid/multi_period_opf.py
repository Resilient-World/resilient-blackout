# Copyright (c) 2026, Resilient World
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Chronological multi-period OPF scheduler.

Provides ``MultiPeriodOPFScheduler``, a rolling-horizon LP co-optimizer
that schedules generation, battery dispatch, and load curtailment across
sequential time steps with generator ramp-rate and battery SOC coupling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog

logger = logging.getLogger(__name__)

_EPS: float = 1e-9


# ---------------------------------------------------------------------------
# MultiPeriodOPFScheduler
# ---------------------------------------------------------------------------


class MultiPeriodOPFScheduler:
    """Multi-period OPF scheduler via large-scale linear programming.

    Co-optimises generation costs and expected unserved energy (EENS)
    over a coupled time horizon:

    .. math::
        \min \sum_{t=1}^{T} \left(
            \sum_{g} C_g \, P_{g,t}
            + \sum_{i} \nu \, L_{\text{shed},i,t}
        \right)

    Subject to:

    - DC power balance per bus
    - Generator output limits
    - Generator ramp-rate limits
    - Battery SOC dynamics with round-trip efficiency
    - Battery energy / power bounds
    - Optional line thermal limits via PTDF

    Parameters
    ----------
    horizon_steps : int
        Number of time steps in the scheduling horizon.  Default 24.
    dt_hours : float
        Duration of each time step in hours.  Default 1.0.
    voll_usd_per_mwh : float
        Value of Lost Load — penalty weight for shed energy in $/MWh.
        Default 10 000.
    max_ramp_pu_per_step : float
        Maximum generator ramp as a fraction of rated capacity per step.
        Default 0.3 (30 %).
    n_cost_segments : int
        Piecewise-linear segments for quadratic generator cost curves.
        Default 3.

    Attributes
    ----------
    horizon_steps : int
    dt_hours : float
    voll : float
    max_ramp_pu : float
    """

    def __init__(
        self,
        horizon_steps: int = 24,
        dt_hours: float = 1.0,
        voll_usd_per_mwh: float = 10_000.0,
        max_ramp_pu_per_step: float = 0.3,
        n_cost_segments: int = 3,
    ) -> None:
        if horizon_steps <= 0:
            raise ValueError(f"horizon_steps must be > 0, got {horizon_steps}")
        if dt_hours <= 0:
            raise ValueError(f"dt_hours must be > 0, got {dt_hours}")

        self.horizon_steps = horizon_steps
        self.dt_hours = dt_hours
        self.voll = voll_usd_per_mwh / 1000.0  # $/kWh
        self.max_ramp_pu = max_ramp_pu_per_step
        self.n_cost_segments = max(1, n_cost_segments)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_schedule(
        self,
        net: Any,
        load_profile: np.ndarray,
        storage_specs: Optional[List[Dict[str, Any]]] = None,
        gen_costs: Optional[np.ndarray] = None,
        ptdf: Optional[np.ndarray] = None,
        line_limits_a: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Solve the multi-period OPF as a single large LP.

        Parameters
        ----------
        net : pandapowerNet
        load_profile : np.ndarray
            Active load per bus per timestep.  Shape ``(T, n_load_buses)``
            or ``(T,)`` if one aggregated load.  Units MW.
        storage_specs : list of dict or None
            Each dict describes a battery:

            - ``bus`` (int)
            - ``p_max_mw`` (float)
            - ``e_max_mwh`` (float)
            - ``e_min_mwh`` (float, default 0)
            - ``e_init_mwh`` (float)
            - ``e_term_mwh`` (float, default ``e_init_mwh``)
            - ``eta_in`` (float, default 0.95)
            - ``eta_out`` (float, default 0.95)
        gen_costs : np.ndarray or None
            Linear cost per generator in $/MWh.  Shape ``(n_gens,)``.
            Defaults to column ``cost_per_mwh`` on ``net.gen`` or 50 $/MWh.
        ptdf : np.ndarray or None
            Power Transfer Distribution Factor matrix.
            Shape ``(n_lines, n_buses)``.  If ``None``, line limits are
            ignored.
        line_limits_a : np.ndarray or None
            Thermal current limits per line (A).  Required if ``ptdf`` is
            given.

        Returns
        -------
        dict
            ``status``, ``gen_schedule``, ``battery_schedule``,
            ``shed_per_bus``, ``line_loading_percent``, ``objective``,
            ``message``.
        """
        T = self.horizon_steps
        dt = self.dt_hours
        net = self._ensure_storage(net)

        # --- topology extraction --------------------------------------
        gen_mask = net.gen["in_service"].values.astype(bool)
        gen_idx = net.gen.index[gen_mask].values
        n_gens = len(gen_idx)
        gen_pmax = net.gen.loc[gen_mask, "max_p_mw"].values.astype(np.float64)
        gen_pmin = net.gen.loc[gen_mask, "min_p_mw"].values.astype(np.float64) if "min_p_mw" in net.gen.columns else np.zeros(n_gens)
        gen_buses = net.gen.loc[gen_mask, "bus"].values.astype(np.int64)

        if gen_costs is None:
            if "cost_per_mwh" in net.gen.columns:
                gen_costs = net.gen.loc[gen_mask, "cost_per_mwh"].values.astype(np.float64)
            else:
                gen_costs = np.full(n_gens, 50.0, dtype=np.float64)
        gen_costs = np.asarray(gen_costs, dtype=np.float64)
        if gen_costs.shape[0] != n_gens:
            raise ValueError(f"gen_costs length ({gen_costs.shape[0]}) != n_gens ({n_gens})")

        load_mask = net.load["in_service"].values.astype(bool)
        load_idx = net.load.index[load_mask].values
        n_loads = len(load_idx)
        load_buses = net.load.loc[load_mask, "bus"].values.astype(np.int64)

        bus_mask = net.bus["in_service"].values.astype(bool)
        bus_idx = net.bus.index[bus_mask].values
        n_buses = len(bus_idx)
        bus_to_col = {int(b): i for i, b in enumerate(bus_idx)}

        # Expand load profile
        load_profile = np.asarray(load_profile, dtype=np.float64)
        if load_profile.ndim == 1:
            if n_loads == 1:
                load_profile = load_profile.reshape(-1, 1)
            else:
                raise ValueError("load_profile must be 2-D when n_loads > 1")
        if load_profile.shape[0] != T:
            raise ValueError(f"load_profile rows ({load_profile.shape[0]}) != horizon_steps ({T})")
        if load_profile.shape[1] != n_loads:
            raise ValueError(f"load_profile cols ({load_profile.shape[1]}) != n_loads ({n_loads})")

        storage_specs = storage_specs or []
        n_storage = len(storage_specs)

        # --- variable layout ------------------------------------------
        # per timestep: gen[n_gens], char[n_storage], disch[n_storage],
        #               soc[n_storage], shed[n_loads]
        n_vars_per_t = n_gens + n_storage + n_storage + n_storage + n_loads
        n_vars = T * n_vars_per_t

        var_map = {
            "gen_start": 0,
            "char_start": n_gens,
            "disch_start": n_gens + n_storage,
            "soc_start": n_gens + 2 * n_storage,
            "shed_start": n_gens + 3 * n_storage,
            "vars_per_t": n_vars_per_t,
        }

        # --- objective ------------------------------------------------
        c = np.zeros(n_vars, dtype=np.float64)
        for t in range(T):
            base = t * n_vars_per_t
            c[base : base + n_gens] = gen_costs * dt  # $/MWh * h = $/kWh scaled
            c[base + var_map["shed_start"] : base + var_map["shed_start"] + n_loads] = self.voll * dt

        # --- inequality constraints (A_ub x <= b_ub) ----------------
        A_rows: List[np.ndarray] = []
        b_vals: List[float] = []

        # Gen max
        for t in range(T):
            base = t * n_vars_per_t
            for g in range(n_gens):
                row = np.zeros(n_vars, dtype=np.float64)
                row[base + g] = 1.0
                A_rows.append(row)
                b_vals.append(gen_pmax[g])

        # Gen min (as -P_g <= -P_min)
        for t in range(T):
            base = t * n_vars_per_t
            for g in range(n_gens):
                row = np.zeros(n_vars, dtype=np.float64)
                row[base + g] = -1.0
                A_rows.append(row)
                b_vals.append(-gen_pmin[g])

        # Battery charge max
        for t in range(T):
            base = t * n_vars_per_t
            for s in range(n_storage):
                p_max = float(storage_specs[s].get("p_max_mw", 1.0))
                row = np.zeros(n_vars, dtype=np.float64)
                row[base + var_map["char_start"] + s] = 1.0
                A_rows.append(row)
                b_vals.append(p_max)

        # Battery discharge max
        for t in range(T):
            base = t * n_vars_per_t
            for s in range(n_storage):
                p_max = float(storage_specs[s].get("p_max_mw", 1.0))
                row = np.zeros(n_vars, dtype=np.float64)
                row[base + var_map["disch_start"] + s] = 1.0
                A_rows.append(row)
                b_vals.append(p_max)

        # Battery SOC max
        for t in range(T):
            base = t * n_vars_per_t
            for s in range(n_storage):
                e_max = float(storage_specs[s].get("e_max_mwh", 10.0))
                row = np.zeros(n_vars, dtype=np.float64)
                row[base + var_map["soc_start"] + s] = 1.0
                A_rows.append(row)
                b_vals.append(e_max)

        # Battery SOC min (as -E <= -E_min)
        for t in range(T):
            base = t * n_vars_per_t
            for s in range(n_storage):
                e_min = float(storage_specs[s].get("e_min_mwh", 0.0))
                row = np.zeros(n_vars, dtype=np.float64)
                row[base + var_map["soc_start"] + s] = -1.0
                A_rows.append(row)
                b_vals.append(-e_min)

        # Load shed <= load
        for t in range(T):
            base = t * n_vars_per_t
            for ld in range(n_loads):
                row = np.zeros(n_vars, dtype=np.float64)
                row[base + var_map["shed_start"] + ld] = 1.0
                A_rows.append(row)
                b_vals.append(load_profile[t, ld])

        # Ramp-rate constraints
        if n_gens > 0:
            # Initial ramp from prev dispatch (assumed P_min at t=0)
            for g in range(n_gens):
                ramp = gen_pmax[g] * self.max_ramp_pu
                base = 0
                row_up = np.zeros(n_vars, dtype=np.float64)
                row_up[base + g] = 1.0
                A_rows.append(row_up)
                b_vals.append(gen_pmin[g] + ramp)

                row_down = np.zeros(n_vars, dtype=np.float64)
                row_down[base + g] = -1.0
                A_rows.append(row_down)
                b_vals.append(-(gen_pmin[g] - ramp))

            for t in range(1, T):
                base = t * n_vars_per_t
                prev_base = (t - 1) * n_vars_per_t
                for g in range(n_gens):
                    ramp = gen_pmax[g] * self.max_ramp_pu
                    row_up = np.zeros(n_vars, dtype=np.float64)
                    row_up[base + g] = 1.0
                    row_up[prev_base + g] = -1.0
                    A_rows.append(row_up)
                    b_vals.append(ramp)

                    row_down = np.zeros(n_vars, dtype=np.float64)
                    row_down[base + g] = -1.0
                    row_down[prev_base + g] = 1.0
                    A_rows.append(row_down)
                    b_vals.append(ramp)

        # PTDF line limits
        if ptdf is not None and line_limits_a is not None:
            n_lines = ptdf.shape[0]
            for t in range(T):
                base = t * n_vars_per_t
                for li in range(n_lines):
                    row_pos = np.zeros(n_vars, dtype=np.float64)
                    row_neg = np.zeros(n_vars, dtype=np.float64)
                    for g in range(n_gens):
                        col = bus_to_col.get(int(gen_buses[g]))
                        if col is not None:
                            row_pos[base + g] = ptdf[li, col]
                            row_neg[base + g] = -ptdf[li, col]
                    for ld in range(n_loads):
                        col = bus_to_col.get(int(load_buses[ld]))
                        if col is not None:
                            row_pos[base + var_map["shed_start"] + ld] = -ptdf[li, col]
                            row_neg[base + var_map["shed_start"] + ld] = ptdf[li, col]
                    limit = float(line_limits_a[li])
                    A_rows.append(row_pos)
                    b_vals.append(limit)
                    A_rows.append(row_neg)
                    b_vals.append(limit)

        A_ub = np.array(A_rows, dtype=np.float64) if A_rows else np.zeros((0, n_vars))
        b_ub = np.array(b_vals, dtype=np.float64) if b_vals else np.zeros(0)

        # --- equality constraints (A_eq x = b_eq) -------------------
        A_eq_rows: List[np.ndarray] = []
        b_eq_vals: List[float] = []

        # Power balance: sum(gen) + sum(discharge) - sum(charge) + sum(shed) = sum(load)
        for t in range(T):
            base = t * n_vars_per_t
            total_load_t = float(np.sum(load_profile[t]))
            row = np.zeros(n_vars, dtype=np.float64)
            row[base : base + n_gens] = 1.0
            for s in range(n_storage):
                row[base + var_map["disch_start"] + s] = 1.0
                row[base + var_map["char_start"] + s] = -1.0
            row[base + var_map["shed_start"] : base + var_map["shed_start"] + n_loads] = 1.0
            A_eq_rows.append(row)
            b_eq_vals.append(total_load_t)

        # Battery SOC dynamics
        for t in range(T):
            base = t * n_vars_per_t
            for s in range(n_storage):
                eta_in = float(storage_specs[s].get("eta_in", 0.95))
                eta_out = float(storage_specs[s].get("eta_out", 0.95))
                row = np.zeros(n_vars, dtype=np.float64)
                row[base + var_map["soc_start"] + s] = 1.0
                if t > 0:
                    prev_base = (t - 1) * n_vars_per_t
                    row[prev_base + var_map["soc_start"] + s] = -1.0
                    row[base + var_map["char_start"] + s] = -eta_in * dt
                    row[base + var_map["disch_start"] + s] = dt / eta_out
                else:
                    e_init = float(storage_specs[s].get("e_init_mwh", 5.0))
                    row[base + var_map["char_start"] + s] = -eta_in * dt
                    row[base + var_map["disch_start"] + s] = dt / eta_out
                    A_eq_rows.append(row)
                    b_eq_vals.append(e_init)
                    continue
                A_eq_rows.append(row)
                b_eq_vals.append(0.0)

        # Terminal SOC constraint (optional)
        for s in range(n_storage):
            e_term = storage_specs[s].get("e_term_mwh", storage_specs[s].get("e_init_mwh", 5.0))
            base = (T - 1) * n_vars_per_t
            row = np.zeros(n_vars, dtype=np.float64)
            row[base + var_map["soc_start"] + s] = 1.0
            A_eq_rows.append(row)
            b_eq_vals.append(float(e_term))

        A_eq = np.array(A_eq_rows, dtype=np.float64) if A_eq_rows else np.zeros((0, n_vars))
        b_eq = np.array(b_eq_vals, dtype=np.float64) if b_eq_vals else np.zeros(0)

        # --- bounds ---------------------------------------------------
        bounds: List[Tuple[Optional[float], Optional[float]]] = []
        for t in range(T):
            for _g in range(n_gens):
                bounds.append((0.0, None))
            for _s in range(n_storage):
                bounds.append((0.0, None))
            for _s in range(n_storage):
                bounds.append((0.0, None))
            for _s in range(n_storage):
                bounds.append((0.0, None))
            for ld in range(n_loads):
                bounds.append((0.0, load_profile[t, ld]))

        # --- solve ----------------------------------------------------
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"disp": False},
        )

        return self._extract_solution(
            result, var_map, T, n_gens, n_storage, n_loads, gen_idx, load_idx, load_buses, net
        )

    # ------------------------------------------------------------------
    # Rolling horizon
    # ------------------------------------------------------------------

    def rolling_horizon(
        self,
        net: Any,
        load_profiles: np.ndarray,
        storage_specs: Optional[List[Dict[str, Any]]] = None,
        gen_costs: Optional[np.ndarray] = None,
        window_steps: Optional[int] = None,
        step_size: int = 1,
    ) -> pd.DataFrame:
        """Run a rolling-horizon MPC schedule.

        Solves a ``window_steps`` horizon, applies the first
        ``step_size`` steps, then rolls forward.

        Parameters
        ----------
        net : pandapowerNet
        load_profiles : np.ndarray
            Full time-series load, shape ``(n_total_steps, n_loads)``.
        window_steps : int or None
            Look-ahead window.  Defaults to ``self.horizon_steps``.
        step_size : int
            Steps to apply before rolling.  Default 1.

        Returns
        -------
        pd.DataFrame
            Hourly schedule with columns:
            ``hour``, ``gen_mw``, ``battery_soc_mwh``, ``shed_mw``,
            ``line_loading_percent``.
        """
        window_steps = window_steps or self.horizon_steps
        n_total = load_profiles.shape[0]
        records: List[Dict[str, Any]] = []

        t = 0
        while t + window_steps <= n_total:
            window_load = load_profiles[t : t + window_steps]
            result = self.build_schedule(
                net,
                window_load,
                storage_specs=storage_specs,
                gen_costs=gen_costs,
            )

            if result["status"] != 0:
                logger.warning("LP failed at t=%d: %s", t, result.get("message", ""))
                break

            # Record first step_size timesteps
            for offset in range(step_size):
                step = t + offset
                if step >= n_total:
                    break
                gen_sched = result["gen_schedule"]
                batt_sched = result["battery_schedule"]
                shed_sched = result["shed_per_bus"]

                total_gen = float(np.sum(gen_sched[offset, :])) if gen_sched.size > 0 else 0.0
                total_batt = 0.0
                for s, sched in batt_sched.items():
                    total_batt += float(sched["e_mwh"][offset])
                total_shed = float(np.sum(shed_sched[offset, :])) if shed_sched.size > 0 else 0.0

                records.append({
                    "hour": step,
                    "total_gen_mw": total_gen,
                    "total_battery_soc_mwh": total_batt,
                    "total_shed_mw": total_shed,
                    "objective": result["objective"],
                })

            t += step_size

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Apply to network
    # ------------------------------------------------------------------

    def apply_to_net(
        self,
        net: Any,
        result: Dict[str, Any],
        timestep: int = 0,
    ) -> None:
        """Write the optimal schedule for *timestep* back into *net*.

        Mutates ``net.gen.p_mw`` and ``net.storage.soc_mwh`` in place.

        Parameters
        ----------
        net : pandapowerNet
        result : dict
            Output from :meth:`build_schedule`.
        timestep : int
            Which timestep to apply.  Default 0.
        """
        gen_sched = result.get("gen_schedule")
        if gen_sched is not None and gen_sched.ndim == 2:
            gen_idx = result.get("gen_index", [])
            for i, gidx in enumerate(gen_idx):
                if gidx in net.gen.index:
                    net.gen.at[gidx, "p_mw"] = float(gen_sched[timestep, i])

        batt_sched = result.get("battery_schedule", {})
        for s, sched in batt_sched.items():
            e = sched.get("e_mwh")
            if e is not None and timestep < len(e):
                if s in net.storage.index:
                    net.storage.at[s, "soc_mwh"] = float(e[timestep])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_storage(self, net: Any) -> Any:
        """Create an empty pandapower storage table if missing."""
        if not hasattr(net, "storage") or net.storage is None or len(net.storage) == 0:
            import pandapower as pp
            # No-op: storage table will exist as empty DataFrame
            if not hasattr(net, "storage") or net.storage is None:
                net.storage = pd.DataFrame(
                    columns=["name", "bus", "p_mw", "q_mvar", "sn_mva",
                             "soc_percent", "min_e_mwh", "max_e_mwh",
                             "in_service"]
                )
        return net

    def _extract_solution(
        self,
        result: Any,
        var_map: Dict[str, int],
        T: int,
        n_gens: int,
        n_storage: int,
        n_loads: int,
        gen_idx: np.ndarray,
        load_idx: np.ndarray,
        load_buses: np.ndarray,
        net: Any,
    ) -> Dict[str, Any]:
        """Parse ``linprog`` result into structured output."""
        vars_per_t = var_map["vars_per_t"]

        if not result.success:
            logger.warning("Multi-period OPF failed: %s", result.message)
            return {
                "status": result.status,
                "gen_schedule": np.zeros((T, n_gens)),
                "battery_schedule": {},
                "shed_per_bus": np.zeros((T, n_loads)),
                "line_loading_percent": np.zeros(T),
                "objective": float("nan"),
                "message": result.message,
                "gen_index": list(gen_idx),
            }

        x = result.x
        gen_sched = np.zeros((T, n_gens), dtype=np.float64)
        shed = np.zeros((T, n_loads), dtype=np.float64)
        batt_sched: Dict[int, Dict[str, np.ndarray]] = {}

        for t in range(T):
            base = t * vars_per_t
            gen_sched[t, :] = x[base : base + n_gens]
            shed[t, :] = x[base + var_map["shed_start"] : base + var_map["shed_start"] + n_loads]

        for s in range(n_storage):
            p_char = np.zeros(T, dtype=np.float64)
            p_disch = np.zeros(T, dtype=np.float64)
            e = np.zeros(T, dtype=np.float64)
            for t in range(T):
                base = t * vars_per_t
                p_char[t] = x[base + var_map["char_start"] + s]
                p_disch[t] = x[base + var_map["disch_start"] + s]
                e[t] = x[base + var_map["soc_start"] + s]
            batt_sched[s] = {
                "p_char_mw": p_char,
                "p_disch_mw": p_disch,
                "e_mwh": e,
            }

        # Approximate line loading from net.line.max_i_ka if available
        line_loading = np.zeros(T)
        if hasattr(net, "res_line") and hasattr(net.res_line, "loading_percent"):
            line_loading = net.res_line["loading_percent"].values.copy()

        return {
            "status": 0,
            "gen_schedule": gen_sched,
            "battery_schedule": batt_sched,
            "shed_per_bus": shed,
            "line_loading_percent": line_loading,
            "objective": float(result.fun),
            "message": result.message,
            "gen_index": list(gen_idx),
        }

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MultiPeriodOPFScheduler(horizon={self.horizon_steps}, "
            f"dt={self.dt_hours}h, voll={self.voll * 1000:.0f})"
        )
