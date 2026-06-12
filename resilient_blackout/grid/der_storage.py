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
Localized resource optimization for islanded microgrids.

Provides ``OptimalIslandDispatch``, which uses ``scipy.optimize.linprog``
to solve a linear programming formulation that minimizes unserved load
across isolated grid islands by optimally dispatching behind-the-meter
battery energy storage systems (BESS) and solar PV generation over a
multi-hour simulation horizon.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, vstack

logger = logging.getLogger(__name__)

_EPS: float = 1e-10


class OptimalIslandDispatch:
    """LP-based optimal dispatch of BESS and PV for islanded microgrids.

    Minimizes unserved load penalty costs subject to power balance,
    battery state-of-charge dynamics, power limits, energy capacity
    constraints, and PV curtailment limits.

    Parameters
    ----------
    horizon_hours : float
        Simulation horizon in hours.  Default 4.
    time_step_minutes : float
        Time step resolution in minutes.  Default 15.
    penalty_cost_usd_per_mwh : float
        Cost of unserved energy in $/MWh.  Default 10 000.

    Attributes
    ----------
    horizon_hours : float
    time_step_minutes : float
    dt_hours : float
    n_timesteps : int
    penalty_cost : float
    """

    def __init__(
        self,
        horizon_hours: float = 4.0,
        time_step_minutes: float = 15.0,
        penalty_cost_usd_per_mwh: float = 10000.0,
    ) -> None:
        if horizon_hours <= 0:
            raise ValueError(f"horizon_hours must be positive, got {horizon_hours}")
        if time_step_minutes <= 0:
            raise ValueError(f"time_step_minutes must be positive, got {time_step_minutes}")

        self.horizon_hours = horizon_hours
        self.time_step_minutes = time_step_minutes
        self.dt_hours = time_step_minutes / 60.0
        self.n_timesteps = max(1, int(horizon_hours / self.dt_hours))
        self.penalty_cost = penalty_cost_usd_per_mwh / 1000.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_and_solve(
        self,
        net: Any,
        island_buses: List[int],
        storage_units: List[Dict[str, Any]],
        pv_profiles: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Build and solve the LP for an islanded microgrid.

        Parameters
        ----------
        net : pandapowerNet
            The network (used to read load and generation at island
            buses).
        island_buses : list of int
            Bus indices belonging to the island.
        storage_units : list of dict
            Each dict describes one BESS unit with keys:

            - ``bus`` (int)
            - ``p_max_mw`` (float)
            - ``e_max_mwh`` (float)
            - ``e_min_mwh`` (float, default 0)
            - ``e_init_mwh`` (float)
            - ``eta_in`` (float, default 0.95)
            - ``eta_out`` (float, default 0.95)
        pv_profiles : dict or None
            ``{bus: np.ndarray}`` of available PV generation (MW) per
            timestep.  Shape ``(n_timesteps,)``.

        Returns
        -------
        dict
            ``{"status": int, "total_shed_mwh": float,
            "shed_per_bus": np.ndarray, "battery_schedule": dict,
            "message": str}``.
        """
        bus_set = set(island_buses)
        n_buses = len(island_buses)
        bus_to_idx = {b: i for i, b in enumerate(island_buses)}
        n_storage = len(storage_units)

        load = np.zeros((self.n_timesteps, n_buses), dtype=np.float64)
        gen = np.zeros((self.n_timesteps, n_buses), dtype=np.float64)

        for i, bid in enumerate(island_buses):
            load_mask = (net.load.bus == bid) & net.load.in_service
            load[:, i] = net.load.loc[load_mask, "p_mw"].sum()

            gen_mask = (net.gen.bus == bid) & net.gen.in_service
            gen[:, i] = net.gen.loc[gen_mask, "p_mw"].sum()

            sgen_mask = (net.sgen.bus == bid) & net.sgen.in_service
            gen[:, i] += net.sgen.loc[sgen_mask, "p_mw"].sum()

            ext_mask = (net.ext_grid.bus == bid) & net.ext_grid.in_service
            gen[:, i] += net.ext_grid.loc[ext_mask, "max_p_mw"].sum()

        pv_avail = np.zeros((self.n_timesteps, n_buses), dtype=np.float64)
        if pv_profiles:
            for bid, profile in pv_profiles.items():
                if bid in bus_to_idx:
                    j = bus_to_idx[bid]
                    pv_avail[: len(profile), j] = np.asarray(profile, dtype=np.float64)[
                        : self.n_timesteps
                    ]

        c, A_ub, b_ub, bounds, var_map = self._build_lp_matrices(
            load, gen, pv_avail, storage_units, n_buses, n_storage
        )

        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )

        return self._extract_solution(
            result, var_map, n_buses, n_storage, island_buses
        )

    # ------------------------------------------------------------------
    # LP matrix construction
    # ------------------------------------------------------------------

    def _build_lp_matrices(
        self,
        load: np.ndarray,
        gen: np.ndarray,
        pv_avail: np.ndarray,
        storage_units: List[Dict[str, Any]],
        n_buses: int,
        n_storage: int,
    ) -> Tuple[
        np.ndarray,
        csc_matrix,
        np.ndarray,
        List[Tuple[Optional[float], Optional[float]]],
        Dict[str, Any],
    ]:
        """Construct the LP cost vector, constraints, and bounds.

        Variable ordering per timestep t:
            shed[0..n_buses-1], pv_used[0..n_buses-1],
            p_char[0..n_storage-1], p_disch[0..n_storage-1],
            e[0..n_storage-1]

        Parameters
        ----------
        load : np.ndarray
            ``(T, n_buses)`` load in MW.
        gen : np.ndarray
            ``(T, n_buses)`` generation in MW.
        pv_avail : np.ndarray
            ``(T, n_buses)`` available PV in MW.
        storage_units : list of dict
        n_buses : int
        n_storage : int

        Returns
        -------
        tuple
            ``(c, A_ub, b_ub, bounds, var_map)``.
        """
        T = self.n_timesteps
        dt = self.dt_hours

        vars_per_t = n_buses + n_buses + n_storage + n_storage + n_storage
        n_vars = T * vars_per_t

        var_map = {
            "shed_start": 0,
            "pv_start": n_buses,
            "char_start": 2 * n_buses,
            "disch_start": 2 * n_buses + n_storage,
            "e_start": 2 * n_buses + 2 * n_storage,
            "vars_per_t": vars_per_t,
        }

        c = np.zeros(n_vars, dtype=np.float64)
        for t in range(T):
            base = t * vars_per_t
            c[base : base + n_buses] = self.penalty_cost * dt

        A_rows: List[csc_matrix] = []
        b_vals: List[float] = []

        for t in range(T):
            base = t * vars_per_t

            for i in range(n_buses):
                row = np.zeros(n_vars, dtype=np.float64)
                row[base + i] = 1.0
                row[base + n_buses + i] = 1.0
                for s in range(n_storage):
                    row[base + var_map["disch_start"] + s] += 1.0
                    row[base + var_map["char_start"] + s] -= 1.0
                A_rows.append(csc_matrix(row))
                b_vals.append(load[t, i] - gen[t, i])

        for t in range(T):
            base = t * vars_per_t
            for i in range(n_buses):
                row = np.zeros(n_vars, dtype=np.float64)
                row[base + n_buses + i] = 1.0
                A_rows.append(csc_matrix(row))
                b_vals.append(pv_avail[t, i])

        for t in range(T):
            base = t * vars_per_t
            for s in range(n_storage):
                su = storage_units[s]
                p_max = float(su.get("p_max_mw", 1.0))

                row_char = np.zeros(n_vars, dtype=np.float64)
                row_char[base + var_map["char_start"] + s] = 1.0
                A_rows.append(csc_matrix(row_char))
                b_vals.append(p_max)

                row_disch = np.zeros(n_vars, dtype=np.float64)
                row_disch[base + var_map["disch_start"] + s] = 1.0
                A_rows.append(csc_matrix(row_disch))
                b_vals.append(p_max)

        for t in range(T):
            base = t * vars_per_t
            for s in range(n_storage):
                su = storage_units[s]
                e_max = float(su.get("e_max_mwh", 10.0))

                row = np.zeros(n_vars, dtype=np.float64)
                row[base + var_map["e_start"] + s] = 1.0
                A_rows.append(csc_matrix(row))
                b_vals.append(e_max)

        for t in range(1, T):
            base = t * vars_per_t
            prev_base = (t - 1) * vars_per_t
            for s in range(n_storage):
                su = storage_units[s]
                eta_in = float(su.get("eta_in", 0.95))
                eta_out = float(su.get("eta_out", 0.95))

                row = np.zeros(n_vars, dtype=np.float64)
                row[base + var_map["e_start"] + s] = 1.0
                row[prev_base + var_map["e_start"] + s] = -1.0
                row[base + var_map["char_start"] + s] = -eta_in * dt
                row[base + var_map["disch_start"] + s] = dt / eta_out
                A_rows.append(csc_matrix(row))
                b_vals.append(0.0)

                row_neg = np.zeros(n_vars, dtype=np.float64)
                row_neg[base + var_map["e_start"] + s] = -1.0
                row_neg[prev_base + var_map["e_start"] + s] = 1.0
                row_neg[base + var_map["char_start"] + s] = eta_in * dt
                row_neg[base + var_map["disch_start"] + s] = -dt / eta_out
                A_rows.append(csc_matrix(row_neg))
                b_vals.append(0.0)

        A_ub = vstack(A_rows, format="csc")
        b_ub = np.array(b_vals, dtype=np.float64)

        bounds: List[Tuple[Optional[float], Optional[float]]] = []
        for t in range(T):
            for _ in range(n_buses):
                bounds.append((0.0, None))
            for _ in range(n_buses):
                bounds.append((0.0, None))
            for _ in range(n_storage):
                bounds.append((0.0, None))
            for _ in range(n_storage):
                bounds.append((0.0, None))
            for s in range(n_storage):
                su = storage_units[s]
                e_min = float(su.get("e_min_mwh", 0.0))
                e_max = float(su.get("e_max_mwh", 10.0))
                if t == 0:
                    e_init = float(su.get("e_init_mwh", e_max * 0.5))
                    bounds.append((e_init, e_init))
                else:
                    bounds.append((e_min, e_max))

        return c, A_ub, b_ub, bounds, var_map

    # ------------------------------------------------------------------
    # Solution extraction
    # ------------------------------------------------------------------

    def _extract_solution(
        self,
        result: Any,
        var_map: Dict[str, Any],
        n_buses: int,
        n_storage: int,
        island_buses: List[int],
    ) -> Dict[str, Any]:
        """Parse linprog result into structured output.

        Parameters
        ----------
        result : OptimizeResult
        var_map : dict
        n_buses : int
        n_storage : int
        island_buses : list of int

        Returns
        -------
        dict
        """
        T = self.n_timesteps
        vars_per_t = var_map["vars_per_t"]

        if not result.success:
            logger.warning("LP solver failed: %s", result.message)
            return {
                "status": result.status,
                "total_shed_mwh": float("nan"),
                "shed_per_bus": np.zeros((T, n_buses)),
                "battery_schedule": {},
                "message": result.message,
            }

        x = result.x
        shed = np.zeros((T, n_buses), dtype=np.float64)
        pv_used = np.zeros((T, n_buses), dtype=np.float64)
        battery_schedule: Dict[int, Dict[str, np.ndarray]] = {}

        for t in range(T):
            base = t * vars_per_t
            shed[t, :] = x[base : base + n_buses]
            pv_used[t, :] = x[base + n_buses : base + 2 * n_buses]

        for s in range(n_storage):
            p_char = np.zeros(T, dtype=np.float64)
            p_disch = np.zeros(T, dtype=np.float64)
            e = np.zeros(T, dtype=np.float64)

            for t in range(T):
                base = t * vars_per_t
                p_char[t] = x[base + var_map["char_start"] + s]
                p_disch[t] = x[base + var_map["disch_start"] + s]
                e[t] = x[base + var_map["e_start"] + s]

            battery_schedule[s] = {
                "p_char_mw": p_char,
                "p_disch_mw": p_disch,
                "e_mwh": e,
            }

        total_shed_mwh = float(np.sum(shed) * self.dt_hours)

        return {
            "status": result.status,
            "total_shed_mwh": total_shed_mwh,
            "shed_per_bus": shed,
            "pv_used_mw": pv_used,
            "battery_schedule": battery_schedule,
            "message": result.message,
        }

    # ------------------------------------------------------------------
    # Cascade integration
    # ------------------------------------------------------------------

    def apply_to_cascade(
        self,
        net: Any,
        islands: List[List[int]],
        storage_units: List[Dict[str, Any]],
        pv_profiles: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[float, Dict[int, float]]:
        """Replace cascade load shedding with optimal DER dispatch.

        For each island, solves the LP to determine optimal BESS/PV
        dispatch and residual load shedding.  Returns total shed and
        per-bus shed dictionary compatible with ``CascadingSimulator``.

        Parameters
        ----------
        net : pandapowerNet
        islands : list of list of int
        storage_units : list of dict
        pv_profiles : dict or None

        Returns
        -------
        tuple of (float, dict)
            Total MWh shed and per-bus shed dictionary.
        """
        total_shed = 0.0
        shed_per_bus: Dict[int, float] = {}

        for bus_group in islands:
            if not bus_group:
                continue

            result = self.build_and_solve(
                net, bus_group, storage_units, pv_profiles
            )

            if result["status"] != 0:
                continue

            shed_matrix = result["shed_per_bus"]
            for t in range(self.n_timesteps):
                for i, bid in enumerate(bus_group):
                    shed_mw = float(shed_matrix[t, i])
                    if shed_mw > _EPS:
                        shed_per_bus[bid] = shed_per_bus.get(bid, 0.0) + shed_mw

            total_shed += float(result["total_shed_mwh"])

        return total_shed, shed_per_bus
