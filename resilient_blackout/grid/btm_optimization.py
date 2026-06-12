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
Stochastic microgrid resource dispatch solver.

Provides ``OptimalStochasticScheduler``, a scenario-based stochastic
linear programming engine that evaluates behind-the-meter energy storage
systems (BTM-ESS) under multi-horizon stochastic grid outages and
computes the Avoided Loss of Load (ALOL) to monetise the resilience
return on investment of storage assets.

Mathematical formulation
------------------------

Given a set of discrete outage scenarios :math:`\\Omega` where each
scenario :math:`\\omega` has duration :math:`d_\\omega`, start hour
:math:`h_\\omega`, and probability :math:`\\pi_\\omega`, the scheduler
solves:

.. math::

    \\min \\sum_{\\omega \\in \\Omega} \\pi_\\omega
    \\sum_{t \\in T} \\left(
        \\lambda_t P_{\\omega, t}^{\\text{grid}}
        + \\nu P_{\\omega, t}^{\\text{shed}}
    \\right) \\Delta t

subject to power balance, BESS state-of-charge dynamics, and grid
availability constraints for each scenario timeline independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, vstack

logger = logging.getLogger(__name__)

_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# OutageScenario
# ---------------------------------------------------------------------------


@dataclass
class OutageScenario:
    """A discrete grid outage scenario for stochastic scheduling.

    Attributes
    ----------
    duration_h : float
        Outage duration in hours.
    start_h : float
        Hour of the study horizon at which the outage begins
        (0-indexed from simulation start).
    probability : float
        Occurrence probability :math:`\\pi_\\omega \\in (0, 1]`.
        The sum of probabilities across all scenarios should equal 1.
    label : str or None
        Optional human-readable name.
    """

    duration_h: float
    start_h: float
    probability: float
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if self.duration_h <= 0:
            raise ValueError(f"duration_h must be positive, got {self.duration_h}")
        if self.start_h < 0:
            raise ValueError(f"start_h must be non-negative, got {self.start_h}")
        if not (0 < self.probability <= 1):
            raise ValueError(
                f"probability must be in (0, 1], got {self.probability}"
            )


# ---------------------------------------------------------------------------
# OptimalStochasticScheduler
# ---------------------------------------------------------------------------


class OptimalStochasticScheduler:
    """Stochastic LP scheduler for BTM-ESS under multi-scenario outages.

    Evaluates behind-the-meter battery and solar PV dispatch across
    multiple probabilistic outage timelines in a single linear program,
    then computes the Avoided Loss of Load (ALOL) by comparing the
    resilient configuration against a no-storage baseline.

    Parameters
    ----------
    horizon_h : float
        Total study horizon in hours.  Default 24.
    dt_minutes : float
        Time-step resolution in minutes.  Default 15.
    grid_price_usd_per_mwh : float or np.ndarray
        Utility electricity price in $/MWh.  If scalar, constant across
        all time steps; if array, shape ``(n_steps,)``.
    penalty_usd_per_mwh : float
        Value of Lost Load (VoLL) penalty for unserved energy in $/MWh.
        Default 10 000.
    bess_p_max_mw : float
        Maximum BESS charge/discharge power in MW.  Default 1.0.
    bess_e_max_mwh : float
        Maximum BESS energy capacity in MWh.  Default 4.0.
    bess_e_init_mwh : float or None
        Initial BESS state of charge in MWh.  If ``None``, defaults to
        50 % of *bess_e_max_mwh*.
    bess_eta_in : float
        Round-trip charge efficiency (0–1).  Default 0.95.
    bess_eta_out : float
        Round-trip discharge efficiency (0–1).  Default 0.95.
    pv_capacity_mw : float
        Installed solar PV capacity in MW (DC nameplate).  Default 0.0.
    pv_profile_pu : np.ndarray or None
        Per-unit PV generation profile, shape ``(n_steps,)``.  Values
        in [0, 1] representing the fraction of nameplate capacity
        available at each time step.  If ``None``, PV is unavailable.

    Attributes
    ----------
    horizon_h : float
    dt_hours : float
    n_steps : int
    grid_price : np.ndarray
    penalty : float
    result_ : dict or None
        Populated after :meth:`solve`.
    baseline_result_ : dict or None
        Populated after :meth:`evaluate_baseline`.
    """

    def __init__(
        self,
        horizon_h: float = 24.0,
        dt_minutes: float = 15.0,
        grid_price_usd_per_mwh: float | np.ndarray = 100.0,
        penalty_usd_per_mwh: float = 10_000.0,
        bess_p_max_mw: float = 1.0,
        bess_e_max_mwh: float = 4.0,
        bess_e_init_mwh: Optional[float] = None,
        bess_eta_in: float = 0.95,
        bess_eta_out: float = 0.95,
        pv_capacity_mw: float = 0.0,
        pv_profile_pu: Optional[np.ndarray] = None,
    ) -> None:
        if horizon_h <= 0:
            raise ValueError(f"horizon_h must be positive, got {horizon_h}")
        if dt_minutes <= 0:
            raise ValueError(f"dt_minutes must be positive, got {dt_minutes}")

        self.horizon_h = horizon_h
        self.dt_hours = dt_minutes / 60.0
        self.n_steps = max(1, int(horizon_h / self.dt_hours))

        if np.isscalar(grid_price_usd_per_mwh):
            self.grid_price = np.full(
                self.n_steps, float(grid_price_usd_per_mwh), dtype=np.float64
            )
        else:
            gp = np.asarray(grid_price_usd_per_mwh, dtype=np.float64)
            if len(gp) < self.n_steps:
                raise ValueError(
                    f"grid_price array length {len(gp)} < n_steps {self.n_steps}"
                )
            self.grid_price = gp[: self.n_steps].copy()

        self.penalty = float(penalty_usd_per_mwh) / 1000.0  # $/kWh for LP

        self.bess_p_max = float(bess_p_max_mw)
        self.bess_e_max = float(bess_e_max_mwh)
        self.bess_e_init = (
            float(bess_e_init_mwh)
            if bess_e_init_mwh is not None
            else self.bess_e_max * 0.5
        )
        self.bess_eta_in = float(bess_eta_in)
        self.bess_eta_out = float(bess_eta_out)

        if not (0 < self.bess_eta_in <= 1):
            raise ValueError(f"bess_eta_in must be in (0, 1], got {self.bess_eta_in}")
        if not (0 < self.bess_eta_out <= 1):
            raise ValueError(f"bess_eta_out must be in (0, 1], got {self.bess_eta_out}")
        if self.bess_e_init < 0 or self.bess_e_init > self.bess_e_max:
            raise ValueError(
                f"bess_e_init {self.bess_e_init} outside [0, {self.bess_e_max}]"
            )

        self.pv_capacity = float(pv_capacity_mw)
        if pv_profile_pu is not None:
            pv_arr = np.asarray(pv_profile_pu, dtype=np.float64)
            if len(pv_arr) < self.n_steps:
                raise ValueError(
                    f"pv_profile_pu length {len(pv_arr)} < n_steps {self.n_steps}"
                )
            self.pv_profile = np.clip(pv_arr[: self.n_steps], 0.0, 1.0)
        else:
            self.pv_profile = np.zeros(self.n_steps, dtype=np.float64)

        self.result_: Optional[Dict[str, Any]] = None
        self.baseline_result_: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        scenarios: List[OutageScenario],
        load_profile_mw: np.ndarray,
    ) -> Dict[str, Any]:
        """Solve the stochastic LP across all outage scenarios.

        Parameters
        ----------
        scenarios : list of OutageScenario
            Discrete outage scenarios with probabilities.  Probabilities
            are normalised to sum to 1 if they do not already.
        load_profile_mw : np.ndarray
            Local load in MW at each time step, shape ``(n_steps,)``.

        Returns
        -------
        dict
            Keys:

            - ``status`` (int) — LP solver status (0 = success).
            - ``expected_cost_usd`` (float) — probability-weighted total
              cost across all scenarios.
            - ``expected_eens_kwh`` (float) — expected unserved energy.
            - ``per_scenario`` (dict) — mapping from scenario index to
              per-scenario results (schedules, costs).
            - ``message`` (str) — solver message.
        """
        load = np.asarray(load_profile_mw, dtype=np.float64)
        if len(load) < self.n_steps:
            raise ValueError(
                f"load_profile_mw length {len(load)} < n_steps {self.n_steps}"
            )
        load = load[: self.n_steps].copy()

        probs = np.array([s.probability for s in scenarios], dtype=np.float64)
        probs /= probs.sum()

        c, A_ub, b_ub, bounds, var_layout = self._build_lp(
            scenarios, probs, load, has_bess=True
        )

        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
            options={"disp": False},
        )

        if not result.success:
            logger.warning("Stochastic LP failed: %s", result.message)
            self.result_ = {
                "status": result.status,
                "expected_cost_usd": float("nan"),
                "expected_eens_kwh": float("nan"),
                "per_scenario": {},
                "message": result.message,
            }
            return self.result_

        parsed = self._extract_solution(result.x, scenarios, probs, var_layout)
        self.result_ = parsed
        return parsed

    def evaluate_baseline(
        self,
        scenarios: List[OutageScenario],
        load_profile_mw: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate the no-storage baseline for ALOL calculation.

        Same as :meth:`solve` but with BESS capacity forced to zero.
        Only PV and grid supply (plus shedding) are available.

        Parameters
        ----------
        scenarios : list of OutageScenario
        load_profile_mw : np.ndarray

        Returns
        -------
        dict
            Same structure as :meth:`solve`.
        """
        load = np.asarray(load_profile_mw, dtype=np.float64)[: self.n_steps].copy()

        probs = np.array([s.probability for s in scenarios], dtype=np.float64)
        probs /= probs.sum()

        c, A_ub, b_ub, bounds, var_layout = self._build_lp(
            scenarios, probs, load, has_bess=False
        )

        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
            options={"disp": False},
        )

        if not result.success:
            logger.warning("Baseline LP failed: %s", result.message)
            self.baseline_result_ = {
                "status": result.status,
                "expected_cost_usd": float("nan"),
                "expected_eens_kwh": float("nan"),
                "per_scenario": {},
                "message": result.message,
            }
            return self.baseline_result_

        parsed = self._extract_solution(result.x, scenarios, probs, var_layout)
        self.baseline_result_ = parsed
        return parsed

    def compute_alol(self) -> Dict[str, float]:
        """Compute Avoided Loss of Load (ALOL) and resilience ROI.

        Requires both :meth:`solve` and :meth:`evaluate_baseline` to
        have been called first.

        Returns
        -------
        dict
            Keys:

            - ``baseline_expected_cost_usd``
            - ``resilient_expected_cost_usd``
            - ``alol_cost_savings_usd`` — absolute cost reduction
            - ``baseline_expected_eens_kwh``
            - ``resilient_expected_eens_kwh``
            - ``alol_eens_reduction_kwh`` — unserved energy avoided
            - ``alol_ratio`` — resilient / baseline cost (lower is better)

        Raises
        ------
        RuntimeError
            If :meth:`solve` or :meth:`evaluate_baseline` have not been
            called.
        """
        if self.result_ is None:
            raise RuntimeError("Call solve() before compute_alol().")
        if self.baseline_result_ is None:
            raise RuntimeError("Call evaluate_baseline() before compute_alol().")

        base_cost = float(self.baseline_result_["expected_cost_usd"])
        resil_cost = float(self.result_["expected_cost_usd"])
        base_eens = float(self.baseline_result_["expected_eens_kwh"])
        resil_eens = float(self.result_["expected_eens_kwh"])

        return {
            "baseline_expected_cost_usd": base_cost,
            "resilient_expected_cost_usd": resil_cost,
            "alol_cost_savings_usd": base_cost - resil_cost,
            "baseline_expected_eens_kwh": base_eens,
            "resilient_expected_eens_kwh": resil_eens,
            "alol_eens_reduction_kwh": base_eens - resil_eens,
            "alol_ratio": resil_cost / base_cost if base_cost > _EPS else float("nan"),
        }

    # ------------------------------------------------------------------
    # LP construction
    # ------------------------------------------------------------------

    def _build_lp(
        self,
        scenarios: List[OutageScenario],
        probs: np.ndarray,
        load: np.ndarray,
        has_bess: bool,
    ) -> Tuple[
        np.ndarray,
        csc_matrix,
        np.ndarray,
        List[Tuple[Optional[float], Optional[float]]],
        Dict[str, Any],
    ]:
        """Construct the full stochastic LP.

        Variable ordering per scenario per timestep:
            P_grid, P_shed, P_char, P_disch, E_soc

        When ``has_bess=False``, P_char, P_disch, and E_soc are omitted
        (or fixed to zero).

        Parameters
        ----------
        scenarios : list of OutageScenario
        probs : np.ndarray
            Normalised probabilities, shape ``(n_scenarios,)``.
        load : np.ndarray
            Load in MW, shape ``(n_steps,)``.
        has_bess : bool
            Whether to include BESS variables.

        Returns
        -------
        tuple
            ``(c, A_ub, b_ub, bounds, var_layout)``.
        """
        T = self.n_steps
        dt = self.dt_hours
        S = len(scenarios)

        if has_bess:
            vars_per_ts = 5  # grid, shed, char, disch, e
        else:
            vars_per_ts = 2  # grid, shed

        n_vars = S * T * vars_per_ts

        var_layout: Dict[str, Any] = {
            "n_scenarios": S,
            "n_steps": T,
            "vars_per_ts": vars_per_ts,
            "has_bess": has_bess,
            "grid_offset": 0,
            "shed_offset": 1,
        }
        if has_bess:
            var_layout.update(
                {
                    "char_offset": 2,
                    "disch_offset": 3,
                    "e_offset": 4,
                }
            )

        def _var_idx(s: int, t: int, v: int) -> int:
            return s * T * vars_per_ts + t * vars_per_ts + v

        c = np.zeros(n_vars, dtype=np.float64)
        for s in range(S):
            for t in range(T):
                c[_var_idx(s, t, 0)] = probs[s] * self.grid_price[t] * dt / 1000.0
                c[_var_idx(s, t, 1)] = probs[s] * self.penalty * dt

        A_rows: List[csc_matrix] = []
        b_vals: List[float] = []

        # --- Power balance ---
        for s in range(S):
            for t in range(T):
                row = np.zeros(n_vars, dtype=np.float64)
                row[_var_idx(s, t, 0)] = 1.0  # P_grid
                row[_var_idx(s, t, 1)] = 1.0  # P_shed
                if has_bess:
                    row[_var_idx(s, t, 2)] = -1.0  # P_char (consumes power)
                    row[_var_idx(s, t, 3)] = 1.0  # P_disch (supplies power)
                pv_avail = self.pv_capacity * self.pv_profile[t]
                A_rows.append(csc_matrix(row))
                b_vals.append(load[t] - pv_avail)

        # --- Grid unavailable during outage ---
        for s, sc in enumerate(scenarios):
            start_step = int(sc.start_h / self.dt_hours)
            end_step = int((sc.start_h + sc.duration_h) / self.dt_hours)
            start_step = max(0, min(start_step, T - 1))
            end_step = max(start_step + 1, min(end_step, T))

            for t in range(start_step, end_step):
                row = np.zeros(n_vars, dtype=np.float64)
                row[_var_idx(s, t, 0)] = 1.0
                A_rows.append(csc_matrix(row))
                b_vals.append(0.0)

        # --- BESS constraints ---
        if has_bess:
            for s in range(S):
                for t in range(T):
                    # P_char <= P_max
                    row = np.zeros(n_vars, dtype=np.float64)
                    row[_var_idx(s, t, 2)] = 1.0
                    A_rows.append(csc_matrix(row))
                    b_vals.append(self.bess_p_max)

                    # P_disch <= P_max
                    row = np.zeros(n_vars, dtype=np.float64)
                    row[_var_idx(s, t, 3)] = 1.0
                    A_rows.append(csc_matrix(row))
                    b_vals.append(self.bess_p_max)

                    # E_soc <= E_max
                    row = np.zeros(n_vars, dtype=np.float64)
                    row[_var_idx(s, t, 4)] = 1.0
                    A_rows.append(csc_matrix(row))
                    b_vals.append(self.bess_e_max)

            # --- SoC dynamics ---
            for s in range(S):
                for t in range(1, T):
                    # E_t = E_{t-1} + eta_in * P_char * dt - P_disch * dt / eta_out
                    row_pos = np.zeros(n_vars, dtype=np.float64)
                    row_pos[_var_idx(s, t, 4)] = 1.0
                    row_pos[_var_idx(s, t - 1, 4)] = -1.0
                    row_pos[_var_idx(s, t, 2)] = -self.bess_eta_in * dt
                    row_pos[_var_idx(s, t, 3)] = dt / self.bess_eta_out
                    A_rows.append(csc_matrix(row_pos))
                    b_vals.append(0.0)

                    row_neg = np.zeros(n_vars, dtype=np.float64)
                    row_neg[_var_idx(s, t, 4)] = -1.0
                    row_neg[_var_idx(s, t - 1, 4)] = 1.0
                    row_neg[_var_idx(s, t, 2)] = self.bess_eta_in * dt
                    row_neg[_var_idx(s, t, 3)] = -dt / self.bess_eta_out
                    A_rows.append(csc_matrix(row_neg))
                    b_vals.append(0.0)

        A_ub = vstack(A_rows, format="csc") if A_rows else csc_matrix((0, n_vars))
        b_ub = np.array(b_vals, dtype=np.float64)

        # --- Bounds ---
        bounds: List[Tuple[Optional[float], Optional[float]]] = []
        for s in range(S):
            for t in range(T):
                bounds.append((0.0, None))  # P_grid
                bounds.append((0.0, load[t]))  # P_shed ≤ load
                if has_bess:
                    bounds.append((0.0, None))  # P_char
                    bounds.append((0.0, None))  # P_disch
                    if t == 0:
                        bounds.append((self.bess_e_init, self.bess_e_init))
                    else:
                        bounds.append((0.0, self.bess_e_max))

        return c, A_ub, b_ub, bounds, var_layout

    # ------------------------------------------------------------------
    # Solution extraction
    # ------------------------------------------------------------------

    def _extract_solution(
        self,
        x: np.ndarray,
        scenarios: List[OutageScenario],
        probs: np.ndarray,
        var_layout: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Parse the LP solution vector into structured output.

        Parameters
        ----------
        x : np.ndarray
            Solution vector from linprog.
        scenarios : list of OutageScenario
        probs : np.ndarray
        var_layout : dict

        Returns
        -------
        dict
        """
        T = self.n_steps
        S = var_layout["n_scenarios"]
        vars_per_ts = var_layout["vars_per_ts"]
        has_bess = var_layout["has_bess"]
        dt = self.dt_hours

        expected_cost = 0.0
        expected_eens = 0.0
        per_scenario: Dict[int, Dict[str, Any]] = {}

        for s in range(S):
            p_grid = np.zeros(T, dtype=np.float64)
            p_shed = np.zeros(T, dtype=np.float64)
            p_char = np.zeros(T, dtype=np.float64)
            p_disch = np.zeros(T, dtype=np.float64)
            e_soc = np.zeros(T, dtype=np.float64)

            for t in range(T):
                base = s * T * vars_per_ts + t * vars_per_ts
                p_grid[t] = x[base + 0]
                p_shed[t] = x[base + 1]
                if has_bess:
                    p_char[t] = x[base + 2]
                    p_disch[t] = x[base + 3]
                    e_soc[t] = x[base + 4]

            cost_grid = float(np.sum(p_grid * self.grid_price * dt / 1000.0))
            cost_shed = float(np.sum(p_shed * self.penalty * 1000.0 * dt))
            total_cost = cost_grid + cost_shed
            eens = float(np.sum(p_shed * dt * 1000.0))  # kWh

            expected_cost += probs[s] * total_cost
            expected_eens += probs[s] * eens

            per_scenario[s] = {
                "label": scenarios[s].label or f"scenario_{s}",
                "probability": float(probs[s]),
                "p_grid_mw": p_grid,
                "p_shed_mw": p_shed,
                "cost_grid_usd": cost_grid,
                "cost_shed_usd": cost_shed,
                "total_cost_usd": total_cost,
                "eens_kwh": eens,
            }
            if has_bess:
                per_scenario[s].update(
                    {
                        "p_char_mw": p_char,
                        "p_disch_mw": p_disch,
                        "e_soc_mwh": e_soc,
                    }
                )

        return {
            "status": 0,
            "expected_cost_usd": expected_cost,
            "expected_eens_kwh": expected_eens,
            "per_scenario": per_scenario,
            "message": "Optimization terminated successfully.",
        }

    # ------------------------------------------------------------------
    # Convenience: results as DataFrames
    # ------------------------------------------------------------------

    def get_schedule_dataframe(self, scenario_index: int = 0) -> pd.DataFrame:
        """Return a single scenario's dispatch schedule as a DataFrame.

        Parameters
        ----------
        scenario_index : int
            Index into the scenario list.  Default 0.

        Returns
        -------
        pd.DataFrame
            Columns: ``p_grid_mw``, ``p_shed_mw``, ``p_char_mw``,
            ``p_disch_mw``, ``e_soc_mwh``, ``load_mw``, ``pv_mw``.

        Raises
        ------
        RuntimeError
            If :meth:`solve` has not been called.
        """
        if self.result_ is None:
            raise RuntimeError("Call solve() before get_schedule_dataframe().")

        sc = self.result_["per_scenario"][scenario_index]
        index = pd.timedelta_range(
            start="0 min",
            periods=self.n_steps,
            freq=f"{int(self.dt_hours * 60)}min",
        )

        df = pd.DataFrame(
            {
                "p_grid_mw": sc["p_grid_mw"],
                "p_shed_mw": sc["p_shed_mw"],
            },
            index=index,
        )

        if "p_char_mw" in sc:
            df["p_char_mw"] = sc["p_char_mw"]
            df["p_disch_mw"] = sc["p_disch_mw"]
            df["e_soc_mwh"] = sc["e_soc_mwh"]

        df["pv_mw"] = self.pv_capacity * self.pv_profile
        df["load_mw"] = df["p_grid_mw"] + df["p_shed_mw"] + df.get("p_char_mw", 0.0) - df.get("p_disch_mw", 0.0) + df["pv_mw"]

        return df

    def __repr__(self) -> str:
        return (
            f"OptimalStochasticScheduler(horizon={self.horizon_h}h, "
            f"dt={self.dt_hours * 60:.0f}min, "
            f"bess={self.bess_p_max}MW/{self.bess_e_max}MWh, "
            f"pv={self.pv_capacity}MW)"
        )
