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
Stochastic capital planning and portfolio optimization.

Provides ``InvestmentPortfolioOptimizer``, a scenario-based linear
programming engine that selects an optimal portfolio of grid-hardening
upgrades to minimise the expected sum of annualised CAPEX and
VoLL-weighted unserved energy costs across probability-weighted
climate scenarios.
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
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ResilienceProject:
    """A candidate grid-hardening investment project.

    Attributes
    ----------
    name : str
        Unique project identifier.
    capex_usd : float
        Up-front capital expenditure in USD.
    opex_delta_usd_per_year : float
        Change in annual operating expenditure (positive = cost
        increase, negative = savings).
    target_asset_ids : list of str
        Asset identifiers that this project hardens.
    failure_rate_multiplier : float
        Factor applied to baseline failure rates when the project is
        active.  Values < 1 reduce failure rates (e.g., 0.5 = 50 %
        reduction).
    recovery_time_multiplier : float
        Factor applied to recovery/repair times.  Values < 1 reduce
        outage duration.
    description : str or None
        Optional human-readable description.
    """

    name: str
    capex_usd: float
    opex_delta_usd_per_year: float = 0.0
    target_asset_ids: List[str] = field(default_factory=list)
    failure_rate_multiplier: float = 1.0
    recovery_time_multiplier: float = 1.0
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if self.capex_usd < 0:
            raise ValueError(f"capex_usd must be non-negative, got {self.capex_usd}")
        if not (0 < self.failure_rate_multiplier <= 1):
            raise ValueError(
                f"failure_rate_multiplier must be in (0, 1], "
                f"got {self.failure_rate_multiplier}"
            )
        if not (0 < self.recovery_time_multiplier <= 1):
            raise ValueError(
                f"recovery_time_multiplier must be in (0, 1], "
                f"got {self.recovery_time_multiplier}"
            )


@dataclass
class ClimateScenario:
    """A probability-weighted climate hazard scenario.

    Attributes
    ----------
    name : str
        Unique scenario identifier.
    probability : float
        Occurrence probability π_s ∈ (0, 1].
    baseline_eens_mwh : float
        Expected Energy Not Served under this scenario with no
        resilience investments (MWh).
    baseline_lolp : float
        Loss of Load Probability under this scenario with no
        investments.
    voll_usd_per_mwh : float
        Value of Lost Load for this scenario in $/MWh.
    description : str or None
    """

    name: str
    probability: float
    baseline_eens_mwh: float = 0.0
    baseline_lolp: float = 0.0
    voll_usd_per_mwh: float = 10_000.0
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if not (0 < self.probability <= 1):
            raise ValueError(
                f"probability must be in (0, 1], got {self.probability}"
            )
        if self.baseline_eens_mwh < 0:
            raise ValueError(
                f"baseline_eens_mwh must be non-negative, got {self.baseline_eens_mwh}"
            )
        if self.voll_usd_per_mwh <= 0:
            raise ValueError(
                f"voll_usd_per_mwh must be positive, got {self.voll_usd_per_mwh}"
            )


# ---------------------------------------------------------------------------
# InvestmentPortfolioOptimizer
# ---------------------------------------------------------------------------


class InvestmentPortfolioOptimizer:
    """Stochastic LP-based portfolio selector for resilience investments.

    Selects a subset of candidate grid-hardening projects to minimise
    the expected sum of annualised CAPEX and VoLL-weighted EENS across
    probability-weighted climate scenarios, subject to a total budget
    constraint.

    The decision variables :math:`z_k \\in [0, 1]` are relaxed to
    continuous for LP efficiency; final selection thresholds at 0.5.

    Parameters
    ----------
    projects : list of ResilienceProject
        Candidate investment projects.
    scenarios : list of ClimateScenario
        Probability-weighted climate hazard scenarios.
    budget_usd : float
        Maximum total CAPEX budget in USD.
    discount_rate : float
        Annual discount rate for CAPEX amortisation.  Default 0.05.
    planning_horizon_years : int
        Planning horizon in years.  Default 20.
    failure_weight : float
        Weight assigned to failure-rate reduction in the risk-reduction
        composite.  Must be in [0, 1].  Default 0.7.
    recovery_weight : float
        Weight assigned to recovery-time reduction.  Must be in [0, 1]
        and sum with *failure_weight* to 1.0.  Default 0.3.

    Attributes
    ----------
    projects : list of ResilienceProject
    scenarios : list of ClimateScenario
    budget_usd : float
    discount_rate : float
    horizon_years : int
    result_ : dict or None
        Populated after :meth:`solve`.
    frontier_ : list of dict or None
        Populated after :meth:`compute_efficient_frontier`.
    """

    def __init__(
        self,
        projects: List[ResilienceProject],
        scenarios: List[ClimateScenario],
        budget_usd: float,
        discount_rate: float = 0.05,
        planning_horizon_years: int = 20,
        failure_weight: float = 0.7,
        recovery_weight: float = 0.3,
    ) -> None:
        if not projects:
            raise ValueError("projects list must not be empty")
        if not scenarios:
            raise ValueError("scenarios list must not be empty")
        if budget_usd < 0:
            raise ValueError(f"budget_usd must be non-negative, got {budget_usd}")
        if not (0 < discount_rate < 1):
            raise ValueError(
                f"discount_rate must be in (0, 1), got {discount_rate}"
            )
        if planning_horizon_years <= 0:
            raise ValueError(
                f"planning_horizon_years must be positive, got {planning_horizon_years}"
            )
        if not (0 <= failure_weight <= 1):
            raise ValueError(
                f"failure_weight must be in [0, 1], got {failure_weight}"
            )
        if not (0 <= recovery_weight <= 1):
            raise ValueError(
                f"recovery_weight must be in [0, 1], got {recovery_weight}"
            )
        if abs(failure_weight + recovery_weight - 1.0) > 1e-9:
            raise ValueError(
                f"failure_weight + recovery_weight must equal 1.0, "
                f"got {failure_weight} + {recovery_weight} = {failure_weight + recovery_weight}"
            )

        self.projects = list(projects)
        self.scenarios = list(scenarios)
        self.budget_usd = float(budget_usd)
        self.discount_rate = float(discount_rate)
        self.horizon_years = int(planning_horizon_years)
        self.failure_weight = float(failure_weight)
        self.recovery_weight = float(recovery_weight)

        self.result_: Optional[Dict[str, Any]] = None
        self.frontier_: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # CAPEX amortisation
    # ------------------------------------------------------------------

    @staticmethod
    def _amortize_capex(
        capex: float,
        discount_rate: float,
        horizon_years: int,
    ) -> float:
        """Compute annualised CAPEX via capital recovery factor.

        .. math::

            C_{\\text{amortized}} = C \\cdot
            \\frac{r(1+r)^n}{(1+r)^n - 1}

        Parameters
        ----------
        capex : float
            Up-front capital cost.
        discount_rate : float
            Annual discount rate.
        horizon_years : int
            Planning horizon in years.

        Returns
        -------
        float
            Annualised cost.
        """
        if capex <= _EPS:
            return 0.0
        r = discount_rate
        n = horizon_years
        crf = r * (1 + r) ** n / ((1 + r) ** n - 1)
        return capex * crf

    # ------------------------------------------------------------------
    # LP construction
    # ------------------------------------------------------------------

    def _build_lp(
        self,
        budget_usd: Optional[float] = None,
    ) -> Tuple[
        np.ndarray,
        csc_matrix,
        np.ndarray,
        List[Tuple[float, float]],
    ]:
        """Construct the portfolio optimisation LP.

        Variables: z_0, ..., z_{K-1} (project selection fractions).

        Objective: min Σ C_k_amortized × z_k + Σ π_s × baseline_risk_s × Π_k(1 - (1-m_k) × z_k)

        For tractability, the risk reduction is linearised:
        residual_risk_s = baseline_risk_s × (1 - Σ_k w_{k,s} × z_k)
        where w_{k,s} is the risk reduction weight of project k on scenario s.

        Parameters
        ----------
        budget_usd : float or None
            Budget constraint.  If ``None``, uses ``self.budget_usd``.

        Returns
        -------
        tuple
            ``(c, A_ub, b_ub, bounds)`` for ``scipy.optimize.linprog``.
        """
        if budget_usd is None:
            budget_usd = self.budget_usd

        K = len(self.projects)
        S = len(self.scenarios)

        probs = np.array([s.probability for s in self.scenarios], dtype=np.float64)
        probs /= probs.sum()

        n_vars = K

        annualised_capex = np.array(
            [self._amortize_capex(p.capex_usd, self.discount_rate, self.horizon_years)
             for p in self.projects],
            dtype=np.float64,
        )

        baseline_risk = np.array(
            [s.baseline_eens_mwh * s.voll_usd_per_mwh for s in self.scenarios],
            dtype=np.float64,
        )

        weights = np.zeros((S, K), dtype=np.float64)
        for s_idx, sc in enumerate(self.scenarios):
            for k_idx, proj in enumerate(self.projects):
                risk_reduction = (
                    (1.0 - proj.failure_rate_multiplier) * self.failure_weight
                    + (1.0 - proj.recovery_time_multiplier) * self.recovery_weight
                )
                coverage = 1.0 if proj.target_asset_ids else 0.0
                weights[s_idx, k_idx] = risk_reduction * coverage

        c = np.zeros(n_vars, dtype=np.float64)
        for k in range(K):
            capex_term = annualised_capex[k] + self.projects[k].opex_delta_usd_per_year
            risk_term = np.sum(probs * baseline_risk * weights[:, k])
            c[k] = capex_term - risk_term

        A_rows: List[csc_matrix] = []
        b_vals: List[float] = []

        capex_array = np.array([p.capex_usd for p in self.projects], dtype=np.float64)
        row_budget = np.zeros(n_vars, dtype=np.float64)
        row_budget[:] = capex_array
        A_rows.append(csc_matrix(row_budget))
        b_vals.append(float(budget_usd))

        for k in range(K):
            row = np.zeros(n_vars, dtype=np.float64)
            row[k] = 1.0
            A_rows.append(csc_matrix(row))
            b_vals.append(1.0)

        A_ub = vstack(A_rows, format="csc")
        b_ub = np.array(b_vals, dtype=np.float64)

        bounds: List[Tuple[float, float]] = [(0.0, 1.0)] * n_vars

        return c, A_ub, b_ub, bounds

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        budget_usd: Optional[float] = None,
        selection_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Solve the portfolio optimisation LP.

        Parameters
        ----------
        budget_usd : float or None
            Override budget constraint.  If ``None``, uses the instance
            default.
        selection_threshold : float
            Threshold for converting continuous z_k to binary selection.
            Default 0.5.

        Returns
        -------
        dict
            Keys:

            - ``status`` (int) — LP solver status.
            - ``selected_projects`` (list of str) — names of selected
              projects.
            - ``z_values`` (np.ndarray) — raw LP solution.
            - ``total_capex_usd`` (float) — total CAPEX of selected
              projects.
            - ``annualised_capex_usd`` (float) — annualised cost.
            - ``expected_risk_cost_usd`` (float) — probability-weighted
              residual risk cost.
            - ``total_expected_cost_usd`` (float) — annualised CAPEX +
              residual risk.
            - ``bcr`` (float) — Benefit-Cost Ratio.
            - ``message`` (str).
        """
        c, A_ub, b_ub, bounds = self._build_lp(budget_usd)

        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
            options={"disp": False},
        )

        if not result.success:
            logger.warning("Portfolio LP failed: %s", result.message)
            self.result_ = {
                "status": result.status,
                "selected_projects": [],
                "z_values": np.zeros(len(self.projects)),
                "total_capex_usd": 0.0,
                "annualised_capex_usd": 0.0,
                "expected_risk_cost_usd": float("nan"),
                "total_expected_cost_usd": float("nan"),
                "bcr": float("nan"),
                "message": result.message,
            }
            return self.result_

        z = result.x
        selected_mask = z >= selection_threshold

        selected_names = [
            self.projects[i].name for i in range(len(self.projects)) if selected_mask[i]
        ]

        total_capex = float(np.sum(
            [self.projects[i].capex_usd for i in range(len(self.projects)) if selected_mask[i]]
        ))

        annualised = float(np.sum(
            [self._amortize_capex(self.projects[i].capex_usd, self.discount_rate, self.horizon_years)
             + self.projects[i].opex_delta_usd_per_year
             for i in range(len(self.projects)) if selected_mask[i]]
        ))

        probs = np.array([s.probability for s in self.scenarios], dtype=np.float64)
        probs /= probs.sum()

        baseline_risk = np.array(
            [s.baseline_eens_mwh * s.voll_usd_per_mwh for s in self.scenarios],
            dtype=np.float64,
        )

        residual_risk = 0.0
        for s_idx, sc in enumerate(self.scenarios):
            risk_reduction_factor = 1.0
            for k_idx, proj in enumerate(self.projects):
                if selected_mask[k_idx]:
                    fr = proj.failure_rate_multiplier
                    rt = proj.recovery_time_multiplier
                    risk_reduction_factor *= (
                        fr * self.failure_weight + rt * self.recovery_weight
                    )
            residual_risk += probs[s_idx] * baseline_risk[s_idx] * risk_reduction_factor

        total_cost = annualised + residual_risk

        baseline_total_risk = float(np.sum(probs * baseline_risk))
        avoided_risk = baseline_total_risk - residual_risk
        bcr = avoided_risk / annualised if annualised > _EPS else float("inf")

        self.result_ = {
            "status": result.status,
            "selected_projects": selected_names,
            "z_values": z.copy(),
            "total_capex_usd": total_capex,
            "annualised_capex_usd": annualised,
            "expected_risk_cost_usd": residual_risk,
            "total_expected_cost_usd": total_cost,
            "bcr": bcr,
            "baseline_risk_cost_usd": baseline_total_risk,
            "avoided_risk_cost_usd": avoided_risk,
            "message": result.message,
        }
        return self.result_

    # ------------------------------------------------------------------
    # Efficient frontier
    # ------------------------------------------------------------------

    def compute_efficient_frontier(
        self,
        n_points: int = 20,
    ) -> List[Dict[str, Any]]:
        """Compute the CAPEX-vs-risk efficient frontier.

        Sweeps the budget constraint from 0 to the total CAPEX of all
        projects, solving the LP at each point.

        Parameters
        ----------
        n_points : int
            Number of points on the frontier.  Default 20.

        Returns
        -------
        list of dict
            Each dict contains ``budget_usd``, ``total_capex_usd``,
            ``expected_risk_cost_usd``, ``total_expected_cost_usd``,
            ``bcr``, ``n_selected``.
        """
        max_budget = float(np.sum([p.capex_usd for p in self.projects]))
        budgets = np.linspace(0, max_budget, n_points)

        frontier: List[Dict[str, Any]] = []
        for budget in budgets:
            result = self.solve(budget_usd=float(budget))
            frontier.append(
                {
                    "budget_usd": float(budget),
                    "total_capex_usd": result["total_capex_usd"],
                    "expected_risk_cost_usd": result["expected_risk_cost_usd"],
                    "total_expected_cost_usd": result["total_expected_cost_usd"],
                    "bcr": result["bcr"],
                    "n_selected": len(result["selected_projects"]),
                }
            )

        self.frontier_ = frontier
        return frontier

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame summarising project selection results.

        Returns
        -------
        pd.DataFrame
            Columns: ``project``, ``capex_usd``, ``annualised_usd``,
            ``z_value``, ``selected``.

        Raises
        ------
        RuntimeError
            If :meth:`solve` has not been called.
        """
        if self.result_ is None:
            raise RuntimeError("Call solve() before summary_dataframe().")

        z = self.result_["z_values"]
        records = []
        for i, proj in enumerate(self.projects):
            records.append(
                {
                    "project": proj.name,
                    "capex_usd": proj.capex_usd,
                    "annualised_usd": self._amortize_capex(
                        proj.capex_usd, self.discount_rate, self.horizon_years
                    ),
                    "z_value": float(z[i]),
                    "selected": float(z[i]) >= 0.5,
                }
            )

        return pd.DataFrame(records)

    def __repr__(self) -> str:
        return (
            f"InvestmentPortfolioOptimizer(n_projects={len(self.projects)}, "
            f"n_scenarios={len(self.scenarios)}, "
            f"budget=${self.budget_usd:,.0f}, "
            f"horizon={self.horizon_years}y)"
        )
