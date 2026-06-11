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
Financial and risk calculation engine.

Provides the ``AvoidedLossCalculator`` class that ties together the hazard
engine, cascading simulator, and economic metrics to compute the return on
investment (ROI) of grid resilience investments.  Supports parallel Monte
Carlo simulation across CPU cores via ``ProcessPoolExecutor``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import newton

from resilient_blackout.core.base import Asset, HazardEvent
from resilient_blackout.core.fragility import ImpactFunction, ImpactFunctionSet
from resilient_blackout.grid.cascade import CascadingSimulator
from resilient_blackout.grid.network import GridModel
from resilient_blackout.utils.geo import map_hazard_to_assets

logger = logging.getLogger(__name__)


@dataclass
class _TrialInput:
    """Pickleable container for a single Monte Carlo trial's inputs."""

    asset_ids: List[str]
    impact_function_ids: List[Optional[str]]
    intensities: List[float]
    grid_model: GridModel
    cascade_tolerance: float
    cascade_max_iter: int
    trial_seed: int


def _execute_single_trial(inputs: _TrialInput) -> Dict[str, Any]:
    """Execute one Monte Carlo trial (top-level function for multiprocessing).

    Parameters
    ----------
    inputs : _TrialInput
        All data needed for the trial.

    Returns
    -------
    dict
        ``{"load_shed_mw": float, "n_failed_assets": int, "failed_asset_ids": list}``.
    """
    rng = np.random.default_rng(inputs.trial_seed)

    failed_assets: List[str] = []
    for aid, if_id, intensity in zip(
        inputs.asset_ids, inputs.impact_function_ids, inputs.intensities
    ):
        if if_id is None or intensity <= 0:
            continue
        prob = _evaluate_failure_static(if_id, intensity)
        if rng.random() < prob:
            failed_assets.append(aid)

    if not failed_assets:
        return {"load_shed_mw": 0.0, "n_failed_assets": 0, "failed_asset_ids": []}

    simulator = CascadingSimulator(
        grid_model=inputs.grid_model,
        tolerance_factor=inputs.cascade_tolerance,
        max_iterations=inputs.cascade_max_iter,
        rng=rng,
    )
    result = simulator.simulate_cascade(failed_assets)
    return {
        "load_shed_mw": result["total_load_shed_mw"],
        "n_failed_assets": len(failed_assets),
        "failed_asset_ids": failed_assets,
    }


# Module-level cache for impact functions used by _execute_single_trial.
# Populated by AvoidedLossCalculator before parallel execution.
_IMPACT_FUNCTION_CACHE: Dict[str, ImpactFunction] = {}


def _evaluate_failure_static(function_id: str, intensity: float) -> float:
    """Look up an impact function and evaluate failure probability.

    Uses the module-level ``_IMPACT_FUNCTION_CACHE`` populated by the
    parent ``AvoidedLossCalculator`` before spawning worker processes.

    Parameters
    ----------
    function_id : str
        Impact function identifier.
    intensity : float
        Hazard intensity.

    Returns
    -------
    float
        Failure probability in [0, 1].
    """
    func = _IMPACT_FUNCTION_CACHE.get(function_id)
    if func is None:
        return 0.0
    return func.evaluate_failure_probability(intensity)


class AvoidedLossCalculator:
    """Computes the financial ROI of grid resilience investments.

    Runs parallel Monte Carlo simulations to estimate Expected Energy Not
    Served (EENS) for baseline and resilient grid configurations, then
    derives avoided losses and performs cost-benefit analysis (NPV, IRR,
    BCR).

    Parameters
    ----------
    baseline_grid : GridModel
        The unmodified (baseline) grid model.
    resilient_grid : GridModel
        The hardened/resilient grid model.
    baseline_assets : list of Asset
        Assets in the baseline configuration.
    resilient_assets : list of Asset
        Assets in the resilient configuration.
    impact_functions : ImpactFunctionSet
        Vulnerability curves keyed by ``function_id``.
    discount_rate : float
        Annual discount rate for NPV calculation (e.g., 0.05 for 5 %).
    planning_horizon : int
        Number of years over which benefits are evaluated.
    n_jobs : int
        Number of parallel worker processes.  ``-1`` uses all available
        cores.  ``1`` runs serially.
    mc_trials : int
        Number of Monte Carlo trials per hazard event.  Default 1000.
    recovery_base_hours : float
        Base recovery duration in hours for a single asset failure.
    recovery_compression : float
        Compression factor for multi-asset recovery (higher values mean
        less additional time per extra failed asset).
    cascade_tolerance : float
        Overload tolerance passed to ``CascadingSimulator``.
    cascade_max_iter : int
        Max cascade iterations passed to ``CascadingSimulator``.

    Attributes
    ----------
    baseline_grid : GridModel
    resilient_grid : GridModel
    baseline_assets : list of Asset
    resilient_assets : list of Asset
    impact_functions : ImpactFunctionSet
    discount_rate : float
    planning_horizon : int
    n_jobs : int
    mc_trials : int
    """

    def __init__(
        self,
        baseline_grid: GridModel,
        resilient_grid: GridModel,
        baseline_assets: List[Asset],
        resilient_assets: List[Asset],
        impact_functions: ImpactFunctionSet,
        discount_rate: float = 0.05,
        planning_horizon: int = 30,
        n_jobs: int = -1,
        mc_trials: int = 1000,
        recovery_base_hours: float = 24.0,
        recovery_compression: float = 10.0,
        cascade_tolerance: float = 1.2,
        cascade_max_iter: int = 50,
    ) -> None:
        if discount_rate < 0:
            raise ValueError(f"discount_rate must be >= 0, got {discount_rate}")
        if planning_horizon < 1:
            raise ValueError(f"planning_horizon must be >= 1, got {planning_horizon}")
        if mc_trials < 1:
            raise ValueError(f"mc_trials must be >= 1, got {mc_trials}")

        self.baseline_grid = baseline_grid
        self.resilient_grid = resilient_grid
        self.baseline_assets = baseline_assets
        self.resilient_assets = resilient_assets
        self.impact_functions = impact_functions
        self.discount_rate = discount_rate
        self.planning_horizon = planning_horizon
        self.n_jobs = n_jobs
        self.mc_trials = mc_trials
        self.recovery_base_hours = recovery_base_hours
        self.recovery_compression = recovery_compression
        self.cascade_tolerance = cascade_tolerance
        self.cascade_max_iter = cascade_max_iter

    # ------------------------------------------------------------------
    # EENS calculation
    # ------------------------------------------------------------------

    def calculate_expected_energy_not_served(
        self,
        grid_model: GridModel,
        assets: List[Asset],
        hazard_events: List[HazardEvent],
    ) -> Dict[str, Any]:
        """Estimate Expected Energy Not Served (EENS) via Monte Carlo.

        For each hazard event, runs ``mc_trials`` trials in parallel.
        Each trial maps hazard intensities to assets, samples physical
        failures, runs the cascading simulator, and computes Energy Not
        Served as ``load_shed_mw * recovery_duration_hours``.

        Recovery duration uses logarithmic compression to model the
        "Recover" state of M-A-R-C:

        .. math::

            T_{rec} = T_{base} \\cdot \\left(1 +
            \\frac{\\ln(1 + N_{failed})}{\\ln(1 + C)}\\right)

        where :math:`C` is ``recovery_compression``.

        Parameters
        ----------
        grid_model : GridModel
            The grid configuration to evaluate.
        assets : list of Asset
            Assets in this configuration.
        hazard_events : list of HazardEvent
            Hazard events to simulate.

        Returns
        -------
        dict
            Keys:

            - ``per_event`` (dict) — ``event_id`` → per-event statistics.
            - ``total_eens_mwh`` (float) — sum of EENS across all events,
              weighted by event frequency.
            - ``total_mean_shed_mw`` (float) — frequency-weighted mean
              load shed.
        """
        results: Dict[str, Any] = {"per_event": {}}
        total_eens = 0.0
        total_mean_shed = 0.0

        for hazard in hazard_events:
            event_stats = self._evaluate_event(grid_model, assets, hazard)
            results["per_event"][hazard.event_id] = event_stats
            total_eens += event_stats["eens_mwh"] * hazard.frequency
            total_mean_shed += event_stats["mean_shed_mw"] * hazard.frequency

        results["total_eens_mwh"] = total_eens
        results["total_mean_shed_mw"] = total_mean_shed
        return results

    def _evaluate_event(
        self,
        grid_model: GridModel,
        assets: List[Asset],
        hazard: HazardEvent,
    ) -> Dict[str, Any]:
        """Run Monte Carlo trials for a single hazard event.

        Parameters
        ----------
        grid_model : GridModel
        assets : list of Asset
        hazard : HazardEvent

        Returns
        -------
        dict
            Per-event statistics.
        """
        intensity_map = map_hazard_to_assets(assets, hazard)

        asset_ids: List[str] = []
        impact_function_ids: List[Optional[str]] = []
        intensities: List[float] = []

        for asset in assets:
            intensity = intensity_map.get(asset.asset_id, 0.0)
            asset_ids.append(asset.asset_id)
            impact_function_ids.append(asset.impact_function_id)
            intensities.append(intensity)

        base_seed = hash(hazard.event_id) & 0x7FFFFFFF

        trial_inputs = [
            _TrialInput(
                asset_ids=asset_ids,
                impact_function_ids=impact_function_ids,
                intensities=intensities,
                grid_model=grid_model,
                cascade_tolerance=self.cascade_tolerance,
                cascade_max_iter=self.cascade_max_iter,
                trial_seed=base_seed + i,
            )
            for i in range(self.mc_trials)
        ]

        # Populate module-level cache for worker processes
        global _IMPACT_FUNCTION_CACHE
        _IMPACT_FUNCTION_CACHE = dict(self.impact_functions)

        if self.n_jobs == 1:
            trial_results = [_execute_single_trial(ti) for ti in trial_inputs]
        else:
            trial_results = self._run_parallel(trial_inputs)

        sheds = np.array([r["load_shed_mw"] for r in trial_results], dtype=np.float64)
        n_failed = np.array([r["n_failed_assets"] for r in trial_results], dtype=np.int32)

        recovery_hours = self._compute_recovery_duration(n_failed)
        ens_per_trial = sheds * recovery_hours

        mean_ens = float(np.mean(ens_per_trial))
        mean_shed = float(np.mean(sheds))
        mean_failed = float(np.mean(n_failed))
        eens_mwh = mean_ens

        return {
            "eens_mwh": eens_mwh,
            "mean_ens_mwh": mean_ens,
            "mean_shed_mw": mean_shed,
            "mean_failed_assets": mean_failed,
            "std_shed_mw": float(np.std(sheds)),
            "p95_shed_mw": float(np.percentile(sheds, 95)),
            "p99_shed_mw": float(np.percentile(sheds, 99)),
            "n_trials": self.mc_trials,
        }

    def _compute_recovery_duration(self, n_failed: np.ndarray) -> np.ndarray:
        """Compute recovery duration with logarithmic compression.

        Parameters
        ----------
        n_failed : np.ndarray
            Number of failed assets per trial.

        Returns
        -------
        np.ndarray
            Recovery duration in hours.
        """
        base = self.recovery_base_hours
        comp = self.recovery_compression
        if comp <= 0:
            return np.full_like(n_failed, base, dtype=np.float64)
        factor = 1.0 + np.log1p(n_failed.astype(np.float64)) / np.log1p(comp)
        return base * factor

    def _run_parallel(self, trial_inputs: List[_TrialInput]) -> List[Dict[str, Any]]:
        """Execute trials in parallel via ProcessPoolExecutor.

        Parameters
        ----------
        trial_inputs : list of _TrialInput

        Returns
        -------
        list of dict
            Trial results in submission order.
        """
        max_workers = self.n_jobs if self.n_jobs > 0 else None
        results: List[Optional[Dict[str, Any]]] = [None] * len(trial_inputs)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_execute_single_trial, ti): i
                for i, ti in enumerate(trial_inputs)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.exception("Trial %d failed; using zero-shed fallback.", idx)
                    results[idx] = {
                        "load_shed_mw": 0.0,
                        "n_failed_assets": 0,
                        "failed_asset_ids": [],
                    }

        return [r for r in results if r is not None]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Avoided loss
    # ------------------------------------------------------------------

    def calculate_avoided_loss(
        self,
        hazard_events: List[HazardEvent],
        voll_by_sector: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compute avoided financial loss between baseline and resilient grids.

        Estimates EENS for both configurations and converts to financial
        risk using the Value of Lost Load (VoLL).

        Parameters
        ----------
        hazard_events : list of HazardEvent
            Hazard events to evaluate.
        voll_by_sector : dict
            Mapping from sector name to VoLL in $/MWh.  The key
            ``"default"`` is used as a fallback.

        Returns
        -------
        dict
            Keys:

            - ``baseline_eens_mwh`` (float)
            - ``resilient_eens_mwh`` (float)
            - ``baseline_risk_usd`` (float)
            - ``resilient_risk_usd`` (float)
            - ``avoided_loss_usd`` (float)
            - ``avoided_eens_mwh`` (float)
            - ``voll_used`` (float) — the VoLL value applied.
        """
        voll = voll_by_sector.get("default", voll_by_sector.get("residential", 10000.0))

        baseline_eens = self.calculate_expected_energy_not_served(
            self.baseline_grid, self.baseline_assets, hazard_events
        )
        resilient_eens = self.calculate_expected_energy_not_served(
            self.resilient_grid, self.resilient_assets, hazard_events
        )

        baseline_eens_val = baseline_eens["total_eens_mwh"]
        resilient_eens_val = resilient_eens["total_eens_mwh"]

        baseline_risk = baseline_eens_val * voll
        resilient_risk = resilient_eens_val * voll
        avoided_loss = baseline_risk - resilient_risk
        avoided_eens = baseline_eens_val - resilient_eens_val

        return {
            "baseline_eens_mwh": baseline_eens_val,
            "resilient_eens_mwh": resilient_eens_val,
            "baseline_risk_usd": baseline_risk,
            "resilient_risk_usd": resilient_risk,
            "avoided_loss_usd": avoided_loss,
            "avoided_eens_mwh": avoided_eens,
            "voll_used": voll,
            "baseline_per_event": baseline_eens["per_event"],
            "resilient_per_event": resilient_eens["per_event"],
        }

    # ------------------------------------------------------------------
    # Cost-benefit analysis
    # ------------------------------------------------------------------

    def run_cost_benefit_analysis(
        self,
        initial_investment: float,
        annual_opex_delta: float,
        hazard_events: List[HazardEvent],
        voll_by_sector: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compute NPV, IRR, and BCR for a resilience investment.

        Parameters
        ----------
        initial_investment : float
            Up-front capital cost of the resilience upgrade (USD).
        annual_opex_delta : float
            Change in annual operating expenditure (positive = cost
            increase, negative = savings).
        hazard_events : list of HazardEvent
            Hazard events to evaluate.
        voll_by_sector : dict
            VoLL by sector (see :meth:`calculate_avoided_loss`).

        Returns
        -------
        dict
            Keys:

            - ``npv`` (float) — Net Present Value.
            - ``irr`` (float or None) — Internal Rate of Return (``None``
              if not computable).
            - ``bcr`` (float) — Benefit-Cost Ratio.
            - ``payback_years`` (float or None) — simple payback period.
            - ``annual_benefit`` (float) — annual avoided loss.
            - ``pv_benefits`` (float) — present value of all benefits.
            - ``pv_costs`` (float) — present value of all costs.
        """
        avoided = self.calculate_avoided_loss(hazard_events, voll_by_sector)
        annual_benefit = avoided["avoided_loss_usd"] - annual_opex_delta

        years = np.arange(1, self.planning_horizon + 1, dtype=np.float64)
        discount_factors = (1.0 + self.discount_rate) ** years
        pv_benefits = np.sum(annual_benefit / discount_factors)
        pv_costs = initial_investment

        npv = pv_benefits - pv_costs
        bcr = pv_benefits / pv_costs if pv_costs > 0 else float("inf")

        irr = self._compute_irr(initial_investment, annual_benefit)
        payback = self._compute_payback(initial_investment, annual_benefit)

        return {
            "npv": npv,
            "irr": irr,
            "bcr": bcr,
            "payback_years": payback,
            "annual_benefit": annual_benefit,
            "pv_benefits": pv_benefits,
            "pv_costs": pv_costs,
            "avoided_loss_detail": avoided,
        }

    def _compute_irr(self, investment: float, annual_benefit: float) -> Optional[float]:
        """Compute IRR via Newton's method on the NPV polynomial.

        Parameters
        ----------
        investment : float
            Initial investment (positive).
        annual_benefit : float
            Annual cash flow.

        Returns
        -------
        float or None
            IRR as a decimal, or ``None`` if not computable.
        """
        if investment <= 0 or annual_benefit <= 0:
            return None

        def npv_func(r: float) -> float:
            if r <= -1.0:
                return -investment
            if abs(r) < 1e-12:
                return annual_benefit * self.planning_horizon - investment
            factor = (1.0 - (1.0 + r) ** -self.planning_horizon) / r
            return annual_benefit * factor - investment

        try:
            return float(newton(npv_func, 0.05, maxiter=100, tol=1e-8))
        except (RuntimeError, ValueError):
            return None

    def _compute_payback(self, investment: float, annual_benefit: float) -> Optional[float]:
        """Compute simple payback period in years.

        Parameters
        ----------
        investment : float
        annual_benefit : float

        Returns
        -------
        float or None
        """
        if annual_benefit <= 0:
            return None
        return investment / annual_benefit
