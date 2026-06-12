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
Global sensitivity analysis and uncertainty quantification engine.

Provides ``GridSensitivityAnalyzer``, which uses the SALib library to
perform Sobol and Morris global sensitivity analysis on the economic
outputs of ``AvoidedLossCalculator``.  Supports quasi-random Sobol
sequence sampling, parallel model evaluation via ``ProcessPoolExecutor``,
and computation of first-order, total-order, and second-order Sobol
indices with confidence intervals.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from SALib.analyze import morris as morris_analyze
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import morris as morris_sample
from SALib.sample import sobol as sobol_sample

from resilient_blackout.core.base import HazardEvent
from resilient_blackout.core.economics import AvoidedLossCalculator

logger = logging.getLogger(__name__)

_DEFAULT_PARAMETER_BOUNDS: Dict[str, List[float]] = {
    "failure_rate_scale": [0.5, 2.0],
    "restoration_duration_hours": [4.0, 72.0],
    "recovery_compression": [2.0, 50.0],
    "voll_residential": [1000.0, 50000.0],
    "voll_commercial": [2000.0, 100000.0],
    "voll_industrial": [5000.0, 200000.0],
    "discount_rate": [0.01, 0.15],
    "cascade_tolerance": [1.05, 1.5],
}


class GridSensitivityAnalyzer:
    """Global sensitivity analysis for grid resilience economics.

    Wraps an ``AvoidedLossCalculator`` and uses SALib's Sobol and Morris
    methods to identify which input parameters most strongly influence
    economic outputs (NPV, avoided loss, BCR).

    Parameters
    ----------
    avoided_loss_calculator : AvoidedLossCalculator
        The economic engine to analyse.
    parameter_bounds : dict or None
        Optional override of default parameter ranges.  Keys must match
        the default parameter names.  Values are ``[lower, upper]``
        lists.

    Attributes
    ----------
    calculator : AvoidedLossCalculator
    problem : dict
        SALib problem definition.
    param_names : list of str
    n_params : int
    """

    def __init__(
        self,
        avoided_loss_calculator: AvoidedLossCalculator,
        parameter_bounds: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        self.calculator = avoided_loss_calculator

        bounds = dict(_DEFAULT_PARAMETER_BOUNDS)
        if parameter_bounds:
            bounds.update(parameter_bounds)

        self.param_names = list(bounds.keys())
        self.n_params = len(self.param_names)

        self.problem: Dict[str, Any] = {
            "num_vars": self.n_params,
            "names": self.param_names,
            "bounds": [bounds[name] for name in self.param_names],
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def generate_quasi_random_samples(self, N: int) -> np.ndarray:
        """Generate Sobol quasi-random samples.

        Uses SALib's Sobol sequence sampler to produce
        :math:`N \\times (2D + 2)` samples, where :math:`D` is the
        number of parameters.

        Parameters
        ----------
        N : int
            Base sample size.  Total samples = ``N * (2*D + 2)``.

        Returns
        -------
        np.ndarray
            Sample matrix of shape ``(N*(2D+2), D)``.
        """
        samples = sobol_sample.sample(self.problem, N)
        logger.info(
            "Generated %d Sobol samples for %d parameters.",
            len(samples), self.n_params,
        )
        return samples

    def generate_morris_samples(
        self, N: int, num_levels: int = 4
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate Morris method samples for screening.

        Parameters
        ----------
        N : int
            Number of trajectories.
        num_levels : int
            Number of grid levels.  Default 4.

        Returns
        -------
        tuple of (np.ndarray, dict)
            Sample matrix and Morris-specific problem dict.
        """
        morris_problem = {**self.problem, "num_levels": num_levels}
        samples = morris_sample.sample(morris_problem, N)
        logger.info(
            "Generated %d Morris samples (%d trajectories, %d levels).",
            len(samples), N, num_levels,
        )
        return samples, morris_problem

    # ------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        parameter_samples: np.ndarray,
        hazard_events: List[HazardEvent],
        parallel: bool = True,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """Evaluate the economic model for each parameter row.

        For each row in ``parameter_samples``, creates a modified
        ``AvoidedLossCalculator`` with the sampled parameters and runs
        ``run_cost_benefit_analysis``.

        Parameters
        ----------
        parameter_samples : np.ndarray
            Shape ``(n_samples, n_params)``.
        hazard_events : list of HazardEvent
            Hazard events to evaluate against.
        parallel : bool
            If ``True``, use ``ProcessPoolExecutor``.
        n_jobs : int
            Number of parallel workers.  ``-1`` uses all cores.

        Returns
        -------
        np.ndarray
            Model outputs of shape ``(n_samples,)``.  Default output
            is NPV in USD.
        """
        n_samples = len(parameter_samples)

        if parallel and n_jobs != 1:
            return self._evaluate_parallel(
                parameter_samples, hazard_events, n_jobs
            )

        outputs = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            outputs[i] = self._evaluate_single(parameter_samples[i], hazard_events)
        return outputs

    def _evaluate_single(
        self,
        params: np.ndarray,
        hazard_events: List[HazardEvent],
    ) -> float:
        """Evaluate one parameter set.

        Parameters
        ----------
        params : np.ndarray
            Single row of parameters.
        hazard_events : list of HazardEvent

        Returns
        -------
        float
            NPV in USD.
        """
        p = dict(zip(self.param_names, params))

        voll = {
            "residential": float(p["voll_residential"]),
            "commercial": float(p["voll_commercial"]),
            "industrial": float(p["voll_industrial"]),
            "default": float(p["voll_residential"]),
        }

        calc = AvoidedLossCalculator(
            baseline_grid=self.calculator.baseline_grid,
            resilient_grid=self.calculator.resilient_grid,
            baseline_assets=self.calculator.baseline_assets,
            resilient_assets=self.calculator.resilient_assets,
            impact_functions=self.calculator.impact_functions,
            discount_rate=float(p["discount_rate"]),
            planning_horizon=self.calculator.planning_horizon,
            n_jobs=1,
            mc_trials=self.calculator.mc_trials,
            recovery_base_hours=float(p["restoration_duration_hours"]),
            recovery_compression=float(p["recovery_compression"]),
            cascade_tolerance=float(p["cascade_tolerance"]),
            cascade_max_iter=self.calculator.cascade_max_iter,
        )

        result = calc.run_cost_benefit_analysis(
            initial_investment=0.0,
            annual_opex_delta=0.0,
            hazard_events=hazard_events,
            voll_by_sector=voll,
        )

        return float(result["npv"])

    def _evaluate_parallel(
        self,
        parameter_samples: np.ndarray,
        hazard_events: List[HazardEvent],
        n_jobs: int,
    ) -> np.ndarray:
        """Parallel model evaluation via ProcessPoolExecutor.

        Parameters
        ----------
        parameter_samples : np.ndarray
        hazard_events : list of HazardEvent
        n_jobs : int

        Returns
        -------
        np.ndarray
        """
        max_workers = n_jobs if n_jobs > 0 else None
        n_samples = len(parameter_samples)
        outputs = np.empty(n_samples, dtype=np.float64)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._evaluate_single, row, hazard_events): i
                for i, row in enumerate(parameter_samples)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    outputs[idx] = future.result()
                except Exception:
                    logger.exception("Evaluation failed for sample %d.", idx)
                    outputs[idx] = np.nan

        return outputs

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def analyze_variance(
        self,
        parameter_samples: np.ndarray,
        model_outputs: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute Sobol sensitivity indices.

        Calculates first-order (S1), total-order (ST), and second-order
        (S2) indices with bootstrap confidence intervals.

        Parameters
        ----------
        parameter_samples : np.ndarray
            Sobol sample matrix.
        model_outputs : np.ndarray
            Model outputs corresponding to each sample row.

        Returns
        -------
        dict
            Keys: ``"S1"``, ``"ST"``, ``"S2"``, ``"S1_conf"``,
            ``"ST_conf"``, ``"S2_conf"``, ``"param_names"``.
        """
        valid_mask = ~np.isnan(model_outputs)
        if not valid_mask.all():
            logger.warning(
                "Dropping %d NaN outputs from %d total.",
                (~valid_mask).sum(), len(model_outputs),
            )
            Y = model_outputs[valid_mask]
            D = self.n_params
            N_base = len(Y) // (2 * D + 2)
            if N_base < 2:
                raise ValueError(
                    "Too few valid outputs for Sobol analysis after "
                    "dropping NaNs."
                )
            adjusted_problem = {**self.problem}
            result = sobol_analyze.analyze(
                adjusted_problem, Y,
                calc_second_order=True,
                print_to_console=False,
            )
        else:
            result = sobol_analyze.analyze(
                self.problem, model_outputs,
                calc_second_order=True,
                print_to_console=False,
            )

        return {
            "S1": result["S1"].tolist(),
            "ST": result["ST"].tolist(),
            "S2": result.get("S2", np.array([])).tolist(),
            "S1_conf": result["S1_conf"].tolist(),
            "ST_conf": result["ST_conf"].tolist(),
            "S2_conf": result.get("S2_conf", np.array([])).tolist(),
            "param_names": self.param_names,
        }

    def analyze_morris(
        self,
        parameter_samples: np.ndarray,
        model_outputs: np.ndarray,
        morris_problem: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute Morris method sensitivity indices.

        Returns mu_star (mean absolute elementary effect), sigma
        (standard deviation), and confidence intervals for screening.

        Parameters
        ----------
        parameter_samples : np.ndarray
        model_outputs : np.ndarray
        morris_problem : dict or None

        Returns
        -------
        dict
            Keys: ``"mu_star"``, ``"sigma"``, ``"mu_star_conf"``,
            ``"param_names"``.
        """
        problem = morris_problem or {**self.problem, "num_levels": 4}

        valid_mask = ~np.isnan(model_outputs)
        if not valid_mask.all():
            Y = model_outputs[valid_mask]
        else:
            Y = model_outputs

        result = morris_analyze.analyze(
            problem,
            parameter_samples[: len(Y)],
            Y,
            print_to_console=False,
        )

        return {
            "mu_star": result["mu_star"].tolist(),
            "sigma": result["sigma"].tolist(),
            "mu_star_conf": result["mu_star_conf"].tolist(),
            "param_names": self.param_names,
        }

    # ------------------------------------------------------------------
    # Full analysis pipeline
    # ------------------------------------------------------------------

    def run_full_analysis(
        self,
        hazard_events: List[HazardEvent],
        N: int = 1024,
        method: str = "sobol",
        parallel: bool = True,
        n_jobs: int = -1,
    ) -> Dict[str, Any]:
        """Run end-to-end sensitivity analysis.

        Parameters
        ----------
        hazard_events : list of HazardEvent
        N : int
            Base sample size.  Default 1024.
        method : str
            ``"sobol"`` or ``"morris"``.
        parallel : bool
        n_jobs : int

        Returns
        -------
        dict
            ``{"method": str, "samples": np.ndarray,
            "outputs": np.ndarray, "indices": dict}``.
        """
        if method == "sobol":
            samples = self.generate_quasi_random_samples(N)
            outputs = self.evaluate_model(
                samples, hazard_events, parallel=parallel, n_jobs=n_jobs
            )
            indices = self.analyze_variance(samples, outputs)
        elif method == "morris":
            samples, morris_prob = self.generate_morris_samples(N)
            outputs = self.evaluate_model(
                samples, hazard_events, parallel=parallel, n_jobs=n_jobs
            )
            indices = self.analyze_morris(samples, outputs, morris_prob)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'sobol' or 'morris'.")

        return {
            "method": method,
            "samples": samples,
            "outputs": outputs,
            "indices": indices,
        }
