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
Global sensitivity analysis module.

Provides ``GlobalSensitivityAnalyzer``, a SALib-based engine for
Sobol variance decomposition and Morris screening of arbitrary
black-box models.  Supports quasi-random Sobol sequence sampling,
parallel model evaluation, computation of first-order (S₁),
second-order (S₂), and total-order (S_T) sensitivity indices with
95 % bootstrap confidence intervals, and built-in visualisation
helpers (bar charts and interaction heatmaps).

All dependencies (SALib, NumPy, Pandas, Matplotlib) are permissively
licensed (MIT / BSD / PSF).
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyze
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import morris as morris_sample
from SALib.sample import sobol as sobol_sample

logger = logging.getLogger(__name__)

_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# GlobalSensitivityAnalyzer
# ---------------------------------------------------------------------------


class GlobalSensitivityAnalyzer:
    """Global sensitivity analysis engine using SALib.

    Defines an input parameter space via SALib's standard problem
    dictionary, generates quasi-random Sobol and Morris samples,
    evaluates a user-supplied black-box model (optionally in parallel),
    and computes Sobol and Morris sensitivity indices with confidence
    intervals.

    Parameters
    ----------
    param_names : list of str
        Ordered list of parameter names.
    param_bounds : list of [float, float]
        Lower and upper bounds per parameter, same order as *param_names*.
    param_groups : list of str or None
        Optional group label per parameter (e.g. ``"failure"``,
        ``"economic"``).  Used for grouped visualisation.
    model_func : callable or None
        Signature ``f(params: np.ndarray) -> float``.  If ``None``,
        must be supplied to :meth:`evaluate_model`.

    Attributes
    ----------
    problem : dict
        SALib problem definition with ``num_vars``, ``names``,
        ``bounds``, and optionally ``groups``.
    param_names : list of str
    param_groups : list of str or None
    n_params : int
    model_func : callable or None
    """

    def __init__(
        self,
        param_names: List[str],
        param_bounds: List[List[float]],
        param_groups: Optional[List[str]] = None,
        model_func: Optional[Callable[[np.ndarray], float]] = None,
    ) -> None:
        if len(param_names) == 0:
            raise ValueError("param_names must not be empty")
        if len(param_bounds) != len(param_names):
            raise ValueError(
                f"param_bounds length ({len(param_bounds)}) != "
                f"param_names length ({len(param_names)})"
            )
        if param_groups is not None and len(param_groups) != len(param_names):
            raise ValueError(
                f"param_groups length ({len(param_groups)}) != "
                f"param_names length ({len(param_names)})"
            )

        for i, (lo, hi) in enumerate(param_bounds):
            if lo >= hi:
                raise ValueError(
                    f"param_bounds[{i}] ('{param_names[i]}'): "
                    f"lower ({lo}) must be < upper ({hi})"
                )

        self.param_names = list(param_names)
        self.param_groups = list(param_groups) if param_groups else None
        self.n_params = len(self.param_names)
        self.model_func = model_func

        self.problem: Dict[str, Any] = {
            "num_vars": self.n_params,
            "names": self.param_names,
            "bounds": [list(b) for b in param_bounds],
        }
        if self.param_groups is not None:
            self.problem["groups"] = self.param_groups

    # ------------------------------------------------------------------
    # Sobol sampling
    # ------------------------------------------------------------------

    def generate_sobol_samples(
        self,
        N: int,
        calc_second_order: bool = True,
    ) -> np.ndarray:
        """Generate Sobol quasi-random parameter combinations.

        Uses SALib's Saltelli cross-sampling scheme.  The total sample
        size scales as :math:`2 \\cdot N \\cdot (K + 1)` when
        *calc_second_order* is ``True``, or :math:`N \\cdot (K + 2)`
        when ``False``, where :math:`K` is the number of variables.

        Parameters
        ----------
        N : int
            Base sample size (power of 2 recommended).
        calc_second_order : bool
            If ``True``, generate samples sufficient for second-order
            (S₂) index computation.  Default ``True``.

        Returns
        -------
        np.ndarray
            Sample matrix of shape ``(N_total, K)``.
        """
        samples = sobol_sample.sample(
            self.problem, N, calc_second_order=calc_second_order
        )
        logger.info(
            "Generated %d Sobol samples for %d parameters (calc_second_order=%s).",
            len(samples), self.n_params, calc_second_order,
        )
        return samples

    # ------------------------------------------------------------------
    # Morris sampling
    # ------------------------------------------------------------------

    def evaluate_morris_screening(
        self,
        N: int,
        num_levels: int = 4,
    ) -> Dict[str, Any]:
        """Perform Morris method screening for parameter prioritisation.

        Generates Morris trajectories and returns the sample matrix
        together with the Morris-specific problem dictionary.

        Parameters
        ----------
        N : int
            Number of trajectories.
        num_levels : int
            Number of grid levels.  Default 4.

        Returns
        -------
        dict
            ``{"samples": np.ndarray, "morris_problem": dict}``.
        """
        morris_problem = {**self.problem, "num_levels": num_levels}
        samples = morris_sample.sample(morris_problem, N)
        logger.info(
            "Generated %d Morris samples (%d trajectories, %d levels).",
            len(samples), N, num_levels,
        )
        return {"samples": samples, "morris_problem": morris_problem}

    # ------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        parameter_samples: np.ndarray,
        model_func: Optional[Callable[[np.ndarray], float]] = None,
        parallel: bool = True,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """Evaluate the model for each parameter row.

        Parameters
        ----------
        parameter_samples : np.ndarray
            Shape ``(n_samples, n_params)``.
        model_func : callable or None
            ``f(params: np.ndarray) -> float``.  Uses instance default
            if ``None``.
        parallel : bool
            If ``True``, use ``ProcessPoolExecutor``.
        n_jobs : int
            Number of parallel workers.  ``-1`` uses all cores.

        Returns
        -------
        np.ndarray
            Model outputs of shape ``(n_samples,)``.
        """
        func = model_func or self.model_func
        if func is None:
            raise ValueError(
                "model_func must be provided either at construction "
                "or to evaluate_model()."
            )

        n_samples = len(parameter_samples)

        if parallel and n_jobs != 1:
            return self._evaluate_parallel(parameter_samples, func, n_jobs)

        outputs = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            outputs[i] = float(func(parameter_samples[i]))
        return outputs

    @staticmethod
    def _evaluate_parallel(
        parameter_samples: np.ndarray,
        model_func: Callable[[np.ndarray], float],
        n_jobs: int,
    ) -> np.ndarray:
        """Parallel model evaluation via ProcessPoolExecutor.

        Parameters
        ----------
        parameter_samples : np.ndarray
        model_func : callable
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
                executor.submit(model_func, row): i
                for i, row in enumerate(parameter_samples)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    outputs[idx] = float(future.result())
                except Exception:
                    logger.exception("Evaluation failed for sample %d.", idx)
                    outputs[idx] = np.nan

        return outputs

    # ------------------------------------------------------------------
    # Sobol analysis
    # ------------------------------------------------------------------

    def analyze_sobol_indices(
        self,
        parameter_samples: np.ndarray,
        model_outputs: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute Sobol sensitivity indices.

        Calculates first-order (S₁), second-order (S₂), and total-order
        (S_T) indices with 95 % bootstrap confidence intervals.

        Parameters
        ----------
        parameter_samples : np.ndarray
            Sobol sample matrix from :meth:`generate_sobol_samples`.
        model_outputs : np.ndarray
            Model outputs corresponding to each sample row.

        Returns
        -------
        dict
            Keys:

            - ``S1`` (np.ndarray) — first-order indices.
            - ``ST`` (np.ndarray) — total-order indices.
            - ``S2`` (np.ndarray) — second-order indices (upper
              triangle).
            - ``S1_conf`` (np.ndarray) — S₁ 95 % confidence intervals.
            - ``ST_conf`` (np.ndarray) — S_T 95 % confidence intervals.
            - ``S2_conf`` (np.ndarray) — S₂ 95 % confidence intervals.
            - ``param_names`` (list of str).
            - ``summary`` (pd.DataFrame) — tabular summary.
        """
        valid_mask = ~np.isnan(model_outputs)
        if not valid_mask.all():
            n_dropped = (~valid_mask).sum()
            logger.warning("Dropping %d NaN outputs from %d total.", n_dropped, len(model_outputs))
            Y = model_outputs[valid_mask]
        else:
            Y = model_outputs

        result = sobol_analyze.analyze(
            self.problem, Y,
            calc_second_order=True,
            print_to_console=False,
        )

        s1 = np.asarray(result["S1"], dtype=np.float64)
        st = np.asarray(result["ST"], dtype=np.float64)
        s1_conf = np.asarray(result["S1_conf"], dtype=np.float64)
        st_conf = np.asarray(result["ST_conf"], dtype=np.float64)
        s2 = np.asarray(result.get("S2", []), dtype=np.float64)
        s2_conf = np.asarray(result.get("S2_conf", []), dtype=np.float64)

        summary = pd.DataFrame({
            "parameter": self.param_names,
            "S1": s1,
            "S1_conf": s1_conf,
            "ST": st,
            "ST_conf": st_conf,
        })
        if self.param_groups is not None:
            summary["group"] = self.param_groups

        summary = summary.sort_values("ST", ascending=False).reset_index(drop=True)

        return {
            "S1": s1,
            "ST": st,
            "S2": s2,
            "S1_conf": s1_conf,
            "ST_conf": st_conf,
            "S2_conf": s2_conf,
            "param_names": self.param_names,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Morris analysis
    # ------------------------------------------------------------------

    def analyze_morris(
        self,
        parameter_samples: np.ndarray,
        model_outputs: np.ndarray,
        morris_problem: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute Morris method sensitivity indices.

        Returns mu_star (mean absolute elementary effect), sigma
        (standard deviation), and confidence intervals.

        Parameters
        ----------
        parameter_samples : np.ndarray
        model_outputs : np.ndarray
        morris_problem : dict or None

        Returns
        -------
        dict
            ``{"mu_star": np.ndarray, "sigma": np.ndarray,
            "mu_star_conf": np.ndarray, "param_names": list,
            "summary": pd.DataFrame}``.
        """
        problem = morris_problem or {**self.problem, "num_levels": 4}

        valid_mask = ~np.isnan(model_outputs)
        Y = model_outputs[valid_mask] if not valid_mask.all() else model_outputs

        result = morris_analyze.analyze(
            problem,
            parameter_samples[: len(Y)],
            Y,
            print_to_console=False,
        )

        mu_star = np.asarray(result["mu_star"], dtype=np.float64)
        sigma = np.asarray(result["sigma"], dtype=np.float64)
        mu_star_conf = np.asarray(result["mu_star_conf"], dtype=np.float64)

        summary = pd.DataFrame({
            "parameter": self.param_names,
            "mu_star": mu_star,
            "mu_star_conf": mu_star_conf,
            "sigma": sigma,
        })
        if self.param_groups is not None:
            summary["group"] = self.param_groups

        summary = summary.sort_values("mu_star", ascending=False).reset_index(drop=True)

        return {
            "mu_star": mu_star,
            "sigma": sigma,
            "mu_star_conf": mu_star_conf,
            "param_names": self.param_names,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_sobol_analysis(
        self,
        N: int = 1024,
        model_func: Optional[Callable[[np.ndarray], float]] = None,
        calc_second_order: bool = True,
        parallel: bool = True,
        n_jobs: int = -1,
    ) -> Dict[str, Any]:
        """Run end-to-end Sobol sensitivity analysis.

        Parameters
        ----------
        N : int
            Base sample size.  Default 1024.
        model_func : callable or None
        calc_second_order : bool
        parallel : bool
        n_jobs : int

        Returns
        -------
        dict
            ``{"samples": np.ndarray, "outputs": np.ndarray,
            "indices": dict}``.
        """
        samples = self.generate_sobol_samples(N, calc_second_order=calc_second_order)
        outputs = self.evaluate_model(
            samples, model_func=model_func, parallel=parallel, n_jobs=n_jobs
        )
        indices = self.analyze_sobol_indices(samples, outputs)
        return {"samples": samples, "outputs": outputs, "indices": indices}

    def run_morris_analysis(
        self,
        N: int = 100,
        num_levels: int = 4,
        model_func: Optional[Callable[[np.ndarray], float]] = None,
        parallel: bool = True,
        n_jobs: int = -1,
    ) -> Dict[str, Any]:
        """Run end-to-end Morris screening analysis.

        Parameters
        ----------
        N : int
            Number of trajectories.  Default 100.
        num_levels : int
            Grid levels.  Default 4.
        model_func : callable or None
        parallel : bool
        n_jobs : int

        Returns
        -------
        dict
            ``{"samples": np.ndarray, "outputs": np.ndarray,
            "indices": dict}``.
        """
        screening = self.evaluate_morris_screening(N, num_levels=num_levels)
        samples = screening["samples"]
        morris_prob = screening["morris_problem"]
        outputs = self.evaluate_model(
            samples, model_func=model_func, parallel=parallel, n_jobs=n_jobs
        )
        indices = self.analyze_morris(samples, outputs, morris_prob)
        return {"samples": samples, "outputs": outputs, "indices": indices}

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def plot_sensitivity_bars(
        indices: Dict[str, Any],
        title: str = "Sobol Sensitivity Indices",
        figsize: Tuple[float, float] = (10, 6),
        top_n: Optional[int] = None,
    ) -> Any:
        """Horizontal bar chart of first-order and total-order indices.

        Parameters
        ----------
        indices : dict
            Output from :meth:`analyze_sobol_indices`.
        title : str
        figsize : tuple of (float, float)
        top_n : int or None
            Limit to top-N parameters by ST.  ``None`` shows all.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        summary = indices.get("summary")
        if summary is None:
            names = indices["param_names"]
            s1 = np.asarray(indices["S1"])
            st = np.asarray(indices["ST"])
            s1_conf = np.asarray(indices["S1_conf"])
            st_conf = np.asarray(indices["ST_conf"])
            summary = pd.DataFrame({
                "parameter": names, "S1": s1, "ST": st,
                "S1_conf": s1_conf, "ST_conf": st_conf,
            }).sort_values("ST", ascending=True)

        if top_n is not None:
            summary = summary.tail(top_n)

        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(len(summary))
        height = 0.35

        ax.barh(y + height / 2, summary["ST"].values, height,
                xerr=summary["ST_conf"].values, label="Total-order (S_T)",
                color="#2196F3", edgecolor="white")
        ax.barh(y - height / 2, summary["S1"].values, height,
                xerr=summary["S1_conf"].values, label="First-order (S₁)",
                color="#4CAF50", edgecolor="white")

        ax.set_yticks(y)
        ax.set_yticklabels(summary["parameter"].values)
        ax.set_xlabel("Sensitivity Index")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.axvline(0, color="black", linewidth=0.5)

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_interaction_heatmap(
        indices: Dict[str, Any],
        title: str = "Second-Order Interaction Effects (S₂)",
        figsize: Tuple[float, float] = (8, 7),
    ) -> Any:
        """Heatmap of second-order Sobol interaction indices.

        Parameters
        ----------
        indices : dict
            Output from :meth:`analyze_sobol_indices`.
        title : str
        figsize : tuple of (float, float)

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        s2 = np.asarray(indices.get("S2", []))
        names = indices["param_names"]
        K = len(names)

        if s2.size == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No second-order indices available.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            return fig

        # Reconstruct full symmetric matrix from upper triangle
        matrix = np.zeros((K, K), dtype=np.float64)
        idx = 0
        for i in range(K):
            for j in range(i + 1, K):
                if idx < len(s2):
                    matrix[i, j] = s2[idx]
                    matrix[j, i] = s2[idx]
                    idx += 1

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)

        ax.set_xticks(np.arange(K))
        ax.set_yticks(np.arange(K))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        ax.set_title(title)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("S₂")

        # Annotate cells
        for i in range(K):
            for j in range(K):
                if i != j and matrix[i, j] > 0:
                    ax.text(j, i, f"{matrix[i, j]:.3f}",
                            ha="center", va="center", fontsize=7)

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_morris_bars(
        indices: Dict[str, Any],
        title: str = "Morris Screening (μ*)",
        figsize: Tuple[float, float] = (10, 6),
        top_n: Optional[int] = None,
    ) -> Any:
        """Horizontal bar chart of Morris mu_star values.

        Parameters
        ----------
        indices : dict
            Output from :meth:`analyze_morris`.
        title : str
        figsize : tuple of (float, float)
        top_n : int or None

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        summary = indices.get("summary")
        if summary is None:
            names = indices["param_names"]
            mu_star = np.asarray(indices["mu_star"])
            mu_star_conf = np.asarray(indices["mu_star_conf"])
            summary = pd.DataFrame({
                "parameter": names, "mu_star": mu_star,
                "mu_star_conf": mu_star_conf,
            }).sort_values("mu_star", ascending=True)

        if top_n is not None:
            summary = summary.tail(top_n)

        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(len(summary))

        ax.barh(y, summary["mu_star"].values,
                xerr=summary["mu_star_conf"].values,
                color="#FF9800", edgecolor="white")

        ax.set_yticks(y)
        ax.set_yticklabels(summary["parameter"].values)
        ax.set_xlabel("μ* (mean absolute elementary effect)")
        ax.set_title(title)
        ax.axvline(0, color="black", linewidth=0.5)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GlobalSensitivityAnalyzer(params={self.n_params}, "
            f"names={self.param_names[:3]}...)"
            if self.n_params > 3
            else f"GlobalSensitivityAnalyzer(params={self.n_params}, "
                 f"names={self.param_names})"
        )
