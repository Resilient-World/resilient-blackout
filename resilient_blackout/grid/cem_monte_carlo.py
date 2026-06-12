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
Accelerated reliability evaluation via Cross-Entropy Method (CEM).

Provides ``CEMMonteCarloSimulator``, which uses importance sampling
with dynamically optimised sampling distributions to accelerate the
estimation of Expected Energy Not Served (EENS) and Loss of Load
Probability (LOLP) for rare joint-outage events.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from resilient_blackout.grid.cascade import CascadingSimulator
from resilient_blackout.grid.network import GridModel

logger = logging.getLogger(__name__)

_DEFAULT_FOR_GEN: float = 0.05
_DEFAULT_FOR_LINE_PER_KM: float = 0.001
_DEFAULT_FOR_TRAFO: float = 0.01
_EPS: float = 1e-12


class CEMMonteCarloSimulator:
    """Cross-Entropy Method for accelerated reliability evaluation.

    Uses importance sampling to bias the sampling distribution toward
    rare joint-outage states that trigger load shedding, then corrects
    the bias via likelihood ratio weights.

    Parameters
    ----------
    grid_model : GridModel
        The grid model to evaluate.
    forced_outage_rates : dict or None
        Optional override for baseline FOR.  Keys: ``"gen"``,
        ``"line"``, ``"trafo"``.  Values are 1-D ``np.ndarray``
        matching the number of elements.  If ``None``, reads from
        pandapower reliability tables or uses IEEE defaults.
    cascade_tolerance : float
        Overload tolerance for the cascading simulator.  Default 1.2.
    cascade_max_iter : int
        Max cascade iterations.  Default 50.
    rho : float
        Elite sample quantile (0–1).  Default 0.1 (top 10%).
    smoothing_alpha : float
        Exponential smoothing factor for parameter updates (0–1).
        Default 0.7.

    Attributes
    ----------
    grid_model : GridModel
    p_gen : np.ndarray
    p_line : np.ndarray
    p_trafo : np.ndarray
    n_gen : int
    n_line : int
    n_trafo : int
    rho : float
    alpha : float
    """

    def __init__(
        self,
        grid_model: GridModel,
        forced_outage_rates: Optional[Dict[str, np.ndarray]] = None,
        cascade_tolerance: float = 1.2,
        cascade_max_iter: int = 50,
        rho: float = 0.1,
        smoothing_alpha: float = 0.7,
    ) -> None:
        if not 0 < rho <= 1:
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        if not 0 < smoothing_alpha <= 1:
            raise ValueError(f"smoothing_alpha must be in (0, 1], got {smoothing_alpha}")

        self.grid_model = grid_model
        self.cascade_tolerance = cascade_tolerance
        self.cascade_max_iter = cascade_max_iter
        self.rho = rho
        self.alpha = smoothing_alpha

        net = grid_model.net
        self.n_gen = len(net.gen)
        self.n_line = len(net.line)
        self.n_trafo = len(net.trafo)

        if forced_outage_rates is not None:
            self.p_gen = np.asarray(forced_outage_rates.get("gen", []), dtype=np.float64)
            self.p_line = np.asarray(forced_outage_rates.get("line", []), dtype=np.float64)
            self.p_trafo = np.asarray(forced_outage_rates.get("trafo", []), dtype=np.float64)
        else:
            self.p_gen, self.p_line, self.p_trafo = self._build_baseline_for(net)

        self.p_gen = np.clip(self.p_gen, _EPS, 1.0 - _EPS)
        self.p_line = np.clip(self.p_line, _EPS, 1.0 - _EPS)
        self.p_trafo = np.clip(self.p_trafo, _EPS, 1.0 - _EPS)

    # ------------------------------------------------------------------
    # Baseline FOR construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_baseline_for(net: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build baseline FOR vectors from pandapower or defaults.

        Parameters
        ----------
        net : pandapowerNet

        Returns
        -------
        tuple of (np.ndarray, np.ndarray, np.ndarray)
        """
        n_gen = len(net.gen)
        n_line = len(net.line)
        n_trafo = len(net.trafo)

        p_gen = np.full(n_gen, _DEFAULT_FOR_GEN, dtype=np.float64)
        if "reliability" in net.gen.columns:
            rel = net.gen.reliability.values
            mask = ~np.isnan(rel)
            p_gen[mask] = rel[mask]

        p_line = np.full(n_line, _DEFAULT_FOR_LINE_PER_KM, dtype=np.float64)
        if "length_km" in net.line.columns:
            p_line = p_line * net.line.length_km.values
        if "reliability" in net.line.columns:
            rel = net.line.reliability.values
            mask = ~np.isnan(rel)
            p_line[mask] = rel[mask]

        p_trafo = np.full(n_trafo, _DEFAULT_FOR_TRAFO, dtype=np.float64)
        if "reliability" in net.trafo.columns:
            rel = net.trafo.reliability.values
            mask = ~np.isnan(rel)
            p_trafo[mask] = rel[mask]

        return p_gen, p_line, p_trafo

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_state(
        p_gen: np.ndarray,
        p_line: np.ndarray,
        p_trafo: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample binary outage states from Bernoulli distributions.

        Parameters
        ----------
        p_gen : np.ndarray
        p_line : np.ndarray
        p_trafo : np.ndarray
        rng : np.random.Generator

        Returns
        -------
        tuple of (np.ndarray, np.ndarray, np.ndarray)
            Binary arrays (1 = failed, 0 = in-service).
        """
        gen_state = (rng.random(len(p_gen)) < p_gen).astype(np.int8)
        line_state = (rng.random(len(p_line)) < p_line).astype(np.int8)
        trafo_state = (rng.random(len(p_trafo)) < p_trafo).astype(np.int8)
        return gen_state, line_state, trafo_state

    def _sample_batch(
        self,
        p_gen: np.ndarray,
        p_line: np.ndarray,
        p_trafo: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of outage states.

        Parameters
        ----------
        p_gen, p_line, p_trafo : np.ndarray
        n_samples : int
        rng : np.random.Generator

        Returns
        -------
        tuple of (np.ndarray, np.ndarray, np.ndarray)
            Each of shape ``(n_samples, n_elements)``.
        """
        gen_batch = rng.random((n_samples, len(p_gen))) < p_gen
        line_batch = rng.random((n_samples, len(p_line))) < p_line
        trafo_batch = rng.random((n_samples, len(p_trafo))) < p_trafo
        return gen_batch.astype(np.int8), line_batch.astype(np.int8), trafo_batch.astype(np.int8)

    # ------------------------------------------------------------------
    # Likelihood ratio
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_likelihood_ratio(
        gen_state: np.ndarray,
        line_state: np.ndarray,
        trafo_state: np.ndarray,
        p_gen_base: np.ndarray,
        p_line_base: np.ndarray,
        p_trafo_base: np.ndarray,
        p_gen_biased: np.ndarray,
        p_line_biased: np.ndarray,
        p_trafo_biased: np.ndarray,
    ) -> float:
        """Compute likelihood ratio W = f(ω) / g(ω).

        Parameters
        ----------
        gen_state, line_state, trafo_state : np.ndarray
            Binary outage states (1-D per element type).
        p_*_base : np.ndarray
            Baseline FOR.
        p_*_biased : np.ndarray
            Importance sampling distribution.

        Returns
        -------
        float
            Likelihood ratio weight.
        """
        log_w = 0.0

        for state, p_base, p_biased in [
            (gen_state, p_gen_base, p_gen_biased),
            (line_state, p_line_base, p_line_biased),
            (trafo_state, p_trafo_base, p_trafo_biased),
        ]:
            p_b = np.clip(p_base, _EPS, 1.0 - _EPS)
            p_v = np.clip(p_biased, _EPS, 1.0 - _EPS)

            mask_fail = state == 1
            mask_ok = state == 0

            log_w += np.sum(np.log(p_b[mask_fail] / p_v[mask_fail]))
            log_w += np.sum(np.log((1.0 - p_b[mask_ok]) / (1.0 - p_v[mask_ok])))

        return float(np.exp(log_w))

    @staticmethod
    def _compute_likelihood_ratios_batch(
        gen_batch: np.ndarray,
        line_batch: np.ndarray,
        trafo_batch: np.ndarray,
        p_gen_base: np.ndarray,
        p_line_base: np.ndarray,
        p_trafo_base: np.ndarray,
        p_gen_biased: np.ndarray,
        p_line_biased: np.ndarray,
        p_trafo_biased: np.ndarray,
    ) -> np.ndarray:
        """Vectorized likelihood ratio for a batch.

        Parameters
        ----------
        *_batch : np.ndarray
            Shape ``(n_samples, n_elements)``.
        p_*_base, p_*_biased : np.ndarray

        Returns
        -------
        np.ndarray
            Likelihood ratios of shape ``(n_samples,)``.
        """
        log_w = np.zeros(len(gen_batch), dtype=np.float64)

        for batch, p_base, p_biased in [
            (gen_batch, p_gen_base, p_gen_biased),
            (line_batch, p_line_base, p_line_biased),
            (trafo_batch, p_trafo_base, p_trafo_biased),
        ]:
            p_b = np.clip(p_base, _EPS, 1.0 - _EPS)
            p_v = np.clip(p_biased, _EPS, 1.0 - _EPS)

            log_w += np.sum(
                batch * np.log(p_b / p_v) + (1 - batch) * np.log((1 - p_b) / (1 - p_v)),
                axis=1,
            )

        return np.exp(log_w)

    # ------------------------------------------------------------------
    # Cascade execution
    # ------------------------------------------------------------------

    def _run_cascade_for_state(
        self,
        gen_state: np.ndarray,
        line_state: np.ndarray,
        trafo_state: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        """Run cascading simulator for a single outage state.

        Parameters
        ----------
        gen_state, line_state, trafo_state : np.ndarray
            Binary outage states.
        rng : np.random.Generator

        Returns
        -------
        float
            Load shed in MW.
        """
        net = copy.deepcopy(self.grid_model.net)

        failed_assets: List[str] = []

        for i in range(len(gen_state)):
            if gen_state[i] and i in net.gen.index:
                net.gen.at[i, "in_service"] = False
                failed_assets.append(f"gen_{i}")

        for i in range(len(line_state)):
            if line_state[i] and i in net.line.index:
                net.line.at[i, "in_service"] = False
                failed_assets.append(f"line_{i}")

        for i in range(len(trafo_state)):
            if trafo_state[i] and i in net.trafo.index:
                net.trafo.at[i, "in_service"] = False
                failed_assets.append(f"trafo_{i}")

        if not failed_assets:
            return 0.0

        simulator = CascadingSimulator(
            grid_model=GridModel(net),
            tolerance_factor=self.cascade_tolerance,
            max_iterations=self.cascade_max_iter,
            rng=rng,
        )
        result = simulator.simulate_cascade(failed_assets)
        return float(result.get("total_load_shed_mw", 0.0))

    # ------------------------------------------------------------------
    # CEM main loop
    # ------------------------------------------------------------------

    def run_cem(
        self,
        n_iterations: int = 5,
        n_samples_per_iter: int = 1000,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute the Cross-Entropy Method optimisation loop.

        Parameters
        ----------
        n_iterations : int
            Number of CEM iterations.  Default 5.
        n_samples_per_iter : int
            Samples per iteration.  Default 1000.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        dict
            ``{"v_gen": np.ndarray, "v_line": np.ndarray,
            "v_trafo": np.ndarray, "gamma_history": list,
            "elite_fraction_history": list}``.
        """
        rng = np.random.default_rng(seed)

        v_gen = self.p_gen.copy()
        v_line = self.p_line.copy()
        v_trafo = self.p_trafo.copy()

        gamma_history: List[float] = []
        elite_frac_history: List[float] = []

        n_elite = max(1, int(n_samples_per_iter * self.rho))

        for iteration in range(n_iterations):
            gen_batch, line_batch, trafo_batch = self._sample_batch(
                v_gen, v_line, v_trafo, n_samples_per_iter, rng
            )

            weights = self._compute_likelihood_ratios_batch(
                gen_batch, line_batch, trafo_batch,
                self.p_gen, self.p_line, self.p_trafo,
                v_gen, v_line, v_trafo,
            )

            load_sheds = np.empty(n_samples_per_iter, dtype=np.float64)
            for i in range(n_samples_per_iter):
                load_sheds[i] = self._run_cascade_for_state(
                    gen_batch[i], line_batch[i], trafo_batch[i], rng
                )

            weighted_sheds = load_sheds * weights

            sorted_idx = np.argsort(weighted_sheds)[::-1]
            elite_idx = sorted_idx[:n_elite]

            gamma = float(weighted_sheds[elite_idx[-1]])
            gamma_history.append(gamma)

            elite_gen = gen_batch[elite_idx].astype(np.float64)
            elite_line = line_batch[elite_idx].astype(np.float64)
            elite_trafo = trafo_batch[elite_idx].astype(np.float64)

            elite_frac = float(len(elite_idx)) / n_samples_per_iter
            elite_frac_history.append(elite_frac)

            mean_gen = np.mean(elite_gen, axis=0)
            mean_line = np.mean(elite_line, axis=0)
            mean_trafo = np.mean(elite_trafo, axis=0)

            v_gen = self.alpha * mean_gen + (1.0 - self.alpha) * v_gen
            v_line = self.alpha * mean_line + (1.0 - self.alpha) * v_line
            v_trafo = self.alpha * mean_trafo + (1.0 - self.alpha) * v_trafo

            v_gen = np.clip(v_gen, _EPS, 1.0 - _EPS)
            v_line = np.clip(v_line, _EPS, 1.0 - _EPS)
            v_trafo = np.clip(v_trafo, _EPS, 1.0 - _EPS)

            logger.info(
                "CEM iter %d/%d — gamma=%.2f, elite_frac=%.3f",
                iteration + 1, n_iterations, gamma, elite_frac,
            )

        return {
            "v_gen": v_gen,
            "v_line": v_line,
            "v_trafo": v_trafo,
            "gamma_history": gamma_history,
            "elite_fraction_history": elite_frac_history,
        }

    # ------------------------------------------------------------------
    # Final estimation
    # ------------------------------------------------------------------

    def estimate_eens_lolp(
        self,
        v_gen: np.ndarray,
        v_line: np.ndarray,
        v_trafo: np.ndarray,
        n_samples: int = 10000,
        recovery_hours: float = 24.0,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Estimate EENS and LOLP using the optimised importance distribution.

        Parameters
        ----------
        v_gen, v_line, v_trafo : np.ndarray
            Optimised importance sampling distributions.
        n_samples : int
            Number of estimation samples.  Default 10000.
        recovery_hours : float
            Recovery duration in hours.  Default 24.
        seed : int or None

        Returns
        -------
        dict
            ``{"eens_mwh": float, "lolp": float, "variance": float,
            "std_error": float, "n_samples": int}``.
        """
        rng = np.random.default_rng(seed)

        gen_batch, line_batch, trafo_batch = self._sample_batch(
            v_gen, v_line, v_trafo, n_samples, rng
        )

        weights = self._compute_likelihood_ratios_batch(
            gen_batch, line_batch, trafo_batch,
            self.p_gen, self.p_line, self.p_trafo,
            v_gen, v_line, v_trafo,
        )

        load_sheds = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            load_sheds[i] = self._run_cascade_for_state(
                gen_batch[i], line_batch[i], trafo_batch[i], rng
            )

        weighted_sheds = load_sheds * weights
        weighted_lolp = (load_sheds > 0).astype(np.float64) * weights

        eens = float(np.mean(weighted_sheds) * recovery_hours)
        lolp = float(np.mean(weighted_lolp))

        var_eens = float(np.var(weighted_sheds * recovery_hours) / n_samples)
        std_error = float(np.sqrt(var_eens))

        return {
            "eens_mwh": eens,
            "lolp": lolp,
            "variance": var_eens,
            "std_error": std_error,
            "n_samples": n_samples,
        }

    # ------------------------------------------------------------------
    # Comparison with standard Monte Carlo
    # ------------------------------------------------------------------

    def compare_with_standard_mc(
        self,
        n_samples: int = 10000,
        recovery_hours: float = 24.0,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compare CEM estimator against standard Monte Carlo.

        Parameters
        ----------
        n_samples : int
            Samples for each method.  Default 10000.
        recovery_hours : float
            Recovery duration in hours.  Default 24.
        seed : int or None

        Returns
        -------
        dict
            ``{"cem_eens_mwh": float, "mc_eens_mwh": float,
            "cem_variance": float, "mc_variance": float,
            "variance_reduction_factor": float}``.
        """
        rng = np.random.default_rng(seed)

        cem_result = self.run_cem(n_iterations=5, n_samples_per_iter=1000, seed=seed)
        cem_est = self.estimate_eens_lolp(
            cem_result["v_gen"], cem_result["v_line"], cem_result["v_trafo"],
            n_samples=n_samples, recovery_hours=recovery_hours, seed=seed,
        )

        gen_batch, line_batch, trafo_batch = self._sample_batch(
            self.p_gen, self.p_line, self.p_trafo, n_samples, rng
        )

        mc_sheds = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            mc_sheds[i] = self._run_cascade_for_state(
                gen_batch[i], line_batch[i], trafo_batch[i], rng
            )

        mc_eens = float(np.mean(mc_sheds) * recovery_hours)
        mc_var = float(np.var(mc_sheds * recovery_hours) / n_samples)

        vrf = mc_var / max(cem_est["variance"], _EPS)

        return {
            "cem_eens_mwh": cem_est["eens_mwh"],
            "mc_eens_mwh": mc_eens,
            "cem_variance": cem_est["variance"],
            "mc_variance": mc_var,
            "variance_reduction_factor": vrf,
        }
