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
Multi-hazard compound risk engine.

Provides ``CompoundHazardEngine`` for modelling the joint probability
and cascading interaction of co-occurring or sequential physical
hazards, including bivariate copula models, conditional hazard
modifiers, and temporal compound evaluation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit

logger = logging.getLogger(__name__)

CopulaType = Literal["gumbel", "clayton", "auto"]
ModifierType = Literal["wind_damage", "flood_weakening", "thermal_stress"]

_EPS: float = 1e-12

_DEFAULT_IGNITION_BETA: np.ndarray = np.array([-4.0, 0.08, 2.5], dtype=np.float64)


# ---------------------------------------------------------------------------
# Copula utilities
# ---------------------------------------------------------------------------

def _empirical_cdf(x: np.ndarray) -> np.ndarray:
    """Transform data to uniform margins via empirical CDF.

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    np.ndarray
        Values in (0, 1).
    """
    n = len(x)
    ranks = stats.rankdata(x, method="average")
    return np.clip(ranks / (n + 1), _EPS, 1.0 - _EPS)


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Kendall's τ correlation coefficient.

    Parameters
    ----------
    x, y : np.ndarray

    Returns
    -------
    float
    """
    tau, _ = stats.kendalltau(x, y)
    return float(tau if tau is not None else 0.0)


def _gumbel_copula_pdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Gumbel copula density.

    Parameters
    ----------
    u, v : np.ndarray
        Uniform marginals in (0, 1).
    theta : float
        Copula parameter (≥ 1).

    Returns
    -------
    np.ndarray
    """
    t = theta
    neg_log_u = (-np.log(u)) ** t
    neg_log_v = (-np.log(v)) ** t
    s = neg_log_u + neg_log_v
    cdf = np.exp(-(s ** (1.0 / t)))
    term1 = cdf * (s ** (2.0 / t - 2.0))
    term2 = (t - 1.0) * (s ** (1.0 / t - 2.0))
    term3 = (neg_log_u * neg_log_v) / (u * v * ((-np.log(u)) * (-np.log(v))))
    return term1 * (term2 + s ** (1.0 / t)) * term3


def _clayton_copula_pdf(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Clayton copula density.

    Parameters
    ----------
    u, v : np.ndarray
    theta : float
        Copula parameter (> 0).

    Returns
    -------
    np.ndarray
    """
    t = theta
    u_t = u ** (-t)
    v_t = v ** (-t)
    s = u_t + v_t - 1.0
    cdf = np.maximum(s, _EPS) ** (-1.0 / t - 2.0)
    return (1.0 + t) * (u * v) ** (-t - 1.0) * cdf


def _gumbel_theta_from_tau(tau: float) -> float:
    """Gumbel θ from Kendall's τ: θ = 1 / (1 - τ).

    Parameters
    ----------
    tau : float

    Returns
    -------
    float
    """
    tau_clipped = max(_EPS, min(1.0 - _EPS, tau))
    return 1.0 / (1.0 - tau_clipped)


def _clayton_theta_from_tau(tau: float) -> float:
    """Clayton θ from Kendall's τ: θ = 2τ / (1 - τ).

    Parameters
    ----------
    tau : float

    Returns
    -------
    float
    """
    tau_clipped = max(_EPS, min(1.0 - _EPS, tau))
    return 2.0 * tau_clipped / (1.0 - tau_clipped)


def _gumbel_sample(n: int, theta: float, rng: np.random.Generator) -> np.ndarray:
    """Sample from Gumbel copula.

    Parameters
    ----------
    n : int
    theta : float
    rng : np.random.Generator

    Returns
    -------
    np.ndarray
        Shape ``(n, 2)`` in (0, 1).
    """
    t = theta
    gamma = rng.standard_exponential(n)
    u1 = rng.random(n)
    u2 = rng.random(n)

    w = np.zeros(n, dtype=np.float64)
    for _ in range(20):
        w_new = u2 ** (u2 * (1.0 - 1.0 / t))
        if np.allclose(w, w_new):
            break
        w = w_new

    v = np.column_stack([u1, w])
    return v


def _clayton_sample(n: int, theta: float, rng: np.random.Generator) -> np.ndarray:
    """Sample from Clayton copula.

    Parameters
    ----------
    n : int
    theta : float
    rng : np.random.Generator

    Returns
    -------
    np.ndarray
        Shape ``(n, 2)`` in (0, 1).
    """
    t = theta
    gamma = rng.gamma(shape=1.0 / t, scale=1.0, size=n)
    u1 = rng.random(n)
    u2 = rng.random(n)

    v1 = (1.0 - np.log(u1) / gamma) ** (-1.0 / t)
    v2 = (1.0 - np.log(u2) / gamma) ** (-1.0 / t)
    return np.column_stack([v1, v2])


def _empirical_tail_dependence(
    u: np.ndarray, v: np.ndarray, q: float = 0.95
) -> Tuple[float, float]:
    """Estimate upper and lower tail dependence coefficients.

    Parameters
    ----------
    u, v : np.ndarray
    q : float
        Quantile threshold.

    Returns
    -------
    tuple of (float, float)
        ``(lambda_lower, lambda_upper)``.
    """
    n = len(u)
    t_low = np.mean((u < (1 - q)) & (v < (1 - q)))
    t_up = np.mean((u > q) & (v > q))
    lambda_lower = t_low / (1 - q) if q < 1 else 0.0
    lambda_upper = t_up / (1 - q) if q < 1 else 0.0
    return float(lambda_lower), float(lambda_upper)


# ---------------------------------------------------------------------------
# CompoundHazardEngine
# ---------------------------------------------------------------------------

class CompoundHazardEngine:
    """Multi-hazard compound risk modelling engine.

    Models joint probability of co-occurring hazards via bivariate
    copulas, applies conditional hazard modifiers to vulnerability
    curves, and evaluates temporal compounding of sequential events.

    Parameters
    ----------
    copula_type : str
        ``"gumbel"``, ``"clayton"``, or ``"auto"``.  Default
        ``"auto"`` selects based on empirical tail dependence.

    Attributes
    ----------
    copula_type : str
    fitted_copula : dict or None
    """

    def __init__(self, copula_type: CopulaType = "auto") -> None:
        if copula_type not in {"gumbel", "clayton", "auto"}:
            raise ValueError(
                f"copula_type must be 'gumbel', 'clayton', or 'auto', got '{copula_type}'"
            )
        self.copula_type: CopulaType = copula_type
        self.fitted_copula: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Copula fitting
    # ------------------------------------------------------------------

    def fit_copula(
        self,
        hazard_a: np.ndarray,
        hazard_b: np.ndarray,
    ) -> Dict[str, Any]:
        """Fit bivariate copula to two hazard intensity series.

        Parameters
        ----------
        hazard_a, hazard_b : np.ndarray
            Hazard intensity values.

        Returns
        -------
        dict
            ``{"copula_type": str, "theta": float, "tau": float,
            "marginals": dict}``.
        """
        a = np.asarray(hazard_a, dtype=np.float64).ravel()
        b = np.asarray(hazard_b, dtype=np.float64).ravel()

        if len(a) != len(b):
            raise ValueError("hazard_a and hazard_b must have the same length")
        if len(a) < 10:
            raise ValueError("At least 10 samples required for copula fitting")

        u = _empirical_cdf(a)
        v = _empirical_cdf(b)

        tau = _kendall_tau(a, b)

        if self.copula_type == "auto":
            lambda_low, lambda_up = _empirical_tail_dependence(u, v)
            selected = "gumbel" if lambda_up > lambda_low else "clayton"
        else:
            selected = self.copula_type

        if selected == "gumbel":
            theta = _gumbel_theta_from_tau(tau)
        else:
            theta = _clayton_theta_from_tau(tau)

        self.fitted_copula = {
            "copula_type": selected,
            "theta": theta,
            "tau": tau,
            "marginals": {
                "a_sorted": np.sort(a),
                "b_sorted": np.sort(b),
                "n": len(a),
            },
        }

        logger.info(
            "Fitted %s copula: theta=%.3f, tau=%.3f", selected, theta, tau
        )
        return self.fitted_copula

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_joint(
        self,
        n_samples: int,
        copula_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample from the fitted copula.

        Parameters
        ----------
        n_samples : int
        copula_params : dict or None
            Uses ``self.fitted_copula`` if ``None``.
        seed : int or None

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            Samples on the original hazard scales.
        """
        params = copula_params or self.fitted_copula
        if params is None:
            raise RuntimeError("No copula fitted. Call fit_copula() first.")

        rng = np.random.default_rng(seed)
        ctype = params["copula_type"]
        theta = params["theta"]

        if ctype == "gumbel":
            uv = _gumbel_sample(n_samples, theta, rng)
        else:
            uv = _clayton_sample(n_samples, theta, rng)

        marginals = params["marginals"]
        a_sorted = marginals["a_sorted"]
        b_sorted = marginals["b_sorted"]
        n_marg = marginals["n"]

        idx_a = np.clip((uv[:, 0] * n_marg).astype(int), 0, n_marg - 1)
        idx_b = np.clip((uv[:, 1] * n_marg).astype(int), 0, n_marg - 1)

        samples_a = a_sorted[idx_a]
        samples_b = b_sorted[idx_b]

        return samples_a, samples_b

    # ------------------------------------------------------------------
    # Joint exceedance
    # ------------------------------------------------------------------

    def joint_exceedance_probability(
        self,
        x_thresh: float,
        y_thresh: float,
        copula_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute P(X > x, Y > y) under the fitted copula.

        Parameters
        ----------
        x_thresh, y_thresh : float
        copula_params : dict or None

        Returns
        -------
        float
        """
        params = copula_params or self.fitted_copula
        if params is None:
            raise RuntimeError("No copula fitted. Call fit_copula() first.")

        marginals = params["marginals"]
        a_sorted = marginals["a_sorted"]
        b_sorted = marginals["b_sorted"]

        u_x = float(np.searchsorted(a_sorted, x_thresh) / len(a_sorted))
        u_y = float(np.searchsorted(b_sorted, y_thresh) / len(b_sorted))

        u_x = max(_EPS, min(1.0 - _EPS, u_x))
        u_y = max(_EPS, min(1.0 - _EPS, u_y))

        ctype = params["copula_type"]
        theta = params["theta"]

        if ctype == "gumbel":
            cdf = np.exp(-((-np.log(u_x)) ** theta + (-np.log(u_y)) ** theta) ** (1.0 / theta))
        else:
            cdf = max(u_x ** (-theta) + u_y ** (-theta) - 1.0, _EPS) ** (-1.0 / theta)

        return float(1.0 - u_x - u_y + cdf)

    # ------------------------------------------------------------------
    # Conditional ignition probability
    # ------------------------------------------------------------------

    @staticmethod
    def conditional_ignition_probability(
        wind_speed_ms: np.ndarray,
        dryness_index: np.ndarray,
        beta: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Logistic model for wildfire ignition probability.

        .. math::

            P(\\text{Ignition} \\mid v, d) =
            \\frac{1}{1 + \\exp(-(\\beta_0 + \\beta_1 v + \\beta_2 d))}

        Parameters
        ----------
        wind_speed_ms : np.ndarray
            Wind speed in m/s.
        dryness_index : np.ndarray
            Dryness index (0–1 or higher).
        beta : np.ndarray or None
            ``[β₀, β₁, β₂]`` coefficients.  Default from literature.

        Returns
        -------
        np.ndarray
            Ignition probabilities in [0, 1].
        """
        b = beta if beta is not None else _DEFAULT_IGNITION_BETA
        v = np.asarray(wind_speed_ms, dtype=np.float64)
        d = np.asarray(dryness_index, dtype=np.float64)
        logit = b[0] + b[1] * v + b[2] * d
        return expit(logit)

    # ------------------------------------------------------------------
    # Vulnerability modification
    # ------------------------------------------------------------------

    @staticmethod
    def modify_vulnerability(
        base_fragility_params: Dict[str, float],
        primary_intensity: float,
        modifier_type: ModifierType = "wind_damage",
    ) -> Dict[str, float]:
        """Adjust fragility curve parameters based on primary hazard.

        Parameters
        ----------
        base_fragility_params : dict
            Base fragility parameters (e.g., ``{"mean": 30.0, "std": 5.0}``).
        primary_intensity : float
            Intensity of the primary hazard.
        modifier_type : str
            ``"wind_damage"``, ``"flood_weakening"``, or
            ``"thermal_stress"``.

        Returns
        -------
        dict
            Modified fragility parameters.
        """
        modified = dict(base_fragility_params)

        if modifier_type == "wind_damage":
            factor = 1.0 + 0.02 * max(0.0, primary_intensity - 20.0)
            modified["mean"] = base_fragility_params.get("mean", 30.0) / factor
            modified["std"] = base_fragility_params.get("std", 5.0) * factor

        elif modifier_type == "flood_weakening":
            factor = 1.0 + 0.5 * max(0.0, primary_intensity - 0.5)
            modified["mean"] = base_fragility_params.get("mean", 30.0) / factor
            modified["std"] = base_fragility_params.get("std", 5.0) * factor

        elif modifier_type == "thermal_stress":
            factor = 1.0 + 0.01 * max(0.0, primary_intensity - 35.0)
            modified["mean"] = base_fragility_params.get("mean", 30.0) / factor
            modified["std"] = base_fragility_params.get("std", 5.0) * factor

        return modified

    # ------------------------------------------------------------------
    # Temporal compound evaluation
    # ------------------------------------------------------------------

    def evaluate_temporal_compound(
        self,
        primary_event: gpd.GeoDataFrame,
        secondary_event: gpd.GeoDataFrame,
        window_days: float = 14.0,
        degradation_lambda: float = 0.1,
    ) -> gpd.GeoDataFrame:
        """Model sequential hazard compounding over a time window.

        Applies exponential degradation to asset resistance when a
        secondary event follows a primary event within the specified
        window.

        Parameters
        ----------
        primary_event : GeoDataFrame
            Primary hazard with ``geometry`` and ``intensity`` columns.
        secondary_event : GeoDataFrame
            Secondary hazard.
        window_days : float
            Maximum separation in days.  Default 14.
        degradation_lambda : float
            Degradation rate.  Default 0.1.

        Returns
        -------
        GeoDataFrame
            Compound intensity field with ``geometry`` and
            ``compound_intensity`` columns.
        """
        primary = primary_event.copy()
        secondary = secondary_event.copy()

        primary["time_delta"] = 0.0
        secondary["time_delta"] = window_days * 0.5

        combined = pd.concat([primary, secondary], ignore_index=True)

        if "intensity" not in combined.columns:
            raise ValueError("Input GeoDataFrames must have an 'intensity' column")

        degradation = np.exp(-degradation_lambda * combined["time_delta"] / window_days)
        combined["compound_intensity"] = combined["intensity"] * degradation

        result = combined[["geometry", "compound_intensity"]].copy()
        return gpd.GeoDataFrame(result, geometry="geometry", crs=primary.crs)

    # ------------------------------------------------------------------
    # Hazard layer merging
    # ------------------------------------------------------------------

    @staticmethod
    def merge_hazard_layers(
        layers: List[gpd.GeoDataFrame],
        weights: Optional[List[float]] = None,
        max_distance: float = 1000.0,
    ) -> gpd.GeoDataFrame:
        """Merge multiple hazard intensity fields into a unified layer.

        Performs spatial join to find coincident hazard intensities
        and combines them via weighted averaging.

        Parameters
        ----------
        layers : list of GeoDataFrame
            Each must have ``geometry`` and ``intensity`` columns.
        weights : list of float or None
            Relative weight per layer.  Defaults to equal weights.
        max_distance : float
            Maximum distance in metres for spatial join.  Default 1000.

        Returns
        -------
        GeoDataFrame
            Merged layer with ``geometry`` and ``compound_intensity``.
        """
        if not layers:
            raise ValueError("At least one hazard layer required")

        if weights is None:
            weights = [1.0] * len(layers)
        if len(weights) != len(layers):
            raise ValueError("weights must match the number of layers")

        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()

        crs = layers[0].crs

        merged = layers[0][["geometry", "intensity"]].copy()
        merged["weighted"] = merged["intensity"] * w[0]
        merged["w_sum"] = w[0]

        for i, layer in enumerate(layers[1:], start=1):
            layer_proj = layer.to_crs(crs) if layer.crs != crs else layer
            joined = gpd.sjoin_nearest(
                merged,
                layer_proj[["geometry", "intensity"]],
                how="left",
                max_distance=max_distance,
                distance_col="dist",
            )
            joined["intensity_right"] = joined["intensity_right"].fillna(0.0)
            joined["weighted"] += joined["intensity_right"] * w[i]
            joined["w_sum"] += w[i]
            merged = joined.drop(columns=["index_right", "dist", "intensity_right"])

        merged["compound_intensity"] = merged["weighted"] / merged["w_sum"]
        result = merged[["geometry", "compound_intensity"]].copy()
        return gpd.GeoDataFrame(result, geometry="geometry", crs=crs)
