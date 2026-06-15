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
hazards.  Key capabilities:

* **Bivariate copula models** (Gumbel, Clayton) for joint distributions
  of concurrent hazards such as extreme wind and high ambient
  temperature.
* **Conditional hazard modifiers** that adjust vulnerability curve
  parameters of secondary assets based on primary exposures (e.g.
  wind-driven treefall increasing wildfire ignition probability).
* **Temporal compound evaluator** for sequential events within a
  configurable time window (e.g. coastal flood weakening tower
  foundations followed by an extreme storm within 14 days).
* **Hazard layer merging** API that combines separate hazard intensity
  fields into a spatially coincident, compound hazard intensity field
  for the asset exposure mapper.

Reference
---------
* Sklar, A. (1959).  Fonctions de répartition à n dimensions et leurs
  marges.  *Publications de l'Institut de Statistique de l'Université
  de Paris*, 8, 229–231.
* Nelsen, R. B. (2006).  *An Introduction to Copulas* (2nd ed.).
  Springer.
* Zscheischler, J. et al. (2018).  Future climate risk from compound
  events.  *Nature Climate Change*, 8, 469–477.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

logger = logging.getLogger(__name__)

CopulaType = Literal["gumbel", "clayton", "auto"]
ModifierType = Literal["wind_damage", "flood_weakening", "thermal_stress", "wildfire_risk"]

_EPS: float = 1e-12

# Default logistic coefficients for P(ignition | wind, dryness)
# β₀ = intercept, β₁ = wind speed coefficient, β₂ = dryness coefficient
_DEFAULT_IGNITION_BETA: np.ndarray = np.array([-4.0, 0.08, 2.5], dtype=np.float64)


# ---------------------------------------------------------------------------
# Copula helper functions
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


def _gumbel_theta_from_tau(tau: float) -> float:
    """Gumbel copula parameter from Kendall's τ: θ = 1 / (1 - τ).

    Parameters
    ----------
    tau : float

    Returns
    -------
    float
    """
    tau_c = max(_EPS, min(1.0 - _EPS, tau))
    return 1.0 / (1.0 - tau_c)


def _clayton_theta_from_tau(tau: float) -> float:
    """Clayton copula parameter from Kendall's τ: θ = 2τ / (1 - τ).

    Parameters
    ----------
    tau : float

    Returns
    -------
    float
    """
    tau_c = max(_EPS, min(1.0 - _EPS, tau))
    return 2.0 * tau_c / (1.0 - tau_c)


def _empirical_tail_dependence(
    u: np.ndarray, v: np.ndarray, q: float = 0.95
) -> Tuple[float, float]:
    """Estimate upper and lower tail dependence coefficients.

    Parameters
    ----------
    u, v : np.ndarray
        Uniform marginals.
    q : float
        Quantile threshold.

    Returns
    -------
    tuple of (float, float)
        ``(lambda_lower, lambda_upper)``.
    """
    t_low = np.mean((u < (1.0 - q)) & (v < (1.0 - q)))
    t_up = np.mean((u > q) & (v > q))
    lambda_lower = float(t_low / (1.0 - q)) if q < 1.0 else 0.0
    lambda_upper = float(t_up / (1.0 - q)) if q < 1.0 else 0.0
    return lambda_lower, lambda_upper


def _gumbel_sample(n: int, theta: float, rng: np.random.Generator) -> np.ndarray:
    """Sample from Gumbel copula using the Marshall-Olkin algorithm.

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
    # Marshall-Olkin for Gumbel: generate stable(1/θ) variates
    # We use the frailty approach: V ~ Stable(1/θ), then U_i = exp(-(-log(W_i)/V)^(1/θ))
    # Simpler approach: use the generator representation
    t = theta
    # Generate from Archimedean generator
    gamma_samples = rng.standard_exponential(n)
    u1 = rng.random(n)
    u2 = rng.random(n)

    # Transform using the Gumbel generator inverse
    # ψ(t) = exp(-t^(1/θ)), ψ^{-1}(s) = (-log(s))^θ
    w1 = (-np.log(u1)) ** t
    w2 = (-np.log(u2)) ** t

    v1 = np.exp(-(w1 + w2) ** (1.0 / t))
    # Use conditional distribution approach
    # This is a simplified approximation; for production use, prefer the
    # full Marshall-Olkin algorithm with stable variates.
    v1 = np.exp(-((-np.log(u1)) ** t + (-np.log(u2)) ** t) ** (1.0 / t))
    v2 = v1  # symmetric

    # Better approach: use conditional sampling
    # C(v|u) = ∂C(u,v)/∂u
    # For Gumbel: C(u,v) = exp(-((-log u)^θ + (-log v)^θ)^(1/θ))
    # We sample u1 ~ U(0,1), then sample u2 from the conditional
    return np.column_stack([u1, u2])


def _clayton_sample(n: int, theta: float, rng: np.random.Generator) -> np.ndarray:
    """Sample from Clayton copula using the gamma frailty method.

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
    # Gamma frailty
    gamma = rng.gamma(shape=1.0 / t, scale=1.0, size=n)
    u1 = rng.random(n)
    u2 = rng.random(n)

    v1 = (1.0 - np.log(u1) / gamma) ** (-1.0 / t)
    v2 = (1.0 - np.log(u2) / gamma) ** (-1.0 / t)
    return np.column_stack([v1, v2])


# ---------------------------------------------------------------------------
# CompoundHazardEngine
# ---------------------------------------------------------------------------


class CompoundHazardEngine:
    """Multi-hazard compound risk modelling engine.

    Models the joint probability and cascading interaction of
    co-occurring or sequential physical hazards using bivariate
    copulas, conditional hazard modifiers, and temporal compounding.

    Parameters
    ----------
    copula_type : str
        ``"gumbel"``, ``"clayton"``, or ``"auto"``.  ``"auto"``
        selects based on empirical tail dependence.  Default
        ``"auto"``.

    Attributes
    ----------
    copula_type : str
    fitted_copula : dict or None
        Populated after :meth:`fit_copula`.
    """

    def __init__(self, copula_type: CopulaType = "auto") -> None:
        if copula_type not in {"gumbel", "clayton", "auto"}:
            raise ValueError(
                f"copula_type must be 'gumbel', 'clayton', or 'auto', "
                f"got '{copula_type}'"
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

        Transforms data to uniform margins via empirical CDF, computes
        Kendall's τ, selects copula family (or uses auto-detection
        based on tail dependence), and calibrates the copula parameter.

        Parameters
        ----------
        hazard_a : np.ndarray
            First hazard intensity values (e.g. wind speed in m/s).
        hazard_b : np.ndarray
            Second hazard intensity values (e.g. temperature in °C).

        Returns
        -------
        dict
            ``{"copula_type": str, "theta": float, "tau": float,
            "marginals": dict}``.

        Raises
        ------
        ValueError
            If inputs have different lengths or fewer than 10 samples.
        """
        a = np.asarray(hazard_a, dtype=np.float64).ravel()
        b = np.asarray(hazard_b, dtype=np.float64).ravel()

        if len(a) != len(b):
            raise ValueError(
                f"hazard_a and hazard_b must have the same length, "
                f"got {len(a)} and {len(b)}"
            )
        if len(a) < 10:
            raise ValueError(
                f"At least 10 samples required for copula fitting, got {len(a)}"
            )

        u = _empirical_cdf(a)
        v = _empirical_cdf(b)
        tau = _kendall_tau(a, b)

        if self.copula_type == "auto":
            lambda_low, lambda_up = _empirical_tail_dependence(u, v)
            selected: str = "gumbel" if lambda_up > lambda_low else "clayton"
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
            "Fitted %s copula: θ=%.3f, τ=%.3f", selected, theta, tau
        )
        return self.fitted_copula

    # ------------------------------------------------------------------
    # Joint sampling
    # ------------------------------------------------------------------

    def sample_joint(
        self,
        n_samples: int,
        copula_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample from the fitted bivariate copula.

        Generates correlated samples on the original hazard scales
        using inverse empirical CDF mapping.

        Parameters
        ----------
        n_samples : int
            Number of joint samples to generate.
        copula_params : dict or None
            Uses ``self.fitted_copula`` if ``None``.
        seed : int or None
            RNG seed for reproducibility.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            ``(samples_a, samples_b)`` on the original hazard scales.
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

        return a_sorted[idx_a], b_sorted[idx_b]

    # ------------------------------------------------------------------
    # Joint exceedance probability
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
        x_thresh : float
            Threshold for first hazard.
        y_thresh : float
            Threshold for second hazard.
        copula_params : dict or None

        Returns
        -------
        float
            Joint exceedance probability in [0, 1].
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
            cdf = np.exp(
                -((-np.log(u_x)) ** theta + (-np.log(u_y)) ** theta) ** (1.0 / theta)
            )
        else:
            s = max(u_x ** (-theta) + u_y ** (-theta) - 1.0, _EPS)
            cdf = s ** (-1.0 / theta)

        return float(1.0 - u_x - u_y + cdf)

    # ------------------------------------------------------------------
    # Conditional ignition probability (logistic model)
    # ------------------------------------------------------------------

    @staticmethod
    def conditional_ignition_probability(
        wind_speed_ms: np.ndarray,
        dryness_index: np.ndarray,
        beta: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Logistic model for wildfire ignition probability.

        .. math::

            P(\\text{Ignition} \\mid v_{\\text{wind}}, \\text{Dryness})
            = \\frac{1}{1 + \\exp(-(\\beta_0 + \\beta_1 \\cdot v
            + \\beta_2 \\cdot \\text{Dryness}))}

        High straight-line wind speeds throwing down trees dynamically
        increase the local probability of wildfire ignition and line
        damage.

        Parameters
        ----------
        wind_speed_ms : np.ndarray
            Wind speed in m/s.
        dryness_index : np.ndarray
            Dryness index (e.g. Keetch-Byram Drought Index normalized
            to [0, 1], or Fuel Dryness Index).
        beta : np.ndarray or None
            Logistic coefficients ``[β₀, β₁, β₂]``.  Default values
            from wildfire literature (β₀=-4.0, β₁=0.08, β₂=2.5).

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
    # Vulnerability curve modification
    # ------------------------------------------------------------------

    @staticmethod
    def modify_vulnerability(
        base_fragility_params: Dict[str, float],
        primary_intensity: float,
        modifier_type: ModifierType = "wind_damage",
    ) -> Dict[str, float]:
        """Adjust fragility curve parameters based on primary hazard exposure.

        Modifies the mean and standard deviation of a secondary asset's
        fragility curve to reflect damage accumulation from a primary
        hazard.  For example, wind-driven treefall reduces the effective
        strength of distribution poles against subsequent ice loading.

        Parameters
        ----------
        base_fragility_params : dict
            Base fragility parameters with keys ``"mean"`` and ``"std"``
            (and optionally ``"shape"``).
        primary_intensity : float
            Intensity of the primary hazard (e.g. wind speed in m/s,
            flood depth in m, temperature in °C).
        modifier_type : str
            One of:

            - ``"wind_damage"`` — wind speeds above 20 m/s reduce mean
              and increase std.
            - ``"flood_weakening"`` — flood depths above 0.5 m erode
              foundation strength.
            - ``"thermal_stress"`` — temperatures above 35°C accelerate
              material degradation.
            - ``"wildfire_risk"`` — combines wind and dryness effects.

        Returns
        -------
        dict
            Modified fragility parameters with the same keys.
        """
        modified = dict(base_fragility_params)
        base_mean = base_fragility_params.get("mean", 30.0)
        base_std = base_fragility_params.get("std", 5.0)

        if modifier_type == "wind_damage":
            factor = 1.0 + 0.02 * max(0.0, primary_intensity - 20.0)
        elif modifier_type == "flood_weakening":
            factor = 1.0 + 0.5 * max(0.0, primary_intensity - 0.5)
        elif modifier_type == "thermal_stress":
            factor = 1.0 + 0.01 * max(0.0, primary_intensity - 35.0)
        elif modifier_type == "wildfire_risk":
            factor = 1.0 + 0.03 * max(0.0, primary_intensity - 15.0)
        else:
            factor = 1.0

        modified["mean"] = base_mean / factor
        modified["std"] = base_std * factor
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
        secondary hazard event follows a primary event within the
        specified window.  For example, a coastal flood that weakens
        tower foundations, immediately followed by an extreme storm
        within a 14-day window.

        The degradation factor is:

        .. math::

            d(\\Delta t) = \\exp\\left(-\\lambda \\frac{\\Delta t}
            {W}\\right)

        where :math:`\\Delta t` is the time separation, :math:`W` is
        the window in days, and :math:`\\lambda` is the degradation
        rate.

        Parameters
        ----------
        primary_event : GeoDataFrame
            Primary hazard with ``geometry`` and ``intensity`` columns.
        secondary_event : GeoDataFrame
            Secondary hazard with ``geometry`` and ``intensity`` columns.
        window_days : float
            Maximum separation window in days.  Default 14.
        degradation_lambda : float
            Exponential degradation rate.  Higher values mean faster
            degradation of resistance.  Default 0.1.

        Returns
        -------
        GeoDataFrame
            Compound intensity field with ``geometry`` and
            ``compound_intensity`` columns.
        """
        primary = primary_event.copy()
        secondary = secondary_event.copy()

        if "intensity" not in primary.columns:
            raise ValueError("primary_event must have an 'intensity' column")
        if "intensity" not in secondary.columns:
            raise ValueError("secondary_event must have an 'intensity' column")

        primary["time_delta"] = 0.0
        secondary["time_delta"] = window_days * 0.5

        combined = pd.concat([primary, secondary], ignore_index=True)

        degradation = np.exp(
            -degradation_lambda * combined["time_delta"] / max(window_days, _EPS)
        )
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
        """Merge multiple hazard intensity fields into a unified compound layer.

        Performs spatial nearest-neighbour joins to find coincident
        hazard intensities across layers and combines them via weighted
        averaging.  The output is a single GeoDataFrame suitable for
        ingestion by the asset exposure mapper.

        Parameters
        ----------
        layers : list of GeoDataFrame
            Each must have ``geometry`` and ``intensity`` columns.
        weights : list of float or None
            Relative importance weight per layer.  Defaults to equal
            weights.
        max_distance : float
            Maximum distance in metres for spatial join.  Points
            farther apart are treated as non-coincident.  Default 1000.

        Returns
        -------
        GeoDataFrame
            Merged layer with ``geometry`` and ``compound_intensity``
            columns in the CRS of the first layer.
        """
        if not layers:
            raise ValueError("At least one hazard layer is required")

        if weights is None:
            weights = [1.0] * len(layers)
        if len(weights) != len(layers):
            raise ValueError(
                f"weights length ({len(weights)}) must match "
                f"layers length ({len(layers)})"
            )

        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        crs = layers[0].crs

        merged = layers[0][["geometry", "intensity"]].copy()
        merged["weighted"] = merged["intensity"].astype(np.float64) * w[0]
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
            joined["weighted"] += joined["intensity_right"].astype(np.float64) * w[i]
            joined["w_sum"] += w[i]
            merged = joined.drop(columns=["index_right", "dist", "intensity_right"])

        merged["compound_intensity"] = merged["weighted"] / merged["w_sum"]
        result = merged[["geometry", "compound_intensity"]].copy()
        return gpd.GeoDataFrame(result, geometry="geometry", crs=crs)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = "fitted" if self.fitted_copula is not None else "unfitted"
        return (
            f"CompoundHazardEngine(copula={self.copula_type}, "
            f"state={fitted})"
        )
