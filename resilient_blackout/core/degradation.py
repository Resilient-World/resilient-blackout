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
Dynamic asset degradation engine.

Implements Physics-of-Failure (PoF) models that dynamically update
log-normal fragility curves based on cumulative thermal stress, ambient
humidity, physical age, and inspection quality.  Provides:

- ``ArrheniusDegradationModel`` — thermal aging for transformers and
  substation equipment.
- ``structural_decay_mu`` — material decay for poles and pylons.
- ``DynamicFragilityAdjuster`` — unified engine that maps asset metadata
  and weather history to updated ``ImpactFunction`` instances, feeding
  directly into Monte Carlo hazard runs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from resilient_blackout.core.base import Asset
from resilient_blackout.core.fragility import ImpactFunction, ImpactFunctionSet

logger = logging.getLogger(__name__)

_KELVIN_OFFSET: float = 273.15
_MIN_MU_FLOOR_RATIO: float = 0.1
_MIN_QUALITY: float = 0.1
_DEFAULT_GAMMA: float = 0.02
_DEFAULT_B: float = 15000.0
_DEFAULT_T_BASELINE_K: float = 293.15

_THERMAL_MATERIALS: frozenset[str] = frozenset({
    "transformer", "substation", "switchgear", "circuit_breaker",
})
_STRUCTURAL_MATERIALS: frozenset[str] = frozenset({
    "wood_pole", "steel_pole", "concrete_pole", "steel_tower",
    "lattice_tower", "pylon",
})


# ---------------------------------------------------------------------------
# Arrhenius thermal degradation model
# ---------------------------------------------------------------------------

class ArrheniusDegradationModel:
    """Arrhenius-based thermal aging acceleration model.

    Computes the acceleration factor :math:`V_r` that relates aging at an
    elevated ambient temperature to aging at a baseline reference
    temperature:

    .. math::

        V_r = \\exp\\!\\left(\\frac{B}{T_{\\text{baseline}}}
        - \\frac{B}{T_{\\text{ambient}}}\\right)

    where :math:`B` is the activation energy constant (K) and
    temperatures are in Kelvin.

    Parameters
    ----------
    B : float
        Activation energy constant in Kelvin.  Default 15 000 K
        (typical for transformer insulation paper).
    T_baseline : float
        Baseline reference temperature in Kelvin.  Default 293.15 K
        (20 °C).

    Attributes
    ----------
    B : float
    T_baseline : float
    """

    def __init__(
        self,
        B: float = _DEFAULT_B,
        T_baseline: float = _DEFAULT_T_BASELINE_K,
    ) -> None:
        if B <= 0:
            raise ValueError(f"Activation energy B must be positive, got {B}")
        if T_baseline <= 0:
            raise ValueError(f"T_baseline must be positive, got {T_baseline}")

        self.B: float = B
        self.T_baseline: float = T_baseline

    def compute_acceleration_factor(
        self, T_ambient: np.ndarray | float
    ) -> np.ndarray | float:
        """Compute the instantaneous acceleration factor.

        Parameters
        ----------
        T_ambient : float or np.ndarray
            Ambient temperature in Kelvin.

        Returns
        -------
        float or np.ndarray
            Acceleration factor :math:`V_r \\ge 1` when
            :math:`T_{\\text{ambient}} > T_{\\text{baseline}}`.
        """
        T = np.asarray(T_ambient, dtype=np.float64)
        inv_baseline = self.B / self.T_baseline
        inv_ambient = self.B / np.maximum(T, 1.0)
        result = np.exp(inv_baseline - inv_ambient)
        if result.ndim == 0:
            return float(result)
        return result

    def compute_equivalent_age(
        self,
        T_series: np.ndarray,
        delta_t_hours: float = 1.0,
    ) -> float:
        """Integrate acceleration over a temperature time series.

        Computes the equivalent thermal age relative to baseline:

        .. math::

            t_{\\text{eq}} = \\sum_t V_r(t) \\cdot \\Delta t

        Parameters
        ----------
        T_series : np.ndarray
            1-D array of temperatures in Kelvin.
        delta_t_hours : float
            Time step between consecutive readings in hours.

        Returns
        -------
        float
            Equivalent age in hours at baseline temperature.
        """
        if len(T_series) == 0:
            return 0.0
        factors = self.compute_acceleration_factor(T_series)
        return float(np.sum(factors) * delta_t_hours)


# ---------------------------------------------------------------------------
# Structural decay function
# ---------------------------------------------------------------------------

def structural_decay_mu(
    mu_0: float,
    age: float,
    gamma: float = _DEFAULT_GAMMA,
    humidity_factor: float = 1.0,
) -> float:
    """Compute the decayed log-median threshold for structural assets.

    .. math::

        \\mu_a(t) = \\mu_a^{(0)} \\cdot
        (1 - \\gamma \\cdot \\text{age} \\cdot h)

    where :math:`\\gamma` is the material decay rate and :math:`h` is an
    optional humidity acceleration factor.

    The result is clamped to a minimum of
    :math:`\\mu_a^{(0)} \\cdot f_{\\text{floor}}` to prevent zero or
    negative thresholds.

    Parameters
    ----------
    mu_0 : float
        Original (as-built) log-median threshold.
    age : float
        Asset age in years.
    gamma : float
        Annual material decay rate.  Default 0.02 (2 % per year).
    humidity_factor : float
        Multiplicative humidity acceleration.  Values > 1 accelerate
        decay (e.g., 1.5 for consistently humid climates).

    Returns
    -------
    float
        Decayed log-median threshold.
    """
    if mu_0 <= 0:
        return mu_0
    if age < 0:
        age = 0.0
    if gamma < 0:
        gamma = 0.0

    decayed = mu_0 * (1.0 - gamma * age * humidity_factor)
    floor = mu_0 * _MIN_MU_FLOOR_RATIO
    return max(decayed, floor)


# ---------------------------------------------------------------------------
# Material-specific decay rates (per year)
# ---------------------------------------------------------------------------

_MATERIAL_GAMMA: Dict[str, float] = {
    "wood_pole": 0.025,
    "steel_pole": 0.008,
    "concrete_pole": 0.012,
    "steel_tower": 0.006,
    "lattice_tower": 0.007,
    "pylon": 0.006,
}


# ---------------------------------------------------------------------------
# Dynamic fragility adjuster
# ---------------------------------------------------------------------------

class DynamicFragilityAdjuster:
    """Unified engine that updates fragility curves for asset degradation.

    Reads asset metadata (installation year, material, quality factor)
    from ``Asset.original_properties`` and optionally consumes hourly
    weather history to compute Physics-of-Failure adjustments to
    ``ImpactFunction`` parameters :math:`\\mu` and :math:`\\sigma`.

    Parameters
    ----------
    impact_function_set : ImpactFunctionSet
        The baseline (as-built) fragility curves.
    weather_history : pd.DataFrame or None
        Optional hourly weather data with columns:

        - ``temperature_c`` — ambient temperature in °C.
        - ``humidity_pct`` — relative humidity in % (0–100).

        If ``None``, a mild default climate is assumed.
    inspection_quality : dict or None
        Optional mapping from ``asset_id`` to a quality factor in
        (0, 1], where 1.0 is as-built condition.  Overrides any
        ``quality_factor`` found in ``Asset.original_properties``.

    Attributes
    ----------
    impact_function_set : ImpactFunctionSet
    arrhenius : ArrheniusDegradationModel
    weather_history : pd.DataFrame or None
    inspection_quality : dict
    """

    def __init__(
        self,
        impact_function_set: ImpactFunctionSet,
        weather_history: Optional[pd.DataFrame] = None,
        inspection_quality: Optional[Dict[str, float]] = None,
    ) -> None:
        self.impact_function_set = impact_function_set
        self.arrhenius = ArrheniusDegradationModel()
        self.weather_history = weather_history
        self.inspection_quality: Dict[str, float] = inspection_quality or {}

        self._mean_temp_k: float = _DEFAULT_T_BASELINE_K
        self._mean_humidity: float = 60.0

        if weather_history is not None:
            self._parse_weather(weather_history)

    def _parse_weather(self, df: pd.DataFrame) -> None:
        """Extract mean temperature and humidity from weather history.

        Parameters
        ----------
        df : pd.DataFrame
            Weather data.
        """
        if "temperature_c" in df.columns and len(df) > 0:
            self._mean_temp_k = float(df["temperature_c"].mean()) + _KELVIN_OFFSET
        if "humidity_pct" in df.columns and len(df) > 0:
            self._mean_humidity = float(df["humidity_pct"].mean())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def adjust_for_asset(
        self,
        asset: Asset,
        current_year: int,
        weather_series: Optional[np.ndarray] = None,
    ) -> ImpactFunction:
        """Produce a degraded ``ImpactFunction`` for a single asset.

        Parameters
        ----------
        asset : Asset
            The asset to evaluate.  Must have ``impact_function_id`` set
            and relevant keys in ``original_properties``.
        current_year : int
            The reference year for age calculation.
        weather_series : np.ndarray or None
            Optional 1-D array of hourly temperatures (Kelvin) specific
            to this asset.  If ``None``, the global weather history mean
            is used.

        Returns
        -------
        ImpactFunction
            A new instance with degraded :math:`\\mu` and :math:`\\sigma`.

        Raises
        ------
        KeyError
            If ``asset.impact_function_id`` is not in the function set.
        ValueError
            If ``asset.impact_function_id`` is ``None``.
        """
        if asset.impact_function_id is None:
            raise ValueError(
                f"Asset '{asset.asset_id}' has no impact_function_id assigned"
            )

        base_func = self.impact_function_set[asset.impact_function_id]
        props = asset.original_properties

        installation_year = int(props.get("installation_year", current_year))
        age = max(0.0, float(current_year - installation_year))

        quality = self.inspection_quality.get(asset.asset_id)
        if quality is None:
            quality = float(props.get("quality_factor", 1.0))
        quality = max(quality, _MIN_QUALITY)

        material = str(props.get("material", "")).lower().replace(" ", "_")

        if material in _THERMAL_MATERIALS:
            return self._adjust_thermal(base_func, age, quality, weather_series)
        elif material in _STRUCTURAL_MATERIALS:
            return self._adjust_structural(base_func, age, quality, material)
        else:
            logger.debug(
                "Material '%s' for asset '%s' not recognised; applying generic aging.",
                material,
                asset.asset_id,
            )
            return self._adjust_generic(base_func, age, quality)

    def adjust_all(
        self,
        assets: List[Asset],
        current_year: int,
    ) -> ImpactFunctionSet:
        """Batch-adjust all assets and return a new ``ImpactFunctionSet``.

        Parameters
        ----------
        assets : list of Asset
            Assets to process.
        current_year : int
            Reference year for age calculation.

        Returns
        -------
        ImpactFunctionSet
            A new set containing the degraded functions.  Functions for
            assets without an ``impact_function_id`` are left unchanged.
        """
        degraded_funcs: Dict[str, ImpactFunction] = {}

        for asset in assets:
            if asset.impact_function_id is None:
                continue
            try:
                degraded = self.adjust_for_asset(asset, current_year)
                degraded_funcs[degraded.function_id] = degraded
            except Exception:
                logger.exception(
                    "Failed to adjust asset '%s'; using baseline function.",
                    asset.asset_id,
                )
                base = self.impact_function_set.get(asset.impact_function_id)
                if base is not None:
                    degraded_funcs[base.function_id] = base

        return ImpactFunctionSet(list(degraded_funcs.values()))

    # ------------------------------------------------------------------
    # Internal adjustment methods
    # ------------------------------------------------------------------

    def _adjust_thermal(
        self,
        base: ImpactFunction,
        age: float,
        quality: float,
        weather_series: Optional[np.ndarray],
    ) -> ImpactFunction:
        """Apply Arrhenius-based thermal degradation.

        Parameters
        ----------
        base : ImpactFunction
        age : float
            Chronological age in years.
        quality : float
            Inspection quality factor (0–1).
        weather_series : np.ndarray or None

        Returns
        -------
        ImpactFunction
        """
        effective_age = age / quality

        if weather_series is not None and len(weather_series) > 0:
            eq_hours = self.arrhenius.compute_equivalent_age(weather_series)
            eq_years = eq_hours / 8760.0
            effective_age = max(effective_age, eq_years)
        else:
            v_r = self.arrhenius.compute_acceleration_factor(self._mean_temp_k)
            effective_age *= float(v_r)

        delta_mu = -0.05 * effective_age
        sigma_factor = 1.0 + 0.03 * effective_age

        func = base.shift_mu(delta_mu)
        func = func.scale_sigma(sigma_factor)
        return func

    def _adjust_structural(
        self,
        base: ImpactFunction,
        age: float,
        quality: float,
        material: str,
    ) -> ImpactFunction:
        """Apply structural decay to log-median threshold.

        Parameters
        ----------
        base : ImpactFunction
        age : float
        quality : float
        material : str

        Returns
        -------
        ImpactFunction
        """
        effective_age = age / quality
        gamma = _MATERIAL_GAMMA.get(material, _DEFAULT_GAMMA)

        humidity_factor = self._mean_humidity / 60.0
        humidity_factor = max(0.5, min(humidity_factor, 2.5))

        new_mu = structural_decay_mu(
            mu_0=base.mu,
            age=effective_age,
            gamma=gamma,
            humidity_factor=humidity_factor,
        )

        delta_mu = new_mu - base.mu
        sigma_factor = 1.0 + 0.02 * effective_age

        func = base.shift_mu(delta_mu)
        func = func.scale_sigma(sigma_factor)
        return func

    def _adjust_generic(
        self,
        base: ImpactFunction,
        age: float,
        quality: float,
    ) -> ImpactFunction:
        """Apply a mild generic aging penalty.

        Parameters
        ----------
        base : ImpactFunction
        age : float
        quality : float

        Returns
        -------
        ImpactFunction
        """
        effective_age = age / quality
        delta_mu = -0.03 * effective_age
        sigma_factor = 1.0 + 0.015 * effective_age

        func = base.shift_mu(delta_mu)
        func = func.scale_sigma(sigma_factor)
        return func
