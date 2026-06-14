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
Wildfire risk and line outage model.

Provides ``WildfireRiskEngine``, a geospatial risk evaluator that
computes dynamic line trip probabilities and conductor thermal derating
when transmission corridors intersect active wildfire flame fronts.
The model uses Shapely geometry operations for proximity analysis and
Poisson-process failure modelling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import (
    LineString,
    MultiPolygon,
    Point,
    Polygon,
)

logger = logging.getLogger(__name__)

_EPS: float = 1e-12

_DEFAULT_BASE_FAILURE_RATE: float = 1e-5  # failures per second
_DEFAULT_VULNERABILITY_GAMMA: float = 0.01  # m/K
_DEFAULT_SMOKE_DERATING: float = 0.5
_DEFAULT_PM25_THRESHOLD: float = 150.0  # µg/m³ (US EPA unhealthy)
_DEFAULT_FLAME_TEMP_K: float = 1073.15  # 800 °C
_DEFAULT_AMBIENT_TEMP_K: float = 298.15  # 25 °C
_DEFAULT_MIN_BUFFER_DISTANCE_M: float = 1.0
_DEFAULT_MAX_EXPONENT: float = 50.0


# ---------------------------------------------------------------------------
# WildfireRiskEngine
# ---------------------------------------------------------------------------


class WildfireRiskEngine:
    """Dynamic wildfire-to-transmission-line risk evaluator.

    Computes time-dependent line trip probabilities and conductor
    thermal derating based on proximity to an active wildfire flame
    front.  The failure rate is modelled as an exponential function
    of flame temperature excess over ambient divided by buffer
    distance, and trip events follow a Poisson process.

    Parameters
    ----------
    base_failure_rate : float
        Baseline failure rate :math:`\\lambda_0` in failures per
        second.  Default 1e-5.
    vulnerability_gamma : float
        Conductor-specific vulnerability scaling parameter
        :math:`\\gamma` in m/K.  Higher values increase sensitivity
        to temperature gradients.  Default 0.01.
    smoke_derating_coefficient : float
        Smoke derating coefficient :math:`\\delta_{\\text{smoke}}`
        (dimensionless, 0–1).  Default 0.5.
    pm25_threshold_ug_m3 : float
        PM₂.₅ concentration threshold in µg/m³ above which derating
        begins.  Default 150 (US EPA "unhealthy" level).
    default_ampacity_a : float
        Baseline conductor ampacity in Amperes.  Default 1000.
    default_flame_temp_k : float
        Default flame temperature in Kelvin when not specified in
        fire front metadata.  Default 1073.15 (800 °C).
    default_ambient_temp_k : float
        Default ambient temperature in Kelvin when not provided.
        Default 298.15 (25 °C).
    min_buffer_distance_m : float
        Minimum buffer distance clamp in metres to prevent
        division-by-zero in the failure-rate exponent.
        Default 1.0.
    max_exponent : float
        Maximum allowed exponent value to prevent numerical
        overflow in the failure-rate calculation.  Default 50.0.

    Attributes
    ----------
    lambda_0 : float
    gamma : float
    delta_smoke : float
    pm25_threshold : float
    I_max0 : float
    default_T_flame : float
    default_T_ambient : float
    """

    def __init__(
        self,
        base_failure_rate: float = _DEFAULT_BASE_FAILURE_RATE,
        vulnerability_gamma: float = _DEFAULT_VULNERABILITY_GAMMA,
        smoke_derating_coefficient: float = _DEFAULT_SMOKE_DERATING,
        pm25_threshold_ug_m3: float = _DEFAULT_PM25_THRESHOLD,
        default_ampacity_a: float = 1000.0,
        default_flame_temp_k: float = _DEFAULT_FLAME_TEMP_K,
        default_ambient_temp_k: float = _DEFAULT_AMBIENT_TEMP_K,
        min_buffer_distance_m: float = _DEFAULT_MIN_BUFFER_DISTANCE_M,
        max_exponent: float = _DEFAULT_MAX_EXPONENT,
    ) -> None:
        if base_failure_rate < 0:
            raise ValueError(
                f"base_failure_rate must be non-negative, got {base_failure_rate}"
            )
        if vulnerability_gamma < 0:
            raise ValueError(
                f"vulnerability_gamma must be non-negative, got {vulnerability_gamma}"
            )
        if not (0 <= smoke_derating_coefficient <= 1):
            raise ValueError(
                f"smoke_derating_coefficient must be in [0, 1], "
                f"got {smoke_derating_coefficient}"
            )
        if pm25_threshold_ug_m3 <= 0:
            raise ValueError(
                f"pm25_threshold_ug_m3 must be positive, got {pm25_threshold_ug_m3}"
            )
        if default_ampacity_a <= 0:
            raise ValueError(
                f"default_ampacity_a must be positive, got {default_ampacity_a}"
            )
        if min_buffer_distance_m <= 0:
            raise ValueError(
                f"min_buffer_distance_m must be positive, got {min_buffer_distance_m}"
            )
        if max_exponent <= 0:
            raise ValueError(
                f"max_exponent must be positive, got {max_exponent}"
            )

        self.lambda_0 = float(base_failure_rate)
        self.gamma = float(vulnerability_gamma)
        self.delta_smoke = float(smoke_derating_coefficient)
        self.pm25_threshold = float(pm25_threshold_ug_m3)
        self.I_max0 = float(default_ampacity_a)
        self.default_T_flame = float(default_flame_temp_k)
        self.default_T_ambient = float(default_ambient_temp_k)
        self.min_buffer_distance = float(min_buffer_distance_m)
        self.max_exponent = float(max_exponent)

    # ------------------------------------------------------------------
    # Core physics
    # ------------------------------------------------------------------

    @staticmethod
    def _minimum_distance(
        line_geom: LineString,
        fire_geom: Union[Polygon, MultiPolygon],
    ) -> float:
        """Compute minimum Euclidean distance between line and fire perimeter.

        Uses Shapely's ``distance`` method which returns the shortest
        distance between the boundaries of two geometries.  If the line
        intersects the fire polygon, the distance is zero.

        Parameters
        ----------
        line_geom : LineString
            The transmission line path.
        fire_geom : Polygon or MultiPolygon
            The wildfire flame front perimeter.

        Returns
        -------
        float
            Minimum distance in the same units as the input geometries
            (typically metres for projected CRS).
        """
        return float(line_geom.distance(fire_geom))

    def _dynamic_failure_rate(
        self,
        T_flame: float,
        T_ambient: float,
        D_buffer: float,
    ) -> float:
        """Compute dynamic failure rate λ_a(t).

        .. math::

            \\lambda_a(t) = \\lambda_0
            \\exp\\left(\\gamma \\frac{T_{\\text{flame}}
            - T_{\\text{ambient}}}{D_{\\text{buffer}}}\\right)

        Parameters
        ----------
        T_flame : float
            Flame temperature (K).
        T_ambient : float
            Ambient air temperature (K).
        D_buffer : float
            Minimum distance from line to fire perimeter (m).

        Returns
        -------
        float
            Dynamic failure rate in failures per second.
        """
        if D_buffer < _EPS:
            D_buffer = self.min_buffer_distance

        delta_T = max(0.0, T_flame - T_ambient)
        exponent = self.gamma * delta_T / D_buffer
        exponent = min(exponent, self.max_exponent)

        return self.lambda_0 * np.exp(exponent)

    @staticmethod
    def _trip_probability(
        lambda_a: float,
        dt_seconds: float,
    ) -> float:
        """Compute Poisson-process trip probability.

        .. math::

            P_{\\text{trip}}(t) = 1 - \\exp(-\\lambda_a(t) \\Delta t)

        Parameters
        ----------
        lambda_a : float
            Dynamic failure rate (failures/s).
        dt_seconds : float
            Time step duration (s).

        Returns
        -------
        float
            Trip probability ∈ [0, 1].
        """
        if lambda_a < 0:
            lambda_a = 0.0
        if dt_seconds < 0:
            dt_seconds = 0.0

        return float(1.0 - np.exp(-lambda_a * dt_seconds))

    def _derate_ampacity(
        self,
        pm25_ug_m3: float,
    ) -> float:
        """Compute smoke-derated conductor ampacity.

        .. math::

            I_{\\max}(t) = I_{\\max, 0}
            \\left(1 - \\delta_{\\text{smoke}}
            \\frac{\\text{PM}_{2.5}}{\\text{PM}_{\\text{threshold}}}
            \\right)

        Parameters
        ----------
        pm25_ug_m3 : float
            PM₂.₅ concentration at the line location (µg/m³).

        Returns
        -------
        float
            Derated ampacity in Amperes.
        """
        if pm25_ug_m3 < 0:
            pm25_ug_m3 = 0.0

        ratio = pm25_ug_m3 / self.pm25_threshold
        derating_factor = 1.0 - self.delta_smoke * ratio
        derating_factor = max(0.0, min(1.0, derating_factor))

        return self.I_max0 * derating_factor

    # ------------------------------------------------------------------
    # Single-line evaluation
    # ------------------------------------------------------------------

    def evaluate_line(
        self,
        line_geometry: LineString,
        fire_front: Union[Polygon, MultiPolygon],
        fire_metadata: Optional[Dict[str, Any]] = None,
        T_ambient: Optional[float] = None,
        dt_seconds: float = 3600.0,
    ) -> Dict[str, Any]:
        """Evaluate wildfire risk for a single transmission line.

        Parameters
        ----------
        line_geometry : LineString
            The transmission line path geometry.
        fire_front : Polygon or MultiPolygon
            The active wildfire flame front.
        fire_metadata : dict or None
            Optional metadata for the fire front.  Supported keys:

            - ``flame_temperature_k`` (float) — flame temperature.
            - ``pm25_ug_m3`` (float) — PM₂.₅ concentration.
        T_ambient : float or None
            Ambient air temperature in Kelvin.  If ``None``, uses the
            engine default.
        dt_seconds : float
            Simulation time step in seconds.  Default 3600 (1 hour).

        Returns
        -------
        dict
            Keys:

            - ``distance_m`` (float) — minimum distance to fire.
            - ``failure_rate_per_s`` (float) — λ_a(t).
            - ``trip_probability`` (float) — P_trip ∈ [0, 1].
            - ``derated_ampacity_a`` (float) — I_max(t).
            - ``intersects_fire`` (bool) — whether line crosses fire.
        """
        meta = fire_metadata or {}

        T_flame = float(meta.get("flame_temperature_k", self.default_T_flame))
        pm25 = float(meta.get("pm25_ug_m3", 0.0))

        if T_ambient is None:
            T_ambient = self.default_T_ambient

        D = self._minimum_distance(line_geometry, fire_front)

        lambda_a = self._dynamic_failure_rate(T_flame, T_ambient, D)
        p_trip = self._trip_probability(lambda_a, dt_seconds)
        I_derated = self._derate_ampacity(pm25)

        intersects = line_geometry.intersects(fire_front)

        return {
            "distance_m": D,
            "failure_rate_per_s": lambda_a,
            "trip_probability": p_trip,
            "derated_ampacity_a": I_derated,
            "intersects_fire": intersects,
        }

    # ------------------------------------------------------------------
    # Network-level evaluation
    # ------------------------------------------------------------------

    def evaluate_network(
        self,
        lines_gdf: gpd.GeoDataFrame,
        fire_front: Union[Polygon, MultiPolygon],
        fire_metadata: Optional[Dict[str, Any]] = None,
        T_ambient: Optional[Union[float, np.ndarray]] = None,
        dt_seconds: float = 3600.0,
    ) -> gpd.GeoDataFrame:
        """Evaluate wildfire risk for all lines in a GeoDataFrame.

        Parameters
        ----------
        lines_gdf : gpd.GeoDataFrame
            Transmission lines with ``geometry`` column containing
            ``LineString`` objects.
        fire_front : Polygon or MultiPolygon
            The active wildfire flame front.
        fire_metadata : dict or None
            Fire metadata (see :meth:`evaluate_line`).
        T_ambient : float, np.ndarray, or None
            Ambient temperature per line.  If array, must match the
            length of *lines_gdf*.
        dt_seconds : float
            Simulation time step in seconds.

        Returns
        -------
        gpd.GeoDataFrame
            Copy of *lines_gdf* with added columns: ``distance_m``,
            ``failure_rate_per_s``, ``trip_probability``,
            ``derated_ampacity_a``, ``intersects_fire``.
        """
        if len(lines_gdf) == 0:
            return lines_gdf.copy()

        result = lines_gdf.copy()

        meta = fire_metadata or {}
        T_flame = float(meta.get("flame_temperature_k", self.default_T_flame))
        pm25 = float(meta.get("pm25_ug_m3", 0.0))

        distances = np.empty(len(lines_gdf), dtype=np.float64)
        for i, geom in enumerate(lines_gdf.geometry):
            if geom is None or geom.is_empty:
                distances[i] = np.inf
            else:
                distances[i] = WildfireRiskEngine._minimum_distance(geom, fire_front)

        if T_ambient is None:
            T_ambient_arr = np.full(len(lines_gdf), self.default_T_ambient, dtype=np.float64)
        elif np.isscalar(T_ambient):
            T_ambient_arr = np.full(len(lines_gdf), float(T_ambient), dtype=np.float64)
        else:
            T_ambient_arr = np.asarray(T_ambient, dtype=np.float64)

        delta_T = np.maximum(0.0, T_flame - T_ambient_arr)
        D_clipped = np.maximum(distances, self.min_buffer_distance)
        exponent = np.minimum(self.gamma * delta_T / D_clipped, self.max_exponent)
        lambda_arr = self.lambda_0 * np.exp(exponent)

        p_trip_arr = 1.0 - np.exp(-lambda_arr * dt_seconds)

        pm25_ratio = pm25 / self.pm25_threshold
        derating_factor = np.clip(1.0 - self.delta_smoke * pm25_ratio, 0.0, 1.0)
        I_derated_arr = np.full(len(lines_gdf), self.I_max0 * derating_factor, dtype=np.float64)

        intersects_arr = np.array(
            [geom.intersects(fire_front) if geom is not None and not geom.is_empty else False
             for geom in lines_gdf.geometry],
            dtype=bool,
        )

        result["distance_m"] = distances
        result["failure_rate_per_s"] = lambda_arr
        result["trip_probability"] = p_trip_arr
        result["derated_ampacity_a"] = I_derated_arr
        result["intersects_fire"] = intersects_arr

        return result

    # ------------------------------------------------------------------
    # Time-series evaluation
    # ------------------------------------------------------------------

    def evaluate_timeseries(
        self,
        line_geometry: LineString,
        fire_fronts: List[Union[Polygon, MultiPolygon]],
        fire_metadata_list: Optional[List[Dict[str, Any]]] = None,
        T_ambient_arr: Optional[np.ndarray] = None,
        dt_seconds: float = 3600.0,
    ) -> pd.DataFrame:
        """Evaluate wildfire risk across a sequence of fire front snapshots.

        Parameters
        ----------
        line_geometry : LineString
            The transmission line path.
        fire_fronts : list of Polygon or MultiPolygon
            Time series of fire front geometries.
        fire_metadata_list : list of dict or None
            Per-timestep fire metadata.
        T_ambient_arr : np.ndarray or None
            Ambient temperature per timestep (K).
        dt_seconds : float
            Time step duration (s).

        Returns
        -------
        pd.DataFrame
            Time series with columns from :meth:`evaluate_line`.
        """
        n = len(fire_fronts)
        if fire_metadata_list is None:
            fire_metadata_list = [None] * n
        if T_ambient_arr is None:
            T_ambient_arr = np.full(n, self.default_T_ambient, dtype=np.float64)

        records: List[Dict[str, Any]] = []
        for i in range(n):
            result = self.evaluate_line(
                line_geometry,
                fire_fronts[i],
                fire_metadata=fire_metadata_list[i],
                T_ambient=float(T_ambient_arr[i]),
                dt_seconds=dt_seconds,
            )
            result["timestep"] = i
            records.append(result)

        return pd.DataFrame(records)

    def __repr__(self) -> str:
        return (
            f"WildfireRiskEngine(λ₀={self.lambda_0:.1e}/s, "
            f"γ={self.gamma:.3f}m/K, "
            f"δ_smoke={self.delta_smoke:.2f}, "
            f"I_max0={self.I_max0:.0f}A)"
        )
