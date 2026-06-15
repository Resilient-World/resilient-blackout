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
Physical storyline scenario generator.

Provides ``ClimateStorylineAdjuster`` for scaling historical weather
disasters to future climate conditions, including the Batts gradient
wind-field model for tropical cyclones, IPCC AR6-based intensity
adjustments, and spatial intersection with grid assets.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from resilient_blackout.core.base import Asset, HazardEvent

logger = logging.getLogger(__name__)

ScenarioType = Literal["ssp245", "ssp585"]

_RHO_AIR: float = 1.15
_OMEGA_EARTH: float = 7.2921e-5

_IPCC_WARMING: Dict[str, Dict[int, float]] = {
    "ssp245": {2030: 1.3, 2050: 2.0, 2070: 2.4, 2100: 2.7},
    "ssp585": {2030: 1.5, 2050: 2.4, 2070: 3.3, 2100: 4.4},
}

_SCALING_PER_DEGREE: Dict[str, float] = {
    "wind": 0.03,
    "flood": 0.07,
    "cold": -3.0,
}

_EPS: float = 1e-12


class ClimateStorylineAdjuster:
    """Scale historical disasters to future climate conditions.

    Uses IPCC AR6 warming projections to adjust hazard intensities
    and the Batts gradient wind-field model to generate tropical
    cyclone wind footprints for deterministic grid stress testing.

    Parameters
    ----------
    climate_scenario : str
        ``"ssp245"`` or ``"ssp585"``.  Default ``"ssp585"``.
    target_year : int
        Future target year (2030–2100).  Default 2050.
    custom_scaling : dict or None
        Optional per-variable scaling factors overriding IPCC defaults.
        Keys: ``"wind"``, ``"flood"``, ``"cold"``.  Values are
        multiplicative (wind, flood) or additive (cold) per °C.

    Attributes
    ----------
    climate_scenario : str
    target_year : int
    warming_delta_c : float
    scaling_factors : dict
    """

    def __init__(
        self,
        climate_scenario: ScenarioType = "ssp585",
        target_year: int = 2050,
        custom_scaling: Optional[Dict[str, float]] = None,
    ) -> None:
        if climate_scenario not in _IPCC_WARMING:
            raise ValueError(
                f"climate_scenario must be one of {list(_IPCC_WARMING.keys())}, "
                f"got '{climate_scenario}'"
            )

        self.climate_scenario: ScenarioType = climate_scenario
        self.target_year = target_year
        self.warming_delta_c = self._interpolate_warming(climate_scenario, target_year)

        self.scaling_factors = dict(_SCALING_PER_DEGREE)
        if custom_scaling:
            self.scaling_factors.update(custom_scaling)

        logger.info(
            "Storyline adjuster: %s, %d → ΔT=%.1f°C",
            climate_scenario, target_year, self.warming_delta_c,
        )

    @staticmethod
    def _interpolate_warming(scenario: str, year: int) -> float:
        """Interpolate warming delta for a given year.

        Parameters
        ----------
        scenario : str
        year : int

        Returns
        -------
        float
        """
        table = _IPCC_WARMING[scenario]
        years = np.array(list(table.keys()), dtype=np.float64)
        deltas = np.array(list(table.values()), dtype=np.float64)
        return float(np.interp(year, years, deltas))

    # ------------------------------------------------------------------
    # Batts gradient wind-field model
    # ------------------------------------------------------------------

    @staticmethod
    def batts_wind_field(
        r_km: np.ndarray,
        delta_p_hpa: float,
        rmax_km: float,
        B: float = 1.3,
        latitude: float = 25.0,
    ) -> np.ndarray:
        """Batts gradient wind-field model for tropical cyclones.

        .. math::

            V = \\sqrt{\\frac{\\Delta p}{\\rho} R B
            \\exp\\left(-\\left(\\frac{R}{r}\\right)^B\\right)
            + \\frac{r^2 f^2}{4}} - \\frac{r f}{2}

        where :math:`R = r_{\\max}`, :math:`r` is radial distance,
        :math:`\\rho` is air density, and :math:`f` is the Coriolis
        parameter.

        Parameters
        ----------
        r_km : np.ndarray
            Radial distances from storm centre in km.
        delta_p_hpa : float
            Central pressure deficit in hPa.
        rmax_km : float
            Radius of maximum winds in km.
        B : float
            Holland B parameter (1.0–2.5).  Default 1.3.
        latitude : float
            Storm centre latitude in degrees.  Default 25.

        Returns
        -------
        np.ndarray
            Gradient wind speed in m/s at each radial distance.
        """
        r = np.asarray(r_km, dtype=np.float64)
        dp = delta_p_hpa * 100.0
        R = rmax_km * 1000.0
        r_m = r * 1000.0

        f = 2.0 * _OMEGA_EARTH * np.sin(np.radians(latitude))

        ratio = np.where(r > _EPS, R / r_m, np.inf)
        exp_term = np.exp(-(ratio ** B))

        term1 = (dp / _RHO_AIR) * (R / r_m) * B * exp_term
        term2 = (r_m * f / 2.0) ** 2

        v = np.sqrt(np.maximum(0.0, term1 + term2)) - r_m * f / 2.0
        return np.maximum(0.0, v)

    # ------------------------------------------------------------------
    # Cyclone footprint generation
    # ------------------------------------------------------------------

    def generate_cyclone_footprint(
        self,
        track_points: gpd.GeoDataFrame,
        delta_p_hpa: float,
        rmax_km: float,
        B: float = 1.3,
        grid_resolution_km: float = 5.0,
        max_radius_km: float = 300.0,
    ) -> gpd.GeoDataFrame:
        """Generate 2-D wind field from cyclone track.

        For each track point, computes the radial wind profile and
        assigns wind speeds to a regular grid within the storm's
        influence radius.

        Parameters
        ----------
        track_points : GeoDataFrame
            Track with ``geometry`` (Point), ``time``, and optionally
            ``latitude`` column.
        delta_p_hpa : float
            Central pressure deficit in hPa.
        rmax_km : float
            Radius of maximum winds in km.
        B : float
            Holland B parameter.  Default 1.3.
        grid_resolution_km : float
            Grid spacing in km.  Default 5.
        max_radius_km : float
            Maximum radius of influence in km.  Default 300.

        Returns
        -------
        GeoDataFrame
            Grid points with ``geometry`` and ``wind_speed_ms`` columns.
        """
        if len(track_points) == 0:
            raise ValueError("track_points must not be empty")

        all_points: List[Dict[str, Any]] = []

        for _, row in track_points.iterrows():
            centre = row.geometry
            lat = row.get("latitude", centre.y)

            radii = np.arange(grid_resolution_km, max_radius_km + grid_resolution_km, grid_resolution_km)
            speeds = self.batts_wind_field(radii, delta_p_hpa, rmax_km, B, lat)

            for r, v in zip(radii, speeds):
                if v < 5.0:
                    continue
                n_azimuth = max(4, int(2 * np.pi * r / grid_resolution_km))
                for angle in np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False):
                    dx = r * np.cos(angle) / 111.32
                    dy = r * np.sin(angle) / (111.32 * np.cos(np.radians(lat)))
                    pt = Point(centre.x + dx, centre.y + dy)
                    all_points.append({"geometry": pt, "wind_speed_ms": v})

        if not all_points:
            logger.warning("Cyclone footprint generated no points above 5 m/s threshold.")
            return gpd.GeoDataFrame(columns=["geometry", "wind_speed_ms"], crs="EPSG:4326")

        result = gpd.GeoDataFrame(all_points, crs="EPSG:4326")
        result = result.dissolve(by=None, aggfunc="max").explode(index_parts=False)
        result = result.loc[result.groupby("geometry")["wind_speed_ms"].idxmax()]
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Historical event adjustment
    # ------------------------------------------------------------------

    def adjust_historical_event(
        self,
        event_data: pd.DataFrame,
        climate_scenario: Optional[ScenarioType] = None,
        target_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """Scale historical hazard intensities to future climate.

        Parameters
        ----------
        event_data : DataFrame
            Must have ``hazard_type``, ``intensity``, and ``units``
            columns.  ``hazard_type`` should be ``"wind"``, ``"flood"``,
            or ``"cold"``.
        climate_scenario : str or None
            Override the instance scenario.
        target_year : int or None
            Override the instance target year.

        Returns
        -------
        DataFrame
            Copy of input with ``intensity_adjusted`` column added.
        """
        result = event_data.copy()

        if climate_scenario is not None and target_year is not None:
            delta = self._interpolate_warming(climate_scenario, target_year)
        else:
            delta = self.warming_delta_c

        for hazard_type in ["wind", "flood", "cold"]:
            mask = result["hazard_type"] == hazard_type
            if not mask.any():
                continue

            factor = self.scaling_factors.get(hazard_type, 0.0)

            if hazard_type == "cold":
                result.loc[mask, "intensity_adjusted"] = (
                    result.loc[mask, "intensity"] + factor * delta
                )
            else:
                result.loc[mask, "intensity_adjusted"] = (
                    result.loc[mask, "intensity"] * (1.0 + factor * delta)
                )

        logger.info(
            "Adjusted %d hazard records with ΔT=%.1f°C.", len(result), delta
        )
        return result

    # ------------------------------------------------------------------
    # Asset intersection
    # ------------------------------------------------------------------

    @staticmethod
    def intersect_with_assets(
        hazard_footprint: gpd.GeoDataFrame,
        assets: List[Asset],
        max_distance_m: float = 5000.0,
    ) -> gpd.GeoDataFrame:
        """Spatially intersect hazard footprint with grid assets.

        Parameters
        ----------
        hazard_footprint : GeoDataFrame
            Must have ``geometry`` and an intensity column.
        assets : list of Asset
        max_distance_m : float
            Maximum search distance in metres.  Default 5000.

        Returns
        -------
        GeoDataFrame
            ``asset_id``, ``intensity``, ``geometry`` for matched
            assets.
        """
        asset_geoms = [
            {"asset_id": a.asset_id, "geometry": a.geom} for a in assets
        ]
        asset_gdf = gpd.GeoDataFrame(asset_geoms, crs="EPSG:4326")

        if hazard_footprint.crs is None:
            hazard_footprint = hazard_footprint.set_crs("EPSG:4326")
        if hazard_footprint.crs != "EPSG:4326":
            hazard_footprint = hazard_footprint.to_crs("EPSG:4326")

        intensity_col = (
            "wind_speed_ms" if "wind_speed_ms" in hazard_footprint.columns else "intensity"
        )
        if intensity_col not in hazard_footprint.columns:
            raise ValueError(
                f"Hazard footprint must have '{intensity_col}' or 'intensity' column"
            )

        joined = gpd.sjoin_nearest(
            asset_gdf,
            hazard_footprint[[intensity_col, "geometry"]],
            how="left",
            max_distance=max_distance_m,
            distance_col="distance_m",
        )

        joined = joined.dropna(subset=[intensity_col])
        joined = joined.rename(columns={intensity_col: "intensity"})
        result = joined[["asset_id", "intensity", "geometry"]].copy()
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # End-to-end storyline scenario
    # ------------------------------------------------------------------

    def generate_storyline_scenario(
        self,
        historical_event: pd.DataFrame,
        assets: List[Asset],
        climate_scenario: Optional[ScenarioType] = None,
        target_year: Optional[int] = None,
        cyclone_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a complete future climate storyline scenario.

        Parameters
        ----------
        historical_event : DataFrame
            Historical hazard data (see :meth:`adjust_historical_event`).
        assets : list of Asset
        climate_scenario : str or None
        target_year : int or None
        cyclone_params : dict or None
            For cyclone events: ``{"delta_p_hpa": float,
            "rmax_km": float, "B": float, "track_points": GeoDataFrame}``.

        Returns
        -------
        dict
            ``{"hazard_event": dict, "adjusted_data": DataFrame,
            "asset_exposure": GeoDataFrame}`` compatible with
            ``HazardEvent`` and ``CascadingSimulator``.
        """
        adjusted = self.adjust_historical_event(
            historical_event, climate_scenario, target_year
        )

        if cyclone_params is not None:
            track = cyclone_params["track_points"]
            dp = cyclone_params["delta_p_hpa"]
            rmax = cyclone_params["rmax_km"]
            B_val = cyclone_params.get("B", 1.3)

            footprint = self.generate_cyclone_footprint(track, dp, rmax, B_val)
            footprint = footprint.rename(columns={"wind_speed_ms": "intensity"})
        else:
            footprint = gpd.GeoDataFrame(
                adjusted[["intensity_adjusted"]].rename(
                    columns={"intensity_adjusted": "intensity"}
                ),
                geometry=gpd.points_from_xy(
                    np.zeros(len(adjusted)), np.zeros(len(adjusted))
                ),
                crs="EPSG:4326",
            )

        asset_exposure = self.intersect_with_assets(footprint, assets)

        hazard_event = {
            "event_id": f"storyline_{self.climate_scenario}_{self.target_year}",
            "name": f"Storyline {self.climate_scenario} {self.target_year}",
            "hazard_type": str(historical_event["hazard_type"].iloc[0]),
            "frequency": 1.0,
            "centroids": footprint,
            "units": "m/s" if cyclone_params else "adjusted",
        }

        return {
            "hazard_event": hazard_event,
            "adjusted_data": adjusted,
            "asset_exposure": asset_exposure,
        }
