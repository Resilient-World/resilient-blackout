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
Hydrological flood vulnerability engine.

Provides ``SubstationFlooder``, a vectorised physical vulnerability
model for electrical substations exposed to localised floodwaters.
The engine uses log-logistic depth-damage curves, accounts for
structural flood protection (levees), active pump mitigation, and
supports geospatial raster-based flood depth sampling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.special import expit
from shapely.geometry import Point

logger = logging.getLogger(__name__)

try:
    import rasterio

    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    _HAS_RASTERIO = False

_EPS: float = 1e-12

_DEFAULT_GAMMA: float = 2.0
_DEFAULT_FFE_M: float = 0.3
_DEFAULT_LEVEE_M: float = 0.0
_DEFAULT_PUMP_RATE_MPS: float = 0.0
_DEFAULT_MAX_EXPONENT: float = 500.0


# ---------------------------------------------------------------------------
# SubstationFlooder
# ---------------------------------------------------------------------------


class SubstationFlooder:
    """Physical flood vulnerability model for electrical substations.

    Models substation failure probability as a function of water depth
    above the First Floor Elevation (FFE) using a cumulative log-logistic
    depth-damage curve.  Accounts for structural levee protection and
    active drainage pump mitigation.

    Parameters
    ----------
    gamma : float
        Vulnerability slope factor :math:`\\gamma` controlling the
        steepness of the log-logistic curve.  Higher values produce a
        sharper transition from safe to failed.  Default 2.0.
    default_ffe_m : float
        Default First Floor Elevation in metres above local ground.
        Used when substations lack explicit FFE data.  Default 0.3.
    default_levee_height_m : float
        Default structural flood protection wall height in metres.
        Default 0.0 (no protection).
    default_pump_rate_m_per_s : float
        Default active drainage pump evacuation rate in m/s (effective
        depth reduction per second).  Default 0.0 (no pumps).
    default_flood_duration_s : float
        Default flood peak duration in seconds for pump mitigation
        calculation.  Default 3600 (1 hour).
    max_exponent : float
        Maximum absolute exponent value for numerical stability
        in the log-logistic curve.  Default 500.0.

    Attributes
    ----------
    gamma : float
    default_ffe_m : float
    default_levee_height_m : float
    default_pump_rate_mps : float
    default_flood_duration_s : float
    """

    def __init__(
        self,
        gamma: float = _DEFAULT_GAMMA,
        default_ffe_m: float = _DEFAULT_FFE_M,
        default_levee_height_m: float = _DEFAULT_LEVEE_M,
        default_pump_rate_m_per_s: float = _DEFAULT_PUMP_RATE_MPS,
        default_flood_duration_s: float = 3600.0,
        max_exponent: float = _DEFAULT_MAX_EXPONENT,
    ) -> None:
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if default_ffe_m < 0:
            raise ValueError(f"default_ffe_m must be non-negative, got {default_ffe_m}")
        if default_levee_height_m < 0:
            raise ValueError(
                f"default_levee_height_m must be non-negative, got {default_levee_height_m}"
            )
        if default_pump_rate_m_per_s < 0:
            raise ValueError(
                f"default_pump_rate_m_per_s must be non-negative, "
                f"got {default_pump_rate_m_per_s}"
            )
        if default_flood_duration_s <= 0:
            raise ValueError(
                f"default_flood_duration_s must be positive, got {default_flood_duration_s}"
            )
        if max_exponent <= 0:
            raise ValueError(
                f"max_exponent must be positive, got {max_exponent}"
            )

        self.gamma = float(gamma)
        self.default_ffe_m = float(default_ffe_m)
        self.default_levee_height_m = float(default_levee_height_m)
        self.default_pump_rate_mps = float(default_pump_rate_m_per_s)
        self.default_flood_duration_s = float(default_flood_duration_s)
        self.max_exponent = float(max_exponent)

    # ------------------------------------------------------------------
    # Core physics
    # ------------------------------------------------------------------

    @staticmethod
    def _effective_depth(
        raw_depth_m: np.ndarray,
        pump_rate_mps: np.ndarray,
        dt_seconds: float,
    ) -> np.ndarray:
        """Compute effective flood depth after pump mitigation.

        .. math::

            d_{\\text{effective}} = \\max(0, d - Q_{\\text{pump}} \\cdot \\Delta t)

        Parameters
        ----------
        raw_depth_m : np.ndarray
            Raw flood depth above ground (m).
        pump_rate_mps : np.ndarray
            Pump evacuation rate per substation (m/s).
        dt_seconds : float
            Flood peak duration (s).

        Returns
        -------
        np.ndarray
            Effective depth after pump mitigation (m).
        """
        pumped_volume = pump_rate_mps * dt_seconds
        return np.maximum(0.0, raw_depth_m - pumped_volume)

    @staticmethod
    def _failure_probability(
        d_effective_m: np.ndarray,
        ffe_m: np.ndarray,
        levee_height_m: np.ndarray,
        gamma: float,
        max_exponent: float = _DEFAULT_MAX_EXPONENT,
    ) -> np.ndarray:
        """Compute substation failure probability via log-logistic curve.

        .. math::

            P_f(d) = \\frac{1}{1 + \\exp(-\\gamma (d - \\text{FFE} - H_{\\text{levee}}))}

        Parameters
        ----------
        d_effective_m : np.ndarray
            Effective water depth above ground (m).
        ffe_m : np.ndarray
            First Floor Elevation per substation (m).
        levee_height_m : np.ndarray
            Levee/protection wall height per substation (m).
        gamma : float
            Vulnerability slope factor.

        Returns
        -------
        np.ndarray
            Failure probability ∈ [0, 1].
        """
        water_above_ffe = d_effective_m - ffe_m - levee_height_m
        exponent = -gamma * water_above_ffe
        exponent = np.clip(exponent, -max_exponent, max_exponent)
        return expit(exponent)

    # ------------------------------------------------------------------
    # Single-substation evaluation
    # ------------------------------------------------------------------

    def evaluate_substation(
        self,
        flood_depth_m: float,
        ffe_m: Optional[float] = None,
        levee_height_m: Optional[float] = None,
        pump_rate_mps: Optional[float] = None,
        flood_duration_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate flood vulnerability for a single substation.

        Parameters
        ----------
        flood_depth_m : float
            Raw flood depth above ground level (m).
        ffe_m : float or None
            First Floor Elevation.  If ``None``, uses engine default.
        levee_height_m : float or None
            Structural protection height.  If ``None``, uses default.
        pump_rate_mps : float or None
            Drainage pump rate.  If ``None``, uses default.
        flood_duration_s : float or None
            Flood peak duration.  If ``None``, uses default.

        Returns
        -------
        dict
            Keys:

            - ``raw_depth_m`` (float)
            - ``effective_depth_m`` (float)
            - ``failure_probability`` (float) — P_f ∈ [0, 1]
            - ``operational`` (bool) — ``True`` if P_f < 0.5
            - ``ffe_m`` (float)
            - ``levee_height_m`` (float)
        """
        ffe = float(ffe_m if ffe_m is not None else self.default_ffe_m)
        levee = float(levee_height_m if levee_height_m is not None else self.default_levee_height_m)
        pump = float(pump_rate_mps if pump_rate_mps is not None else self.default_pump_rate_mps)
        dt = float(flood_duration_s if flood_duration_s is not None else self.default_flood_duration_s)

        d_eff = max(0.0, flood_depth_m - pump * dt)
        water_above_ffe = d_eff - ffe - levee
        exponent = -self.gamma * water_above_ffe
        exponent = max(-self.max_exponent, min(self.max_exponent, exponent))
        p_f = float(expit(float(exponent)))

        return {
            "raw_depth_m": float(flood_depth_m),
            "effective_depth_m": float(d_eff),
            "failure_probability": p_f,
            "operational": bool(p_f < 0.5),
            "ffe_m": ffe,
            "levee_height_m": levee,
        }

    # ------------------------------------------------------------------
    # Batch substation evaluation
    # ------------------------------------------------------------------

    def evaluate_substations(
        self,
        substations_gdf: gpd.GeoDataFrame,
        flood_depths_m: Union[np.ndarray, Dict[str, float]],
        flood_duration_s: Optional[float] = None,
        ffe_col: str = "ffe_m",
        levee_col: str = "levee_height_m",
        pump_col: str = "pump_rate_mps",
        id_col: str = "substation_id",
    ) -> gpd.GeoDataFrame:
        """Vectorised flood vulnerability evaluation for multiple substations.

        Parameters
        ----------
        substations_gdf : gpd.GeoDataFrame
            Substation locations with optional columns for FFE, levee
            height, and pump rate.
        flood_depths_m : np.ndarray or dict
            Flood depth per substation.  If array, must match the
            length of *substations_gdf*.  If dict, keys are substation
            IDs matching *id_col*.
        flood_duration_s : float or None
            Flood peak duration.  If ``None``, uses engine default.
        ffe_col : str
            Column name for FFE in *substations_gdf*.  Default
            ``"ffe_m"``.
        levee_col : str
            Column name for levee height.  Default ``"levee_height_m"``.
        pump_col : str
            Column name for pump rate.  Default ``"pump_rate_mps"``.
        id_col : str
            Column name for substation identifier.  Default
            ``"substation_id"``.

        Returns
        -------
        gpd.GeoDataFrame
            Copy of *substations_gdf* with added columns:
            ``raw_depth_m``, ``effective_depth_m``,
            ``failure_probability``, ``operational``.
        """
        n = len(substations_gdf)
        if n == 0:
            return substations_gdf.copy()

        result = substations_gdf.copy()

        if isinstance(flood_depths_m, dict):
            depths = np.array(
                [float(flood_depths_m.get(str(row[id_col]), 0.0))
                 for _, row in substations_gdf.iterrows()],
                dtype=np.float64,
            )
        else:
            depths = np.asarray(flood_depths_m, dtype=np.float64)
            if len(depths) != n:
                raise ValueError(
                    f"flood_depths_m length {len(depths)} != substations count {n}"
                )

        ffe = (
            substations_gdf[ffe_col].values.astype(np.float64)
            if ffe_col in substations_gdf.columns
            else np.full(n, self.default_ffe_m, dtype=np.float64)
        )

        levee = (
            substations_gdf[levee_col].values.astype(np.float64)
            if levee_col in substations_gdf.columns
            else np.full(n, self.default_levee_height_m, dtype=np.float64)
        )

        pump = (
            substations_gdf[pump_col].values.astype(np.float64)
            if pump_col in substations_gdf.columns
            else np.full(n, self.default_pump_rate_mps, dtype=np.float64)
        )

        dt = float(flood_duration_s if flood_duration_s is not None else self.default_flood_duration_s)

        d_eff = self._effective_depth(depths, pump, dt)
        p_f = self._failure_probability(d_eff, ffe, levee, self.gamma, self.max_exponent)

        result["raw_depth_m"] = depths
        result["effective_depth_m"] = d_eff
        result["failure_probability"] = p_f
        result["operational"] = p_f < 0.5

        return result

    # ------------------------------------------------------------------
    # Geospatial raster interface
    # ------------------------------------------------------------------

    def evaluate_flood_impact(
        self,
        substations_gdf: gpd.GeoDataFrame,
        flood_source: Union[str, Path, np.ndarray, Dict[str, float]],
        flood_duration_s: Optional[float] = None,
        raster_band: int = 1,
        **kwargs: Any,
    ) -> gpd.GeoDataFrame:
        """Evaluate flood impact on substations from a flood depth source.

        Accepts either a GeoTIFF raster path (requires ``rasterio``) or
        pre-extracted depth values as an array or dict.

        Parameters
        ----------
        substations_gdf : gpd.GeoDataFrame
            Substation locations.
        flood_source : str, Path, np.ndarray, or dict
            Flood depth data.  If a file path ending in ``.tif`` or
            ``.tiff``, reads via ``rasterio``.  Otherwise treated as
            pre-extracted depth values.
        flood_duration_s : float or None
            Flood peak duration.
        raster_band : int
            Raster band index for depth values (1-based).  Default 1.
        **kwargs
            Passed to :meth:`evaluate_substations`.

        Returns
        -------
        gpd.GeoDataFrame
            See :meth:`evaluate_substations`.
        """
        if isinstance(flood_source, (str, Path)):
            path = Path(flood_source)
            if path.suffix.lower() in (".tif", ".tiff"):
                depths = self._sample_raster(substations_gdf, path, band=raster_band)
            else:
                raise ValueError(
                    f"Unrecognised flood source file type: {path.suffix}. "
                    f"Use .tif/.tiff for raster or pass array/dict directly."
                )
        elif isinstance(flood_source, np.ndarray):
            depths = flood_source
        elif isinstance(flood_source, dict):
            depths = flood_source
        else:
            raise TypeError(
                f"flood_source must be str, Path, np.ndarray, or dict, "
                f"got {type(flood_source).__name__}"
            )

        return self.evaluate_substations(
            substations_gdf,
            flood_depths_m=depths,
            flood_duration_s=flood_duration_s,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Raster sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_raster(
        substations_gdf: gpd.GeoDataFrame,
        raster_path: Path,
        band: int = 1,
    ) -> np.ndarray:
        """Sample flood depth values from a GeoTIFF raster at substation points.

        Parameters
        ----------
        substations_gdf : gpd.GeoDataFrame
        raster_path : Path
            Path to GeoTIFF raster.
        band : int
            1-based band index.

        Returns
        -------
        np.ndarray
            Sampled depth values, shape ``(n_substations,)``.

        Raises
        ------
        ImportError
            If ``rasterio`` is not installed.
        """
        if not _HAS_RASTERIO:
            raise ImportError(
                "rasterio is required for GeoTIFF sampling. "
                "Install with: pip install rasterio"
            )

        with rasterio.open(raster_path) as src:
            gdf_reprojected = substations_gdf.to_crs(src.crs)

            coords = [
                (geom.x, geom.y)
                for geom in gdf_reprojected.geometry
                if geom is not None and not geom.is_empty
            ]

            if not coords:
                return np.zeros(len(substations_gdf), dtype=np.float64)

            sampled = np.array(
                [float(val[0]) if not src.nodata or val[0] != src.nodata else 0.0
                 for val in src.sample(coords, indexes=band)],
                dtype=np.float64,
            )

        return sampled

    # ------------------------------------------------------------------
    # Time-series evaluation
    # ------------------------------------------------------------------

    def evaluate_timeseries(
        self,
        substation_id: str,
        flood_depths_m: np.ndarray,
        ffe_m: Optional[float] = None,
        levee_height_m: Optional[float] = None,
        pump_rate_mps: Optional[float] = None,
        flood_duration_s: Optional[float] = None,
    ) -> pd.DataFrame:
        """Evaluate flood vulnerability across a time series of depths.

        Parameters
        ----------
        substation_id : str
            Identifier for the substation.
        flood_depths_m : np.ndarray
            Flood depth at each time step (m).
        ffe_m, levee_height_m, pump_rate_mps, flood_duration_s :
            See :meth:`evaluate_substation`.

        Returns
        -------
        pd.DataFrame
            Time series with columns from :meth:`evaluate_substation`
            plus ``timestep``.
        """
        ffe = float(ffe_m if ffe_m is not None else self.default_ffe_m)
        levee = float(levee_height_m if levee_height_m is not None else self.default_levee_height_m)
        pump = float(pump_rate_mps if pump_rate_mps is not None else self.default_pump_rate_mps)
        dt = float(flood_duration_s if flood_duration_s is not None else self.default_flood_duration_s)

        depths = np.asarray(flood_depths_m, dtype=np.float64)
        n = len(depths)

        pump_arr = np.full(n, pump, dtype=np.float64)
        ffe_arr = np.full(n, ffe, dtype=np.float64)
        levee_arr = np.full(n, levee, dtype=np.float64)

        d_eff = self._effective_depth(depths, pump_arr, dt)
        p_f = self._failure_probability(d_eff, ffe_arr, levee_arr, self.gamma, self.max_exponent)

        return pd.DataFrame(
            {
                "substation_id": [substation_id] * n,
                "timestep": np.arange(n),
                "raw_depth_m": depths,
                "effective_depth_m": d_eff,
                "failure_probability": p_f,
                "operational": p_f < 0.5,
            }
        )

    def __repr__(self) -> str:
        return (
            f"SubstationFlooder(γ={self.gamma:.2f}, "
            f"FFE={self.default_ffe_m:.2f}m, "
            f"levee={self.default_levee_height_m:.2f}m, "
            f"pump={self.default_pump_rate_mps:.4f}m/s)"
        )
