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
Physical tropical cyclone wind-field simulator.

Implements the Batts gradient wind equation with Holland pressure
profile, empirical radius-of-maximum-wind and B-parameter formulas,
height-conversion scaling, and a circular sub-region method for
evaluating wind-speed time series at grid asset locations along a
storm track.

Reference
---------
* Batts, M. E., M. R. Cordes, L. R. Russell, J. R. Shaver, and
  E. Simiu (1980).  Hurricane wind speeds in the United States.
  NBS Building Science Series 124.
* Holland, G. J. (1980).  An analytic model of the wind and pressure
  profiles in hurricanes.  *Monthly Weather Review*, 108(8),
  1212–1218.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_RHO_AIR: float = 1.15  # kg/m³ — standard near-surface air density
_OMEGA_EARTH: float = 7.2921e-5  # rad/s — Earth angular velocity
_P_AMBIENT_HPA: float = 1013.25  # hPa — standard ambient surface pressure
_H_G: float = 10.0  # m — gradient height reference
_EPS: float = 1e-12

# Default surface roughness and wind shear
_DEFAULT_K_S: float = 0.9
_DEFAULT_ALPHA_SHEAR: float = 0.143  # open terrain (ASCE 7)

# Default buffer radius for asset exposure
_DEFAULT_BUFFER_RADIUS_KM: float = 200.0


# ---------------------------------------------------------------------------
# TyphoonWindSimulator
# ---------------------------------------------------------------------------


class TyphoonWindSimulator:
    """Tropical cyclone gradient wind-field simulator.

    Computes the Batts gradient wind speed at arbitrary distances from
    a storm centre using the Holland pressure profile, scales wind
    speeds to engineering heights via a power-law shear profile, and
    evaluates hourly wind-speed time series at grid asset locations
    using a circular sub-region intersection method.

    Parameters
    ----------
    rho_air : float
        Near-surface air density in kg/m³.  Default 1.15.
    p_ambient_hpa : float
        Ambient surface pressure far from the storm in hPa.
        Default 1013.25.
    k_s : float
        Surface roughness correction coefficient (0–1).  Default 0.9.
    alpha_shear : float
        Wind shear exponent for power-law height profile.
        Default 0.143 (open terrain, ASCE 7).
    h_g : float
        Gradient height reference in metres.  Default 10.0.
    buffer_radius_km : float
        Default buffer circle radius in km for asset exposure
        evaluation.  Default 200.0.

    Attributes
    ----------
    rho_air : float
    p_ambient_hpa : float
    k_s : float
    alpha_shear : float
    h_g : float
    buffer_radius_km : float
    """

    def __init__(
        self,
        rho_air: float = _RHO_AIR,
        p_ambient_hpa: float = _P_AMBIENT_HPA,
        k_s: float = _DEFAULT_K_S,
        alpha_shear: float = _DEFAULT_ALPHA_SHEAR,
        h_g: float = _H_G,
        buffer_radius_km: float = _DEFAULT_BUFFER_RADIUS_KM,
    ) -> None:
        if rho_air <= 0:
            raise ValueError(f"rho_air must be positive, got {rho_air}")
        if p_ambient_hpa <= 0:
            raise ValueError(f"p_ambient_hpa must be positive, got {p_ambient_hpa}")
        if not (0 < k_s <= 1):
            raise ValueError(f"k_s must be in (0, 1], got {k_s}")
        if alpha_shear <= 0:
            raise ValueError(f"alpha_shear must be positive, got {alpha_shear}")
        if h_g <= 0:
            raise ValueError(f"h_g must be positive, got {h_g}")
        if buffer_radius_km <= 0:
            raise ValueError(f"buffer_radius_km must be positive, got {buffer_radius_km}")

        self.rho_air = float(rho_air)
        self.p_ambient_hpa = float(p_ambient_hpa)
        self.k_s = float(k_s)
        self.alpha_shear = float(alpha_shear)
        self.h_g = float(h_g)
        self.buffer_radius_km = float(buffer_radius_km)

    # ------------------------------------------------------------------
    # Empirical parameterisation
    # ------------------------------------------------------------------

    @staticmethod
    def holland_b(pc: Union[float, np.ndarray]) -> np.ndarray:
        r"""Holland pressure-profile shape parameter *B*.

        .. math::

            B = 1.5 + \frac{980 - p_c}{120}

        Parameters
        ----------
        pc : float or np.ndarray
            Central pressure in hPa.

        Returns
        -------
        np.ndarray
            Holland *B* parameter (dimensionless, typically 1.0–2.5).
        """
        pc = np.asarray(pc, dtype=np.float64)
        return 1.5 + (980.0 - pc) / 120.0

    @staticmethod
    def radius_of_maximum_wind(
        pc: Union[float, np.ndarray],
        lat: Union[float, np.ndarray],
        vt: Union[float, np.ndarray] = 0.0,
    ) -> np.ndarray:
        r"""Empirical radius of maximum wind speed :math:`R_{\max}`.

        .. math::

            R_{\max} = 28.52 \tanh\!\bigl(0.0873(\varphi - 28)\bigr)
            + 12.22 \exp\!\left(\frac{p_c - 1013.2}{33.86}\right)
            + 0.2 v_t + 37.22

        where :math:`\varphi` is latitude in degrees and :math:`v_t`
        is the storm translation speed in km/h.

        Parameters
        ----------
        pc : float or np.ndarray
            Central pressure in hPa.
        lat : float or np.ndarray
            Latitude in degrees.
        vt : float or np.ndarray
            Storm translation speed in km/h.  Default 0.

        Returns
        -------
        np.ndarray
            Radius of maximum wind in km.
        """
        pc = np.asarray(pc, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        vt = np.asarray(vt, dtype=np.float64)
        return (
            28.52 * np.tanh(0.0873 * (lat - 28.0))
            + 12.22 * np.exp((pc - 1013.2) / 33.86)
            + 0.2 * vt
            + 37.22
        )

    # ------------------------------------------------------------------
    # Coriolis parameter
    # ------------------------------------------------------------------

    @staticmethod
    def coriolis(lat: Union[float, np.ndarray]) -> np.ndarray:
        r"""Coriolis parameter :math:`f = 2\Omega \sin\varphi`.

        Parameters
        ----------
        lat : float or np.ndarray
            Latitude in degrees.

        Returns
        -------
        np.ndarray
            Coriolis parameter in rad/s.
        """
        lat = np.asarray(lat, dtype=np.float64)
        return 2.0 * _OMEGA_EARTH * np.sin(np.radians(lat))

    # ------------------------------------------------------------------
    # Gradient wind speed
    # ------------------------------------------------------------------

    def calculate_gradient_wind_speed(
        self,
        r: Union[float, np.ndarray],
        pc: float,
        vt: float = 0.0,
        lat: float = 25.0,
    ) -> np.ndarray:
        r"""Batts gradient wind speed at radial distance *r*.

        .. math::

            V = \sqrt{\frac{\Delta p}{\rho} \frac{R}{r} B
            \exp\!\left(-\left(\frac{R}{r}\right)^B\right)
            + \frac{r^2 f^2}{4}} - \frac{r f}{2}

        where :math:`\Delta p = p_{\text{ambient}} - p_c`,
        :math:`R = R_{\max}` is computed via
        :meth:`radius_of_maximum_wind`, and :math:`B` via
        :meth:`holland_b`.

        Parameters
        ----------
        r : float or np.ndarray
            Radial distance(s) from storm centre in km.
        pc : float
            Central pressure in hPa.
        vt : float
            Storm translation speed in km/h.  Default 0.
        lat : float
            Latitude of storm centre in degrees.  Default 25.

        Returns
        -------
        np.ndarray
            Gradient wind speed in m/s at each *r*.
        """
        r = np.asarray(r, dtype=np.float64)
        dp = (self.p_ambient_hpa - pc) * 100.0  # Pa

        B = float(self.holland_b(pc))
        R = float(self.radius_of_maximum_wind(pc, lat, vt))
        f = float(self.coriolis(lat))

        r_m = r * 1000.0  # km → m
        R_m = R * 1000.0  # km → m

        # ratio = R / r, guarded against division by zero
        ratio = np.where(r > _EPS, R_m / r_m, np.inf)
        exp_term = np.exp(-(ratio ** B))

        term1 = (dp / self.rho_air) * (R_m / r_m) * B * exp_term
        term2 = (r_m * f / 2.0) ** 2

        v = np.sqrt(np.maximum(0.0, term1 + term2)) - r_m * f / 2.0
        return np.maximum(0.0, v)

    # ------------------------------------------------------------------
    # Height conversion
    # ------------------------------------------------------------------

    def height_convert(
        self,
        v_gradient: Union[float, np.ndarray],
        h: float,
    ) -> np.ndarray:
        r"""Scale gradient wind speed to height *h*.

        .. math::

            v_h = V \cdot k_s \cdot \left(\frac{h}{h_g}\right)^{\alpha}

        Parameters
        ----------
        v_gradient : float or np.ndarray
            Gradient wind speed(s) in m/s.
        h : float
            Target height in metres (e.g. hub height, pylon height).

        Returns
        -------
        np.ndarray
            Wind speed at height *h* in m/s.
        """
        v_gradient = np.asarray(v_gradient, dtype=np.float64)
        return v_gradient * self.k_s * (h / self.h_g) ** self.alpha_shear

    # ------------------------------------------------------------------
    # Circular sub-region asset exposure
    # ------------------------------------------------------------------

    def evaluate_asset_exposure(
        self,
        asset_lon: float,
        asset_lat: float,
        track_df: pd.DataFrame,
        target_height_m: float = 10.0,
        buffer_radius_km: Optional[float] = None,
    ) -> pd.DataFrame:
        """Evaluate hourly wind-speed series at an asset location.

        Defines a buffer circle of radius *buffer_radius_km* around the
        asset.  For each storm track point whose centre falls within
        the circle, computes the gradient wind speed at the asset's
        radial distance, scales it to *target_height_m*, and returns
        a time series.

        Parameters
        ----------
        asset_lon : float
            Asset longitude (degrees).
        asset_lat : float
            Asset latitude (degrees).
        track_df : pd.DataFrame
            Storm track with columns:

            - ``lon`` (float) — storm centre longitude.
            - ``lat`` (float) — storm centre latitude.
            - ``pc`` (float) — central pressure in hPa.
            - ``vt`` (float, optional) — translation speed in km/h.
            - ``time`` (optional) — timestamp.
        target_height_m : float
            Target height for wind speed in metres.  Default 10.
        buffer_radius_km : float or None
            Buffer circle radius in km.  If ``None``, uses
            ``self.buffer_radius_km``.

        Returns
        -------
        pd.DataFrame
            Columns: ``time``, ``distance_km``, ``gradient_wind_ms``,
            ``wind_speed_ms``, ``pc``, ``lat``, ``lon``.
            Only rows where the storm centre is within the buffer
            circle.
        """
        buf = buffer_radius_km if buffer_radius_km is not None else self.buffer_radius_km

        if len(track_df) == 0:
            return pd.DataFrame(
                columns=["time", "distance_km", "gradient_wind_ms",
                         "wind_speed_ms", "pc", "lat", "lon"]
            )

        required = {"lon", "lat", "pc"}
        missing = required - set(track_df.columns)
        if missing:
            raise ValueError(f"track_df missing required columns: {missing}")

        lons = track_df["lon"].values.astype(np.float64)
        lats = track_df["lat"].values.astype(np.float64)
        pcs = track_df["pc"].values.astype(np.float64)
        vts = track_df["vt"].values.astype(np.float64) if "vt" in track_df.columns else np.zeros(len(track_df))

        # Haversine distances from asset to each track point (vectorised)
        distances_km = _haversine_km(asset_lon, asset_lat, lons, lats)

        # Mask: storm centre within buffer circle
        mask = distances_km <= buf
        if not np.any(mask):
            return pd.DataFrame(
                columns=["time", "distance_km", "gradient_wind_ms",
                         "wind_speed_ms", "pc", "lat", "lon"]
            )

        idx = np.where(mask)[0]
        r_km = distances_km[idx]
        pc_vals = pcs[idx]
        lat_vals = lats[idx]
        vt_vals = vts[idx]

        # Compute gradient wind speed for each qualifying track point
        v_grad = np.empty(len(idx), dtype=np.float64)
        for i in range(len(idx)):
            v_grad[i] = float(
                self.calculate_gradient_wind_speed(
                    r_km[i], float(pc_vals[i]), float(vt_vals[i]), float(lat_vals[i])
                )
            )

        # Height conversion
        v_h = self.height_convert(v_grad, target_height_m)

        result = pd.DataFrame({
            "distance_km": r_km,
            "gradient_wind_ms": v_grad,
            "wind_speed_ms": v_h,
            "pc": pc_vals,
            "lat": lat_vals,
            "lon": lons[idx],
        })

        if "time" in track_df.columns:
            result.insert(0, "time", track_df["time"].values[idx])

        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Batch asset exposure
    # ------------------------------------------------------------------

    def evaluate_assets(
        self,
        assets: List[Dict[str, Any]],
        track_df: pd.DataFrame,
        target_height_m: float = 10.0,
        buffer_radius_km: Optional[float] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Evaluate wind-speed time series for multiple assets.

        Parameters
        ----------
        assets : list of dict
            Each dict must have ``"lon"`` and ``"lat"`` keys.
            Optionally ``"id"`` for identification.
        track_df : pd.DataFrame
            See :meth:`evaluate_asset_exposure`.
        target_height_m : float
            Target height in metres.  Default 10.
        buffer_radius_km : float or None
            Buffer circle radius in km.

        Returns
        -------
        dict
            Mapping ``asset_id → DataFrame``.  Assets with no exposure
            are omitted.
        """
        result: Dict[str, pd.DataFrame] = {}
        for i, asset in enumerate(assets):
            aid = asset.get("id", f"asset_{i}")
            df = self.evaluate_asset_exposure(
                asset["lon"], asset["lat"],
                track_df,
                target_height_m=target_height_m,
                buffer_radius_km=buffer_radius_km,
            )
            if len(df) > 0:
                result[aid] = df
        return result

    # ------------------------------------------------------------------
    # Full wind-field profile
    # ------------------------------------------------------------------

    def wind_profile(
        self,
        pc: float,
        vt: float = 0.0,
        lat: float = 25.0,
        r_max_km: float = 300.0,
        n_points: int = 200,
    ) -> pd.DataFrame:
        """Compute radial wind-speed profile from storm centre.

        Parameters
        ----------
        pc : float
            Central pressure in hPa.
        vt : float
            Translation speed in km/h.
        lat : float
            Latitude in degrees.
        r_max_km : float
            Maximum radial distance in km.
        n_points : int
            Number of radial sample points.

        Returns
        -------
        pd.DataFrame
            Columns: ``r_km``, ``gradient_wind_ms``.
        """
        r = np.linspace(_EPS, r_max_km, n_points, dtype=np.float64)
        v = self.calculate_gradient_wind_speed(r, pc, vt, lat)
        return pd.DataFrame({"r_km": r, "gradient_wind_ms": v})

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TyphoonWindSimulator(ρ={self.rho_air:.2f}kg/m³, "
            f"k_s={self.k_s:.2f}, α={self.alpha_shear:.3f}, "
            f"buffer={self.buffer_radius_km:.0f}km)"
        )


# ---------------------------------------------------------------------------
# Vectorised haversine distance
# ---------------------------------------------------------------------------


def _haversine_km(
    lon0: float,
    lat0: float,
    lons: np.ndarray,
    lats: np.ndarray,
) -> np.ndarray:
    """Haversine great-circle distance in km (vectorised).

    Parameters
    ----------
    lon0, lat0 : float
        Reference point in degrees.
    lons, lats : np.ndarray
        Target points in degrees.

    Returns
    -------
    np.ndarray
        Distances in km.
    """
    R_earth = 6371.0
    dlat = np.radians(lats - lat0)
    dlon = np.radians(lons - lon0)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat0)) * np.cos(np.radians(lats)) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R_earth * c
