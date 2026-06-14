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
Thermodynamic wire-icing simulator.

Provides ``MakkonenIcer``, a vectorised time-series ice accretion model
for overhead line conductors based on the Makkonen (2000) thermodynamic
framework.  The solver performs numerical integration across hourly
weather records, computing collision efficiency, accretion efficiency
via surface heat balance, dynamic conductor diameter, and mechanical
failure probability under combined ice and wind loading.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.constants import Stefan_Boltzmann, g

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_KELVIN_OFFSET: float = 273.15

_LATENT_HEAT_FUSION: float = 3.34e5  # J/kg
_SPECIFIC_HEAT_WATER: float = 4186.0  # J/(kg·K)
_SPECIFIC_HEAT_AIR: float = 1005.0  # J/(kg·K)
_GAS_CONSTANT_DRY_AIR: float = 287.058  # J/(kg·K)
_GAS_CONSTANT_WATER_VAPOUR: float = 461.5  # J/(kg·K)

_DEFAULT_ICE_DENSITY: Dict[str, float] = {
    "glaze": 917.0,
    "rime": 500.0,
}

_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# MakkonenIcer
# ---------------------------------------------------------------------------


class MakkonenIcer:
    """Time-series ice accretion simulator for overhead line conductors.

    Implements the Makkonen (2000) thermodynamic model for atmospheric
    icing on cylindrical structures.  The solver numerically integrates
    the ice mass accumulation rate:

    .. math::

        \\frac{dM}{dt} = \\alpha_1 \\alpha_2 \\alpha_3 w v D(t)

    where :math:`\\alpha_1` is collision efficiency, :math:`\\alpha_2`
    is sticking efficiency, :math:`\\alpha_3` is accretion (freezing)
    efficiency, :math:`w` is liquid water content, :math:`v` is wind
    speed, and :math:`D(t)` is the dynamic iced-conductor diameter.

    Parameters
    ----------
    conductor_diameter_m : float
        Bare conductor diameter in metres.  Default 0.0281 (ACSR
        "Drake").
    ice_type : str
        ``"glaze"`` (ρ = 917 kg/m³) or ``"rime"`` (ρ = 500 kg/m³).
        Default ``"glaze"``.
    ice_density_kg_m3 : float or None
        Explicit ice density override.  If provided, *ice_type* is
        ignored.
    span_length_m : float
        Span length between towers in metres.  Default 300.
    max_tension_n : float
        Maximum allowable conductor tension in Newtons.  Default
        70 000 (typical for ACSR Drake).
    conductor_mass_per_m : float
        Bare conductor mass per unit length in kg/m.  Default 1.628
        (ACSR Drake).
    droplet_mvd_m : float
        Median Volume Diameter of supercooled droplets in metres.
        Default 20e-6 (20 µm).
    sticking_efficiency : float
        Collection/sticking efficiency :math:`\\alpha_2`.  Default 1.0
        (supercooled water / wet snow).
    surface_emissivity : float
        Conductor surface emissivity for radiative cooling.  Default
        0.9.
    surface_roughness_m : float
        Equivalent sand-grain roughness in metres for convective heat
        transfer.  Default 5e-4.

    Attributes
    ----------
    conductor_diameter_m : float
    ice_density : float
    span_length_m : float
    max_tension_n : float
    conductor_mass_per_m : float
    droplet_mvd : float
    alpha_2 : float
    emissivity : float
    roughness_m : float
    result_ : pd.DataFrame or None
        Populated after :meth:`run`.
    """

    def __init__(
        self,
        conductor_diameter_m: float = 0.0281,
        ice_type: str = "glaze",
        ice_density_kg_m3: Optional[float] = None,
        span_length_m: float = 300.0,
        max_tension_n: float = 70_000.0,
        conductor_mass_per_m: float = 1.628,
        droplet_mvd_m: float = 20e-6,
        sticking_efficiency: float = 1.0,
        surface_emissivity: float = 0.9,
        surface_roughness_m: float = 5e-4,
    ) -> None:
        if conductor_diameter_m <= 0:
            raise ValueError(
                f"conductor_diameter_m must be positive, got {conductor_diameter_m}"
            )

        self.conductor_diameter_m = float(conductor_diameter_m)

        if ice_density_kg_m3 is not None:
            self.ice_density = float(ice_density_kg_m3)
        elif ice_type in _DEFAULT_ICE_DENSITY:
            self.ice_density = _DEFAULT_ICE_DENSITY[ice_type]
        else:
            raise ValueError(
                f"Unknown ice_type '{ice_type}'.  Use 'glaze', 'rime', "
                f"or provide ice_density_kg_m3."
            )
        self.ice_type = ice_type

        if span_length_m <= 0:
            raise ValueError(f"span_length_m must be positive, got {span_length_m}")
        self.span_length_m = float(span_length_m)

        if max_tension_n <= 0:
            raise ValueError(f"max_tension_n must be positive, got {max_tension_n}")
        self.max_tension_n = float(max_tension_n)

        if conductor_mass_per_m < 0:
            raise ValueError(
                f"conductor_mass_per_m must be non-negative, got {conductor_mass_per_m}"
            )
        self.conductor_mass_per_m = float(conductor_mass_per_m)

        if droplet_mvd_m <= 0:
            raise ValueError(f"droplet_mvd_m must be positive, got {droplet_mvd_m}")
        self.droplet_mvd = float(droplet_mvd_m)

        if not (0 < sticking_efficiency <= 1):
            raise ValueError(
                f"sticking_efficiency must be in (0, 1], got {sticking_efficiency}"
            )
        self.alpha_2 = float(sticking_efficiency)

        if not (0 < surface_emissivity <= 1):
            raise ValueError(
                f"surface_emissivity must be in (0, 1], got {surface_emissivity}"
            )
        self.emissivity = float(surface_emissivity)

        if surface_roughness_m <= 0:
            raise ValueError(
                f"surface_roughness_m must be positive, got {surface_roughness_m}"
            )
        self.roughness_m = float(surface_roughness_m)

        self.result_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Collision efficiency α₁ — Finstad et al. (1988)
    # ------------------------------------------------------------------

    @staticmethod
    def _collision_efficiency(
        droplet_mvd: float,
        wind_speed: np.ndarray,
        diameter: np.ndarray,
        air_density: float = 1.225,
        air_dynamic_viscosity: float = 1.81e-5,
        water_density: float = 1000.0,
    ) -> np.ndarray:
        """Compute collision (impingement) efficiency α₁.

        Uses the Finstad et al. (1988) parameterisation for the
        collision efficiency of droplets on a cylinder in potential
        flow, as a function of the Stokes number *K* and the droplet
        Reynolds number *Re_d*.

        Parameters
        ----------
        droplet_mvd : float
            Median Volume Diameter (m).
        wind_speed : np.ndarray
            Wind speed perpendicular to conductor (m/s).
        diameter : np.ndarray
            Current conductor + ice diameter (m).
        air_density : float
            Air density (kg/m³).  Default 1.225.
        air_dynamic_viscosity : float
            Dynamic viscosity of air (Pa·s).  Default 1.81e-5.
        water_density : float
            Density of water droplets (kg/m³).  Default 1000.

        Returns
        -------
        np.ndarray
            Collision efficiency α₁ ∈ [0, 1].
        """
        K = (water_density * droplet_mvd**2 * wind_speed) / (
            9.0 * air_dynamic_viscosity * diameter
        )
        K = np.maximum(K, _EPS)

        Re_d = (air_density * droplet_mvd * wind_speed) / air_dynamic_viscosity
        Re_d = np.maximum(Re_d, _EPS)

        phi = (Re_d**2) / K
        phi = np.maximum(phi, _EPS)

        alpha = (
            1.0
            / (1.0 + 0.096 * phi**0.4987)
            * (1.0 - 0.0286 * np.exp(-0.0124 * phi**0.5))
        )

        return np.clip(alpha, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Accretion efficiency α₃ — heat balance
    # ------------------------------------------------------------------

    @staticmethod
    def _accretion_efficiency(
        T_air_c: np.ndarray,
        wind_speed: np.ndarray,
        diameter: np.ndarray,
        lwc_g_m3: np.ndarray,
        alpha_1: np.ndarray,
        pressure_pa: float = 101_325.0,
        relative_humidity: float = 0.85,
        emissivity: float = 0.9,
    ) -> np.ndarray:
        """Compute accretion (freezing) efficiency α₃ via heat balance.

        Solves the surface heat balance equation for a cylindrical
        conductor:

        .. math::

            Q_f = Q_c + Q_e + Q_r + Q_w

        where :math:`Q_f` is latent heat released by freezing,
        :math:`Q_c` is convective heat loss, :math:`Q_e` is evaporative
        cooling, :math:`Q_r` is radiative cooling, and :math:`Q_w` is
        the heat required to warm impinging droplets to 0 °C.

        Parameters
        ----------
        T_air_c : np.ndarray
            Air temperature in °C.
        wind_speed : np.ndarray
            Wind speed (m/s).
        diameter : np.ndarray
            Conductor + ice diameter (m).
        lwc_g_m3 : np.ndarray
            Liquid water content (g/m³).
        alpha_1 : np.ndarray
            Collision efficiency.
        pressure_pa : float
            Atmospheric pressure (Pa).  Default 101 325.
        relative_humidity : float
            Relative humidity (0–1).  Default 0.85.

        Returns
        -------
        np.ndarray
            Accretion efficiency α₃ ∈ [0, 1].
        """
        T_k = T_air_c + _KELVIN_OFFSET

        # --- Convective heat transfer coefficient ---
        k_air = 2.42e-2 + 7.2e-5 * T_air_c  # W/(m·K)
        nu_air = 1.32e-5 + 9.5e-8 * T_air_c  # m²/s
        Re = wind_speed * diameter / np.maximum(nu_air, _EPS)
        Pr = 0.71  # Prandtl number for air

        Nu = 0.032 * Re**0.85  # turbulent cylinder correlation
        h_c = Nu * k_air / diameter  # W/(m²·K)

        # --- Convective heat loss (W/m²) ---
        Q_c = h_c * (0.0 - T_air_c)  # surface at 0 °C

        # --- Evaporative cooling (W/m²) ---
        e_sat_0 = 611.2  # saturation vapour pressure at 0 °C (Pa)
        e_sat_air = 611.2 * np.exp(
            17.67 * T_air_c / (T_air_c + 243.5)
        )
        e_sat_air = np.where(T_air_c > 0, e_sat_air, e_sat_0)

        e_air = relative_humidity * e_sat_air

        h_e = 0.622 * h_c / (_SPECIFIC_HEAT_AIR * pressure_pa)  # kg/(m²·s·Pa)
        Q_e = h_e * _LATENT_HEAT_FUSION * (e_sat_0 - e_air)

        # --- Radiative cooling (W/m²) ---
        Q_r = Stefan_Boltzmann * emissivity * (T_k**4 - (_KELVIN_OFFSET) ** 4)

        # --- Droplet warming (W/m²) ---
        lwc_kg_m3 = lwc_g_m3 / 1000.0
        impingement_flux = alpha_1 * lwc_kg_m3 * wind_speed  # kg/(m²·s)
        Q_w = impingement_flux * _SPECIFIC_HEAT_WATER * (0.0 - T_air_c)

        # --- Freezing heat source (W/m²) ---
        Q_f_max = impingement_flux * _LATENT_HEAT_FUSION

        # --- Heat balance ---
        Q_cooling = Q_c + Q_e + Q_r + Q_w
        Q_cooling = np.maximum(Q_cooling, _EPS)

        alpha_3 = Q_cooling / np.maximum(Q_f_max, _EPS)

        return np.clip(alpha_3, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Main integration loop
    # ------------------------------------------------------------------

    def run(
        self,
        weather_df: pd.DataFrame,
        time_col: str = "timestamp",
        temp_col: str = "temperature_c",
        wind_col: str = "wind_speed_mps",
        lwc_col: str = "liquid_water_content_g_m3",
        pressure_col: Optional[str] = None,
        humidity_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Execute time-series ice accretion integration.

        Performs forward Euler integration of the Makkonen equation
        across the provided hourly weather records.

        Parameters
        ----------
        weather_df : pd.DataFrame
            Hourly weather data.  Must contain columns for temperature,
            wind speed, and liquid water content.
        time_col : str
            Column name for timestamps.  Default ``"timestamp"``.
        temp_col : str
            Column name for air temperature in °C.
        wind_col : str
            Column name for wind speed in m/s.
        lwc_col : str
            Column name for liquid water content in g/m³.
        pressure_col : str or None
            Column name for atmospheric pressure in Pa.  If ``None``,
            uses standard sea-level pressure.
        humidity_col : str or None
            Column name for relative humidity (0–1).  If ``None``,
            defaults to 0.85.

        Returns
        -------
        pd.DataFrame
            Time series with columns: ``ice_mass_kg_per_m``,
            ``diameter_m``, ``alpha_1``, ``alpha_3``, ``ice_thickness_m``,
            ``total_mass_kg``, ``tension_n``, ``tension_ratio``.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        n = len(weather_df)
        if n == 0:
            raise ValueError("weather_df is empty")

        for col, name in [
            (temp_col, "temperature"),
            (wind_col, "wind speed"),
            (lwc_col, "liquid water content"),
        ]:
            if col not in weather_df.columns:
                raise ValueError(f"Missing required column '{col}' ({name})")

        T = weather_df[temp_col].values.astype(np.float64)
        V = weather_df[wind_col].values.astype(np.float64)
        LWC = weather_df[lwc_col].values.astype(np.float64)

        if pressure_col and pressure_col in weather_df.columns:
            P = weather_df[pressure_col].values.astype(np.float64)
        else:
            P = np.full(n, 101_325.0, dtype=np.float64)

        if humidity_col and humidity_col in weather_df.columns:
            RH = weather_df[humidity_col].values.astype(np.float64)
        else:
            RH = np.full(n, 0.85, dtype=np.float64)

        timestamps = weather_df[time_col].values if time_col in weather_df.columns else np.arange(n)

        dt_seconds = self._infer_timestep(timestamps)

        M = np.zeros(n + 1, dtype=np.float64)
        D = np.full(n + 1, self.conductor_diameter_m, dtype=np.float64)
        alpha_1_arr = np.zeros(n, dtype=np.float64)
        alpha_3_arr = np.zeros(n, dtype=np.float64)

        for i in range(n):
            D_current = D[i]

            alpha_1_arr[i] = 1.0
            if V[i] > 0.1 and LWC[i] > _EPS and T[i] < 0.5:
                alpha_1_arr[i] = float(
                    self._collision_efficiency(
                        self.droplet_mvd,
                        V[i],
                        D_current,
                    )
                )

            alpha_3_arr[i] = 1.0
            if V[i] > 0.1 and LWC[i] > _EPS and T[i] < 0.5:
                alpha_3_arr[i] = float(
                    self._accretion_efficiency(
                        T[i],
                        V[i],
                        D_current,
                        LWC[i],
                        alpha_1_arr[i],
                        pressure_pa=float(P[i]),
                        relative_humidity=float(RH[i]),
                        emissivity=self.emissivity,
                    )
                )

            dM_dt = (
                alpha_1_arr[i]
                * self.alpha_2
                * alpha_3_arr[i]
                * (LWC[i] / 1000.0)
                * V[i]
                * D_current
            )

            M[i + 1] = M[i] + dM_dt * dt_seconds

            D[i + 1] = np.sqrt(
                self.conductor_diameter_m**2
                + 4.0 * M[i + 1] / (np.pi * self.ice_density)
            )

        ice_thickness = (D[1:] - self.conductor_diameter_m) / 2.0
        total_mass_per_m = self.conductor_mass_per_m + M[1:]
        total_weight_per_m = total_mass_per_m * g

        tension_n = total_weight_per_m * self.span_length_m**2 / (8.0 * 1.0)
        tension_ratio = tension_n / self.max_tension_n

        df = pd.DataFrame(
            {
                "ice_mass_kg_per_m": M[1:],
                "diameter_m": D[1:],
                "alpha_1": alpha_1_arr,
                "alpha_3": alpha_3_arr,
                "ice_thickness_m": ice_thickness,
                "total_mass_kg_per_m": total_mass_per_m,
                "tension_n": tension_n,
                "tension_ratio": tension_ratio,
            }
        )

        if time_col in weather_df.columns:
            df.insert(0, time_col, weather_df[time_col].values)

        self.result_ = df
        return df

    # ------------------------------------------------------------------
    # Mechanical failure probability
    # ------------------------------------------------------------------

    def calculate_ice_failure_threshold(
        self,
        gust_wind_speed_mps: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute mechanical failure probability under ice + wind load.

        Evaluates whether the combined weight of the conductor and
        accumulated ice, together with concurrent wind gust loading,
        exceeds the structural tension limit of the span.

        The total horizontal tension is estimated as:

        .. math::

            T = \\frac{w_{\\text{total}} L^2}{8 d}

        where :math:`w_{\\text{total}}` includes the vertical weight of
        conductor + ice and the horizontal wind pressure on the iced
        cross-section.

        Parameters
        ----------
        gust_wind_speed_mps : float
            Concurrent gust wind speed in m/s.  Default 0.

        Returns
        -------
        dict
            Keys:

            - ``failure_probability`` (float) — 0 or 1 (deterministic
              threshold).
            - ``max_tension_n`` (float) — peak tension over the
              simulation.
            - ``tension_limit_n`` (float) — allowable tension.
            - ``safety_factor`` (float) — limit / peak tension.
            - ``max_ice_thickness_m`` (float).
            - ``max_ice_mass_kg_per_m`` (float).
        """
        if self.result_ is None:
            raise RuntimeError("Call run() before calculate_ice_failure_threshold().")

        max_ice_mass = float(self.result_["ice_mass_kg_per_m"].max())
        max_thickness = float(self.result_["ice_thickness_m"].max())
        max_diameter = float(self.result_["diameter_m"].max())

        total_mass = self.conductor_mass_per_m + max_ice_mass
        vertical_weight = total_mass * g  # N/m

        if gust_wind_speed_mps > 0:
            air_density = 1.225
            drag_coeff = 1.0
            wind_pressure = 0.5 * air_density * drag_coeff * gust_wind_speed_mps**2
            wind_force_per_m = wind_pressure * max_diameter
            resultant_force = np.sqrt(vertical_weight**2 + wind_force_per_m**2)
        else:
            resultant_force = vertical_weight

        sag_m = self.span_length_m * 0.02
        peak_tension = resultant_force * self.span_length_m**2 / (8.0 * sag_m)

        safety_factor = self.max_tension_n / peak_tension if peak_tension > _EPS else float("inf")
        failed = peak_tension > self.max_tension_n

        return {
            "failure_probability": 1.0 if failed else 0.0,
            "max_tension_n": float(peak_tension),
            "tension_limit_n": self.max_tension_n,
            "safety_factor": float(safety_factor),
            "max_ice_thickness_m": max_thickness,
            "max_ice_mass_kg_per_m": max_ice_mass,
            "gust_wind_speed_mps": gust_wind_speed_mps,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_timestep(timestamps: np.ndarray) -> float:
        """Infer time step in seconds from timestamp array.

        Parameters
        ----------
        timestamps : np.ndarray

        Returns
        -------
        float
            Time step in seconds.  Defaults to 3600 if inference fails.
        """
        if len(timestamps) < 2:
            return 3600.0

        try:
            ts = pd.to_datetime(timestamps)
            dt = (ts[1] - ts[0]).total_seconds()
            if dt > 0:
                return float(dt)
        except Exception:
            pass

        return 3600.0

    def __repr__(self) -> str:
        return (
            f"MakkonenIcer(D₀={self.conductor_diameter_m * 1000:.1f}mm, "
            f"ice={self.ice_type} ρ={self.ice_density:.0f}kg/m³, "
            f"span={self.span_length_m:.0f}m, "
            f"T_max={self.max_tension_n / 1000:.1f}kN)"
        )
