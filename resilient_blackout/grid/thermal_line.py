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
Dynamic Line Rating (DLR) and thermal margin engine.

Implements the IEEE 738 standard for steady-state conductor thermal rating,
a transient thermal inertia model for cascading-overload response windows,
and a ``DLRGridController`` that dynamically overwrites pandapower line
thermal capacities based on geographically coincident microclimatic
conditions.

All computations are fully vectorised for high-performance Monte Carlo
simulations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from resilient_blackout.grid.network import GridModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_STEFAN_BOLTZMANN: float = 5.670367e-8  # W·m⁻²·K⁻⁴
_KELVIN_OFFSET: float = 273.15

# Default conductor properties (ACSR "Drake" or similar)
_DEFAULT_DIAMETER_M: float = 0.0281
_DEFAULT_EMISSIVITY: float = 0.7
_DEFAULT_ABSORPTIVITY: float = 0.7
_DEFAULT_MAX_COND_TEMP_C: float = 100.0
_DEFAULT_RESISTANCE_OHM_PER_KM: float = 0.05

# Air properties at film temperature ~50°C
_AIR_THERMAL_CONDUCTIVITY: float = 0.028  # W/(m·K)
_AIR_KINEMATIC_VISCOSITY: float = 1.8e-5  # m²/s

# Thermal mass defaults (J/(kg·K) * kg/m → J/(K·m))
_DEFAULT_THERMAL_CAPACITY: float = 500.0  # J/(K·m) for ACSR


# ---------------------------------------------------------------------------
# IEEE 738 steady-state ampacity
# ---------------------------------------------------------------------------

def calculate_dynamic_ampacity(
    ambient_temp_c: np.ndarray | float,
    wind_speed_mps: np.ndarray | float,
    wind_angle_deg: np.ndarray | float = 90.0,
    solar_radiation_w_m2: np.ndarray | float = 1000.0,
    conductor_resistance_ohms_per_km: np.ndarray | float = _DEFAULT_RESISTANCE_OHM_PER_KM,
    max_cond_temp_c: float = _DEFAULT_MAX_COND_TEMP_C,
    conductor_diameter_m: float = _DEFAULT_DIAMETER_M,
    emissivity: float = _DEFAULT_EMISSIVITY,
    absorptivity: float = _DEFAULT_ABSORPTIVITY,
) -> np.ndarray:
    """Compute the IEEE 738 steady-state dynamic ampacity.

    Solves the heat-balance equation:

    .. math::

        q_c + q_r = q_s + I^2 R(T_{\\text{max}})

    where :math:`q_c` is convective cooling, :math:`q_r` is radiative
    cooling, and :math:`q_s` is solar heat gain.

    Parameters
    ----------
    ambient_temp_c : float or np.ndarray
        Ambient air temperature in °C.
    wind_speed_mps : float or np.ndarray
        Wind speed in m/s.
    wind_angle_deg : float or np.ndarray
        Angle between wind direction and conductor axis in degrees.
        Default 90° (perpendicular).
    solar_radiation_w_m2 : float or np.ndarray
        Total solar radiation in W/m².  Default 1000.
    conductor_resistance_ohms_per_km : float or np.ndarray
        AC resistance at ``max_cond_temp_c`` in Ω/km.
    max_cond_temp_c : float
        Maximum allowable conductor temperature in °C.  Default 100.
    conductor_diameter_m : float
        Conductor outer diameter in metres.  Default 0.0281 (ACSR Drake).
    emissivity : float
        Radiative emissivity (0–1).  Default 0.7.
    absorptivity : float
        Solar absorptivity (0–1).  Default 0.7.

    Returns
    -------
    np.ndarray
        Dynamic ampacity in Amperes.  Scalar input returns 0-d array.
    """
    T_a = np.asarray(ambient_temp_c, dtype=np.float64) + _KELVIN_OFFSET
    T_c = max_cond_temp_c + _KELVIN_OFFSET
    V_w = np.maximum(np.asarray(wind_speed_mps, dtype=np.float64), 0.0)
    phi = np.asarray(wind_angle_deg, dtype=np.float64)
    Q_s = np.asarray(solar_radiation_w_m2, dtype=np.float64)
    R = np.asarray(conductor_resistance_ohms_per_km, dtype=np.float64) / 1000.0  # Ω/m
    D = conductor_diameter_m

    q_r = _radiative_cooling(T_c, T_a, D, emissivity)
    q_s = _solar_heat_gain(Q_s, D, absorptivity)
    q_c = _convective_cooling(T_c, T_a, D, V_w, phi)

    numerator = q_c + q_r - q_s
    numerator = np.maximum(numerator, 0.0)
    denominator = np.maximum(R, 1e-12)

    I = np.sqrt(numerator / denominator)
    return I


def _radiative_cooling(
    T_c: np.ndarray,
    T_a: np.ndarray,
    D: float,
    emissivity: float,
) -> np.ndarray:
    """Compute radiative cooling per unit length (W/m).

    Parameters
    ----------
    T_c : np.ndarray
        Conductor temperature (K).
    T_a : np.ndarray
        Ambient temperature (K).
    D : float
        Conductor diameter (m).
    emissivity : float

    Returns
    -------
    np.ndarray
        Radiative cooling in W/m.
    """
    area_per_m = np.pi * D
    return emissivity * _STEFAN_BOLTZMANN * area_per_m * (T_c**4 - T_a**4)


def _solar_heat_gain(
    Q_s: np.ndarray,
    D: float,
    absorptivity: float,
) -> np.ndarray:
    """Compute absorbed solar heat gain per unit length (W/m).

    Parameters
    ----------
    Q_s : np.ndarray
        Solar radiation (W/m²).
    D : float
        Conductor diameter (m).
    absorptivity : float

    Returns
    -------
    np.ndarray
        Solar heat gain in W/m.
    """
    return absorptivity * Q_s * D


def _convective_cooling(
    T_c: np.ndarray,
    T_a: np.ndarray,
    D: float,
    V_w: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Compute convective cooling per unit length (W/m).

    Uses natural convection for wind speeds < 0.5 m/s and forced
    convection otherwise, per IEEE 738.

    Parameters
    ----------
    T_c : np.ndarray
        Conductor temperature (K).
    T_a : np.ndarray
        Ambient temperature (K).
    D : float
        Conductor diameter (m).
    V_w : np.ndarray
        Wind speed (m/s).
    phi : np.ndarray
        Wind angle (degrees).

    Returns
    -------
    np.ndarray
        Convective cooling in W/m.
    """
    T_film = (T_c + T_a) / 2.0
    k_f = _air_thermal_conductivity(T_film)
    nu = _air_kinematic_viscosity(T_film)

    Re = V_w * D / np.maximum(nu, 1e-12)
    phi_rad = np.radians(phi)

    K_angle = 1.194 - np.cos(phi_rad) + 0.194 * np.cos(2 * phi_rad) + 0.368 * np.sin(2 * phi_rad)

    q_forced = K_angle * (1.01 + 1.35 * Re**0.52) * k_f * (T_c - T_a)
    q_natural = 3.645 * _air_density(T_film)**0.5 * D**0.75 * (T_c - T_a)**1.25

    return np.where(V_w >= 0.5, q_forced, np.maximum(q_forced, q_natural))


def _air_thermal_conductivity(T_film: np.ndarray) -> np.ndarray:
    """Temperature-dependent air thermal conductivity (W/(m·K)).

    Parameters
    ----------
    T_film : np.ndarray
        Film temperature (K).

    Returns
    -------
    np.ndarray
    """
    return 2.42e-2 + 7.2e-5 * (T_film - _KELVIN_OFFSET)


def _air_kinematic_viscosity(T_film: np.ndarray) -> np.ndarray:
    """Temperature-dependent air kinematic viscosity (m²/s).

    Parameters
    ----------
    T_film : np.ndarray
        Film temperature (K).

    Returns
    -------
    np.ndarray
    """
    return 1.32e-5 + 9.5e-8 * (T_film - _KELVIN_OFFSET)


def _air_density(T_film: np.ndarray) -> np.ndarray:
    """Approximate air density (kg/m³) at film temperature.

    Parameters
    ----------
    T_film : np.ndarray
        Film temperature (K).

    Returns
    -------
    np.ndarray
    """
    return 1.293 - 0.00425 * (T_film - _KELVIN_OFFSET) + 1.0e-5 * (T_film - _KELVIN_OFFSET)**2


# ---------------------------------------------------------------------------
# Transient conductor temperature (thermal inertia)
# ---------------------------------------------------------------------------

def transient_conductor_temperature(
    ambient_temp_c: float,
    wind_speed_mps: float,
    wind_angle_deg: float,
    solar_radiation_w_m2: float,
    current_a: float,
    conductor_resistance_ohms_per_km: float,
    initial_temp_c: float,
    duration_seconds: float,
    conductor_diameter_m: float = _DEFAULT_DIAMETER_M,
    emissivity: float = _DEFAULT_EMISSIVITY,
    absorptivity: float = _DEFAULT_ABSORPTIVITY,
    thermal_capacity_j_per_k_m: float = _DEFAULT_THERMAL_CAPACITY,
    time_step_s: float = 60.0,
) -> float:
    """Simulate transient conductor temperature under constant conditions.

    Models the 5–15 minute thermal lag using explicit Euler integration
    of the lumped-capacitance heat equation:

    .. math::

        \\frac{dT}{dt} = \\frac{q_s + I^2 R(T) - q_c(T) - q_r(T)}{m C_p}

    Parameters
    ----------
    ambient_temp_c : float
        Ambient temperature (°C).
    wind_speed_mps : float
        Wind speed (m/s).
    wind_angle_deg : float
        Wind angle (degrees).
    solar_radiation_w_m2 : float
        Solar radiation (W/m²).
    current_a : float
        Conductor current (A).
    conductor_resistance_ohms_per_km : float
        AC resistance (Ω/km) at reference temperature.
    initial_temp_c : float
        Starting conductor temperature (°C).
    duration_seconds : float
        Simulation duration (s).  Typically 300–900 for 5–15 min.
    conductor_diameter_m : float
    emissivity : float
    absorptivity : float
    thermal_capacity_j_per_k_m : float
        Lumped thermal capacity per unit length (J/(K·m)).
    time_step_s : float
        Integration step (s).  Default 60.

    Returns
    -------
    float
        Conductor temperature (°C) after ``duration_seconds``.
    """
    T = initial_temp_c
    R_per_m = conductor_resistance_ohms_per_km / 1000.0
    D = conductor_diameter_m

    n_steps = max(1, int(duration_seconds / time_step_s))
    dt = duration_seconds / n_steps

    for _ in range(n_steps):
        T_k = T + _KELVIN_OFFSET
        T_a_k = ambient_temp_c + _KELVIN_OFFSET

        q_s = _solar_heat_gain(np.array(solar_radiation_w_m2), D, absorptivity)
        q_r = _radiative_cooling(np.array(T_k), np.array(T_a_k), D, emissivity)
        q_c = _convective_cooling(
            np.array(T_k), np.array(T_a_k), D,
            np.array(wind_speed_mps), np.array(wind_angle_deg),
        )

        joule_heating = current_a**2 * R_per_m
        net_heat = float(q_s + joule_heating - q_c - q_r)
        dT = net_heat * dt / thermal_capacity_j_per_k_m
        T += dT

    return T


# ---------------------------------------------------------------------------
# DLR Grid Controller
# ---------------------------------------------------------------------------

class DLRGridController:
    """Dynamically overwrites pandapower line thermal ratings.

    Uses per-line microclimatic weather data and IEEE 738 to compute
    dynamic ampacity limits, optionally incorporating transient thermal
    inertia for cascading-overload response windows.

    Parameters
    ----------
    grid_model : GridModel
        The grid model whose line ratings will be modified.
    line_weather : dict or None
        Mapping from pandapower line index to a dict of weather
        parameters.  Each dict may contain:

        - ``ambient_temp_c`` (float or np.ndarray)
        - ``wind_speed_mps`` (float or np.ndarray)
        - ``wind_angle_deg`` (float or np.ndarray, default 90)
        - ``solar_radiation_w_m2`` (float or np.ndarray, default 1000)

        If an array is provided, it is indexed by simulation timestep.
    default_weather : dict or None
        Fallback weather for lines not present in ``line_weather``.
        Same keys as above.  Defaults to mild conditions (25°C, 0.6 m/s).
    max_cond_temp_c : float
        Maximum conductor temperature (°C).  Default 100.
    conductor_diameter_m : float
        Default diameter for lines without explicit geometry.
    emissivity : float
    absorptivity : float

    Attributes
    ----------
    grid_model : GridModel
    line_weather : dict
    default_weather : dict
    """

    def __init__(
        self,
        grid_model: GridModel,
        line_weather: Optional[Dict[int, Dict[str, Any]]] = None,
        default_weather: Optional[Dict[str, Any]] = None,
        max_cond_temp_c: float = _DEFAULT_MAX_COND_TEMP_C,
        conductor_diameter_m: float = _DEFAULT_DIAMETER_M,
        emissivity: float = _DEFAULT_EMISSIVITY,
        absorptivity: float = _DEFAULT_ABSORPTIVITY,
    ) -> None:
        self.grid_model = grid_model
        self.line_weather = line_weather or {}
        self.default_weather = default_weather or {
            "ambient_temp_c": 25.0,
            "wind_speed_mps": 0.6,
            "wind_angle_deg": 90.0,
            "solar_radiation_w_m2": 1000.0,
        }
        self.max_cond_temp_c = max_cond_temp_c
        self.conductor_diameter_m = conductor_diameter_m
        self.emissivity = emissivity
        self.absorptivity = absorptivity

        self._line_resistance: Dict[int, float] = {}
        self._line_diameter: Dict[int, float] = {}
        self._build_line_properties()

    def _build_line_properties(self) -> None:
        """Extract conductor properties from the pandapower network."""
        net = self.grid_model.net
        for idx in net.line.index:
            if idx in net.line.index and net.line.at[idx, "in_service"]:
                self._line_resistance[idx] = float(
                    net.line.at[idx, "r_ohm_per_km"]
                )
                self._line_diameter[idx] = self.conductor_diameter_m

    def _get_weather(self, line_id: int, timestep: int = 0) -> Dict[str, float]:
        """Resolve weather for a given line and timestep.

        Parameters
        ----------
        line_id : int
        timestep : int

        Returns
        -------
        dict
            Scalar weather values.
        """
        raw = self.line_weather.get(line_id, self.default_weather)

        def _resolve(key: str, default: float) -> float:
            val = raw.get(key, default)
            if isinstance(val, np.ndarray):
                return float(val[timestep % len(val)])
            return float(val)

        return {
            "ambient_temp_c": _resolve("ambient_temp_c", 25.0),
            "wind_speed_mps": _resolve("wind_speed_mps", 0.6),
            "wind_angle_deg": _resolve("wind_angle_deg", 90.0),
            "solar_radiation_w_m2": _resolve("solar_radiation_w_m2", 1000.0),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_dynamic_ratings(self, timestep: int = 0) -> None:
        """Overwrite ``max_i_ka`` for all in-service lines.

        Computes IEEE 738 ampacity using weather at the given timestep
        and writes the result (converted to kA) into the pandapower
        network.

        Parameters
        ----------
        timestep : int
            Index into the weather arrays.  Default 0.
        """
        net = self.grid_model.net

        for idx in net.line.index:
            if not net.line.at[idx, "in_service"]:
                continue

            w = self._get_weather(idx, timestep)
            R = self._line_resistance.get(idx, _DEFAULT_RESISTANCE_OHM_PER_KM)
            D = self._line_diameter.get(idx, self.conductor_diameter_m)

            ampacity_a = calculate_dynamic_ampacity(
                ambient_temp_c=w["ambient_temp_c"],
                wind_speed_mps=w["wind_speed_mps"],
                wind_angle_deg=w["wind_angle_deg"],
                solar_radiation_w_m2=w["solar_radiation_w_m2"],
                conductor_resistance_ohms_per_km=R,
                max_cond_temp_c=self.max_cond_temp_c,
                conductor_diameter_m=D,
                emissivity=self.emissivity,
                absorptivity=self.absorptivity,
            )

            net.line.at[idx, "max_i_ka"] = float(ampacity_a) / 1000.0

    def apply_transient_ratings(
        self,
        timestep: int,
        line_currents: Dict[int, float],
        dt_minutes: float = 10.0,
    ) -> None:
        """Apply transient-aware ratings accounting for thermal inertia.

        For each line, simulates conductor temperature evolution over
        ``dt_minutes`` at the given current, then computes the ampacity
        that would reach ``max_cond_temp_c`` from that elevated
        temperature.

        Parameters
        ----------
        timestep : int
            Weather timestep index.
        line_currents : dict
            Mapping from line index to current in Amperes.
        dt_minutes : float
            Thermal inertia window in minutes.  Default 10.
        """
        net = self.grid_model.net
        dt_s = dt_minutes * 60.0

        for idx in net.line.index:
            if not net.line.at[idx, "in_service"]:
                continue

            w = self._get_weather(idx, timestep)
            R = self._line_resistance.get(idx, _DEFAULT_RESISTANCE_OHM_PER_KM)
            D = self._line_diameter.get(idx, self.conductor_diameter_m)
            I = line_currents.get(idx, 0.0)

            initial_temp = float(net.line.at[idx, "max_i_ka"]) * 1000.0
            if initial_temp < 1.0:
                initial_temp = w["ambient_temp_c"]

            final_temp = transient_conductor_temperature(
                ambient_temp_c=w["ambient_temp_c"],
                wind_speed_mps=w["wind_speed_mps"],
                wind_angle_deg=w["wind_angle_deg"],
                solar_radiation_w_m2=w["solar_radiation_w_m2"],
                current_a=I,
                conductor_resistance_ohms_per_km=R,
                initial_temp_c=initial_temp,
                duration_seconds=dt_s,
                conductor_diameter_m=D,
                emissivity=self.emissivity,
                absorptivity=self.absorptivity,
            )

            margin_c = self.max_cond_temp_c - final_temp
            if margin_c <= 0:
                net.line.at[idx, "max_i_ka"] = 0.0
            else:
                effective_max_temp = final_temp + margin_c * 0.5
                ampacity_a = calculate_dynamic_ampacity(
                    ambient_temp_c=w["ambient_temp_c"],
                    wind_speed_mps=w["wind_speed_mps"],
                    wind_angle_deg=w["wind_angle_deg"],
                    solar_radiation_w_m2=w["solar_radiation_w_m2"],
                    conductor_resistance_ohms_per_km=R,
                    max_cond_temp_c=effective_max_temp,
                    conductor_diameter_m=D,
                    emissivity=self.emissivity,
                    absorptivity=self.absorptivity,
                )
                net.line.at[idx, "max_i_ka"] = float(ampacity_a) / 1000.0

    def get_thermal_margins(
        self,
        timestep: int,
        line_currents: Dict[int, float],
    ) -> Dict[int, float]:
        """Compute thermal margin as (rating - current) / rating.

        Parameters
        ----------
        timestep : int
        line_currents : dict
            Line index → current (A).

        Returns
        -------
        dict
            Line index → margin fraction.  Negative means overloaded.
        """
        self.apply_dynamic_ratings(timestep)
        net = self.grid_model.net
        margins: Dict[int, float] = {}

        for idx in net.line.index:
            if not net.line.at[idx, "in_service"]:
                continue
            rating_a = float(net.line.at[idx, "max_i_ka"]) * 1000.0
            current_a = line_currents.get(idx, 0.0)
            if rating_a > 0:
                margins[idx] = (rating_a - current_a) / rating_a
            else:
                margins[idx] = -1.0

        return margins
