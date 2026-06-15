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
Physical dynamic line rating and thermal margin engine.

Implements the IEEE 738-2012 standard for steady-state and transient
conductor thermal rating.  Provides:

* ``calculate_steady_state_ampacity`` — solves the heat-balance
  equation for maximum allowable current with temperature-dependent
  resistance.
* ``line_thermal_inertia`` — vectorised transient heat accumulation
  over consecutive time steps using the lumped-capacitance model.
* ``DLRGridController`` — dynamically overwrites pandapower line
  thermal capacities (``max_i_ka``) based on geographically coincident
  microclimatic conditions during simulation timesteps.

All computations are fully vectorised with NumPy for high-throughput
Monte Carlo and QSTS simulations.

Reference
---------
IEEE Std 738-2012 — *IEEE Standard for Calculating the Current-
Temperature Relationship of Bare Overhead Conductors*.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_STEFAN_BOLTZMANN: float = 5.670367e-8  # W·m⁻²·K⁻⁴
_KELVIN_OFFSET: float = 273.15  # K

# Default conductor properties (ACSR "Drake" 795 kcmil)
_DEFAULT_DIAMETER_M: float = 0.0281  # m
_DEFAULT_EMISSIVITY: float = 0.7
_DEFAULT_ABSORPTIVITY: float = 0.7
_DEFAULT_MAX_COND_TEMP_C: float = 100.0  # °C
_DEFAULT_RESISTANCE_OHM_PER_KM: float = 0.05  # Ω/km at 20°C
_DEFAULT_TEMP_COEFF: float = 0.00403  # 1/K (aluminium)
_DEFAULT_THERMAL_CAPACITY: float = 500.0  # J/(K·m) for ACSR

_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# Air property functions (temperature-dependent)
# ---------------------------------------------------------------------------


def _air_thermal_conductivity(T_film_c: np.ndarray) -> np.ndarray:
    """Air thermal conductivity k_f (W/(m·K)) at film temperature.

    Parameters
    ----------
    T_film_c : np.ndarray
        Film temperature in °C = (T_c + T_a) / 2.

    Returns
    -------
    np.ndarray
    """
    return 2.42e-2 + 7.2e-5 * T_film_c


def _air_kinematic_viscosity(T_film_c: np.ndarray) -> np.ndarray:
    """Air kinematic viscosity ν (m²/s) at film temperature.

    Parameters
    ----------
    T_film_c : np.ndarray

    Returns
    -------
    np.ndarray
    """
    return 1.32e-5 + 9.5e-8 * T_film_c


def _air_density(T_film_c: np.ndarray) -> np.ndarray:
    """Air density ρ (kg/m³) at film temperature.

    Parameters
    ----------
    T_film_c : np.ndarray

    Returns
    -------
    np.ndarray
    """
    return 1.293 - 0.00425 * T_film_c + 1.0e-5 * T_film_c ** 2


# ---------------------------------------------------------------------------
# IEEE 738 heat transfer components
# ---------------------------------------------------------------------------


def _radiative_cooling(
    T_c_k: np.ndarray,
    T_a_k: np.ndarray,
    D_m: float,
    emissivity: float,
) -> np.ndarray:
    """Radiative cooling q_r per unit length (W/m).

    Parameters
    ----------
    T_c_k : np.ndarray
        Conductor temperature (K).
    T_a_k : np.ndarray
        Ambient temperature (K).
    D_m : float
        Conductor diameter (m).
    emissivity : float

    Returns
    -------
    np.ndarray
    """
    area_per_m = np.pi * D_m
    return emissivity * _STEFAN_BOLTZMANN * area_per_m * (T_c_k ** 4 - T_a_k ** 4)


def _solar_heat_gain(
    Q_s: np.ndarray,
    D_m: float,
    absorptivity: float,
) -> np.ndarray:
    """Absorbed solar heat gain q_s per unit length (W/m).

    Parameters
    ----------
    Q_s : np.ndarray
        Solar radiation (W/m²).
    D_m : float
    absorptivity : float

    Returns
    -------
    np.ndarray
    """
    return absorptivity * Q_s * D_m


def _convective_cooling(
    T_c_k: np.ndarray,
    T_a_k: np.ndarray,
    D_m: float,
    V_w: np.ndarray,
    phi_deg: np.ndarray,
) -> np.ndarray:
    """Convective cooling q_c per unit length (W/m).

    Uses forced convection for wind ≥ 0.5 m/s, natural convection
    otherwise, per IEEE 738 §4.4.

    Parameters
    ----------
    T_c_k : np.ndarray
    T_a_k : np.ndarray
    D_m : float
    V_w : np.ndarray
        Wind speed (m/s).
    phi_deg : np.ndarray
        Wind angle relative to conductor axis (degrees).

    Returns
    -------
    np.ndarray
    """
    T_film_c = (T_c_k + T_a_k) / 2.0 - _KELVIN_OFFSET
    k_f = _air_thermal_conductivity(T_film_c)
    nu = _air_kinematic_viscosity(T_film_c)

    Re = V_w * D_m / np.maximum(nu, _EPS)
    phi_rad = np.radians(phi_deg)

    K_angle = (
        1.194
        - np.cos(phi_rad)
        + 0.194 * np.cos(2.0 * phi_rad)
        + 0.368 * np.sin(2.0 * phi_rad)
    )

    q_forced = K_angle * (1.01 + 1.35 * Re ** 0.52) * k_f * (T_c_k - T_a_k)

    rho_f = _air_density(T_film_c)
    q_natural = 3.645 * rho_f ** 0.5 * D_m ** 0.75 * np.abs(T_c_k - T_a_k) ** 1.25

    return np.where(V_w >= 0.5, q_forced, np.maximum(q_forced, q_natural))


# ---------------------------------------------------------------------------
# Steady-state ampacity (IEEE 738)
# ---------------------------------------------------------------------------


def calculate_steady_state_ampacity(
    ambient_temp_c: np.ndarray | float,
    wind_speed_mps: np.ndarray | float,
    wind_angle_deg: np.ndarray | float = 90.0,
    solar_radiation_w_m2: np.ndarray | float = 1000.0,
    conductor_resistance_ohms_per_km: np.ndarray | float = _DEFAULT_RESISTANCE_OHM_PER_KM,
    max_cond_temp_c: float = _DEFAULT_MAX_COND_TEMP_C,
    conductor_diameter_m: float = _DEFAULT_DIAMETER_M,
    emissivity: float = _DEFAULT_EMISSIVITY,
    absorptivity: float = _DEFAULT_ABSORPTIVITY,
    temp_coeff_resistance: float = _DEFAULT_TEMP_COEFF,
    ref_temp_c: float = 20.0,
) -> np.ndarray:
    """Compute IEEE 738 steady-state ampacity with temperature-dependent R.

    Solves the steady-state heat balance equation:

    .. math::

        q_c + q_r = q_s + I^2 R(T_{\\text{max}})

    where :math:`q_c` is convective cooling, :math:`q_r` is radiative
    cooling, :math:`q_s` is solar heat gain, and :math:`R(T_{\\text{max}})`
    is the conductor AC resistance adjusted to the maximum conductor
    temperature via the temperature coefficient of resistance:

    .. math::

        R(T) = R_{\\text{ref}} \\left[1 + \\alpha (T - T_{\\text{ref}})\\right]

    Parameters
    ----------
    ambient_temp_c : float or np.ndarray
        Ambient air temperature in °C.
    wind_speed_mps : float or np.ndarray
        Wind speed in m/s.
    wind_angle_deg : float or np.ndarray
        Angle between wind direction and conductor axis in degrees.
        Default 90° (perpendicular, maximum cooling).
    solar_radiation_w_m2 : float or np.ndarray
        Total solar irradiance in W/m².  Default 1000.
    conductor_resistance_ohms_per_km : float or np.ndarray
        AC resistance at *ref_temp_c* in Ω/km.
    max_cond_temp_c : float
        Maximum allowable conductor temperature in °C.  Default 100.
    conductor_diameter_m : float
        Conductor outer diameter in m.  Default 0.0281 (ACSR Drake).
    emissivity : float
        Radiative emissivity ε ∈ (0, 1].  Default 0.7.
    absorptivity : float
        Solar absorptivity α_s ∈ (0, 1].  Default 0.7.
    temp_coeff_resistance : float
        Temperature coefficient of resistance α_R in 1/K.  Default
        0.00403 (aluminium).
    ref_temp_c : float
        Reference temperature for *conductor_resistance_ohms_per_km*
        in °C.  Default 20.

    Returns
    -------
    np.ndarray
        Steady-state ampacity in Amperes.  Scalar inputs produce a
        0-d array.
    """
    T_a = np.asarray(ambient_temp_c, dtype=np.float64) + _KELVIN_OFFSET
    T_c = max_cond_temp_c + _KELVIN_OFFSET
    V_w = np.maximum(np.asarray(wind_speed_mps, dtype=np.float64), 0.0)
    phi = np.asarray(wind_angle_deg, dtype=np.float64)
    Q_s = np.asarray(solar_radiation_w_m2, dtype=np.float64)
    R_ref = np.asarray(conductor_resistance_ohms_per_km, dtype=np.float64) / 1000.0
    D = conductor_diameter_m

    # Temperature-adjusted resistance at max conductor temperature
    R_Tmax = R_ref * (1.0 + temp_coeff_resistance * (max_cond_temp_c - ref_temp_c))

    q_r = _radiative_cooling(T_c, T_a, D, emissivity)
    q_s = _solar_heat_gain(Q_s, D, absorptivity)
    q_c = _convective_cooling(T_c, T_a, D, V_w, phi)

    numerator = q_c + q_r - q_s
    numerator = np.maximum(numerator, 0.0)
    denominator = np.maximum(R_Tmax, _EPS)

    return np.sqrt(numerator / denominator)


# ---------------------------------------------------------------------------
# Transient thermal inertia
# ---------------------------------------------------------------------------


def line_thermal_inertia(
    ambient_temp_c: np.ndarray,
    wind_speed_mps: np.ndarray,
    wind_angle_deg: np.ndarray,
    solar_radiation_w_m2: np.ndarray,
    current_a: np.ndarray,
    conductor_resistance_ohms_per_km: np.ndarray,
    initial_temp_c: np.ndarray,
    time_step_s: float,
    conductor_diameter_m: float = _DEFAULT_DIAMETER_M,
    emissivity: float = _DEFAULT_EMISSIVITY,
    absorptivity: float = _DEFAULT_ABSORPTIVITY,
    thermal_time_constant_s: float = 600.0,
    temp_coeff_resistance: float = _DEFAULT_TEMP_COEFF,
    ref_temp_c: float = 20.0,
) -> np.ndarray:
    """Vectorised transient conductor temperature over one time step.

    Models heat accumulation using the lumped-capacitance equation:

    .. math::

        \\tau \\frac{dT_c}{dt} = q_s + I(t)^2 R(T_c) - q_c - q_r

    where :math:`\\tau` is the conductor thermal time constant
    (thermal capacity per unit length in J/(K·m)) and :math:`T_c`
    is the conductor temperature in °C.

    Uses a single explicit Euler step.  All inputs may be arrays of
    shape ``(n_lines,)`` for simultaneous evaluation.

    Parameters
    ----------
    ambient_temp_c : np.ndarray
        Ambient temperature in °C, shape ``(n_lines,)``.
    wind_speed_mps : np.ndarray
        Wind speed in m/s, shape ``(n_lines,)``.
    wind_angle_deg : np.ndarray
        Wind angle in degrees, shape ``(n_lines,)``.
    solar_radiation_w_m2 : np.ndarray
        Solar radiation in W/m², shape ``(n_lines,)``.
    current_a : np.ndarray
        Conductor current in Amperes, shape ``(n_lines,)``.
    conductor_resistance_ohms_per_km : np.ndarray
        AC resistance at *ref_temp_c* in Ω/km, shape ``(n_lines,)``.
    initial_temp_c : np.ndarray
        Conductor temperature at start of step in °C, shape
        ``(n_lines,)``.
    time_step_s : float
        Integration time step in seconds.
    conductor_diameter_m : float
        Conductor diameter in m.
    emissivity : float
    absorptivity : float
    thermal_time_constant_s : float
        Lumped thermal capacity τ in J/(K·m).  Default 600 for ACSR.
    temp_coeff_resistance : float
        TCR α_R in 1/K.  Default 0.00403.
    ref_temp_c : float
        Reference temperature for resistance in °C.  Default 20.

    Returns
    -------
    np.ndarray
        Conductor temperatures in °C after one time step, shape
        ``(n_lines,)``.
    """
    T_c = np.asarray(initial_temp_c, dtype=np.float64)
    T_a = np.asarray(ambient_temp_c, dtype=np.float64)
    V_w = np.maximum(np.asarray(wind_speed_mps, dtype=np.float64), 0.0)
    phi = np.asarray(wind_angle_deg, dtype=np.float64)
    Q_s = np.asarray(solar_radiation_w_m2, dtype=np.float64)
    I = np.asarray(current_a, dtype=np.float64)
    R_ref = np.asarray(conductor_resistance_ohms_per_km, dtype=np.float64) / 1000.0
    D = conductor_diameter_m

    T_c_k = T_c + _KELVIN_OFFSET
    T_a_k = T_a + _KELVIN_OFFSET

    q_s = _solar_heat_gain(Q_s, D, absorptivity)
    q_r = _radiative_cooling(T_c_k, T_a_k, D, emissivity)
    q_c = _convective_cooling(T_c_k, T_a_k, D, V_w, phi)

    R_T = R_ref * (1.0 + temp_coeff_resistance * (T_c - ref_temp_c))
    joule_heating = I ** 2 * R_T

    net_heat = q_s + joule_heating - q_c - q_r
    dT = net_heat * time_step_s / thermal_time_constant_s

    return T_c + dT


# ---------------------------------------------------------------------------
# DLR Grid Controller
# ---------------------------------------------------------------------------


class DLRGridController:
    """Dynamically updates pandapower line thermal capacities.

    Uses per-line microclimatic weather data and IEEE 738 to compute
    dynamic ampacity limits, writing them into ``net.line.max_i_ka``
    at each simulation timestep.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network whose line ratings will be modified.
    line_weather : dict or None
        Mapping from pandapower line index to a dict of weather
        parameters.  Each dict may contain:

        - ``ambient_temp_c`` (float or np.ndarray)
        - ``wind_speed_mps`` (float or np.ndarray)
        - ``wind_angle_deg`` (float or np.ndarray, default 90)
        - ``solar_radiation_w_m2`` (float or np.ndarray, default 1000)

        If an array is provided, it is indexed by simulation timestep.
    default_weather : dict or None
        Fallback weather for lines not present in *line_weather*.
        Defaults to mild conditions (25°C, 0.6 m/s, 90°, 1000 W/m²).
    max_cond_temp_c : float
        Maximum conductor temperature in °C.  Default 100.
    conductor_diameter_m : float
        Default diameter for lines without explicit geometry.
    emissivity : float
    absorptivity : float
    temp_coeff_resistance : float
        TCR for aluminium in 1/K.  Default 0.00403.

    Attributes
    ----------
    net : pandapowerNet
    line_weather : dict
    default_weather : dict
    """

    def __init__(
        self,
        net: Any,
        line_weather: Optional[Dict[int, Dict[str, Any]]] = None,
        default_weather: Optional[Dict[str, Any]] = None,
        max_cond_temp_c: float = _DEFAULT_MAX_COND_TEMP_C,
        conductor_diameter_m: float = _DEFAULT_DIAMETER_M,
        emissivity: float = _DEFAULT_EMISSIVITY,
        absorptivity: float = _DEFAULT_ABSORPTIVITY,
        temp_coeff_resistance: float = _DEFAULT_TEMP_COEFF,
    ) -> None:
        self.net = net
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
        self.temp_coeff_resistance = temp_coeff_resistance

        self._line_resistance: Dict[int, float] = {}
        self._line_diameter: Dict[int, float] = {}
        self._build_line_properties()

    def _build_line_properties(self) -> None:
        """Extract conductor properties from the pandapower network."""
        for idx in self.net.line.index:
            if self.net.line.at[idx, "in_service"]:
                self._line_resistance[idx] = float(
                    self.net.line.at[idx, "r_ohm_per_km"]
                )
                self._line_diameter[idx] = self.conductor_diameter_m

    def _get_weather(self, line_id: int, timestep: int = 0) -> Dict[str, float]:
        """Resolve scalar weather for a given line and timestep.

        Parameters
        ----------
        line_id : int
        timestep : int

        Returns
        -------
        dict
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
        for idx in self.net.line.index:
            if not self.net.line.at[idx, "in_service"]:
                continue

            w = self._get_weather(idx, timestep)
            R = self._line_resistance.get(idx, _DEFAULT_RESISTANCE_OHM_PER_KM)
            D = self._line_diameter.get(idx, self.conductor_diameter_m)

            ampacity_a = calculate_steady_state_ampacity(
                ambient_temp_c=w["ambient_temp_c"],
                wind_speed_mps=w["wind_speed_mps"],
                wind_angle_deg=w["wind_angle_deg"],
                solar_radiation_w_m2=w["solar_radiation_w_m2"],
                conductor_resistance_ohms_per_km=R,
                max_cond_temp_c=self.max_cond_temp_c,
                conductor_diameter_m=D,
                emissivity=self.emissivity,
                absorptivity=self.absorptivity,
                temp_coeff_resistance=self.temp_coeff_resistance,
            )

            self.net.line.at[idx, "max_i_ka"] = float(ampacity_a) / 1000.0

        logger.info(
            "Applied DLR ratings for timestep %d (%d lines).",
            timestep, len(self._line_resistance),
        )

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
            Line index → current in Amperes.

        Returns
        -------
        dict
            Line index → margin fraction.  Negative means overloaded.
        """
        self.apply_dynamic_ratings(timestep)
        margins: Dict[int, float] = {}

        for idx in self.net.line.index:
            if not self.net.line.at[idx, "in_service"]:
                continue
            rating_a = float(self.net.line.at[idx, "max_i_ka"]) * 1000.0
            current_a = line_currents.get(idx, 0.0)
            if rating_a > 0:
                margins[idx] = (rating_a - current_a) / rating_a
            else:
                margins[idx] = -1.0

        return margins

    def __repr__(self) -> str:
        return (
            f"DLRGridController(lines={len(self._line_resistance)}, "
            f"max_temp={self.max_cond_temp_c}°C)"
        )
