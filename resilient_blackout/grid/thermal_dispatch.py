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
Quasi-steady-state time-series (QSTS) thermal dispatch solver.

Provides ``QSTSThermalDispatcher``, a chronological simulator that
co-optimises operating cost and expected energy not served (EENS) across
sequential hourly or sub-hourly time steps while tracking transient
conductor temperatures via the lumped-capacitance heat equation.

Architecture
------------
1. **ConductorThermalTracker** — vectorised Euler integration of the
   transient heat equation for all lines simultaneously.
2. **PredictiveDispatchController** — model predictive control (MPC)
   using ``scipy.optimize.linprog`` with a rolling look-ahead horizon.
3. **QSTSThermalDispatcher** — main orchestrator that loops through
   time steps, runs power flow, updates temperatures, trips overheated
   lines, invokes the MPC controller, and records chronological profiles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, vstack

from resilient_blackout.grid.network import GridModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_STEFAN_BOLTZMANN: float = 5.670367e-8
_KELVIN_OFFSET: float = 273.15

_DEFAULT_DIAMETER_M: float = 0.0281
_DEFAULT_EMISSIVITY: float = 0.7
_DEFAULT_ABSORPTIVITY: float = 0.7
_DEFAULT_MAX_COND_TEMP_C: float = 100.0
_DEFAULT_THERMAL_CAPACITY: float = 500.0
_DEFAULT_TEMP_COEFF_RESISTANCE: float = 0.00403  # 1/K for aluminium

_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# Vectorised IEEE 738 cooling functions (per-line arrays)
# ---------------------------------------------------------------------------

def _radiative_cooling_vector(
    T_c: np.ndarray,
    T_a: np.ndarray,
    D: np.ndarray,
    emissivity: np.ndarray,
) -> np.ndarray:
    area_per_m = np.pi * D
    return emissivity * _STEFAN_BOLTZMANN * area_per_m * (T_c**4 - T_a**4)


def _solar_heat_gain_vector(
    Q_s: np.ndarray,
    D: np.ndarray,
    absorptivity: np.ndarray,
) -> np.ndarray:
    return absorptivity * Q_s * D


def _air_thermal_conductivity(T_film: np.ndarray) -> np.ndarray:
    return 2.42e-2 + 7.2e-5 * (T_film - _KELVIN_OFFSET)


def _air_kinematic_viscosity(T_film: np.ndarray) -> np.ndarray:
    return 1.32e-5 + 9.5e-8 * (T_film - _KELVIN_OFFSET)


def _air_density(T_film: np.ndarray) -> np.ndarray:
    return 1.293 - 0.00425 * (T_film - _KELVIN_OFFSET) + 1.0e-5 * (T_film - _KELVIN_OFFSET) ** 2


def _convective_cooling_vector(
    T_c: np.ndarray,
    T_a: np.ndarray,
    D: np.ndarray,
    V_w: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    T_film = (T_c + T_a) / 2.0
    k_f = _air_thermal_conductivity(T_film)
    nu = _air_kinematic_viscosity(T_film)

    Re = V_w * D / np.maximum(nu, _EPS)
    phi_rad = np.radians(phi)

    K_angle = (
        1.194 - np.cos(phi_rad) + 0.194 * np.cos(2 * phi_rad) + 0.368 * np.sin(2 * phi_rad)
    )

    q_forced = K_angle * (1.01 + 1.35 * Re**0.52) * k_f * (T_c - T_a)
    q_natural = 3.645 * _air_density(T_film) ** 0.5 * D**0.75 * (T_c - T_a) ** 1.25

    return np.where(V_w >= 0.5, q_forced, np.maximum(q_forced, q_natural))


# ---------------------------------------------------------------------------
# ConductorThermalTracker
# ---------------------------------------------------------------------------

@dataclass
class _LineThermalState:
    """Per-line thermal parameters and current state."""

    indices: np.ndarray
    diameter_m: np.ndarray
    emissivity: np.ndarray
    absorptivity: np.ndarray
    r_ref_ohm_per_m: np.ndarray
    thermal_capacity: np.ndarray
    temp_coeff: np.ndarray
    max_temp_c: np.ndarray
    T_c: np.ndarray


class ConductorThermalTracker:
    """Vectorised transient conductor temperature integrator.

    Tracks the lumped-capacitance heat equation for all lines
    simultaneously:

    .. math::

        \\tau \\frac{dT_c}{dt} = q_s + I^2 R(T_c) - q_c - q_r

    where :math:`\\tau` is the thermal capacity per unit length
    (J/(K·m)).

    Parameters
    ----------
    grid_model : GridModel
        The pandapower network providing line geometry and resistance.
    max_cond_temp_c : float
        Default maximum conductor temperature in °C.
    conductor_diameter_m : float
        Default diameter for lines without explicit geometry.
    emissivity : float
    absorptivity : float
    thermal_capacity_j_per_k_m : float
        Lumped thermal capacity per unit length (J/(K·m)).
    temp_coeff_resistance : float
        Temperature coefficient of resistance (1/K) for aluminium.
    """

    def __init__(
        self,
        grid_model: GridModel,
        max_cond_temp_c: float = _DEFAULT_MAX_COND_TEMP_C,
        conductor_diameter_m: float = _DEFAULT_DIAMETER_M,
        emissivity: float = _DEFAULT_EMISSIVITY,
        absorptivity: float = _DEFAULT_ABSORPTIVITY,
        thermal_capacity_j_per_k_m: float = _DEFAULT_THERMAL_CAPACITY,
        temp_coeff_resistance: float = _DEFAULT_TEMP_COEFF_RESISTANCE,
    ) -> None:
        self.grid_model = grid_model
        self._default_max_temp = max_cond_temp_c
        self._default_diameter = conductor_diameter_m
        self._default_emissivity = emissivity
        self._default_absorptivity = absorptivity
        self._default_thermal_capacity = thermal_capacity_j_per_k_m
        self._default_temp_coeff = temp_coeff_resistance

        self._state: Optional[_LineThermalState] = None
        self._n_lines: int = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise(
        self,
        initial_temp_c: Optional[np.ndarray] = None,
        ambient_temp_c: float = 25.0,
    ) -> None:
        """Build the per-line thermal state from the pandapower network.

        Parameters
        ----------
        initial_temp_c : np.ndarray or None
            Starting conductor temperatures in °C.  If ``None``, defaults
            to *ambient_temp_c*.
        ambient_temp_c : float
            Fallback initial temperature when *initial_temp_c* is not
            provided.
        """
        net = self.grid_model.net
        in_service = net.line["in_service"].values.astype(bool)
        line_indices = net.line.index[in_service].values

        self._n_lines = len(line_indices)
        if self._n_lines == 0:
            self._state = None
            return

        D = np.full(self._n_lines, self._default_diameter, dtype=np.float64)
        eps_arr = np.full(self._n_lines, self._default_emissivity, dtype=np.float64)
        abs_arr = np.full(self._n_lines, self._default_absorptivity, dtype=np.float64)
        r_ref = np.zeros(self._n_lines, dtype=np.float64)
        tc = np.full(self._n_lines, self._default_thermal_capacity, dtype=np.float64)
        alpha = np.full(self._n_lines, self._default_temp_coeff, dtype=np.float64)
        tmax = np.full(self._n_lines, self._default_max_temp, dtype=np.float64)

        for i, idx in enumerate(line_indices):
            r_ref[i] = float(net.line.at[idx, "r_ohm_per_km"]) / 1000.0

        if initial_temp_c is not None:
            T_init = np.asarray(initial_temp_c, dtype=np.float64)
        else:
            T_init = np.full(self._n_lines, ambient_temp_c, dtype=np.float64)

        self._state = _LineThermalState(
            indices=line_indices.astype(np.int64),
            diameter_m=D,
            emissivity=eps_arr,
            absorptivity=abs_arr,
            r_ref_ohm_per_m=r_ref,
            thermal_capacity=tc,
            temp_coeff=alpha,
            max_temp_c=tmax,
            T_c=T_init.copy(),
        )

    # ------------------------------------------------------------------
    # Temperature-dependent resistance
    # ------------------------------------------------------------------

    def _resistance_at_temp(self) -> np.ndarray:
        """Compute per-line resistance at current conductor temperature."""
        s = self._state
        return s.r_ref_ohm_per_m * (1.0 + s.temp_coeff * (s.T_c - 20.0))

    # ------------------------------------------------------------------
    # Single Euler step
    # ------------------------------------------------------------------

    def step(
        self,
        line_currents_a: np.ndarray,
        ambient_temp_c: np.ndarray,
        wind_speed_mps: np.ndarray,
        wind_angle_deg: np.ndarray,
        solar_radiation_w_m2: np.ndarray,
        dt_seconds: float,
    ) -> np.ndarray:
        """Advance conductor temperatures by one Euler step.

        Parameters
        ----------
        line_currents_a : np.ndarray
            Current in Amperes for each tracked line, shape ``(n_lines,)``.
        ambient_temp_c : np.ndarray
            Ambient temperature in °C per line.
        wind_speed_mps : np.ndarray
            Wind speed in m/s per line.
        wind_angle_deg : np.ndarray
            Wind angle in degrees per line.
        solar_radiation_w_m2 : np.ndarray
            Solar radiation in W/m² per line.
        dt_seconds : float
            Integration time step in seconds.

        Returns
        -------
        np.ndarray
            Updated conductor temperatures in °C, shape ``(n_lines,)``.
        """
        s = self._state
        if s is None:
            return np.array([], dtype=np.float64)

        T_k = s.T_c + _KELVIN_OFFSET
        T_a_k = ambient_temp_c + _KELVIN_OFFSET

        q_s = _solar_heat_gain_vector(solar_radiation_w_m2, s.diameter_m, s.absorptivity)
        q_r = _radiative_cooling_vector(T_k, T_a_k, s.diameter_m, s.emissivity)
        q_c = _convective_cooling_vector(
            T_k, T_a_k, s.diameter_m, wind_speed_mps, wind_angle_deg,
        )

        R_T = self._resistance_at_temp()
        joule_heating = line_currents_a**2 * R_T

        net_heat = q_s + joule_heating - q_c - q_r
        dT = net_heat * dt_seconds / s.thermal_capacity
        s.T_c += dT

        return s.T_c.copy()

    # ------------------------------------------------------------------
    # Trip detection
    # ------------------------------------------------------------------

    def detect_trips(self) -> np.ndarray:
        """Return indices of lines whose temperature exceeds the maximum.

        Returns
        -------
        np.ndarray
            Pandapower line indices that have overheated.
        """
        s = self._state
        if s is None:
            return np.array([], dtype=np.int64)

        mask = s.T_c >= s.max_temp_c
        return s.indices[mask]

    def get_temperatures(self) -> np.ndarray:
        """Return current conductor temperatures in °C."""
        if self._state is None:
            return np.array([], dtype=np.float64)
        return self._state.T_c.copy()

    def get_line_indices(self) -> np.ndarray:
        """Return the pandapower line indices being tracked."""
        if self._state is None:
            return np.array([], dtype=np.int64)
        return self._state.indices.copy()

    @property
    def n_lines(self) -> int:
        return self._n_lines


# ---------------------------------------------------------------------------
# PredictiveDispatchController
# ---------------------------------------------------------------------------

class PredictiveDispatchController:
    """MPC-based dispatch controller using linear programming.

    At each time step, solves a rolling-horizon LP that co-optimises
    generation cost and EENS penalty subject to:

    - Power balance (DC approximation)
    - Generator output limits
    - Generator ramp-rate constraints
    - Line thermal limits (forecast from the thermal tracker)
    - Load curtailment bounds

    Only the first step of the optimal plan is applied; the horizon
    then rolls forward.

    Parameters
    ----------
    grid_model : GridModel
    look_ahead_steps : int
        Number of future time steps in the MPC horizon.  Default 4.
    voll_usd_per_mwh : float
        Value of Lost Load in $/MWh, used as the EENS penalty weight.
        Default 10 000.
    max_ramp_rate_pu_per_step : float
        Maximum generator ramp rate as a fraction of capacity per step.
        Default 0.3.
    """

    def __init__(
        self,
        grid_model: GridModel,
        look_ahead_steps: int = 4,
        voll_usd_per_mwh: float = 10_000.0,
        max_ramp_rate_pu_per_step: float = 0.3,
    ) -> None:
        self.grid_model = grid_model
        self.look_ahead = max(1, look_ahead_steps)
        self.voll = voll_usd_per_mwh
        self.max_ramp_pu = max_ramp_rate_pu_per_step

        self._gen_indices: np.ndarray = np.array([], dtype=np.int64)
        self._gen_buses: np.ndarray = np.array([], dtype=np.int64)
        self._gen_pmax: np.ndarray = np.array([], dtype=np.float64)
        self._gen_cost: np.ndarray = np.array([], dtype=np.float64)
        self._n_gens: int = 0

        self._load_indices: np.ndarray = np.array([], dtype=np.int64)
        self._load_buses: np.ndarray = np.array([], dtype=np.int64)
        self._n_loads: int = 0

        self._bus_indices: np.ndarray = np.array([], dtype=np.int64)
        self._n_buses: int = 0

        self._prev_gen: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """Extract generator, load, and bus topology from the network."""
        net = self.grid_model.net

        gen_mask = net.gen["in_service"].values.astype(bool)
        self._gen_indices = net.gen.index[gen_mask].values
        self._gen_buses = net.gen.loc[gen_mask, "bus"].values.astype(np.int64)
        self._gen_pmax = net.gen.loc[gen_mask, "max_p_mw"].values.astype(np.float64)
        self._n_gens = len(self._gen_indices)

        if "cost_per_mwh" in net.gen.columns:
            self._gen_cost = net.gen.loc[gen_mask, "cost_per_mwh"].values.astype(np.float64)
        else:
            self._gen_cost = np.full(self._n_gens, 50.0, dtype=np.float64)

        load_mask = net.load["in_service"].values.astype(bool)
        self._load_indices = net.load.index[load_mask].values
        self._load_buses = net.load.loc[load_mask, "bus"].values.astype(np.int64)
        self._n_loads = len(self._load_indices)

        bus_mask = net.bus["in_service"].values.astype(bool)
        self._bus_indices = net.bus.index[bus_mask].values
        self._n_buses = len(self._bus_indices)

        self._prev_gen = np.zeros(self._n_gens, dtype=np.float64)

    # ------------------------------------------------------------------
    # Build and solve LP
    # ------------------------------------------------------------------

    def solve_step(
        self,
        dt_hours: float,
        load_mw: np.ndarray,
        line_limits_a: np.ndarray,
        ptdf: Optional[np.ndarray] = None,
        current_gen_mw: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Solve the MPC LP for the current time step.

        Parameters
        ----------
        dt_hours : float
            Duration of one time step in hours.
        load_mw : np.ndarray
            Active load per load element in MW, shape ``(n_loads,)``.
        line_limits_a : np.ndarray
            Thermal current limits per line in Amperes, shape
            ``(n_lines,)``.
        ptdf : np.ndarray or None
            Power Transfer Distribution Factor matrix, shape
            ``(n_lines, n_buses)``.  If ``None``, line constraints are
            omitted.
        current_gen_mw : np.ndarray or None
            Current generator dispatch in MW, shape ``(n_gens,)``.
            Used for ramp-rate constraints.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            - ``gen_mw`` — optimal generator dispatch (n_gens,)
            - ``curtail_mw`` — load curtailment per load (n_loads,)
            - ``objective_value`` — total cost for this step in USD
        """
        if current_gen_mw is not None:
            self._prev_gen = np.asarray(current_gen_mw, dtype=np.float64)

        H = self.look_ahead
        G = self._n_gens
        L = self._n_loads

        n_vars_per_step = G + L
        n_vars = H * n_vars_per_step

        c_obj = np.zeros(n_vars, dtype=np.float64)
        for h in range(H):
            base = h * n_vars_per_step
            c_obj[base : base + G] = self._gen_cost * dt_hours
            c_obj[base + G : base + G + L] = self.voll * dt_hours

        A_rows: List[np.ndarray] = []
        b_vals: List[float] = []

        total_load_mw = float(np.sum(load_mw))

        for h in range(H):
            base = h * n_vars_per_step

            row = np.zeros(n_vars, dtype=np.float64)
            row[base : base + G] = 1.0
            row[base + G : base + G + L] = -1.0
            A_rows.append(row)
            b_vals.append(total_load_mw)

        for h in range(H):
            base = h * n_vars_per_step
            for g in range(G):
                row = np.zeros(n_vars, dtype=np.float64)
                row[base + g] = 1.0
                A_rows.append(row)
                b_vals.append(self._gen_pmax[g])

        if self._prev_gen is not None:
            for h in range(H):
                base = h * n_vars_per_step
                for g in range(G):
                    prev = self._prev_gen[g] if h == 0 else 0.0
                    ramp_limit = self._gen_pmax[g] * self.max_ramp_pu

                    row_up = np.zeros(n_vars, dtype=np.float64)
                    row_up[base + g] = 1.0
                    A_rows.append(row_up)
                    b_vals.append(prev + ramp_limit)

                    row_down = np.zeros(n_vars, dtype=np.float64)
                    row_down[base + g] = -1.0
                    A_rows.append(row_down)
                    b_vals.append(-(prev - ramp_limit))

        if ptdf is not None and line_limits_a is not None:
            n_lines = ptdf.shape[0]
            bus_to_col = {int(b): i for i, b in enumerate(self._bus_indices)}
            for h in range(H):
                base = h * n_vars_per_step
                for li in range(n_lines):
                    row_pos = np.zeros(n_vars, dtype=np.float64)
                    row_neg = np.zeros(n_vars, dtype=np.float64)
                    for g in range(G):
                        bus = int(self._gen_buses[g])
                        col = bus_to_col.get(bus)
                        if col is not None:
                            row_pos[base + g] = ptdf[li, col]
                            row_neg[base + g] = -ptdf[li, col]
                    for ld in range(L):
                        bus = int(self._load_buses[ld])
                        col = bus_to_col.get(bus)
                        if col is not None:
                            row_pos[base + G + ld] = -ptdf[li, col]
                            row_neg[base + G + ld] = ptdf[li, col]
                    limit = line_limits_a[li]
                    A_rows.append(row_pos)
                    b_vals.append(limit)
                    A_rows.append(row_neg)
                    b_vals.append(limit)

        bounds = [(0.0, None)] * n_vars
        for h in range(H):
            base = h * n_vars_per_step
            for ld in range(L):
                bounds[base + G + ld] = (0.0, float(load_mw[ld]))

        A_ub = np.array(A_rows, dtype=np.float64) if A_rows else np.zeros((0, n_vars))
        b_ub = np.array(b_vals, dtype=np.float64) if b_vals else np.zeros(0)

        try:
            result = linprog(
                c_obj,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method="highs",
                options={"disp": False},
            )
        except Exception:
            logger.warning("LP solver failed; returning zero dispatch.")
            return (
                np.zeros(G, dtype=np.float64),
                np.zeros(L, dtype=np.float64),
                0.0,
            )

        if not result.success:
            logger.warning("LP did not converge: %s; returning zero dispatch.", result.message)
            return (
                np.zeros(G, dtype=np.float64),
                np.zeros(L, dtype=np.float64),
                0.0,
            )

        x = result.x
        gen_mw = x[:G].copy()
        curtail_mw = x[G : G + L].copy()
        obj_val = float(result.fun)

        self._prev_gen = gen_mw.copy()

        return gen_mw, curtail_mw, obj_val


# ---------------------------------------------------------------------------
# QSTSThermalDispatcher
# ---------------------------------------------------------------------------

@dataclass
class QSTSConfig:
    """Configuration for the QSTS thermal dispatcher.

    Attributes
    ----------
    dt_minutes : float
        Time step duration in minutes.  Default 15.
    n_steps : int
        Number of time steps to simulate.
    look_ahead_steps : int
        MPC horizon length in steps.  Default 4.
    voll_usd_per_mwh : float
        Value of Lost Load in $/MWh.  Default 10 000.
    max_cond_temp_c : float
        Maximum conductor temperature before trip in °C.  Default 100.
    thermal_capacity_j_per_k_m : float
        Lumped thermal capacity (J/(K·m)).  Default 500.
    temp_coeff_resistance : float
        TCR for aluminium (1/K).  Default 0.00403.
    max_ramp_rate_pu_per_step : float
        Generator ramp limit as fraction of Pmax per step.  Default 0.3.
    enable_mpc : bool
        If ``False``, skip predictive control (passive thermal tracking).
    """

    dt_minutes: float = 15.0
    n_steps: int = 96
    look_ahead_steps: int = 4
    voll_usd_per_mwh: float = 10_000.0
    max_cond_temp_c: float = _DEFAULT_MAX_COND_TEMP_C
    thermal_capacity_j_per_k_m: float = _DEFAULT_THERMAL_CAPACITY
    temp_coeff_resistance: float = _DEFAULT_TEMP_COEFF_RESISTANCE
    max_ramp_rate_pu_per_step: float = 0.3
    enable_mpc: bool = True


class QSTSThermalDispatcher:
    """Chronological QSTS solver with transient thermal dynamics.

    Co-optimises operating cost and EENS across sequential time steps
    while tracking conductor heat accumulation and triggering automatic
    line trips when temperature limits are exceeded.

    Parameters
    ----------
    grid_model : GridModel
        The pandapower network to simulate.
    config : QSTSConfig
        Simulation configuration.

    Attributes
    ----------
    grid_model : GridModel
    config : QSTSConfig
    profiles : dict
        Populated after :meth:`run` completes.
    """

    def __init__(
        self,
        grid_model: GridModel,
        config: Optional[QSTSConfig] = None,
    ) -> None:
        self.grid_model = grid_model
        self.config = config or QSTSConfig()

        self.tracker = ConductorThermalTracker(
            grid_model,
            max_cond_temp_c=self.config.max_cond_temp_c,
            thermal_capacity_j_per_k_m=self.config.thermal_capacity_j_per_k_m,
            temp_coeff_resistance=self.config.temp_coeff_resistance,
        )
        self.controller = PredictiveDispatchController(
            grid_model,
            look_ahead_steps=self.config.look_ahead_steps,
            voll_usd_per_mwh=self.config.voll_usd_per_mwh,
            max_ramp_rate_pu_per_step=self.config.max_ramp_rate_pu_per_step,
        )

        self.profiles: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def run(
        self,
        load_profile_mw: Optional[np.ndarray] = None,
        weather_profile: Optional[Dict[str, np.ndarray]] = None,
        initial_temps_c: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Execute the full QSTS simulation.

        Parameters
        ----------
        load_profile_mw : np.ndarray or None
            Active load per load element per step, shape
            ``(n_steps, n_loads)``.  If ``None``, uses the static load
            values from the pandapower network repeated for all steps.
        weather_profile : dict or None
            Keys: ``ambient_temp_c``, ``wind_speed_mps``,
            ``wind_angle_deg``, ``solar_radiation_w_m2``.  Each value is
            an array of shape ``(n_steps, n_lines)``.  If ``None``, uses
            mild defaults (25°C, 0.6 m/s, 90°, 1000 W/m²).
        initial_temps_c : np.ndarray or None
            Initial conductor temperatures in °C, shape ``(n_lines,)``.
            Defaults to ambient temperature.

        Returns
        -------
        dict
            Chronological profiles keyed by:

            - ``line_temperatures`` — ``(n_steps, n_lines)`` °C
            - ``generator_dispatch`` — ``(n_steps, n_gens)`` MW
            - ``load_curtailment`` — ``(n_steps, n_loads)`` MW
            - ``dispatch_cost`` — ``(n_steps,)`` USD
            - ``eens_mwh`` — ``(n_steps,)`` MWh
            - ``line_trips`` — list of ``(step, line_idx)``
            - ``converged`` — ``(n_steps,)`` bool
        """
        cfg = self.config
        dt_hours = cfg.dt_minutes / 60.0
        dt_seconds = cfg.dt_minutes * 60.0
        n_steps = cfg.n_steps

        self.tracker.initialise(initial_temp_c=initial_temps_c)
        self.controller.initialise()

        net = self.grid_model.net
        n_lines = self.tracker.n_lines
        n_gens = self.controller._n_gens
        n_loads = self.controller._n_loads
        line_indices = self.tracker.get_line_indices()

        if load_profile_mw is None:
            load_profile_mw = np.tile(
                net.load.loc[net.load["in_service"], "p_mw"].values.astype(np.float64),
                (n_steps, 1),
            )

        if weather_profile is None:
            weather_profile = {
                "ambient_temp_c": np.full((n_steps, n_lines), 25.0, dtype=np.float64),
                "wind_speed_mps": np.full((n_steps, n_lines), 0.6, dtype=np.float64),
                "wind_angle_deg": np.full((n_steps, n_lines), 90.0, dtype=np.float64),
                "solar_radiation_w_m2": np.full((n_steps, n_lines), 1000.0, dtype=np.float64),
            }

        T_profile = np.full((n_steps, n_lines), np.nan, dtype=np.float64)
        P_gen_profile = np.full((n_steps, n_gens), np.nan, dtype=np.float64)
        P_curtail_profile = np.full((n_steps, n_loads), 0.0, dtype=np.float64)
        cost_profile = np.zeros(n_steps, dtype=np.float64)
        eens_profile = np.zeros(n_steps, dtype=np.float64)
        converged_profile = np.zeros(n_steps, dtype=bool)
        trips: List[Tuple[int, int]] = []

        line_to_pos: Dict[int, int] = {int(idx): i for i, idx in enumerate(line_indices)}

        for step in range(n_steps):
            loads_mw = load_profile_mw[step].copy()

            for ld_idx, ld_row in enumerate(self.controller._load_indices):
                net.load.at[ld_row, "p_mw"] = float(loads_mw[ld_idx])

            pf_result = self.grid_model.run_baseline_power_flow()
            converged_profile[step] = pf_result["converged"]

            currents_a = np.zeros(n_lines, dtype=np.float64)
            if pf_result["converged"] and hasattr(net, "res_line"):
                for li, line_idx in enumerate(line_indices):
                    i_ka = float(net.res_line.at[line_idx, "i_ka"])
                    currents_a[li] = i_ka * 1000.0

            w_amb = weather_profile["ambient_temp_c"][step]
            w_wind = weather_profile["wind_speed_mps"][step]
            w_angle = weather_profile["wind_angle_deg"][step]
            w_solar = weather_profile["solar_radiation_w_m2"][step]

            T_profile[step] = self.tracker.step(
                line_currents_a=currents_a,
                ambient_temp_c=w_amb,
                wind_speed_mps=w_wind,
                wind_angle_deg=w_angle,
                solar_radiation_w_m2=w_solar,
                dt_seconds=dt_seconds,
            )

            tripped = self.tracker.detect_trips()
            for li in tripped:
                if net.line.at[li, "in_service"]:
                    net.line.at[li, "in_service"] = False
                    trips.append((step, int(li)))
                    logger.info("Step %d: line %d tripped on overtemperature.", step, li)

            if cfg.enable_mpc and pf_result["converged"]:
                T_current = self.tracker.get_temperatures()
                margin = self.config.max_cond_temp_c - T_current
                near_limit = np.any((margin < 15.0) & (margin > 0))

                if near_limit or len(tripped) > 0:
                    line_limits_a = np.full(n_lines, 9999.0, dtype=np.float64)
                    for li_pos, li_idx in enumerate(line_indices):
                        if net.line.at[li_idx, "in_service"]:
                            rating_ka = float(net.line.at[li_idx, "max_i_ka"])
                            line_limits_a[li_pos] = rating_ka * 1000.0

                    gen_mw, curtail_mw, obj_val = self.controller.solve_step(
                        dt_hours=dt_hours,
                        load_mw=loads_mw,
                        line_limits_a=line_limits_a,
                        ptdf=None,
                    )

                    for g_idx, g_row in enumerate(self.controller._gen_indices):
                        net.gen.at[g_row, "p_mw"] = float(gen_mw[g_idx])

                    for ld_idx, ld_row in enumerate(self.controller._load_indices):
                        curtail = float(curtail_mw[ld_idx])
                        net.load.at[ld_row, "p_mw"] = max(0.0, float(loads_mw[ld_idx]) - curtail)

                    pf_result = self.grid_model.run_baseline_power_flow()
                    converged_profile[step] = pf_result["converged"]

                    P_gen_profile[step] = gen_mw
                    P_curtail_profile[step] = curtail_mw
                    cost_profile[step] = obj_val
                    eens_profile[step] = float(np.sum(curtail_mw)) * dt_hours
                else:
                    gen_mw = np.zeros(n_gens, dtype=np.float64)
                    for g_idx, g_row in enumerate(self.controller._gen_indices):
                        gen_mw[g_idx] = float(net.gen.at[g_row, "p_mw"])
                    P_gen_profile[step] = gen_mw
                    cost_profile[step] = float(np.sum(gen_mw * self.controller._gen_cost * dt_hours))
                    eens_profile[step] = 0.0
            else:
                gen_mw = np.zeros(n_gens, dtype=np.float64)
                for g_idx, g_row in enumerate(self.controller._gen_indices):
                    gen_mw[g_idx] = float(net.gen.at[g_row, "p_mw"])
                P_gen_profile[step] = gen_mw
                cost_profile[step] = float(np.sum(gen_mw * self.controller._gen_cost * dt_hours))

        self.profiles = {
            "line_temperatures": T_profile,
            "generator_dispatch": P_gen_profile,
            "load_curtailment": P_curtail_profile,
            "dispatch_cost": cost_profile,
            "eens_mwh": eens_profile,
            "line_trips": trips,
            "converged": converged_profile,
        }
        return self.profiles

    # ------------------------------------------------------------------
    # Profile access
    # ------------------------------------------------------------------

    def get_profiles(self) -> Dict[str, pd.DataFrame]:
        """Return chronological profiles as pandas DataFrames.

        Returns
        -------
        dict
            Keys match :meth:`run` return values, each value is a
            ``pd.DataFrame`` with time-step rows.
        """
        if not self.profiles:
            raise RuntimeError("Call run() before get_profiles().")

        cfg = self.config
        index = pd.timedelta_range(
            start="0 min", periods=cfg.n_steps, freq=f"{cfg.dt_minutes}min"
        )

        result: Dict[str, pd.DataFrame] = {}

        T = self.profiles["line_temperatures"]
        if T.shape[1] > 0:
            result["line_temperatures"] = pd.DataFrame(
                T, index=index,
                columns=[f"line_{i}" for i in self.tracker.get_line_indices()],
            )

        G = self.profiles["generator_dispatch"]
        if G.shape[1] > 0:
            result["generator_dispatch"] = pd.DataFrame(
                G, index=index,
                columns=[f"gen_{i}" for i in self.controller._gen_indices],
            )

        C = self.profiles["load_curtailment"]
        if C.shape[1] > 0:
            result["load_curtailment"] = pd.DataFrame(
                C, index=index,
                columns=[f"load_{i}" for i in self.controller._load_indices],
            )

        result["dispatch_cost"] = pd.DataFrame(
            {"cost_usd": self.profiles["dispatch_cost"]}, index=index,
        )
        result["eens_mwh"] = pd.DataFrame(
            {"eens_mwh": self.profiles["eens_mwh"]}, index=index,
        )
        result["converged"] = pd.DataFrame(
            {"converged": self.profiles["converged"]}, index=index,
        )

        trips_df = pd.DataFrame(
            self.profiles["line_trips"], columns=["step", "line_idx"],
        )
        result["line_trips"] = trips_df

        return result

    def __repr__(self) -> str:
        return (
            f"QSTSThermalDispatcher(n_steps={self.config.n_steps}, "
            f"dt={self.config.dt_minutes}min, "
            f"mpc={self.config.enable_mpc})"
        )
