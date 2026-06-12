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
Overcurrent protection and operator reaction module.

Implements IEC 60255 / IEEE Inverse Definite Minimum Time (IDMT) relay
characteristics, a cumulative-timer cascading protection engine, and a
curative operator response module that models grid operator intervention
during cascading emergencies.

All constants align with NERC protection standards and IEEE C37.112.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from resilient_blackout.grid.network import GridModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IEEE C37.112 / IEC 60255 IDMT curve parameters
# ---------------------------------------------------------------------------

_IDMT_CURVES: Dict[str, Tuple[float, float]] = {
    "standard_inverse": (0.02, 0.14),
    "very_inverse": (1.0, 13.5),
    "extremely_inverse": (2.0, 80.0),
    "long_time_inverse": (1.0, 120.0),
}

_MIN_TRIP_TIME_S: float = 0.02
_DEFAULT_T_D: float = 0.1
_DEFAULT_I_PICKUP: float = 1.0


# ---------------------------------------------------------------------------
# Relay model
# ---------------------------------------------------------------------------

class RelayModel:
    """IEC 60255 / IEEE IDMT overcurrent relay characteristic.

    Computes tripping time using the standard inverse-time equation:

    .. math::

        t_{\\text{trip}} = \\frac{\\beta \\cdot T_d}
        {(I / I_{\\text{pickup}})^{\\alpha} - 1}

    Parameters
    ----------
    curve_type : str
        One of ``"standard_inverse"``, ``"very_inverse"``,
        ``"extremely_inverse"``, ``"long_time_inverse"``.
    T_d : float
        Time multiplier setting (0.025–1.2 typical).  Default 0.1.
    I_pickup : float
        Pickup current in Amperes.  Default 1.0.

    Attributes
    ----------
    curve_type : str
    alpha : float
    beta : float
    T_d : float
    I_pickup : float
    """

    def __init__(
        self,
        curve_type: str = "standard_inverse",
        T_d: float = _DEFAULT_T_D,
        I_pickup: float = _DEFAULT_I_PICKUP,
    ) -> None:
        curve_key = curve_type.lower().replace("-", "_").replace(" ", "_")
        if curve_key not in _IDMT_CURVES:
            raise ValueError(
                f"Unknown curve_type '{curve_type}'. "
                f"Choose from: {list(_IDMT_CURVES.keys())}"
            )

        self.curve_type = curve_key
        self.alpha, self.beta = _IDMT_CURVES[curve_key]
        self.T_d = T_d
        self.I_pickup = I_pickup

    def trip_time(self, I_line: float) -> float:
        """Compute tripping time for a single current value.

        Parameters
        ----------
        I_line : float
            Line current in Amperes.

        Returns
        -------
        float
            Trip time in seconds.  Returns ``inf`` if
            :math:`I \\le I_{\\text{pickup}}`.
        """
        if I_line <= self.I_pickup:
            return float("inf")
        ratio = I_line / self.I_pickup
        t = (self.beta * self.T_d) / (ratio**self.alpha - 1.0)
        return max(t, _MIN_TRIP_TIME_S)

    def trip_times(self, I_lines: np.ndarray) -> np.ndarray:
        """Vectorized tripping time computation.

        Parameters
        ----------
        I_lines : np.ndarray
            Array of line currents in Amperes.

        Returns
        -------
        np.ndarray
            Trip times in seconds.  ``inf`` where current is below
            pickup.
        """
        I = np.asarray(I_lines, dtype=np.float64)
        ratio = I / self.I_pickup
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self.beta * self.T_d) / (ratio**self.alpha - 1.0)
        t = np.where(I > self.I_pickup, t, np.inf)
        t = np.maximum(t, _MIN_TRIP_TIME_S)
        return t


# ---------------------------------------------------------------------------
# Cascading protection engine
# ---------------------------------------------------------------------------

class CascadingProtectionEngine:
    """Cumulative-timer protection engine for cascading failures.

    Tracks per-line trip timers that accumulate over simulation time
    steps.  When a line's accumulated timer reaches 1.0, the line trips.
    Timers reset when overloads clear, modeling relay reset
    characteristics.

    Parameters
    ----------
    grid_model : GridModel
        The grid model providing line count and ratings.
    relay_configs : dict or None
        Optional per-line relay settings.  Keys are line indices;
        values are dicts with ``curve_type``, ``T_d``, ``I_pickup``.
    default_curve : str
        Default IDMT curve for unconfigured lines.
    default_T_d : float
        Default time multiplier.
    time_step_s : float
        Base simulation time step in seconds.  Default 1.0.

    Attributes
    ----------
    relays : dict
        Line index → ``RelayModel``.
    trip_timers : dict
        Line index → accumulated trip fraction (0–1).
    """

    def __init__(
        self,
        grid_model: GridModel,
        relay_configs: Optional[Dict[int, Dict[str, Any]]] = None,
        default_curve: str = "standard_inverse",
        default_T_d: float = _DEFAULT_T_D,
        time_step_s: float = 1.0,
    ) -> None:
        self.grid_model = grid_model
        self.time_step_s = time_step_s
        self.relay_configs = relay_configs or {}
        self.default_curve = default_curve
        self.default_T_d = default_T_d

        self.relays: Dict[int, RelayModel] = {}
        self.trip_timers: Dict[int, float] = {}
        self._build_relays()

    def _build_relays(self) -> None:
        """Instantiate relay models for all lines."""
        net = self.grid_model.net
        for idx in net.line.index:
            cfg = self.relay_configs.get(idx, {})
            curve = cfg.get("curve_type", self.default_curve)
            T_d = cfg.get("T_d", self.default_T_d)
            I_pickup = cfg.get(
                "I_pickup",
                float(net.line.at[idx, "max_i_ka"]) * 1000.0 * 1.2,
            )
            self.relays[idx] = RelayModel(
                curve_type=curve, T_d=T_d, I_pickup=I_pickup
            )
            self.trip_timers[idx] = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        net: Any,
        overloaded_lines: List[int],
        line_currents: Dict[int, float],
        dt_s: Optional[float] = None,
    ) -> List[int]:
        """Advance protection timers and return newly tripped lines.

        Parameters
        ----------
        net : pandapowerNet
            The network (used to set ``in_service=False`` on tripped
            lines).
        overloaded_lines : list of int
            Line indices currently exceeding their rating.
        line_currents : dict
            Line index → current in Amperes.
        dt_s : float or None
            Time step in seconds.  Defaults to ``self.time_step_s``.

        Returns
        -------
        list of int
            Line indices that tripped during this step.
        """
        dt = dt_s if dt_s is not None else self.time_step_s
        tripped: List[int] = []

        overloaded_set = set(overloaded_lines)

        for idx in self.relays:
            if idx not in overloaded_set:
                if self.trip_timers.get(idx, 0.0) > 0:
                    self.trip_timers[idx] = max(0.0, self.trip_timers[idx] - dt / 10.0)
                continue

            if idx not in net.line.index or not net.line.at[idx, "in_service"]:
                continue

            I = line_currents.get(idx, 0.0)
            relay = self.relays[idx]
            t_trip = relay.trip_time(I)

            if np.isinf(t_trip):
                continue

            self.trip_timers[idx] += dt / t_trip

            if self.trip_timers[idx] >= 1.0:
                net.line.at[idx, "in_service"] = False
                tripped.append(idx)
                self.trip_timers[idx] = 0.0
                logger.info(
                    "Protection trip: line %d (I=%.0f A, pickup=%.0f A, "
                    "timer=%.3f)",
                    idx, I, relay.I_pickup, self.trip_timers[idx] + dt / t_trip,
                )

        return tripped

    def reset(self) -> None:
        """Clear all accumulated trip timers."""
        for idx in self.trip_timers:
            self.trip_timers[idx] = 0.0


# ---------------------------------------------------------------------------
# Operator response module
# ---------------------------------------------------------------------------

class OperatorResponseModule:
    """Curative operator intervention during cascading emergencies.

    Models the grid operator's ability to isolate faults and redistribute
    power after a defined intervention window (e.g., 5 minutes of thermal
    inertia).  Uses heuristic topology search to find stabilizing actions.

    Parameters
    ----------
    grid_model : GridModel
        The grid model.
    intervention_window_s : float
        Minimum cascade elapsed time before operator can act (seconds).
        Default 300 (5 minutes), matching the DLR thermal inertia
        window.
    max_search_iterations : int
        Maximum number of topology search attempts per intervention.

    Attributes
    ----------
    grid_model : GridModel
    intervention_window_s : float
    max_search_iterations : int
    intervention_count : int
    """

    def __init__(
        self,
        grid_model: GridModel,
        intervention_window_s: float = 300.0,
        max_search_iterations: int = 10,
    ) -> None:
        self.grid_model = grid_model
        self.intervention_window_s = intervention_window_s
        self.max_search_iterations = max_search_iterations
        self.intervention_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attempt_intervention(
        self,
        net: Any,
        cascade_elapsed_s: float,
        overloaded_lines: List[int],
        tripped_lines: List[int],
    ) -> Dict[str, Any]:
        """Attempt curative operator action if the intervention window
        has elapsed.

        Parameters
        ----------
        net : pandapowerNet
            The current network state.
        cascade_elapsed_s : float
            Total elapsed time since cascade began.
        overloaded_lines : list of int
            Currently overloaded line indices.
        tripped_lines : list of int
            Already-tripped line indices.

        Returns
        -------
        dict
            ``{"action": str or None, "line": int or None,
            "stabilized": bool}``.
        """
        if cascade_elapsed_s < self.intervention_window_s:
            return {"action": None, "line": None, "stabilized": False}

        if not overloaded_lines:
            return {"action": None, "line": None, "stabilized": True}

        result = self._search_topology_fix(net, overloaded_lines)
        if result is not None:
            self.intervention_count += 1
            return result

        return {"action": None, "line": None, "stabilized": False}

    def _search_topology_fix(
        self,
        net: Any,
        overloaded_lines: List[int],
    ) -> Optional[Dict[str, Any]]:
        """Try isolating each overloaded line to find a stabilizing action.

        For each overloaded line, tests whether disconnecting it clears
        all remaining overloads.  Returns the first successful action.

        Parameters
        ----------
        net : pandapowerNet
        overloaded_lines : list of int

        Returns
        -------
        dict or None
        """
        import pandapower as pp

        sorted_lines = sorted(
            overloaded_lines,
            key=lambda lidx: (
                net.res_line.at[lidx, "loading_percent"]
                if hasattr(net, "res_line") and lidx in net.res_line.index
                else 0.0
            ),
            reverse=True,
        )

        for attempt, candidate in enumerate(sorted_lines):
            if attempt >= self.max_search_iterations:
                break

            test_net = copy.deepcopy(net)
            if candidate in test_net.line.index:
                test_net.line.at[candidate, "in_service"] = False

            try:
                pp.runpp(test_net, numba=False)
            except pp.LoadflowNotConverged:
                try:
                    pp.rundcpp(test_net)
                except Exception:
                    continue

            if hasattr(test_net, "res_line"):
                remaining_overloads = (
                    test_net.res_line.loading_percent > 100.0
                ) & test_net.line.in_service
                if not remaining_overloads.any():
                    net.line.at[candidate, "in_service"] = False
                    logger.info(
                        "Operator isolated line %d — cascade stabilized.", candidate
                    )
                    return {
                        "action": "isolated",
                        "line": candidate,
                        "stabilized": True,
                    }

        return None
