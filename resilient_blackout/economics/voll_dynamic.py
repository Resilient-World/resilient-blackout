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
Advanced non-linear economic damage estimator.

Provides ``DynamicVoLLCalculator`` for modelling duration-dependent,
time-of-use, and seasonally adjusted Value of Lost Load (VoLL) with
event cost triggers, suitable for integration with Monte Carlo and
storyline engines.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_BASE_VOLL: Dict[str, float] = {
    "residential": 10000.0,
    "commercial": 25000.0,
    "industrial": 75000.0,
}

_DEFAULT_GAMMA: Dict[str, float] = {
    "residential": 1.15,
    "commercial": 1.08,
    "industrial": 1.05,
}

_DEFAULT_SEASONAL: Dict[int, float] = {
    1: 1.20, 2: 1.20, 3: 1.00, 4: 1.00, 5: 1.00, 6: 1.10,
    7: 1.30, 8: 1.30, 9: 1.00, 10: 1.00, 11: 1.00, 12: 1.15,
}

_EPS: float = 1e-10


def _default_tou_matrix() -> np.ndarray:
    """Build default 24×7 time-of-use multiplier matrix.

    Returns
    -------
    np.ndarray
        Shape ``(24, 7)``.  Rows = hour (0–23), cols = day (Mon=0..Sun=6).
    """
    tou = np.ones((24, 7), dtype=np.float64)

    tou[0:6, :] = 0.70
    tou[6:9, :] = 0.90
    tou[9:16, :] = 1.00
    tou[16:21, :] = 1.30
    tou[21:24, :] = 1.10

    tou[:, 5] *= 0.95
    tou[:, 6] *= 0.85

    return tou


# ---------------------------------------------------------------------------
# DynamicVoLLCalculator
# ---------------------------------------------------------------------------

class DynamicVoLLCalculator:
    """Duration-dependent and time-of-use escalating VoLL calculator.

    Models non-linear societal disruption costs that grow with outage
    duration, vary by time of day and season, and include fixed event
    overhead costs.

    Parameters
    ----------
    base_voll : dict or None
        Base VoLL in $/MWh per sector.  Defaults to literature values.
    gamma_params : dict or None
        Duration escalation exponent per sector (γ > 1).
    tou_matrix : np.ndarray or None
        24×7 array of hourly multipliers.  Default from typical load
        profiles.
    seasonal_factors : dict or None
        ``{month: float}`` multipliers.  Default winter/summer peaks.
    event_cost_trigger : float
        Fixed cost per outage event in USD.  Default 0.
    config_path : str or None
        Optional path to YAML or JSON config file overriding all
        parameters.

    Attributes
    ----------
    base_voll : dict
    gamma : dict
    tou_matrix : np.ndarray
    seasonal : dict
    event_cost_trigger : float
    """

    def __init__(
        self,
        base_voll: Optional[Dict[str, float]] = None,
        gamma_params: Optional[Dict[str, float]] = None,
        tou_matrix: Optional[np.ndarray] = None,
        seasonal_factors: Optional[Dict[int, float]] = None,
        event_cost_trigger: float = 0.0,
        config_path: Optional[str] = None,
    ) -> None:
        if config_path is not None:
            cfg = self.load_config(config_path)
            base_voll = cfg.get("base_voll", base_voll)
            gamma_params = cfg.get("gamma_params", gamma_params)
            tou_matrix = cfg.get("tou_matrix", tou_matrix)
            seasonal_factors = cfg.get("seasonal_factors", seasonal_factors)
            event_cost_trigger = cfg.get("event_cost_trigger", event_cost_trigger)

        self.base_voll = dict(_DEFAULT_BASE_VOLL)
        if base_voll:
            self.base_voll.update(base_voll)

        self.gamma = dict(_DEFAULT_GAMMA)
        if gamma_params:
            self.gamma.update(gamma_params)

        if tou_matrix is not None:
            self.tou_matrix = np.asarray(tou_matrix, dtype=np.float64)
            if self.tou_matrix.shape != (24, 7):
                raise ValueError(
                    f"tou_matrix must be (24, 7), got {self.tou_matrix.shape}"
                )
        else:
            self.tou_matrix = _default_tou_matrix()

        self.seasonal = dict(_DEFAULT_SEASONAL)
        if seasonal_factors:
            self.seasonal.update(seasonal_factors)

        self.event_cost_trigger = event_cost_trigger

        self._validate()

    def _validate(self) -> None:
        """Validate parameter ranges."""
        for sector, g in self.gamma.items():
            if g < 1.0:
                raise ValueError(
                    f"gamma for '{sector}' must be >= 1.0, got {g}"
                )
        if self.event_cost_trigger < 0:
            raise ValueError(
                f"event_cost_trigger must be non-negative, got {self.event_cost_trigger}"
            )

    # ------------------------------------------------------------------
    # Single-point computation
    # ------------------------------------------------------------------

    def compute_dynamic_voll(
        self,
        sector: str,
        duration_hours: float,
        month: int,
        hour: int,
        day_of_week: int = 0,
    ) -> float:
        """Compute dynamic VoLL for a single outage scenario.

        .. math::

            \\text{VoLL}(t) = \\text{VoLL}_{\\text{base}}
            \\times t^{\\gamma}
            \\times \\text{TOU}(h, d)
            \\times \\text{Seasonal}(m)

        Parameters
        ----------
        sector : str
            ``"residential"``, ``"commercial"``, or ``"industrial"``.
        duration_hours : float
            Outage duration in hours.
        month : int
            Month (1–12).
        hour : int
            Hour of day (0–23).
        day_of_week : int
            Day of week (0=Mon, 6=Sun).  Default 0.

        Returns
        -------
        float
            Dynamic VoLL in $/MWh.
        """
        base = self.base_voll.get(sector, self.base_voll.get("residential", 10000.0))
        g = self.gamma.get(sector, 1.0)

        duration_factor = max(1.0, duration_hours) ** g

        h = max(0, min(23, hour))
        d = max(0, min(6, day_of_week))
        tou_factor = float(self.tou_matrix[h, d])

        m = max(1, min(12, month))
        seasonal_factor = self.seasonal.get(m, 1.0)

        return base * duration_factor * tou_factor * seasonal_factor

    # ------------------------------------------------------------------
    # Vectorized batch computation
    # ------------------------------------------------------------------

    def compute_dynamic_voll_batch(
        self,
        sectors: Union[List[str], np.ndarray],
        durations: np.ndarray,
        months: np.ndarray,
        hours: np.ndarray,
        days_of_week: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Vectorized dynamic VoLL for batches of outage scenarios.

        Parameters
        ----------
        sectors : list of str or np.ndarray
            Sector per scenario.
        durations : np.ndarray
            Duration in hours per scenario.
        months : np.ndarray
            Month (1–12) per scenario.
        hours : np.ndarray
            Hour (0–23) per scenario.
        days_of_week : np.ndarray or None
            Day (0–6) per scenario.  Default 0.

        Returns
        -------
        np.ndarray
            Dynamic VoLL values in $/MWh.
        """
        n = len(durations)
        result = np.empty(n, dtype=np.float64)

        if days_of_week is None:
            days_of_week = np.zeros(n, dtype=np.int32)

        durs = np.asarray(durations, dtype=np.float64)
        mons = np.asarray(months, dtype=np.int32)
        hrs = np.asarray(hours, dtype=np.int32)
        dows = np.asarray(days_of_week, dtype=np.int32)

        hrs = np.clip(hrs, 0, 23)
        dows = np.clip(dows, 0, 6)
        mons = np.clip(mons, 1, 12)

        for sector in set(sectors):
            mask = np.array([s == sector for s in sectors])
            if not mask.any():
                continue

            base = self.base_voll.get(sector, self.base_voll.get("residential", 10000.0))
            g = self.gamma.get(sector, 1.0)

            dur_factor = np.maximum(1.0, durs[mask]) ** g
            tou_factor = self.tou_matrix[hrs[mask], dows[mask]]

            seasonal_arr = np.array(
                [self.seasonal.get(m, 1.0) for m in mons[mask]],
                dtype=np.float64,
            )

            result[mask] = base * dur_factor * tou_factor * seasonal_arr

        return result

    # ------------------------------------------------------------------
    # Event cost
    # ------------------------------------------------------------------

    def compute_event_cost(self, n_failed_assets: int) -> float:
        """Compute fixed event overhead cost.

        Parameters
        ----------
        n_failed_assets : int
            Number of failed assets.

        Returns
        -------
        float
            Event cost in USD.
        """
        if self.event_cost_trigger <= 0:
            return 0.0
        return self.event_cost_trigger * max(1, n_failed_assets)

    # ------------------------------------------------------------------
    # Dynamic avoided loss
    # ------------------------------------------------------------------

    def compute_avoided_loss_dynamic(
        self,
        baseline_eens_mwh: float,
        resilient_eens_mwh: float,
        sector_mix: Dict[str, float],
        duration_hours: float,
        month: int = 6,
        hour: int = 14,
        day_of_week: int = 2,
        n_failed_assets_baseline: int = 0,
        n_failed_assets_resilient: int = 0,
    ) -> Dict[str, Any]:
        """Compute avoided loss with dynamic VoLL weighting.

        Parameters
        ----------
        baseline_eens_mwh : float
            EENS for baseline configuration.
        resilient_eens_mwh : float
            EENS for resilient configuration.
        sector_mix : dict
            Fraction of load per sector (must sum to ~1.0).
        duration_hours : float
            Representative outage duration.
        month, hour, day_of_week : int
            Temporal context.
        n_failed_assets_baseline, n_failed_assets_resilient : int
            Failed asset counts for event cost calculation.

        Returns
        -------
        dict
            ``{"avoided_loss_usd": float, "dynamic_voll_used": dict,
            "event_cost_savings_usd": float}``.
        """
        total = sum(sector_mix.values())
        if total < _EPS:
            raise ValueError("sector_mix fractions must sum to > 0")

        dynamic_voll: Dict[str, float] = {}
        weighted_voll = 0.0

        for sector, frac in sector_mix.items():
            dv = self.compute_dynamic_voll(sector, duration_hours, month, hour, day_of_week)
            dynamic_voll[sector] = dv
            weighted_voll += dv * frac / total

        baseline_risk = baseline_eens_mwh * weighted_voll
        resilient_risk = resilient_eens_mwh * weighted_voll
        avoided_loss = baseline_risk - resilient_risk

        event_cost_baseline = self.compute_event_cost(n_failed_assets_baseline)
        event_cost_resilient = self.compute_event_cost(n_failed_assets_resilient)
        event_cost_savings = event_cost_baseline - event_cost_resilient

        return {
            "avoided_loss_usd": avoided_loss + event_cost_savings,
            "dynamic_voll_used": dynamic_voll,
            "weighted_voll_usd_per_mwh": weighted_voll,
            "event_cost_savings_usd": event_cost_savings,
        }

    # ------------------------------------------------------------------
    # Configuration I/O
    # ------------------------------------------------------------------

    @classmethod
    def load_config(cls, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file.

        Parameters
        ----------
        config_path : str

        Returns
        -------
        dict
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    return yaml.safe_load(f)
                except ImportError:
                    raise ImportError(
                        "PyYAML is required to read YAML config files. "
                        "Install with: pip install pyyaml"
                    )
            else:
                return json.load(f)

    def to_config(self, config_path: str) -> None:
        """Export current configuration to a JSON file.

        Parameters
        ----------
        config_path : str
        """
        cfg = {
            "base_voll": self.base_voll,
            "gamma_params": self.gamma,
            "tou_matrix": self.tou_matrix.tolist(),
            "seasonal_factors": self.seasonal,
            "event_cost_trigger": self.event_cost_trigger,
        }

        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)

        logger.info("DynamicVoLL configuration exported to %s", path)
