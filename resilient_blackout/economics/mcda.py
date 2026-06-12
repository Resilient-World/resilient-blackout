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
Decision support and environmental justice engine.

Provides ``EquityWeightedVoLLCalculator`` for adjusting Value of Lost
Load by CDC Social Vulnerability Index (SVI), and
``MultiCriteriaDecisionSolver`` for ranking resilience investment
pathways using TOPSIS or Simple Additive Weighting (SAW) with
configurable criteria weights and risk aversion.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from resilient_blackout.core.base import Asset

logger = logging.getLogger(__name__)

MethodType = Literal["topsis", "saw"]
CriteriaType = Literal["benefit", "cost"]

_DEFAULT_CRITERIA_WEIGHTS: Dict[str, float] = {
    "npv_usd": 0.30,
    "equity_index": 0.30,
    "health_safety_score": 0.25,
    "implementation_cost_usd": 0.15,
}

_DEFAULT_CRITERIA_TYPES: Dict[str, CriteriaType] = {
    "npv_usd": "benefit",
    "equity_index": "benefit",
    "health_safety_score": "benefit",
    "implementation_cost_usd": "cost",
}


# ---------------------------------------------------------------------------
# Equity-weighted VoLL calculator
# ---------------------------------------------------------------------------

class EquityWeightedVoLLCalculator:
    """Adjusts nominal VoLL by CDC Social Vulnerability Index (SVI).

    .. math::

        \\text{VoLL}_{\\text{equity}}(a) = \\text{VoLL}_{\\text{base}}(a)
        \\times (1 + \\omega \\times \\text{SVI}_a)

    where :math:`\\omega` controls the strength of the equity adjustment
    and :math:`\\text{SVI}_a \\in [0, 1]` is the social vulnerability
    percentile for the census tract containing asset :math:`a`.

    Parameters
    ----------
    base_voll_by_sector : dict
        Nominal VoLL in $/MWh keyed by sector name (e.g.,
        ``"residential"``, ``"commercial"``, ``"industrial"``).
    svi_data : dict or None
        Mapping from ``asset_id`` to SVI percentile (0–1).  If
        ``None``, a neutral SVI of 0.5 is assumed for all assets.
    omega : float
        Equity weighting factor.  0 = no adjustment; higher values
        give more weight to vulnerable communities.  Default 1.0.

    Attributes
    ----------
    base_voll : dict
    svi_data : dict
    omega : float
    """

    def __init__(
        self,
        base_voll_by_sector: Dict[str, float],
        svi_data: Optional[Dict[str, float]] = None,
        omega: float = 1.0,
    ) -> None:
        if omega < 0:
            raise ValueError(f"omega must be non-negative, got {omega}")

        self.base_voll = dict(base_voll_by_sector)
        self.svi_data: Dict[str, float] = dict(svi_data) if svi_data else {}
        self.omega = omega

    def compute_equity_voll(
        self,
        asset_id: str,
        sector: str = "residential",
    ) -> float:
        """Compute equity-adjusted VoLL for a single asset.

        Parameters
        ----------
        asset_id : str
            Asset identifier.
        sector : str
            Sector name for base VoLL lookup.

        Returns
        -------
        float
            Equity-adjusted VoLL in $/MWh.
        """
        base = self.base_voll.get(sector, self.base_voll.get("default", 10000.0))
        svi = self.svi_data.get(asset_id, 0.5)
        svi = max(0.0, min(1.0, svi))
        return base * (1.0 + self.omega * svi)

    def compute_equity_voll_batch(
        self,
        asset_ids: List[str],
        sector: str = "residential",
    ) -> np.ndarray:
        """Vectorized equity-adjusted VoLL for multiple assets.

        Parameters
        ----------
        asset_ids : list of str
        sector : str

        Returns
        -------
        np.ndarray
            Equity-adjusted VoLL values in $/MWh.
        """
        base = self.base_voll.get(sector, self.base_voll.get("default", 10000.0))
        svi_vals = np.array(
            [self.svi_data.get(aid, 0.5) for aid in asset_ids],
            dtype=np.float64,
        )
        svi_vals = np.clip(svi_vals, 0.0, 1.0)
        return base * (1.0 + self.omega * svi_vals)

    def build_voll_by_sector(
        self,
        assets: List[Asset],
    ) -> Dict[str, float]:
        """Aggregate equity-adjusted VoLL by sector from an asset list.

        Reads sector from ``asset.original_properties["sector"]``,
        defaulting to ``"residential"``.

        Parameters
        ----------
        assets : list of Asset

        Returns
        -------
        dict
            Sector → mean equity-adjusted VoLL in $/MWh.
        """
        sector_volls: Dict[str, List[float]] = {}

        for asset in assets:
            sector = str(asset.original_properties.get("sector", "residential"))
            voll = self.compute_equity_voll(asset.asset_id, sector)
            sector_volls.setdefault(sector, []).append(voll)

        return {
            sector: float(np.mean(volls))
            for sector, volls in sector_volls.items()
        }


# ---------------------------------------------------------------------------
# Multi-criteria decision solver
# ---------------------------------------------------------------------------

class MultiCriteriaDecisionSolver:
    """Multi-criteria ranking of resilience investment pathways.

    Supports TOPSIS (Technique for Order of Preference by Similarity to
    Ideal Solution) and SAW (Simple Additive Weighting) with configurable
    criteria weights, criteria types (benefit/cost), and risk aversion.

    Parameters
    ----------
    criteria_weights : dict or None
        Weight per criterion.  Defaults to equal weights across the
        four standard criteria.
    criteria_types : dict or None
        ``"benefit"`` or ``"cost"`` per criterion.
    method : str
        ``"topsis"`` or ``"saw"``.  Default ``"topsis"``.
    risk_aversion_gamma : float
        Exponential utility parameter for risk-averse decision making.
        0 = risk-neutral (linear utility).  Higher values = more
        risk-averse.  Default 0.

    Attributes
    ----------
    weights : dict
    types : dict
    method : str
    gamma : float
    """

    def __init__(
        self,
        criteria_weights: Optional[Dict[str, float]] = None,
        criteria_types: Optional[Dict[str, CriteriaType]] = None,
        method: MethodType = "topsis",
        risk_aversion_gamma: float = 0.0,
    ) -> None:
        self.weights = dict(criteria_weights) if criteria_weights else dict(_DEFAULT_CRITERIA_WEIGHTS)
        self.types: Dict[str, CriteriaType] = (
            dict(criteria_types) if criteria_types else dict(_DEFAULT_CRITERIA_TYPES)
        )
        self.method: MethodType = method
        self.gamma = risk_aversion_gamma

        self._criteria_order = list(self.weights.keys())
        self._n_criteria = len(self._criteria_order)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_scenarios(
        self,
        scenarios: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate and rank a list of investment scenarios.

        Each scenario must contain keys matching the configured criteria.

        Parameters
        ----------
        scenarios : list of dict
            Each dict has at minimum ``"name"`` (str) plus numeric
            values for each criterion.

        Returns
        -------
        dict
            ``{"rankings": list of dict, "scores": np.ndarray,
            "method": str}``.  Rankings are sorted best-to-worst.
        """
        if not scenarios:
            return {"rankings": [], "scores": np.array([]), "method": self.method}

        matrix = self._build_decision_matrix(scenarios)
        matrix = self._apply_risk_aversion(matrix)

        w = np.array([self.weights[c] for c in self._criteria_order], dtype=np.float64)
        types_list = [self.types[c] for c in self._criteria_order]

        if self.method == "topsis":
            scores = self._topsis(matrix, w, types_list)
        else:
            scores = self._saw(matrix, w, types_list)

        rankings = sorted(
            [
                {"name": s["name"], "score": float(scores[i]), **s}
                for i, s in enumerate(scenarios)
            ],
            key=lambda x: x["score"],
            reverse=True,
        )

        return {
            "rankings": rankings,
            "scores": scores,
            "method": self.method,
        }

    def set_risk_aversion(self, gamma: float) -> None:
        """Update the risk aversion parameter.

        Parameters
        ----------
        gamma : float
            New exponential utility parameter.
        """
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        self.gamma = gamma

    # ------------------------------------------------------------------
    # Internal: decision matrix
    # ------------------------------------------------------------------

    def _build_decision_matrix(self, scenarios: List[Dict[str, Any]]) -> np.ndarray:
        """Extract criteria values into a numpy matrix.

        Parameters
        ----------
        scenarios : list of dict

        Returns
        -------
        np.ndarray
            Shape ``(n_scenarios, n_criteria)``.
        """
        n = len(scenarios)
        matrix = np.empty((n, self._n_criteria), dtype=np.float64)

        for i, scenario in enumerate(scenarios):
            for j, criterion in enumerate(self._criteria_order):
                matrix[i, j] = float(scenario.get(criterion, 0.0))

        return matrix

    def _apply_risk_aversion(self, matrix: np.ndarray) -> np.ndarray:
        """Apply exponential utility transform for risk aversion.

        For benefit criteria: :math:`U(x) = (1 - e^{-\\gamma x}) / (1 - e^{-\\gamma})`.
        For cost criteria, the input is negated first.

        Parameters
        ----------
        matrix : np.ndarray

        Returns
        -------
        np.ndarray
        """
        if self.gamma <= 0:
            return matrix.copy()

        result = matrix.copy()
        denom = 1.0 - np.exp(-self.gamma)

        for j, criterion in enumerate(self._criteria_order):
            if self.types[criterion] == "benefit":
                result[:, j] = (1.0 - np.exp(-self.gamma * matrix[:, j])) / denom
            else:
                negated = -matrix[:, j]
                result[:, j] = -(1.0 - np.exp(-self.gamma * negated)) / denom

        return result

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_vector(matrix: np.ndarray) -> np.ndarray:
        """Vector (Euclidean) normalization for TOPSIS.

        Parameters
        ----------
        matrix : np.ndarray

        Returns
        -------
        np.ndarray
        """
        norms = np.sqrt(np.sum(matrix**2, axis=0))
        norms = np.maximum(norms, 1e-12)
        return matrix / norms

    @staticmethod
    def _normalize_minmax(matrix: np.ndarray, types_list: List[CriteriaType]) -> np.ndarray:
        """Min-max normalization for SAW.

        Parameters
        ----------
        matrix : np.ndarray
        types_list : list of str

        Returns
        -------
        np.ndarray
        """
        n, m = matrix.shape
        result = np.empty_like(matrix)

        for j in range(m):
            col = matrix[:, j]
            col_min = np.min(col)
            col_max = np.max(col)
            denom = col_max - col_min

            if denom < 1e-12:
                result[:, j] = 0.5
            elif types_list[j] == "benefit":
                result[:, j] = (col - col_min) / denom
            else:
                result[:, j] = (col_max - col) / denom

        return result

    # ------------------------------------------------------------------
    # TOPSIS
    # ------------------------------------------------------------------

    def _topsis(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types_list: List[CriteriaType],
    ) -> np.ndarray:
        """Compute TOPSIS closeness scores.

        Parameters
        ----------
        matrix : np.ndarray
        weights : np.ndarray
        types_list : list of str

        Returns
        -------
        np.ndarray
            Closeness scores in [0, 1].  Higher is better.
        """
        normed = self._normalize_vector(matrix)
        weighted = normed * weights

        ideal = np.empty(self._n_criteria, dtype=np.float64)
        anti_ideal = np.empty(self._n_criteria, dtype=np.float64)

        for j, ctype in enumerate(types_list):
            col = weighted[:, j]
            if ctype == "benefit":
                ideal[j] = np.max(col)
                anti_ideal[j] = np.min(col)
            else:
                ideal[j] = np.min(col)
                anti_ideal[j] = np.max(col)

        d_plus = np.sqrt(np.sum((weighted - ideal) ** 2, axis=1))
        d_minus = np.sqrt(np.sum((weighted - anti_ideal) ** 2, axis=1))

        denom = d_plus + d_minus
        denom = np.maximum(denom, 1e-12)
        return d_minus / denom

    # ------------------------------------------------------------------
    # SAW
    # ------------------------------------------------------------------

    def _saw(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        types_list: List[CriteriaType],
    ) -> np.ndarray:
        """Compute SAW scores.

        Parameters
        ----------
        matrix : np.ndarray
        weights : np.ndarray
        types_list : list of str

        Returns
        -------
        np.ndarray
            SAW scores.  Higher is better.
        """
        normed = self._normalize_minmax(matrix, types_list)
        return np.sum(normed * weights, axis=1)
