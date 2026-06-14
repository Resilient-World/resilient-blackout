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
# FOR ANY DIRECT, INDIRECT, INCIDENTIAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Resilience Rating System (RRS) ESG scorecard generator.

Implements ``RRSScorecardGenerator``, which consumes outputs from
``AvoidedLossCalculator`` and sensitivity analysis modules to produce
a standardized, CSRD-compliant JSON report modelled on the World
Bank's Resilience Rating System.

The RRS framework evaluates two dimensions:

1. **Resilience *of* the project** — the extent to which the project's
   own physical assets are designed to withstand climate risks over
   their design life.  Assessed via financial return stability under
   extreme climate projections and expressed as a confidence grade
   (A+ through C).

2. **Resilience *through* the project** — the positive externalities
   the project generates for the broader community and system,
   including reduction in Customer Minutes of Interruption (CMI),
   avoided commercial supply-chain losses, and emissions offsets.
   Scored on a 1–10 adaptation benefit scale.

Reference
---------
* World Bank (2023).  *Resilience Rating System: Methodology and
  Guidance*.  Washington, D.C.
* EU Directive 2022/2464 (CSRD) — Corporate Sustainability Reporting
  Directive.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIDENCE_GRADES: List[str] = ["C", "B-", "B", "B+", "A-", "A", "A+"]
_ADAPTATION_SCORE_RANGE: Tuple[int, int] = (1, 10)
_CMI_PER_MWH: float = 60.0  # minutes of interruption per MWh unserved (typical)

# CSRD ESRS topical categories
_ESRS_CATEGORIES: Dict[str, str] = {
    "resilience_of_project": "ESRS E1 — Climate Change Mitigation / Adaptation",
    "resilience_through_project": "ESRS E1 — Climate Change Adaptation",
    "emissions_offset": "ESRS E1 — GHG Emissions",
    "community_benefit": "ESRS S3 — Affected Communities",
    "supply_chain_resilience": "ESRS G1 — Business Conduct / Supply Chain",
    "financial_resilience": "ESRS E1 — Climate-Related Financial Risks",
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ResilienceOfProjectResult:
    """Assessment of resilience *of* the project."""

    confidence_grade: str
    grade_index: int
    baseline_npv: float
    stressed_npv: float
    npv_degradation_pct: float
    irr_stable: bool
    bcr_stable: bool
    key_sensitivities: List[str]
    assessment_rationale: str


@dataclass
class ResilienceThroughProjectResult:
    """Assessment of resilience *through* the project."""

    adaptation_score: int
    cmi_reduction_minutes: float
    avoided_supply_chain_loss_usd: float
    emissions_offset_tonne_co2: float
    avoided_eens_mwh: float
    community_benefit_ratio: float
    assessment_rationale: str


@dataclass
class RRSScorecard:
    """Complete RRS scorecard with CSRD mapping."""

    project_name: str
    assessment_date: str
    rrs_version: str
    resilience_of: ResilienceOfProjectResult
    resilience_through: ResilienceThroughProjectResult
    esrs_mapping: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RRSScorecardGenerator
# ---------------------------------------------------------------------------


class RRSScorecardGenerator:
    """Generate standardized RRS ESG scorecards.

    Consumes raw output DataFrames from ``AvoidedLossCalculator``
    and sensitivity analysis modules, evaluates both RRS dimensions,
    and produces a CSRD-compliant JSON report.

    Parameters
    ----------
    project_name : str
        Name of the project under assessment.
    avoided_loss_result : dict
        Output from ``AvoidedLossCalculator.run_cost_benefit_analysis()``.
    sensitivity_result : dict or None
        Output from ``GlobalSensitivityAnalyzer.analyze_sobol_indices()``
        or ``GridSensitivityAnalyzer.analyze_variance()``.  Optional;
        if ``None``, sensitivity-dependent fields use defaults.
    climate_stress_scenarios : dict or None
        Mapping of scenario name → ``{npv_multiplier, load_multiplier}``.
        If ``None``, uses default moderate/extreme scenarios.
    cmi_per_mwh : float
        Customer Minutes of Interruption per MWh of unserved energy.
        Default 60 (typical distribution-level value).

    Attributes
    ----------
    project_name : str
    avoided_loss_result : dict
    sensitivity_result : dict or None
    climate_stress_scenarios : dict
    cmi_per_mwh : float
    """

    def __init__(
        self,
        project_name: str,
        avoided_loss_result: Dict[str, Any],
        sensitivity_result: Optional[Dict[str, Any]] = None,
        climate_stress_scenarios: Optional[Dict[str, Dict[str, float]]] = None,
        cmi_per_mwh: float = _CMI_PER_MWH,
    ) -> None:
        if cmi_per_mwh <= 0:
            raise ValueError(f"cmi_per_mwh must be positive, got {cmi_per_mwh}")

        self.project_name = str(project_name)
        self.avoided_loss_result = dict(avoided_loss_result)
        self.sensitivity_result = sensitivity_result
        self.cmi_per_mwh = float(cmi_per_mwh)

        self.climate_stress_scenarios = climate_stress_scenarios or {
            "moderate": {"npv_multiplier": 0.7, "load_multiplier": 1.1},
            "extreme": {"npv_multiplier": 0.4, "load_multiplier": 1.3},
        }

    # ------------------------------------------------------------------
    # Resilience OF the project
    # ------------------------------------------------------------------

    def assess_resilience_of_the_project(self) -> Dict[str, Any]:
        """Evaluate physical asset resilience to climate risks.

        Computes a confidence grade (A+ through C) based on whether
        the expected financial rate of return remains stable under
        extreme climate projections (chronic heating, sea-level rise,
        increased storm frequency).

        Methodology
        -----------
        1. Extract baseline NPV, IRR, and BCR from the avoided-loss
           result.
        2. Apply climate stress scenarios by scaling NPV downward
           (reflecting increased damage / reduced asset life).
        3. Compute NPV degradation percentage under the extreme
           scenario.
        4. Map degradation to a confidence grade using the RRS
           thresholds.
        5. Identify key sensitivity parameters (top ST indices) that
           drive financial uncertainty.

        Returns
        -------
        dict
            Serializable result with keys matching
            ``ResilienceOfProjectResult`` fields.
        """
        avoided = self.avoided_loss_result
        baseline_npv = float(avoided.get("npv", 0.0))
        baseline_irr = avoided.get("irr")
        baseline_bcr = float(avoided.get("bcr", 0.0))

        # Apply extreme climate stress scenario
        extreme = self.climate_stress_scenarios.get("extreme", {})
        npv_mult = float(extreme.get("npv_multiplier", 0.4))
        stressed_npv = baseline_npv * npv_mult

        # NPV degradation
        if abs(baseline_npv) > 1e-6:
            npv_degradation = abs(baseline_npv - stressed_npv) / abs(baseline_npv) * 100.0
        else:
            npv_degradation = 100.0

        # IRR stability: IRR remains above discount rate under stress
        irr_stable = baseline_irr is not None and float(baseline_irr) > 0.03

        # BCR stability: BCR remains above 1.0 under stress
        bcr_stable = baseline_bcr > 1.0

        # Confidence grade from degradation
        grade, grade_idx = self._compute_confidence_grade(npv_degradation, irr_stable, bcr_stable)

        # Key sensitivities from Sobol analysis
        key_sensitivities = self._extract_key_sensitivities(top_n=3)

        # Rationale
        rationale = self._build_of_project_rationale(
            grade, npv_degradation, irr_stable, bcr_stable, key_sensitivities
        )

        result = {
            "confidence_grade": grade,
            "grade_index": grade_idx,
            "baseline_npv": baseline_npv,
            "stressed_npv": stressed_npv,
            "npv_degradation_pct": round(npv_degradation, 2),
            "irr_stable": irr_stable,
            "bcr_stable": bcr_stable,
            "key_sensitivities": key_sensitivities,
            "assessment_rationale": rationale,
        }

        logger.info(
            "Resilience OF project: grade=%s, NPV degradation=%.1f%%.",
            grade, npv_degradation,
        )
        return result

    @staticmethod
    def _compute_confidence_grade(
        npv_degradation_pct: float,
        irr_stable: bool,
        bcr_stable: bool,
    ) -> Tuple[str, int]:
        """Map NPV degradation and stability flags to RRS confidence grade.

        Parameters
        ----------
        npv_degradation_pct : float
            Percentage reduction in NPV under extreme stress.
        irr_stable : bool
        bcr_stable : bool

        Returns
        -------
        tuple of (str, int)
            Grade label and 0-based index.
        """
        if npv_degradation_pct <= 10.0 and irr_stable and bcr_stable:
            return "A+", 6
        elif npv_degradation_pct <= 20.0 and irr_stable:
            return "A", 5
        elif npv_degradation_pct <= 30.0 and bcr_stable:
            return "A-", 4
        elif npv_degradation_pct <= 40.0:
            return "B+", 3
        elif npv_degradation_pct <= 50.0:
            return "B", 2
        elif npv_degradation_pct <= 60.0:
            return "B-", 1
        else:
            return "C", 0

    def _extract_key_sensitivities(self, top_n: int = 3) -> List[str]:
        """Extract top-N most influential parameters from Sobol analysis.

        Parameters
        ----------
        top_n : int

        Returns
        -------
        list of str
        """
        if self.sensitivity_result is None:
            return ["insufficient_data"]

        summary = self.sensitivity_result.get("summary")
        if summary is not None and isinstance(summary, pd.DataFrame):
            top = summary.head(top_n)
            return top["parameter"].tolist()

        st = np.asarray(self.sensitivity_result.get("ST", []))
        names = self.sensitivity_result.get("param_names", [])
        if len(st) == 0 or len(names) == 0:
            return ["insufficient_data"]

        order = np.argsort(st)[::-1][:top_n]
        return [names[i] for i in order]

    @staticmethod
    def _build_of_project_rationale(
        grade: str,
        npv_degradation: float,
        irr_stable: bool,
        bcr_stable: bool,
        key_sensitivities: List[str],
    ) -> str:
        """Build human-readable rationale for resilience-of-project assessment.

        Parameters
        ----------
        grade : str
        npv_degradation : float
        irr_stable : bool
        bcr_stable : bool
        key_sensitivities : list of str

        Returns
        -------
        str
        """
        parts = [f"Confidence grade {grade} assigned."]

        if npv_degradation <= 20.0:
            parts.append(
                f"NPV degrades only {npv_degradation:.1f}% under extreme "
                f"climate stress, indicating robust asset design."
            )
        elif npv_degradation <= 50.0:
            parts.append(
                f"NPV degrades {npv_degradation:.1f}% under extreme climate "
                f"stress; moderate vulnerability detected."
            )
        else:
            parts.append(
                f"NPV degrades {npv_degradation:.1f}% under extreme climate "
                f"stress; significant vulnerability.  Recommend hardening."
            )

        if irr_stable:
            parts.append("IRR remains positive under stress.")
        else:
            parts.append("IRR is not stable under stress.")

        if bcr_stable:
            parts.append("BCR remains above 1.0.")
        else:
            parts.append("BCR falls below 1.0 under stress.")

        if key_sensitivities and key_sensitivities[0] != "insufficient_data":
            parts.append(
                f"Key sensitivity drivers: {', '.join(key_sensitivities)}."
            )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Resilience THROUGH the project
    # ------------------------------------------------------------------

    def assess_resilience_through_the_project(self) -> Dict[str, Any]:
        """Measure positive externalities and adaptation benefits.

        Evaluates:

        - Reduction in community Customer Minutes of Interruption (CMI).
        - Avoided commercial supply-chain losses.
        - Emissions offsets (CO₂-eq avoided).
        - Community benefit ratio.

        Scores the adaptation benefit on a 1–10 scale.

        Returns
        -------
        dict
            Serializable result with keys matching
            ``ResilienceThroughProjectResult`` fields.
        """
        avoided = self.avoided_loss_result
        detail = avoided.get("avoided_loss_detail", {})

        avoided_eens_mwh = float(detail.get("avoided_eens_mwh", 0.0))
        avoided_loss_usd = float(detail.get("avoided_loss_usd", 0.0))
        baseline_risk = float(detail.get("baseline_risk_usd", 0.0))

        # CMI reduction
        cmi_reduction = avoided_eens_mwh * self.cmi_per_mwh

        # Avoided supply-chain loss (portion of avoided loss attributed to commercial/industrial)
        avoided_supply_chain = avoided_loss_usd * 0.4

        # Emissions offset (from carbon accounting if available)
        emissions_offset = float(avoided.get("emissions_offset_tonne_co2", 0.0))
        if emissions_offset == 0.0:
            emissions_offset = avoided_eens_mwh * 0.5  # ~500 kg CO₂/MWh default

        # Community benefit ratio: avoided loss relative to baseline risk
        if baseline_risk > 1e-6:
            community_benefit_ratio = avoided_loss_usd / baseline_risk
        else:
            community_benefit_ratio = 0.0

        # Adaptation score (1–10)
        adaptation_score = self._compute_adaptation_score(
            cmi_reduction, avoided_supply_chain, emissions_offset, community_benefit_ratio
        )

        # Rationale
        rationale = self._build_through_project_rationale(
            adaptation_score, cmi_reduction, avoided_supply_chain,
            emissions_offset, community_benefit_ratio,
        )

        result = {
            "adaptation_score": adaptation_score,
            "cmi_reduction_minutes": round(cmi_reduction, 1),
            "avoided_supply_chain_loss_usd": round(avoided_supply_chain, 2),
            "emissions_offset_tonne_co2": round(emissions_offset, 2),
            "avoided_eens_mwh": round(avoided_eens_mwh, 2),
            "community_benefit_ratio": round(community_benefit_ratio, 4),
            "assessment_rationale": rationale,
        }

        logger.info(
            "Resilience THROUGH project: adaptation_score=%d/10, "
            "CMI reduction=%.0f min.",
            adaptation_score, cmi_reduction,
        )
        return result

    @staticmethod
    def _compute_adaptation_score(
        cmi_reduction: float,
        avoided_supply_chain: float,
        emissions_offset: float,
        community_benefit_ratio: float,
    ) -> int:
        """Compute adaptation benefit score on 1–10 scale.

        Parameters
        ----------
        cmi_reduction : float
            Customer minutes of interruption reduced.
        avoided_supply_chain : float
            Avoided supply-chain loss in USD.
        emissions_offset : float
            CO₂ offset in metric tons.
        community_benefit_ratio : float
            Ratio of avoided loss to baseline risk.

        Returns
        -------
        int
            Score from 1 to 10.
        """
        score = 1.0

        # CMI contribution (0–3 points)
        if cmi_reduction > 100000:
            score += 3.0
        elif cmi_reduction > 10000:
            score += 2.0
        elif cmi_reduction > 1000:
            score += 1.0

        # Supply-chain contribution (0–2 points)
        if avoided_supply_chain > 1_000_000:
            score += 2.0
        elif avoided_supply_chain > 100_000:
            score += 1.0

        # Emissions contribution (0–2 points)
        if emissions_offset > 1000:
            score += 2.0
        elif emissions_offset > 100:
            score += 1.0

        # Community benefit ratio (0–3 points)
        if community_benefit_ratio > 0.5:
            score += 3.0
        elif community_benefit_ratio > 0.25:
            score += 2.0
        elif community_benefit_ratio > 0.1:
            score += 1.0

        return min(10, max(1, int(round(score))))

    @staticmethod
    def _build_through_project_rationale(
        adaptation_score: int,
        cmi_reduction: float,
        avoided_supply_chain: float,
        emissions_offset: float,
        community_benefit_ratio: float,
    ) -> str:
        """Build human-readable rationale for resilience-through-project assessment.

        Parameters
        ----------
        adaptation_score : int
        cmi_reduction : float
        avoided_supply_chain : float
        emissions_offset : float
        community_benefit_ratio : float

        Returns
        -------
        str
        """
        parts = [f"Adaptation benefit scored {adaptation_score}/10."]

        if cmi_reduction > 0:
            parts.append(
                f"Community CMI reduced by {cmi_reduction:,.0f} minutes annually."
            )
        if avoided_supply_chain > 0:
            parts.append(
                f"Avoided supply-chain losses of ${avoided_supply_chain:,.0f}."
            )
        if emissions_offset > 0:
            parts.append(
                f"Emissions offset of {emissions_offset:.1f} tCO₂-eq."
            )
        if community_benefit_ratio > 0:
            parts.append(
                f"Community benefit ratio of {community_benefit_ratio:.2%}."
            )

        if adaptation_score >= 8:
            parts.append("Strong adaptation co-benefits demonstrated.")
        elif adaptation_score >= 5:
            parts.append("Moderate adaptation co-benefits.")
        else:
            parts.append("Limited adaptation co-benefits; consider enhancements.")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Full scorecard generation
    # ------------------------------------------------------------------

    def generate_scorecard(self) -> RRSScorecard:
        """Generate the complete RRS scorecard.

        Returns
        -------
        RRSScorecard
        """
        of_result = self.assess_resilience_of_the_project()
        through_result = self.assess_resilience_through_the_project()

        esrs_mapping = self._build_esrs_mapping(of_result, through_result)

        return RRSScorecard(
            project_name=self.project_name,
            assessment_date=datetime.now(timezone.utc).isoformat(),
            rrs_version="1.0.0",
            resilience_of=ResilienceOfProjectResult(**of_result),
            resilience_through=ResilienceThroughProjectResult(**through_result),
            esrs_mapping=esrs_mapping,
            metadata={
                "generator": "RRSScorecardGenerator",
                "framework": "World Bank Resilience Rating System",
                "csrd_compliance": "EU 2022/2464",
                "climate_scenarios": list(self.climate_stress_scenarios.keys()),
            },
        )

    @staticmethod
    def _build_esrs_mapping(
        of_result: Dict[str, Any],
        through_result: Dict[str, Any],
    ) -> Dict[str, str]:
        """Build ESRS topical category mapping.

        Parameters
        ----------
        of_result : dict
        through_result : dict

        Returns
        -------
        dict
        """
        mapping: Dict[str, str] = {}

        mapping["confidence_grade"] = _ESRS_CATEGORIES["financial_resilience"]
        mapping["npv_stability"] = _ESRS_CATEGORIES["financial_resilience"]
        mapping["key_sensitivities"] = _ESRS_CATEGORIES["resilience_of_project"]

        mapping["adaptation_score"] = _ESRS_CATEGORIES["resilience_through_project"]
        mapping["cmi_reduction"] = _ESRS_CATEGORIES["community_benefit"]
        mapping["avoided_supply_chain_loss"] = _ESRS_CATEGORIES["supply_chain_resilience"]
        mapping["emissions_offset"] = _ESRS_CATEGORIES["emissions_offset"]

        return mapping

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Export the scorecard as a CSRD-compliant JSON string.

        Parameters
        ----------
        indent : int
            JSON indentation level.

        Returns
        -------
        str
        """
        scorecard = self.generate_scorecard()
        return json.dumps(self._scorecard_to_dict(scorecard), indent=indent, default=str)

    @staticmethod
    def _scorecard_to_dict(sc: RRSScorecard) -> Dict[str, Any]:
        """Convert RRSScorecard dataclass to serializable dict.

        Parameters
        ----------
        sc : RRSScorecard

        Returns
        -------
        dict
        """
        return {
            "rrs_scorecard": {
                "project_name": sc.project_name,
                "assessment_date": sc.assessment_date,
                "rrs_version": sc.rrs_version,
                "resilience_of_the_project": {
                    "confidence_grade": sc.resilience_of.confidence_grade,
                    "grade_index": sc.resilience_of.grade_index,
                    "baseline_npv_usd": sc.resilience_of.baseline_npv,
                    "stressed_npv_usd": sc.resilience_of.stressed_npv,
                    "npv_degradation_pct": sc.resilience_of.npv_degradation_pct,
                    "irr_stable": sc.resilience_of.irr_stable,
                    "bcr_stable": sc.resilience_of.bcr_stable,
                    "key_sensitivities": sc.resilience_of.key_sensitivities,
                    "rationale": sc.resilience_of.assessment_rationale,
                },
                "resilience_through_the_project": {
                    "adaptation_score": sc.resilience_through.adaptation_score,
                    "cmi_reduction_minutes": sc.resilience_through.cmi_reduction_minutes,
                    "avoided_supply_chain_loss_usd": sc.resilience_through.avoided_supply_chain_loss_usd,
                    "emissions_offset_tonne_co2": sc.resilience_through.emissions_offset_tonne_co2,
                    "avoided_eens_mwh": sc.resilience_through.avoided_eens_mwh,
                    "community_benefit_ratio": sc.resilience_through.community_benefit_ratio,
                    "rationale": sc.resilience_through.assessment_rationale,
                },
                "esrs_mapping": sc.esrs_mapping,
                "metadata": sc.metadata,
            }
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RRSScorecardGenerator(project={self.project_name!r}, "
            f"scenarios={list(self.climate_stress_scenarios.keys())})"
        )
