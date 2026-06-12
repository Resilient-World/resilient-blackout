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
Double-materiality reporting and Resilience Rating System (RRS) scorecard.

Provides ``RRSReportGenerator``, a reporting engine modelled on the
World Bank Group's RRS methodology.  It consumes raw output from
``AvoidedLossCalculator`` and sensitivity modules to produce
standardised metrics tracking ESG and Green Taxonomy alignment,
including bond-style confidence grades, community impact scores, and
regulatory alignment tags.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_EMISSION_FACTOR_TCO2_PER_MWH: float = 0.4
_DEFAULT_CUSTOMERS_PER_MW: float = 500.0

_GRADE_THRESHOLDS: List[Tuple[float, str]] = [
    (0.10, "AAA"),
    (0.20, "AA"),
    (0.35, "A"),
    (0.50, "BBB"),
    (0.75, "BB"),
    (1.00, "B"),
]

_REGULATORY_ALIGNMENT: Dict[str, Dict[str, str]] = {
    "eu_taxonomy": {
        "framework": "EU Taxonomy for Sustainable Activities",
        "objective": "Climate Change Adaptation",
        "dnsht_criteria": "Do No Significant Harm — assessed via avoided emissions",
        "minimum_safeguards": "Social vulnerability weighting applied",
    },
    "tcfd": {
        "framework": "Task Force on Climate-related Financial Disclosures",
        "governance": "Board-level resilience oversight implied",
        "strategy": "Climate scenario analysis across planning horizon",
        "risk_management": "Physical risk quantified via Monte Carlo cascade",
        "metrics_and_targets": "EENS, VoLL, BCR, NPV reported",
    },
    "issb_s2": {
        "framework": "IFRS S2 Climate-related Disclosures",
        "physical_risks": "Chronic and acute hazards modelled",
        "resilience_assessment": "RRS grade assigned",
    },
    "gri": {
        "framework": "Global Reporting Initiative",
        "standard": "GRI 201: Economic Performance",
        "disclosures": "201-2: Financial implications of climate change",
    },
}


# ---------------------------------------------------------------------------
# RRS Report Generator
# ---------------------------------------------------------------------------

class RRSReportGenerator:
    """World Bank RRS-aligned double-materiality reporting engine.

    Generates a standardised JSON scorecard from ``AvoidedLossCalculator``
    outputs, covering financial resilience (resilience *of* the project)
    and community-level benefits (resilience *through* the project).

    Parameters
    ----------
    project_name : str
        Name of the project or investment being assessed.
    planning_horizon : int
        Planning horizon in years.  Default 20.
    discount_rate : float
        Annual discount rate.  Default 0.05.
    climate_scenarios : list of dict or None
        Climate scenarios for chronic stress testing.  Each dict may
        contain ``name``, ``temp_increase_c``, ``sea_level_rise_m``,
        ``heatwave_freq_multiplier``.

    Attributes
    ----------
    project_name : str
    planning_horizon : int
    discount_rate : float
    climate_scenarios : list of dict
    report : dict or None
    """

    def __init__(
        self,
        project_name: str,
        planning_horizon: int = 20,
        discount_rate: float = 0.05,
        climate_scenarios: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.project_name = project_name
        self.planning_horizon = planning_horizon
        self.discount_rate = discount_rate
        self.climate_scenarios = climate_scenarios or [
            {"name": "RCP 4.5", "temp_increase_c": 2.0, "sea_level_rise_m": 0.3, "heatwave_freq_multiplier": 1.5},
            {"name": "RCP 8.5", "temp_increase_c": 4.0, "sea_level_rise_m": 0.8, "heatwave_freq_multiplier": 3.0},
        ]
        self.report: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Resilience OF the project
    # ------------------------------------------------------------------

    def assess_resilience_of_the_project(
        self,
        avoided_loss_result: Dict[str, Any],
        sensitivity_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assess financial resilience of the project under climate stress.

        Computes the probability-weighted asset rate of return and
        Physical Survival Index (PSI), then assigns a bond-style
        confidence grade (AAA through C) based on NPV volatility.

        Parameters
        ----------
        avoided_loss_result : dict
            Output from ``AvoidedLossCalculator.run_cost_benefit_analysis``.
        sensitivity_result : dict or None
            Optional output from ``GridSensitivityAnalyzer``.

        Returns
        -------
        dict
            ``{"grade": str, "npv_cv": float, "psi": float,
            "rate_of_return": float, "npv": float}``.
        """
        npv = float(avoided_loss_result.get("npv", 0.0))
        bcr = float(avoided_loss_result.get("bcr", 0.0))
        avoided_loss = float(avoided_loss_result.get("avoided_loss_usd", 0.0))

        if sensitivity_result and "indices" in sensitivity_result:
            st_indices = sensitivity_result["indices"].get("ST", [])
            npv_uncertainty = float(np.mean(st_indices)) if st_indices else 0.1
        else:
            npv_uncertainty = 0.15

        npv_cv = npv_uncertainty

        grade = self._assign_grade(npv_cv)

        psi = self._compute_psi(avoided_loss_result)

        if npv > 0:
            rate_of_return = (avoided_loss / max(abs(npv), 1.0)) * bcr
        else:
            rate_of_return = 0.0

        return {
            "grade": grade,
            "npv_cv": npv_cv,
            "psi": psi,
            "rate_of_return": rate_of_return,
            "npv": npv,
        }

    @staticmethod
    def _assign_grade(npv_cv: float) -> str:
        """Map NPV coefficient of variation to a bond-style grade.

        Parameters
        ----------
        npv_cv : float
            Coefficient of variation of NPV.

        Returns
        -------
        str
            Grade from ``"AAA"`` to ``"C"``.
        """
        for threshold, grade in _GRADE_THRESHOLDS:
            if npv_cv < threshold:
                return grade
        return "C"

    def _compute_psi(self, avoided_loss_result: Dict[str, Any]) -> float:
        """Compute Physical Survival Index under worst-case climate scenario.

        Parameters
        ----------
        avoided_loss_result : dict

        Returns
        -------
        float
            PSI in [0, 1].
        """
        baseline_eens = float(avoided_loss_result.get("baseline_eens_mwh", 0.0))
        resilient_eens = float(avoided_loss_result.get("resilient_eens_mwh", 0.0))

        if baseline_eens <= 0:
            return 1.0

        worst_case = self.climate_scenarios[-1] if self.climate_scenarios else {"heatwave_freq_multiplier": 1.0}
        multiplier = float(worst_case.get("heatwave_freq_multiplier", 1.0))

        stressed_baseline = baseline_eens * multiplier
        stressed_resilient = resilient_eens * multiplier

        if stressed_baseline <= 0:
            return 1.0

        survival_ratio = 1.0 - (stressed_resilient / stressed_baseline)
        return max(0.0, min(1.0, survival_ratio))

    # ------------------------------------------------------------------
    # Resilience THROUGH the project
    # ------------------------------------------------------------------

    def assess_resilience_through_the_project(
        self,
        avoided_loss_result: Dict[str, Any],
        community_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Quantify community-level benefits of the resilience investment.

        Computes absolute reduction in Customer Minutes of Interruption
        (CMI), avoided commercial supply chain losses, emissions offsets
        from storage/renewable integration, and a composite community
        impact score.

        Parameters
        ----------
        avoided_loss_result : dict
            Output from ``AvoidedLossCalculator.run_cost_benefit_analysis``.
        community_data : dict or None
            Optional community-specific data with keys:

            - ``n_customers`` (int) — total customers served.
            - ``supply_chain_value_per_mwh`` (float) — $/MWh of supply
              chain disruption cost.
            - ``renewable_mwh`` (float) — MWh of renewable energy
              integrated.

        Returns
        -------
        dict
            ``{"cmi_reduction_minutes": float,
            "community_impact_score": float,
            "avoided_supply_chain_loss_usd": float,
            "emissions_offset_tco2": float}``.
        """
        avoided_eens = float(avoided_loss_result.get("avoided_eens_mwh", 0.0))

        cd = community_data or {}
        n_customers = int(cd.get("n_customers", int(avoided_eens * _DEFAULT_CUSTOMERS_PER_MW)))
        supply_chain_value = float(cd.get("supply_chain_value_per_mwh", 5000.0))
        renewable_mwh = float(cd.get("renewable_mwh", avoided_eens * 0.3))

        if n_customers > 0:
            cmi_reduction = (avoided_eens * 60.0) / n_customers
        else:
            cmi_reduction = 0.0

        avoided_supply_chain = avoided_eens * supply_chain_value
        emissions_offset = renewable_mwh * _DEFAULT_EMISSION_FACTOR_TCO2_PER_MWH

        cmi_score = min(100.0, cmi_reduction * 2.0) if cmi_reduction < 50 else 100.0
        supply_score = min(100.0, (avoided_supply_chain / 1e6) * 10.0)
        emissions_score = min(100.0, emissions_offset * 2.0)

        community_impact_score = (cmi_score * 0.4 + supply_score * 0.3 + emissions_score * 0.3)

        return {
            "cmi_reduction_minutes": cmi_reduction,
            "community_impact_score": round(community_impact_score, 1),
            "avoided_supply_chain_loss_usd": avoided_supply_chain,
            "emissions_offset_tco2": emissions_offset,
        }

    # ------------------------------------------------------------------
    # Full report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        avoided_loss_result: Dict[str, Any],
        sensitivity_result: Optional[Dict[str, Any]] = None,
        community_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Produce a unified RRS JSON report.

        Parameters
        ----------
        avoided_loss_result : dict
            From ``AvoidedLossCalculator.run_cost_benefit_analysis``.
        sensitivity_result : dict or None
            From ``GridSensitivityAnalyzer.run_full_analysis``.
        community_data : dict or None
            See :meth:`assess_resilience_through_the_project`.

        Returns
        -------
        dict
            Full RRS scorecard.
        """
        resilience_of = self.assess_resilience_of_the_project(
            avoided_loss_result, sensitivity_result
        )
        resilience_through = self.assess_resilience_through_the_project(
            avoided_loss_result, community_data
        )

        kpis = {
            "expected_annual_loss_usd": float(
                avoided_loss_result.get("baseline_risk_usd", 0.0)
            ),
            "system_wide_voll_usd_per_mwh": float(
                avoided_loss_result.get("voll_used", 0.0)
            ),
            "bcr": float(avoided_loss_result.get("bcr", 0.0)),
            "npv_usd": float(avoided_loss_result.get("npv", 0.0)),
            "irr": avoided_loss_result.get("irr"),
            "avoided_eens_mwh": float(
                avoided_loss_result.get("avoided_eens_mwh", 0.0)
            ),
            "avoided_loss_usd": float(
                avoided_loss_result.get("avoided_loss_usd", 0.0)
            ),
        }

        sensitivity_summary: Optional[Dict[str, Any]] = None
        if sensitivity_result and "indices" in sensitivity_result:
            indices = sensitivity_result["indices"]
            param_names = indices.get("param_names", [])
            st_vals = indices.get("ST", [])
            if param_names and st_vals:
                ranked = sorted(
                    zip(param_names, st_vals), key=lambda x: x[1], reverse=True
                )
                sensitivity_summary = {
                    "method": sensitivity_result.get("method", "sobol"),
                    "top_parameters": [
                        {"parameter": name, "total_order_index": float(val)}
                        for name, val in ranked[:5]
                    ],
                }

        self.report = {
            "report_metadata": {
                "generator": "Resilience Rating System (RRS) v1.0",
                "methodology": "World Bank Group RRS Framework",
                "project_name": self.project_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "planning_horizon_years": self.planning_horizon,
                "discount_rate": self.discount_rate,
                "climate_scenarios": [
                    s.get("name", "Unknown") for s in self.climate_scenarios
                ],
            },
            "key_performance_indicators": kpis,
            "resilience_of_the_project": resilience_of,
            "resilience_through_the_project": resilience_through,
            "regulatory_alignment": _REGULATORY_ALIGNMENT,
            "sensitivity_analysis": sensitivity_summary,
        }

        return self.report

    def export_json(self, filepath: str) -> None:
        """Write the report to a JSON file.

        Parameters
        ----------
        filepath : str
            Destination path.
        """
        if self.report is None:
            raise RuntimeError("No report generated. Call generate_report() first.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.report, f, indent=2, default=str)

        logger.info("RRS report exported to %s", path)
