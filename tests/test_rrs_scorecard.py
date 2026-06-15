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

"""Unit tests for ``resilient_blackout.reporting.rrs_scorecard``."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from resilient_blackout.reporting.rrs_scorecard import RRSReportGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def strong_avoided_loss() -> dict:
    """A project with strong financial returns."""
    return {
        "npv": 5_000_000.0,
        "irr": 0.12,
        "bcr": 2.5,
        "annual_benefit": 500_000.0,
        "pv_benefits": 6_000_000.0,
        "pv_costs": 2_400_000.0,
        "avoided_loss_usd": 15_000_000.0,
        "avoided_eens_mwh": 1500.0,
        "baseline_eens_mwh": 2000.0,
        "resilient_eens_mwh": 500.0,
        "baseline_risk_usd": 20_000_000.0,
        "resilient_risk_usd": 5_000_000.0,
        "voll_used": 10000.0,
    }


@pytest.fixture
def weak_avoided_loss() -> dict:
    """A project with marginal financial returns."""
    return {
        "npv": 100_000.0,
        "irr": 0.04,
        "bcr": 1.1,
        "annual_benefit": 10_000.0,
        "pv_benefits": 200_000.0,
        "pv_costs": 180_000.0,
        "avoided_loss_usd": 500_000.0,
        "avoided_eens_mwh": 50.0,
        "baseline_eens_mwh": 200.0,
        "resilient_eens_mwh": 150.0,
        "baseline_risk_usd": 2_000_000.0,
        "resilient_risk_usd": 1_500_000.0,
        "voll_used": 10000.0,
    }


@pytest.fixture
def sensitivity_result() -> dict:
    """Sobol sensitivity analysis output."""
    return {
        "method": "sobol",
        "indices": {
            "S1": [0.45, 0.15, 0.05],
            "ST": [0.60, 0.25, 0.10],
            "S1_conf": [0.02, 0.02, 0.01],
            "ST_conf": [0.03, 0.02, 0.01],
            "param_names": ["failure_rate", "restoration_time", "voll"],
        },
    }


@pytest.fixture
def community_data() -> dict:
    """Community-level data for through-project assessment."""
    return {
        "n_customers": 50000,
        "supply_chain_value_per_mwh": 5000.0,
        "renewable_mwh": 500.0,
    }


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestInit:
    """Validation of constructor and parameter handling."""

    def test_default_construction(self) -> None:
        gen = RRSReportGenerator("Test Project")
        assert gen.project_name == "Test Project"
        assert gen.planning_horizon == 20
        assert gen.discount_rate == 0.05
        assert len(gen.climate_scenarios) == 2
        assert gen.climate_scenarios[0]["name"] == "RCP 4.5"

    def test_custom_params(self) -> None:
        gen = RRSReportGenerator("Test", planning_horizon=30, discount_rate=0.07)
        assert gen.planning_horizon == 30
        assert gen.discount_rate == 0.07

    def test_custom_scenarios(self) -> None:
        scenarios = [{"name": "Custom", "temp_increase_c": 1.5, "sea_level_rise_m": 0.2, "heatwave_freq_multiplier": 1.2}]
        gen = RRSReportGenerator("Test", climate_scenarios=scenarios)
        assert len(gen.climate_scenarios) == 1
        assert gen.climate_scenarios[0]["name"] == "Custom"

    def test_report_initially_none(self) -> None:
        gen = RRSReportGenerator("Test")
        assert gen.report is None


# ---------------------------------------------------------------------------
# Resilience OF the project
# ---------------------------------------------------------------------------


class TestResilienceOfProject:
    """Validation of resilience-of-project assessment."""

    def test_strong_project_gets_high_grade(self, strong_avoided_loss: dict) -> None:
        gen = RRSReportGenerator("Strong")
        result = gen.assess_resilience_of_the_project(strong_avoided_loss)
        assert result["grade"] in ("AAA", "AA", "A")
        assert result["npv"] == 5_000_000.0
        assert result["psi"] >= 0.0

    def test_weak_project_gets_lower_grade(self, weak_avoided_loss: dict) -> None:
        gen = RRSReportGenerator("Weak")
        result = gen.assess_resilience_of_the_project(weak_avoided_loss)
        assert result["grade"] in ("BBB", "BB", "B", "C")

    def test_with_sensitivity(self, strong_avoided_loss: dict, sensitivity_result: dict) -> None:
        gen = RRSReportGenerator("Test")
        result = gen.assess_resilience_of_the_project(
            strong_avoided_loss, sensitivity_result
        )
        assert "grade" in result
        assert "npv_cv" in result

    def test_without_sensitivity(self, strong_avoided_loss: dict) -> None:
        gen = RRSReportGenerator("Test")
        result = gen.assess_resilience_of_the_project(strong_avoided_loss)
        assert result["npv_cv"] == 0.15

    def test_psi_range(self, strong_avoided_loss: dict) -> None:
        gen = RRSReportGenerator("Test")
        result = gen.assess_resilience_of_the_project(strong_avoided_loss)
        assert 0.0 <= result["psi"] <= 1.0


# ---------------------------------------------------------------------------
# Resilience THROUGH the project
# ---------------------------------------------------------------------------


class TestResilienceThroughProject:
    """Validation of resilience-through-project assessment."""

    def test_with_community_data(
        self, strong_avoided_loss: dict, community_data: dict
    ) -> None:
        gen = RRSReportGenerator("Test")
        result = gen.assess_resilience_through_the_project(
            strong_avoided_loss, community_data
        )
        assert result["cmi_reduction_minutes"] > 0
        assert result["community_impact_score"] > 0
        assert result["avoided_supply_chain_loss_usd"] > 0
        assert result["emissions_offset_tco2"] > 0

    def test_without_community_data(self, strong_avoided_loss: dict) -> None:
        gen = RRSReportGenerator("Test")
        result = gen.assess_resilience_through_the_project(strong_avoided_loss)
        assert result["cmi_reduction_minutes"] >= 0
        assert result["community_impact_score"] >= 0

    def test_weak_project_lower_scores(self, weak_avoided_loss: dict) -> None:
        gen = RRSReportGenerator("Test")
        result_strong = gen.assess_resilience_through_the_project(
            {"avoided_eens_mwh": 1500.0}
        )
        result_weak = gen.assess_resilience_through_the_project(
            {"avoided_eens_mwh": 50.0}
        )
        assert result_strong["cmi_reduction_minutes"] > result_weak["cmi_reduction_minutes"]


# ---------------------------------------------------------------------------
# Full report generation
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Validation of full report generation."""

    def test_generates_report(
        self, strong_avoided_loss: dict, sensitivity_result: dict, community_data: dict
    ) -> None:
        gen = RRSReportGenerator("Test")
        report = gen.generate_report(
            strong_avoided_loss, sensitivity_result, community_data
        )
        assert "report_metadata" in report
        assert "key_performance_indicators" in report
        assert "resilience_of_the_project" in report
        assert "resilience_through_the_project" in report
        assert "regulatory_alignment" in report
        assert "sensitivity_analysis" in report
        assert report["report_metadata"]["project_name"] == "Test"

    def test_report_stored(self, strong_avoided_loss: dict) -> None:
        gen = RRSReportGenerator("Test")
        gen.generate_report(strong_avoided_loss)
        assert gen.report is not None

    def test_kpis_present(self, strong_avoided_loss: dict) -> None:
        gen = RRSReportGenerator("Test")
        report = gen.generate_report(strong_avoided_loss)
        kpis = report["key_performance_indicators"]
        assert kpis["npv_usd"] == 5_000_000.0
        assert kpis["bcr"] == 2.5
        assert kpis["avoided_eens_mwh"] == 1500.0

    def test_regulatory_alignment(self, strong_avoided_loss: dict) -> None:
        gen = RRSReportGenerator("Test")
        report = gen.generate_report(strong_avoided_loss)
        reg = report["regulatory_alignment"]
        assert "eu_taxonomy" in reg
        assert "tcfd" in reg
        assert "issb_s2" in reg
        assert "gri" in reg

    def test_sensitivity_summary(
        self, strong_avoided_loss: dict, sensitivity_result: dict
    ) -> None:
        gen = RRSReportGenerator("Test")
        report = gen.generate_report(strong_avoided_loss, sensitivity_result)
        assert report["sensitivity_analysis"] is not None
        assert report["sensitivity_analysis"]["method"] == "sobol"
        assert len(report["sensitivity_analysis"]["top_parameters"]) > 0

    def test_no_sensitivity(self, strong_avoided_loss: dict) -> None:
        gen = RRSReportGenerator("Test")
        report = gen.generate_report(strong_avoided_loss)
        assert report["sensitivity_analysis"] is None


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


class TestJSONExport:
    """Validation of JSON export."""

    def test_export_json(
        self, strong_avoided_loss: dict, sensitivity_result: dict, community_data: dict
    ) -> None:
        gen = RRSReportGenerator("Test")
        gen.generate_report(strong_avoided_loss, sensitivity_result, community_data)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            gen.export_json(tmp_path)
            with open(tmp_path) as f:
                data = json.load(f)
            assert data["report_metadata"]["project_name"] == "Test"
            assert "resilience_of_the_project" in data
        finally:
            Path(tmp_path).unlink()

    def test_export_without_report_raises(self) -> None:
        gen = RRSReportGenerator("Test")
        with pytest.raises(RuntimeError, match="No report generated"):
            gen.export_json("/tmp/test.json")


# ---------------------------------------------------------------------------
# Grade assignment
# ---------------------------------------------------------------------------


class TestGradeAssignment:
    """Validation of grade assignment logic."""

    def test_all_grades_reachable(self) -> None:
        grades = set()
        for cv in [0.05, 0.15, 0.25, 0.40, 0.60, 0.80, 1.0]:
            grade = RRSReportGenerator._assign_grade(cv)
            grades.add(grade)
        assert len(grades) >= 5

    def test_low_cv_gets_aaa(self) -> None:
        assert RRSReportGenerator._assign_grade(0.05) == "AAA"

    def test_high_cv_gets_c(self) -> None:
        assert RRSReportGenerator._assign_grade(1.5) == "C"
