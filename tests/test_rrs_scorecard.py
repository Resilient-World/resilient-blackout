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

import numpy as np
import pandas as pd
import pytest

from resilient_blackout.reporting.rrs_scorecard import (
    RRSScorecard,
    RRSScorecardGenerator,
)


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
        "avoided_loss_detail": {
            "baseline_eens_mwh": 2000.0,
            "resilient_eens_mwh": 500.0,
            "baseline_risk_usd": 20_000_000.0,
            "resilient_risk_usd": 5_000_000.0,
            "avoided_loss_usd": 15_000_000.0,
            "avoided_eens_mwh": 1500.0,
            "voll_used": 10000.0,
        },
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
        "avoided_loss_detail": {
            "baseline_eens_mwh": 200.0,
            "resilient_eens_mwh": 150.0,
            "baseline_risk_usd": 2_000_000.0,
            "resilient_risk_usd": 1_500_000.0,
            "avoided_loss_usd": 500_000.0,
            "avoided_eens_mwh": 50.0,
            "voll_used": 10000.0,
        },
    }


@pytest.fixture
def sensitivity_result() -> dict:
    """Sobol sensitivity analysis output."""
    return {
        "S1": np.array([0.45, 0.15, 0.05]),
        "ST": np.array([0.60, 0.25, 0.10]),
        "S1_conf": np.array([0.02, 0.02, 0.01]),
        "ST_conf": np.array([0.03, 0.02, 0.01]),
        "param_names": ["failure_rate", "restoration_time", "voll"],
        "summary": pd.DataFrame({
            "parameter": ["failure_rate", "restoration_time", "voll"],
            "S1": [0.45, 0.15, 0.05],
            "ST": [0.60, 0.25, 0.10],
            "S1_conf": [0.02, 0.02, 0.01],
            "ST_conf": [0.03, 0.02, 0.01],
        }).sort_values("ST", ascending=False).reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestInit:
    """Validation of constructor and parameter handling."""

    def test_default_construction(self, strong_avoided_loss: dict) -> None:
        gen = RRSScorecardGenerator("Test Project", strong_avoided_loss)
        assert gen.project_name == "Test Project"
        assert gen.cmi_per_mwh == 60.0
        assert "moderate" in gen.climate_stress_scenarios
        assert "extreme" in gen.climate_stress_scenarios

    def test_custom_cmi(self, strong_avoided_loss: dict) -> None:
        gen = RRSScorecardGenerator("Test", strong_avoided_loss, cmi_per_mwh=45.0)
        assert gen.cmi_per_mwh == 45.0

    def test_invalid_cmi_raises(self, strong_avoided_loss: dict) -> None:
        with pytest.raises(ValueError, match="cmi_per_mwh"):
            RRSScorecardGenerator("Test", strong_avoided_loss, cmi_per_mwh=-1.0)

    def test_custom_scenarios(self, strong_avoided_loss: dict) -> None:
        scenarios = {"mild": {"npv_multiplier": 0.9, "load_multiplier": 1.05}}
        gen = RRSScorecardGenerator(
            "Test", strong_avoided_loss, climate_stress_scenarios=scenarios
        )
        assert "mild" in gen.climate_stress_scenarios
        assert "extreme" not in gen.climate_stress_scenarios

    def test_repr(self, strong_avoided_loss: dict) -> None:
        gen = RRSScorecardGenerator("Test", strong_avoided_loss)
        r = repr(gen)
        assert "RRSScorecardGenerator" in r
        assert "Test" in r


# ---------------------------------------------------------------------------
# Resilience OF the project
# ---------------------------------------------------------------------------


class TestResilienceOfProject:
    """Validation of resilience-of-project assessment."""

    def test_strong_project_gets_high_grade(self, strong_avoided_loss: dict) -> None:
        gen = RRSScorecardGenerator("Strong", strong_avoided_loss)
        result = gen.assess_resilience_of_the_project()
        assert result["confidence_grade"] in ("A+", "A", "A-")
        assert result["irr_stable"] is True
        assert result["bcr_stable"] is True

    def test_weak_project_gets_low_grade(self, weak_avoided_loss: dict) -> None:
        gen = RRSScorecardGenerator("Weak", weak_avoided_loss)
        result = gen.assess_resilience_of_the_project()
        assert result["confidence_grade"] in ("B", "B-", "C")
        assert result["npv_degradation_pct"] > 0

    def test_includes_key_sensitivities(
        self, strong_avoided_loss: dict, sensitivity_result: dict
    ) -> None:
        gen = RRSScorecardGenerator(
            "Test", strong_avoided_loss, sensitivity_result=sensitivity_result
        )
        result = gen.assess_resilience_of_the_project()
        assert len(result["key_sensitivities"]) > 0
        assert "failure_rate" in result["key_sensitivities"]

    def test_no_sensitivity_fallback(self, strong_avoided_loss: dict) -> None:
        gen = RRSScorecardGenerator("Test", strong_avoided_loss)
        result = gen.assess_resilience_of_the_project()
        assert result["key_sensitivities"] == ["insufficient_data"]

    def test_rationale_is_string(self, strong_avoided_loss: dict) -> None:
        gen = RRSScorecardGenerator("Test", strong_avoided_loss)
        result = gen.assess_resilience_of_the_project()
        assert isinstance(result["assessment_rationale"], str)
        assert len(result["assessment_rationale"]) > 0

    def test_stressed_npv_lower_than_baseline(self, strong_avoided_loss: dict) -> None:
        gen = RRSScorecardGenerator("Test", strong_avoided_loss)
        result = gen.assess_resilience_of_the_project()
        assert result["stressed_npv"] < result["baseline_npv"]


# ---------------------------------------------------------------------------
# Resilience THROUGH the project
# ---------------------------------------------------------------------------


class TestResilienceThroughProject:
    """Validation of resilience-through-project assessment."""

    def test_strong_project_high_adaptation_score(
        self, strong_avoided_loss: dict
    ) -> None:
        gen = RRSScorecardGenerator("Strong", strong_avoided_loss)
        result = gen.assess_resilience_through_the_project()
        assert result["adaptation_score"] >= 5
        assert result["cmi_reduction_minutes"] > 0

    def test_weak_project_low_adaptation_score(
        self, weak_avoided_loss: dict
    ) -> None:
        gen = RRSScorecardGenerator("Weak", weak_avoided_loss)
        result = gen.assess_resilience_through_the_project()
        assert result["adaptation_score"] <= 5

    def test_cmi_computation(self, strong_avoided_loss: dict) -> None:
        gen = RRSScorecardGenerator("Test", strong_avoided_loss, cmi_per_mwh=60.0)
        result = gen.assess_resilience_through_the_project()
        expected_cmi = 1500.0 * 60.0
        assert result["cmi_reduction_minutes"] == pytest.approx(expected_cmi, rel=1e-6)

    def test_emissions_offset(self, strong_avoided_loss: dict) -> None:
        gen = RRSScorecardGenerator("Test", strong_avoided_loss)
        result = gen.assess_resilience_through_the_project()
        assert result["emissions_offset_tonne_co2"] > 0

    def test_community_benefit_ratio(self, strong_avoided_loss: dict) -> None:
        gen = RRSScorecardGenerator("Test", strong_avoided_loss)
        result = gen.assess_resilience_through_the_project()
        assert 0.0 < result["community_benefit_ratio"] <= 1.0

    def test_rationale_is_string(self, strong_avoided_loss: dict) -> None:
        gen = RRSScorecardGenerator("Test", strong_avoided_loss)
        result = gen.assess_resilience_through_the_project()
        assert isinstance(result["assessment_rationale"], str)
        assert len(result["assessment_rationale"]) > 0


# ---------------------------------------------------------------------------
# Full scorecard
# ---------------------------------------------------------------------------


class TestGenerateScorecard:
    """Validation of full scorecard generation."""

    def test_generates_scorecard(self, strong_avoided_loss: dict) -> None:
        gen = RRSScorecardGenerator("Test", strong_avoided_loss)
        sc = gen.generate_scorecard()
        assert isinstance(sc, RRSScorecard)
        assert sc.project_name == "Test"
        assert sc.rrs_version == "1.0.0"
        assert sc.resilience_of.confidence_grade in (
            "A+", "A", "A-", "B+", "B", "B-", "C"
        )
        assert 1 <= sc.resilience_through.adaptation_score <= 10

    def test_esrs_mapping(self, strong_avoided_loss: dict) -> None:
        gen = RRSScorecardGenerator("Test", strong_avoided_loss)
        sc = gen.generate_scorecard()
        assert "confidence_grade" in sc.esrs_mapping
        assert "adaptation_score" in sc.esrs_mapping
        assert "cmi_reduction" in sc.esrs_mapping
        assert "emissions_offset" in sc.esrs_mapping
        assert "ESRS" in sc.esrs_mapping["confidence_grade"]

    def test_metadata(self, strong_avoided_loss: dict) -> None:
        gen = RRSScorecardGenerator("Test", strong_avoided_loss)
        sc = gen.generate_scorecard()
        assert sc.metadata["framework"] == "World Bank Resilience Rating System"
        assert "climate_scenarios" in sc.metadata


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


class TestJSONExport:
    """Validation of JSON serialization."""

    def test_to_json_valid(self, strong_avoided_loss: dict) -> None:
        gen = RRSScorecardGenerator("Test", strong_avoided_loss)
        json_str = gen.to_json()
        parsed = json.loads(json_str)
        assert "rrs_scorecard" in parsed
        assert parsed["rrs_scorecard"]["project_name"] == "Test"

    def test_json_contains_both_dimensions(
        self, strong_avoided_loss: dict
    ) -> None:
        gen = RRSScorecardGenerator("Test", strong_avoided_loss)
        json_str = gen.to_json()
        parsed = json.loads(json_str)
        sc = parsed["rrs_scorecard"]
        assert "resilience_of_the_project" in sc
        assert "resilience_through_the_project" in sc
        assert "esrs_mapping" in sc

    def test_json_roundtrip(self, strong_avoided_loss: dict) -> None:
        gen = RRSScorecardGenerator("Test", strong_avoided_loss)
        json_str = gen.to_json()
        parsed = json.loads(json_str)
        sc = parsed["rrs_scorecard"]
        assert sc["resilience_of_the_project"]["confidence_grade"] is not None
        assert sc["resilience_through_the_project"]["adaptation_score"] is not None


# ---------------------------------------------------------------------------
# Confidence grade edge cases
# ---------------------------------------------------------------------------


class TestConfidenceGrade:
    """Validation of confidence grade computation."""

    def test_all_grades_reachable(self) -> None:
        grades = set()
        for deg in [5, 15, 25, 35, 45, 55, 65]:
            grade, _ = RRSScorecardGenerator._compute_confidence_grade(
                deg, irr_stable=True, bcr_stable=True
            )
            grades.add(grade)
        assert len(grades) >= 5

    def test_unstable_irr_downgrades(self) -> None:
        grade_stable, _ = RRSScorecardGenerator._compute_confidence_grade(
            15.0, irr_stable=True, bcr_stable=True
        )
        grade_unstable, _ = RRSScorecardGenerator._compute_confidence_grade(
            15.0, irr_stable=False, bcr_stable=True
        )
        assert grade_stable != grade_unstable

    def test_unstable_bcr_downgrades(self) -> None:
        grade_stable, _ = RRSScorecardGenerator._compute_confidence_grade(
            25.0, irr_stable=True, bcr_stable=True
        )
        grade_unstable, _ = RRSScorecardGenerator._compute_confidence_grade(
            25.0, irr_stable=True, bcr_stable=False
        )
        assert grade_stable != grade_unstable
