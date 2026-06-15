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
End-to-end integration tests for the resilient-blackout pipeline.

Validates the full analysis chain — hazard application, fragility
evaluation, CEM Monte Carlo reliability estimation, financial NPV/BCR
calculation, and RRS ESG scorecard generation — against published
benchmark values for the IEEE 24-bus RTS and a synthetic SimBench-like
HV urban grid.

All tests use standard pytest patterns and are designed for GitHub
Actions CI pipelines.  No external commercial solvers are required.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pytest

try:
    import pandapower as pp

    _HAS_PANDAPOWER = True
except ImportError:  # pragma: no cover
    _HAS_PANDAPOWER = False

from resilient_blackout.core.base import HazardEvent
from resilient_blackout.core.economics import AvoidedLossCalculator
from resilient_blackout.core.fragility import ImpactFunctionSet
from resilient_blackout.grid.cem_monte_carlo import CEMMonteCarloSimulator
from resilient_blackout.grid.network import GridModel
from resilient_blackout.reporting.rrs_scorecard import RRSReportGenerator

from tests.validation_harness import (
    DEFAULT_TOLERANCE,
    IEEE24_RTS_BENCHMARKS,
    SIMBENCH_HV_BENCHMARKS,
    assert_eens_tolerance,
    assert_financial_metrics_valid,
    assert_metric_tolerance,
    assert_power_flow_valid,
    assert_rrs_scorecard_valid,
    generate_hazard_intensity_map,
)

pytestmark = pytest.mark.skipif(
    not _HAS_PANDAPOWER, reason="pandapower not installed"
)


# ---------------------------------------------------------------------------
# End-to-end pipeline on IEEE 24-bus RTS
# ---------------------------------------------------------------------------


class TestEndToEndPipelineIEEE24:
    """Full pipeline integration tests on the IEEE 24-bus RTS."""

    def test_grid_loads_and_runs_power_flow(
        self, ieee24_rts_grid: GridModel
    ) -> None:
        """IEEE 24-bus RTS should load and converge in AC power flow."""
        result = ieee24_rts_grid.run_baseline_power_flow()
        assert_power_flow_valid(result, label="IEEE24")

        assert len(ieee24_rts_grid.net.bus) == IEEE24_RTS_BENCHMARKS["n_buses"]
        assert len(ieee24_rts_grid.net.line) == IEEE24_RTS_BENCHMARKS["n_lines"]

    def test_wind_fragility_on_ieee24(
        self,
        ieee24_rts_grid: GridModel,
        synthetic_wind_hazard: HazardEvent,
        wind_fragility_set: ImpactFunctionSet,
    ) -> None:
        """Wind fragility curves should produce valid failure probabilities."""
        intensities = generate_hazard_intensity_map(
            ieee24_rts_grid, synthetic_wind_hazard
        )

        failure_probs: Dict[str, float] = {}
        for asset_id, intensity in intensities.items():
            func = wind_fragility_set.get_function(
                hazard_type="wind", asset_type="transmission_line"
            )
            if func is not None:
                failure_probs[asset_id] = func.evaluate(intensity)

        assert len(failure_probs) > 0, "No fragility evaluations produced"

        for prob in failure_probs.values():
            assert 0.0 <= prob <= 1.0, f"Failure probability {prob} outside [0, 1]"
            assert not np.isnan(prob)

    def test_cem_monte_carlo_eens_lolp(
        self, ieee24_rts_grid: GridModel
    ) -> None:
        """CEM Monte Carlo should estimate EENS and LOLP on IEEE 24-bus."""
        simulator = CEMMonteCarloSimulator(
            ieee24_rts_grid,
            cascade_tolerance=1.2,
            cascade_max_iter=30,
            rho=0.2,
            smoothing_alpha=0.7,
        )

        cem_result = simulator.run_cem(
            n_iterations=3,
            n_samples_per_iter=200,
            seed=42,
        )

        assert "v_gen" in cem_result
        assert "v_line" in cem_result
        assert len(cem_result["gamma_history"]) == 3

        estimation = simulator.estimate_eens_lolp(
            cem_result["v_gen"],
            cem_result["v_line"],
            cem_result["v_trafo"],
            n_samples=500,
            recovery_hours=24.0,
            seed=123,
        )

        assert estimation["eens_mwh"] >= 0.0
        assert 0.0 <= estimation["lolp"] <= 1.0
        assert estimation["std_error"] >= 0.0

    def test_financial_npv_bcr(
        self,
        ieee24_rts_grid: GridModel,
        synthetic_wind_hazard: HazardEvent,
    ) -> None:
        """Financial analysis should produce positive NPV and BCR > 1."""
        calculator = AvoidedLossCalculator(
            grid_model=ieee24_rts_grid,
            planning_horizon=20,
            discount_rate=0.05,
        )

        voll_by_sector = {
            "residential": 10000.0,
            "commercial": 25000.0,
            "industrial": 75000.0,
        }

        result = calculator.run_cost_benefit_analysis(
            initial_investment=5_000_000.0,
            annual_opex_delta=-50_000.0,
            hazard_events=[synthetic_wind_hazard],
            voll_by_sector=voll_by_sector,
        )

        assert "npv" in result
        assert "bcr" in result
        assert "irr" in result

        assert not np.isnan(result["npv"])
        assert not np.isnan(result["bcr"])

    def test_rrs_scorecard_generation(
        self,
        ieee24_rts_grid: GridModel,
        synthetic_wind_hazard: HazardEvent,
    ) -> None:
        """RRS scorecard should generate valid ESG report."""
        calculator = AvoidedLossCalculator(
            grid_model=ieee24_rts_grid,
            planning_horizon=20,
            discount_rate=0.05,
        )

        voll_by_sector = {
            "residential": 10000.0,
            "commercial": 25000.0,
            "industrial": 75000.0,
        }

        cba_result = calculator.run_cost_benefit_analysis(
            initial_investment=5_000_000.0,
            annual_opex_delta=-50_000.0,
            hazard_events=[synthetic_wind_hazard],
            voll_by_sector=voll_by_sector,
        )

        reporter = RRSReportGenerator(
            project_name="IEEE 24-bus RTS Resilience Upgrade",
            planning_horizon=20,
            discount_rate=0.05,
        )

        scorecard = reporter.generate_report(cba_result)

        assert_rrs_scorecard_valid(scorecard)

        assert "resilience_of_the_project" in scorecard
        assert "resilience_through_the_project" in scorecard

        resilience_of = scorecard["resilience_of_the_project"]
        assert "grade" in resilience_of
        assert resilience_of["grade"] in {
            "AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D",
        }

    def test_full_pipeline_wind_end_to_end(
        self,
        ieee24_rts_grid: GridModel,
        synthetic_wind_hazard: HazardEvent,
        wind_fragility_set: ImpactFunctionSet,
    ) -> None:
        """Complete end-to-end pipeline: wind hazard → fragility → CEM → finance → RRS."""
        intensities = generate_hazard_intensity_map(
            ieee24_rts_grid, synthetic_wind_hazard
        )

        failure_probs: Dict[str, float] = {}
        for asset_id, intensity in intensities.items():
            func = wind_fragility_set.get_function(
                hazard_type="wind", asset_type="transmission_line"
            )
            if func is not None:
                failure_probs[asset_id] = func.evaluate(intensity)

        assert len(failure_probs) > 0

        simulator = CEMMonteCarloSimulator(
            ieee24_rts_grid,
            cascade_tolerance=1.2,
            cascade_max_iter=30,
            rho=0.2,
        )

        cem_result = simulator.run_cem(
            n_iterations=3, n_samples_per_iter=200, seed=42
        )

        estimation = simulator.estimate_eens_lolp(
            cem_result["v_gen"],
            cem_result["v_line"],
            cem_result["v_trafo"],
            n_samples=500,
            recovery_hours=24.0,
            seed=123,
        )

        assert estimation["eens_mwh"] >= 0.0
        assert 0.0 <= estimation["lolp"] <= 1.0

        calculator = AvoidedLossCalculator(
            grid_model=ieee24_rts_grid,
            planning_horizon=20,
            discount_rate=0.05,
        )

        voll_by_sector = {
            "residential": 10000.0,
            "commercial": 25000.0,
            "industrial": 75000.0,
        }

        cba_result = calculator.run_cost_benefit_analysis(
            initial_investment=5_000_000.0,
            annual_opex_delta=-50_000.0,
            hazard_events=[synthetic_wind_hazard],
            voll_by_sector=voll_by_sector,
        )

        assert_financial_metrics_valid(cba_result["npv"], cba_result["bcr"])

        reporter = RRSReportGenerator(
            project_name="IEEE 24-bus RTS Wind Resilience",
            planning_horizon=20,
            discount_rate=0.05,
        )

        scorecard = reporter.generate_report(cba_result)
        assert_rrs_scorecard_valid(scorecard)

        pf_result = ieee24_rts_grid.run_baseline_power_flow()
        assert_power_flow_valid(pf_result, label="IEEE24-final")

        assert_metric_tolerance(
            float(pf_result["total_losses_mw"]),
            IEEE24_RTS_BENCHMARKS["expected_total_losses_mw_max"] * 0.5,
            tolerance=1.0,
            label="IEEE24-losses",
        )

    def test_full_pipeline_flood_end_to_end(
        self,
        ieee24_rts_grid: GridModel,
        synthetic_flood_hazard: HazardEvent,
        flood_fragility_set: ImpactFunctionSet,
    ) -> None:
        """Complete end-to-end pipeline: flood hazard → fragility → CEM → finance → RRS."""
        intensities = generate_hazard_intensity_map(
            ieee24_rts_grid, synthetic_flood_hazard
        )

        failure_probs: Dict[str, float] = {}
        for asset_id, intensity in intensities.items():
            func = flood_fragility_set.get_function(
                hazard_type="flood", asset_type="substation"
            )
            if func is not None:
                failure_probs[asset_id] = func.evaluate(intensity)

        assert len(failure_probs) > 0

        simulator = CEMMonteCarloSimulator(
            ieee24_rts_grid,
            cascade_tolerance=1.2,
            cascade_max_iter=30,
            rho=0.2,
        )

        cem_result = simulator.run_cem(
            n_iterations=3, n_samples_per_iter=200, seed=99
        )

        estimation = simulator.estimate_eens_lolp(
            cem_result["v_gen"],
            cem_result["v_line"],
            cem_result["v_trafo"],
            n_samples=500,
            recovery_hours=48.0,
            seed=456,
        )

        assert estimation["eens_mwh"] >= 0.0
        assert 0.0 <= estimation["lolp"] <= 1.0

        calculator = AvoidedLossCalculator(
            grid_model=ieee24_rts_grid,
            planning_horizon=20,
            discount_rate=0.05,
        )

        voll_by_sector = {
            "residential": 10000.0,
            "commercial": 25000.0,
            "industrial": 75000.0,
        }

        cba_result = calculator.run_cost_benefit_analysis(
            initial_investment=3_000_000.0,
            annual_opex_delta=-30_000.0,
            hazard_events=[synthetic_flood_hazard],
            voll_by_sector=voll_by_sector,
        )

        assert_financial_metrics_valid(cba_result["npv"], cba_result["bcr"])

        reporter = RRSReportGenerator(
            project_name="IEEE 24-bus RTS Flood Resilience",
            planning_horizon=20,
            discount_rate=0.05,
        )

        scorecard = reporter.generate_report(cba_result)
        assert_rrs_scorecard_valid(scorecard)


# ---------------------------------------------------------------------------
# End-to-end pipeline on SimBench-like HV urban grid
# ---------------------------------------------------------------------------


class TestEndToEndPipelineSimBench:
    """Full pipeline integration tests on a SimBench-like HV urban grid."""

    def test_grid_loads_and_runs_power_flow(
        self, simbench_hv_grid: GridModel
    ) -> None:
        """SimBench-like HV grid should load and converge in AC power flow."""
        result = simbench_hv_grid.run_baseline_power_flow()
        assert_power_flow_valid(result, label="SimBench")

        n_buses = len(simbench_hv_grid.net.bus)
        n_lines = len(simbench_hv_grid.net.line)
        assert n_buses == SIMBENCH_HV_BENCHMARKS["n_buses"]
        assert n_lines >= 30, f"Expected ≥30 lines, got {n_lines}"

    def test_wind_fragility_on_simbench(
        self,
        simbench_hv_grid: GridModel,
        synthetic_wind_hazard: HazardEvent,
        wind_fragility_set: ImpactFunctionSet,
    ) -> None:
        """Wind fragility should produce valid failure probabilities."""
        intensities = generate_hazard_intensity_map(
            simbench_hv_grid, synthetic_wind_hazard
        )

        failure_probs: Dict[str, float] = {}
        for asset_id, intensity in intensities.items():
            func = wind_fragility_set.get_function(
                hazard_type="wind", asset_type="transmission_line"
            )
            if func is not None:
                failure_probs[asset_id] = func.evaluate(intensity)

        assert len(failure_probs) > 0
        for prob in failure_probs.values():
            assert 0.0 <= prob <= 1.0

    def test_cem_monte_carlo_on_simbench(
        self, simbench_hv_grid: GridModel
    ) -> None:
        """CEM Monte Carlo should estimate EENS and LOLP on SimBench grid."""
        simulator = CEMMonteCarloSimulator(
            simbench_hv_grid,
            cascade_tolerance=1.2,
            cascade_max_iter=30,
            rho=0.2,
        )

        cem_result = simulator.run_cem(
            n_iterations=3,
            n_samples_per_iter=200,
            seed=42,
        )

        estimation = simulator.estimate_eens_lolp(
            cem_result["v_gen"],
            cem_result["v_line"],
            cem_result["v_trafo"],
            n_samples=500,
            recovery_hours=24.0,
            seed=123,
        )

        assert estimation["eens_mwh"] >= 0.0
        assert 0.0 <= estimation["lolp"] <= 1.0

    def test_full_pipeline_simbench(
        self,
        simbench_hv_grid: GridModel,
        synthetic_wind_hazard: HazardEvent,
        wind_fragility_set: ImpactFunctionSet,
    ) -> None:
        """Complete pipeline on SimBench-like grid."""
        intensities = generate_hazard_intensity_map(
            simbench_hv_grid, synthetic_wind_hazard
        )

        failure_probs: Dict[str, float] = {}
        for asset_id, intensity in intensities.items():
            func = wind_fragility_set.get_function(
                hazard_type="wind", asset_type="transmission_line"
            )
            if func is not None:
                failure_probs[asset_id] = func.evaluate(intensity)

        assert len(failure_probs) > 0

        simulator = CEMMonteCarloSimulator(
            simbench_hv_grid,
            cascade_tolerance=1.2,
            cascade_max_iter=30,
            rho=0.2,
        )

        cem_result = simulator.run_cem(
            n_iterations=3, n_samples_per_iter=200, seed=42
        )

        estimation = simulator.estimate_eens_lolp(
            cem_result["v_gen"],
            cem_result["v_line"],
            cem_result["v_trafo"],
            n_samples=500,
            recovery_hours=24.0,
            seed=123,
        )

        assert estimation["eens_mwh"] >= 0.0

        calculator = AvoidedLossCalculator(
            grid_model=simbench_hv_grid,
            planning_horizon=20,
            discount_rate=0.05,
        )

        voll_by_sector = {
            "residential": 10000.0,
            "commercial": 25000.0,
            "industrial": 75000.0,
        }

        cba_result = calculator.run_cost_benefit_analysis(
            initial_investment=3_000_000.0,
            annual_opex_delta=-30_000.0,
            hazard_events=[synthetic_wind_hazard],
            voll_by_sector=voll_by_sector,
        )

        assert_financial_metrics_valid(cba_result["npv"], cba_result["bcr"])

        reporter = RRSReportGenerator(
            project_name="SimBench HV Urban Resilience",
            planning_horizon=20,
            discount_rate=0.05,
        )

        scorecard = reporter.generate_report(cba_result)
        assert_rrs_scorecard_valid(scorecard)

        pf_result = simbench_hv_grid.run_baseline_power_flow()
        assert_power_flow_valid(pf_result, label="SimBench-final")


# ---------------------------------------------------------------------------
# Financial metrics validation
# ---------------------------------------------------------------------------


class TestFinancialMetrics:
    """Validation of NPV, BCR, and IRR calculations."""

    def test_npv_positive_for_viable_investment(
        self,
        ieee24_rts_grid: GridModel,
        synthetic_wind_hazard: HazardEvent,
    ) -> None:
        """A resilience investment with positive avoided loss should yield NPV > 0."""
        calculator = AvoidedLossCalculator(
            grid_model=ieee24_rts_grid,
            planning_horizon=20,
            discount_rate=0.05,
        )

        result = calculator.run_cost_benefit_analysis(
            initial_investment=1_000_000.0,
            annual_opex_delta=0.0,
            hazard_events=[synthetic_wind_hazard],
            voll_by_sector={"residential": 10000.0},
        )

        assert_financial_metrics_valid(result["npv"], result["bcr"])

    def test_bcr_exceeds_one(
        self,
        ieee24_rts_grid: GridModel,
        synthetic_wind_hazard: HazardEvent,
    ) -> None:
        """BCR should exceed 1.0 for a cost-effective investment."""
        calculator = AvoidedLossCalculator(
            grid_model=ieee24_rts_grid,
            planning_horizon=20,
            discount_rate=0.05,
        )

        result = calculator.run_cost_benefit_analysis(
            initial_investment=1_000_000.0,
            annual_opex_delta=0.0,
            hazard_events=[synthetic_wind_hazard],
            voll_by_sector={"residential": 10000.0},
        )

        assert result["bcr"] > 1.0, f"BCR {result['bcr']:.3f} should exceed 1.0"

    def test_irr_computable(
        self,
        ieee24_rts_grid: GridModel,
        synthetic_wind_hazard: HazardEvent,
    ) -> None:
        """IRR should be computable (not None) for a viable investment."""
        calculator = AvoidedLossCalculator(
            grid_model=ieee24_rts_grid,
            planning_horizon=20,
            discount_rate=0.05,
        )

        result = calculator.run_cost_benefit_analysis(
            initial_investment=1_000_000.0,
            annual_opex_delta=0.0,
            hazard_events=[synthetic_wind_hazard],
            voll_by_sector={"residential": 10000.0},
        )

        assert result["irr"] is not None, "IRR should be computable"
        assert result["irr"] > 0.0, f"IRR {result['irr']} should be positive"


# ---------------------------------------------------------------------------
# RRS ESG reporting validation
# ---------------------------------------------------------------------------


class TestRRSReporting:
    """Validation of RRS scorecard generation and ESG alignment."""

    def test_scorecard_contains_required_sections(
        self,
        ieee24_rts_grid: GridModel,
        synthetic_wind_hazard: HazardEvent,
    ) -> None:
        """RRS scorecard must contain resilience_of and resilience_through sections."""
        calculator = AvoidedLossCalculator(
            grid_model=ieee24_rts_grid,
            planning_horizon=20,
            discount_rate=0.05,
        )

        cba_result = calculator.run_cost_benefit_analysis(
            initial_investment=5_000_000.0,
            annual_opex_delta=-50_000.0,
            hazard_events=[synthetic_wind_hazard],
            voll_by_sector={"residential": 10000.0},
        )

        reporter = RRSReportGenerator(
            project_name="Test Project",
            planning_horizon=20,
            discount_rate=0.05,
        )

        scorecard = reporter.generate_report(cba_result)

        assert "resilience_of_the_project" in scorecard
        assert "resilience_through_the_project" in scorecard
        assert "kpis" in scorecard
        assert "regulatory_alignment" in scorecard

    def test_scorecard_json_serializable(
        self,
        ieee24_rts_grid: GridModel,
        synthetic_wind_hazard: HazardEvent,
    ) -> None:
        """RRS scorecard must be JSON-serializable."""
        calculator = AvoidedLossCalculator(
            grid_model=ieee24_rts_grid,
            planning_horizon=20,
            discount_rate=0.05,
        )

        cba_result = calculator.run_cost_benefit_analysis(
            initial_investment=5_000_000.0,
            annual_opex_delta=-50_000.0,
            hazard_events=[synthetic_wind_hazard],
            voll_by_sector={"residential": 10000.0},
        )

        reporter = RRSReportGenerator(
            project_name="Test Project",
            planning_horizon=20,
            discount_rate=0.05,
        )

        scorecard = reporter.generate_report(cba_result)

        json_str = json.dumps(scorecard, default=str)
        parsed = json.loads(json_str)

        assert isinstance(parsed, dict)
        assert parsed["project_name"] == "Test Project"

    def test_grade_assignment(
        self,
        ieee24_rts_grid: GridModel,
        synthetic_wind_hazard: HazardEvent,
    ) -> None:
        """Grade should be a valid RRS bond-style grade."""
        calculator = AvoidedLossCalculator(
            grid_model=ieee24_rts_grid,
            planning_horizon=20,
            discount_rate=0.05,
        )

        cba_result = calculator.run_cost_benefit_analysis(
            initial_investment=5_000_000.0,
            annual_opex_delta=-50_000.0,
            hazard_events=[synthetic_wind_hazard],
            voll_by_sector={"residential": 10000.0},
        )

        reporter = RRSReportGenerator(
            project_name="Test Project",
            planning_horizon=20,
            discount_rate=0.05,
        )

        scorecard = reporter.generate_report(cba_result)
        grade = scorecard["resilience_of_the_project"]["grade"]

        valid_grades = {"AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"}
        assert grade in valid_grades, f"Invalid grade: {grade}"

    def test_regulatory_alignment_present(
        self,
        ieee24_rts_grid: GridModel,
        synthetic_wind_hazard: HazardEvent,
    ) -> None:
        """Scorecard should reference EU Taxonomy, TCFD, ISSB, and GRI."""
        calculator = AvoidedLossCalculator(
            grid_model=ieee24_rts_grid,
            planning_horizon=20,
            discount_rate=0.05,
        )

        cba_result = calculator.run_cost_benefit_analysis(
            initial_investment=5_000_000.0,
            annual_opex_delta=-50_000.0,
            hazard_events=[synthetic_wind_hazard],
            voll_by_sector={"residential": 10000.0},
        )

        reporter = RRSReportGenerator(
            project_name="Test Project",
            planning_horizon=20,
            discount_rate=0.05,
        )

        scorecard = reporter.generate_report(cba_result)
        alignment = scorecard.get("regulatory_alignment", {})

        expected_frameworks = {"eu_taxonomy", "tcfd", "issb_s2", "gri"}
        for framework in expected_frameworks:
            assert framework in alignment, f"Missing regulatory framework: {framework}"


# ---------------------------------------------------------------------------
# Power flow validation
# ---------------------------------------------------------------------------


class TestPowerFlowValidation:
    """Validation of AC/DC power flow convergence and electrical validity."""

    def test_ac_power_flow_converges_ieee24(
        self, ieee24_rts_grid: GridModel
    ) -> None:
        """AC power flow must converge on IEEE 24-bus RTS."""
        result = ieee24_rts_grid.run_baseline_power_flow()
        assert result["converged"], "AC power flow did not converge on IEEE 24-bus"

    def test_ac_power_flow_converges_simbench(
        self, simbench_hv_grid: GridModel
    ) -> None:
        """AC power flow must converge on SimBench-like grid."""
        result = simbench_hv_grid.run_baseline_power_flow()
        assert result["converged"], "AC power flow did not converge on SimBench grid"

    def test_voltage_magnitudes_in_range_ieee24(
        self, ieee24_rts_grid: GridModel
    ) -> None:
        """All bus voltages must be within [0.95, 1.05] pu on IEEE 24-bus."""
        result = ieee24_rts_grid.run_baseline_power_flow()
        for i, vm in enumerate(result["vm_pu"]):
            assert 0.95 <= vm <= 1.05, f"Bus {i} voltage {vm:.4f} pu out of range"

    def test_voltage_magnitudes_in_range_simbench(
        self, simbench_hv_grid: GridModel
    ) -> None:
        """All bus voltages must be within [0.95, 1.05] pu on SimBench grid."""
        result = simbench_hv_grid.run_baseline_power_flow()
        for i, vm in enumerate(result["vm_pu"]):
            assert 0.95 <= vm <= 1.05, f"Bus {i} voltage {vm:.4f} pu out of range"

    def test_line_loadings_within_limits_ieee24(
        self, ieee24_rts_grid: GridModel
    ) -> None:
        """Line loadings must not exceed 100% under baseline conditions."""
        result = ieee24_rts_grid.run_baseline_power_flow()
        for i, loading in enumerate(result["loading_percent"]):
            assert loading <= 100.0, f"Line {i} loading {loading:.1f}% exceeds 100%"

    def test_total_losses_reasonable_ieee24(
        self, ieee24_rts_grid: GridModel
    ) -> None:
        """Total active power losses should be within IEEE 24-bus benchmark."""
        result = ieee24_rts_grid.run_baseline_power_flow()
        losses = float(result["total_losses_mw"])
        max_expected = IEEE24_RTS_BENCHMARKS["expected_total_losses_mw_max"]
        assert losses <= max_expected, (
            f"Losses {losses:.2f} MW exceed benchmark max {max_expected} MW"
        )


# ---------------------------------------------------------------------------
# Hazard intensity mapping
# ---------------------------------------------------------------------------


class TestHazardIntensityMapping:
    """Validation of hazard-to-asset intensity mapping."""

    def test_wind_intensity_map_ieee24(
        self,
        ieee24_rts_grid: GridModel,
        synthetic_wind_hazard: HazardEvent,
    ) -> None:
        """Wind intensity map should cover all buses and lines."""
        intensities = generate_hazard_intensity_map(
            ieee24_rts_grid, synthetic_wind_hazard
        )

        n_buses = len(ieee24_rts_grid.net.bus)
        n_lines = len(ieee24_rts_grid.net.line)
        assert len(intensities) >= n_buses + n_lines

        for val in intensities.values():
            assert val >= 0.0
            assert not np.isnan(val)

    def test_flood_intensity_map_ieee24(
        self,
        ieee24_rts_grid: GridModel,
        synthetic_flood_hazard: HazardEvent,
    ) -> None:
        """Flood intensity map should cover all substations."""
        intensities = generate_hazard_intensity_map(
            ieee24_rts_grid, synthetic_flood_hazard
        )

        assert len(intensities) > 0
        for val in intensities.values():
            assert val >= 0.0
            assert not np.isnan(val)
