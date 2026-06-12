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

"""Unit tests for ``resilient_blackout.economics.investment_planner``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resilient_blackout.economics.investment_planner import (
    ClimateScenario,
    InvestmentPortfolioOptimizer,
    ResilienceProject,
)


def _make_sample_projects() -> list:
    return [
        ResilienceProject(
            name="Underground Line A",
            capex_usd=2_000_000,
            target_asset_ids=["line_1", "line_2"],
            failure_rate_multiplier=0.3,
            recovery_time_multiplier=0.5,
        ),
        ResilienceProject(
            name="Substation Flood Wall",
            capex_usd=1_500_000,
            target_asset_ids=["sub_1"],
            failure_rate_multiplier=0.8,
            recovery_time_multiplier=0.4,
        ),
        ResilienceProject(
            name="Vegetation Management Zone B",
            capex_usd=500_000,
            opex_delta_usd_per_year=20_000,
            target_asset_ids=["line_3", "line_4", "line_5"],
            failure_rate_multiplier=0.6,
            recovery_time_multiplier=0.9,
        ),
        ResilienceProject(
            name="Seismic Retrofit Substation C",
            capex_usd=3_000_000,
            target_asset_ids=["sub_2", "sub_3"],
            failure_rate_multiplier=0.4,
            recovery_time_multiplier=0.3,
        ),
    ]


def _make_sample_scenarios() -> list:
    return [
        ClimateScenario(
            name="Moderate Wind (RCP 4.5)",
            probability=0.5,
            baseline_eens_mwh=500.0,
            voll_usd_per_mwh=10_000,
        ),
        ClimateScenario(
            name="Extreme Wind (RCP 8.5)",
            probability=0.3,
            baseline_eens_mwh=2000.0,
            voll_usd_per_mwh=15_000,
        ),
        ClimateScenario(
            name="Flood Event",
            probability=0.2,
            baseline_eens_mwh=800.0,
            voll_usd_per_mwh=12_000,
        ),
    ]


class TestResilienceProject:
    """Validation of ResilienceProject dataclass."""

    def test_valid_construction(self) -> None:
        p = ResilienceProject(name="Test", capex_usd=1_000_000)
        assert p.name == "Test"
        assert p.capex_usd == 1_000_000
        assert p.failure_rate_multiplier == 1.0

    def test_negative_capex_raises(self) -> None:
        with pytest.raises(ValueError, match="capex_usd"):
            ResilienceProject(name="Bad", capex_usd=-100)

    def test_invalid_failure_multiplier_raises(self) -> None:
        with pytest.raises(ValueError, match="failure_rate_multiplier"):
            ResilienceProject(name="Bad", capex_usd=100, failure_rate_multiplier=1.5)

    def test_invalid_recovery_multiplier_raises(self) -> None:
        with pytest.raises(ValueError, match="recovery_time_multiplier"):
            ResilienceProject(name="Bad", capex_usd=100, recovery_time_multiplier=0.0)


class TestClimateScenario:
    """Validation of ClimateScenario dataclass."""

    def test_valid_construction(self) -> None:
        s = ClimateScenario(name="Test", probability=0.5)
        assert s.name == "Test"
        assert s.probability == 0.5

    def test_invalid_probability_raises(self) -> None:
        with pytest.raises(ValueError, match="probability"):
            ClimateScenario(name="Bad", probability=1.5)

    def test_negative_eens_raises(self) -> None:
        with pytest.raises(ValueError, match="baseline_eens_mwh"):
            ClimateScenario(name="Bad", probability=0.5, baseline_eens_mwh=-10)

    def test_negative_voll_raises(self) -> None:
        with pytest.raises(ValueError, match="voll_usd_per_mwh"):
            ClimateScenario(name="Bad", probability=0.5, voll_usd_per_mwh=-100)


class TestAmortizeCapex:
    """Validation of CAPEX amortisation."""

    def test_zero_capex(self) -> None:
        result = InvestmentPortfolioOptimizer._amortize_capex(0, 0.05, 20)
        assert result == 0.0

    def test_positive_capex(self) -> None:
        result = InvestmentPortfolioOptimizer._amortize_capex(1_000_000, 0.05, 20)
        assert result > 0
        assert result < 1_000_000

    def test_longer_horizon_reduces_annual(self) -> None:
        short = InvestmentPortfolioOptimizer._amortize_capex(1_000_000, 0.05, 10)
        long = InvestmentPortfolioOptimizer._amortize_capex(1_000_000, 0.05, 30)
        assert long < short


class TestInit:
    """Validation of optimizer construction."""

    def test_valid_construction(self) -> None:
        opt = InvestmentPortfolioOptimizer(
            _make_sample_projects(),
            _make_sample_scenarios(),
            budget_usd=5_000_000,
        )
        assert len(opt.projects) == 4
        assert len(opt.scenarios) == 3
        assert opt.budget_usd == 5_000_000

    def test_empty_projects_raises(self) -> None:
        with pytest.raises(ValueError, match="projects"):
            InvestmentPortfolioOptimizer(
                [], _make_sample_scenarios(), budget_usd=1_000_000
            )

    def test_empty_scenarios_raises(self) -> None:
        with pytest.raises(ValueError, match="scenarios"):
            InvestmentPortfolioOptimizer(
                _make_sample_projects(), [], budget_usd=1_000_000
            )

    def test_negative_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="budget_usd"):
            InvestmentPortfolioOptimizer(
                _make_sample_projects(), _make_sample_scenarios(), budget_usd=-100
            )

    def test_invalid_discount_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="discount_rate"):
            InvestmentPortfolioOptimizer(
                _make_sample_projects(),
                _make_sample_scenarios(),
                budget_usd=1_000_000,
                discount_rate=1.5,
            )


class TestSolve:
    """Validation of LP solve and results."""

    def test_solve_returns_dict(self) -> None:
        opt = InvestmentPortfolioOptimizer(
            _make_sample_projects(),
            _make_sample_scenarios(),
            budget_usd=5_000_000,
        )
        result = opt.solve()
        assert isinstance(result, dict)
        assert "selected_projects" in result
        assert "z_values" in result
        assert "total_capex_usd" in result
        assert "bcr" in result

    def test_budget_constraint_enforced(self) -> None:
        opt = InvestmentPortfolioOptimizer(
            _make_sample_projects(),
            _make_sample_scenarios(),
            budget_usd=5_000_000,
        )
        result = opt.solve()
        assert result["total_capex_usd"] <= 5_000_000 + _EPS

    def test_zero_budget_selects_nothing(self) -> None:
        opt = InvestmentPortfolioOptimizer(
            _make_sample_projects(),
            _make_sample_scenarios(),
            budget_usd=0,
        )
        result = opt.solve()
        assert len(result["selected_projects"]) == 0
        assert result["total_capex_usd"] == 0.0

    def test_unlimited_budget_selects_all(self) -> None:
        total_capex = sum(p.capex_usd for p in _make_sample_projects())
        opt = InvestmentPortfolioOptimizer(
            _make_sample_projects(),
            _make_sample_scenarios(),
            budget_usd=total_capex * 2,
        )
        result = opt.solve()
        assert len(result["selected_projects"]) == 4

    def test_bcr_computed(self) -> None:
        opt = InvestmentPortfolioOptimizer(
            _make_sample_projects(),
            _make_sample_scenarios(),
            budget_usd=5_000_000,
        )
        result = opt.solve()
        assert result["bcr"] > 0
        assert not np.isnan(result["bcr"])

    def test_risk_reduction_with_projects(self) -> None:
        opt = InvestmentPortfolioOptimizer(
            _make_sample_projects(),
            _make_sample_scenarios(),
            budget_usd=5_000_000,
        )
        result = opt.solve()
        assert result["expected_risk_cost_usd"] <= result["baseline_risk_cost_usd"]


class TestEfficientFrontier:
    """Validation of efficient frontier computation."""

    def test_returns_list(self) -> None:
        opt = InvestmentPortfolioOptimizer(
            _make_sample_projects(),
            _make_sample_scenarios(),
            budget_usd=5_000_000,
        )
        frontier = opt.compute_efficient_frontier(n_points=10)
        assert isinstance(frontier, list)
        assert len(frontier) == 10

    def test_frontier_entries_have_keys(self) -> None:
        opt = InvestmentPortfolioOptimizer(
            _make_sample_projects(),
            _make_sample_scenarios(),
            budget_usd=5_000_000,
        )
        frontier = opt.compute_efficient_frontier(n_points=5)
        for entry in frontier:
            assert "budget_usd" in entry
            assert "total_capex_usd" in entry
            assert "expected_risk_cost_usd" in entry
            assert "bcr" in entry

    def test_risk_decreases_with_budget(self) -> None:
        opt = InvestmentPortfolioOptimizer(
            _make_sample_projects(),
            _make_sample_scenarios(),
            budget_usd=5_000_000,
        )
        frontier = opt.compute_efficient_frontier(n_points=10)
        risks = [e["expected_risk_cost_usd"] for e in frontier]
        assert risks[-1] <= risks[0] + _EPS


class TestSummaryDataframe:
    """Validation of summary DataFrame."""

    def test_returns_dataframe(self) -> None:
        opt = InvestmentPortfolioOptimizer(
            _make_sample_projects(),
            _make_sample_scenarios(),
            budget_usd=5_000_000,
        )
        opt.solve()
        df = opt.summary_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "project" in df.columns
        assert "z_value" in df.columns
        assert "selected" in df.columns

    def test_raises_without_solve(self) -> None:
        opt = InvestmentPortfolioOptimizer(
            _make_sample_projects(),
            _make_sample_scenarios(),
            budget_usd=5_000_000,
        )
        with pytest.raises(RuntimeError, match="Call solve()"):
            opt.summary_dataframe()


class TestRepr:
    """Validation of string representation."""

    def test_repr_includes_key_params(self) -> None:
        opt = InvestmentPortfolioOptimizer(
            _make_sample_projects(),
            _make_sample_scenarios(),
            budget_usd=5_000_000,
            planning_horizon_years=25,
        )
        r = repr(opt)
        assert "n_projects=4" in r
        assert "n_scenarios=3" in r
        assert "25y" in r


_EPS: float = 1e-12
