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

"""Unit tests for ``resilient_blackout.grid.carbon_accounting``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resilient_blackout.grid.carbon_accounting import CarbonAccountingEngine


# ---------------------------------------------------------------------------
# Helpers: build minimal pandapower networks
# ---------------------------------------------------------------------------


def _make_simple_net():
    """Build a minimal 3-bus network with coal + gas generators."""
    import pandapower as pp

    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=110, name="Bus 0")
    b1 = pp.create_bus(net, vn_kv=110, name="Bus 1")
    b2 = pp.create_bus(net, vn_kv=110, name="Bus 2")

    pp.create_line(net, b0, b1, length_km=10, std_type="149-AL1/24-ST1A 110.0")
    pp.create_line(net, b1, b2, length_km=10, std_type="149-AL1/24-ST1A 110.0")

    pp.create_gen(net, b0, p_mw=50, min_p_mw=0, max_p_mw=80, name="Coal Gen 1", slack=True)
    pp.create_gen(net, b1, p_mw=30, min_p_mw=0, max_p_mw=60, name="Gas Gen 2", slack=False)

    pp.create_load(net, b1, p_mw=40, name="Load 1")
    pp.create_load(net, b2, p_mw=30, name="Load 2")

    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=30)
    pp.create_poly_cost(net, 1, "gen", cp1_eur_per_mw=50)

    return net


def _make_resilient_net():
    """Build a resilient variant with storage added at bus 2."""
    import pandapower as pp

    net = _make_simple_net()
    pp.create_storage(
        net, 2, p_mw=0, q_mvar=0, sn_mva=20,
        soc_percent=50, min_e_mwh=0, max_e_mwh=20,
        name="Battery 1",
    )
    pp.create_poly_cost(net, 0, "storage", cp1_eur_per_mw=10)
    return net


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestInit:
    """Validation of constructor and parameter handling."""

    def test_default_construction(self) -> None:
        engine = CarbonAccountingEngine()
        assert engine.fuel_map is None
        assert engine.default_co2_factor == 600.0

    def test_custom_fuel_map(self) -> None:
        fm = {0: {"fuel": "coal", "co2_factor": 950.0}}
        engine = CarbonAccountingEngine(fuel_map=fm, default_co2_factor=500.0)
        assert engine.fuel_map == fm
        assert engine.default_co2_factor == 500.0

    def test_invalid_co2_factor_raises(self) -> None:
        with pytest.raises(ValueError, match="default_co2_factor"):
            CarbonAccountingEngine(default_co2_factor=-1.0)

    def test_repr(self) -> None:
        engine = CarbonAccountingEngine()
        r = repr(engine)
        assert "CarbonAccountingEngine" in r
        assert "600" in r


# ---------------------------------------------------------------------------
# Generator enrichment
# ---------------------------------------------------------------------------


class TestEnrichGenerators:
    """Validation of generator fuel-type enrichment."""

    def test_adds_columns(self) -> None:
        net = _make_simple_net()
        engine = CarbonAccountingEngine()
        engine.enrich_generators(net)
        assert "fuel_type" in net.gen.columns
        assert "co2_kg_per_mwh" in net.gen.columns
        assert "fuel_type" in net.sgen.columns
        assert "co2_kg_per_mwh" in net.sgen.columns

    def test_detects_coal(self) -> None:
        net = _make_simple_net()
        engine = CarbonAccountingEngine()
        engine.enrich_generators(net)
        assert net.gen.at[0, "fuel_type"] == "coal"
        assert net.gen.at[0, "co2_kg_per_mwh"] == 950.0

    def test_detects_gas(self) -> None:
        net = _make_simple_net()
        engine = CarbonAccountingEngine()
        engine.enrich_generators(net)
        assert net.gen.at[1, "fuel_type"] == "gas"
        assert net.gen.at[1, "co2_kg_per_mwh"] == 450.0

    def test_fuel_map_overrides(self) -> None:
        net = _make_simple_net()
        fm = {0: {"fuel": "solar", "co2_factor": 0.0}}
        engine = CarbonAccountingEngine(fuel_map=fm)
        engine.enrich_generators(net)
        assert net.gen.at[0, "fuel_type"] == "solar"
        assert net.gen.at[0, "co2_kg_per_mwh"] == 0.0
        # Gen 1 still auto-detected
        assert net.gen.at[1, "fuel_type"] == "gas"


# ---------------------------------------------------------------------------
# System emissions
# ---------------------------------------------------------------------------


class TestSystemEmissions:
    """Validation of system emissions computation."""

    def test_after_power_flow(self) -> None:
        import pandapower as pp

        net = _make_simple_net()
        engine = CarbonAccountingEngine()
        pp.runpp(net)
        result = engine.compute_system_emissions(net)
        assert "total_kg_co2" in result
        assert "total_tonne_co2" in result
        assert result["total_kg_co2"] > 0
        assert "per_gen" in result
        assert "per_sgen" in result

    def test_emissions_match_fuel(self) -> None:
        import pandapower as pp

        net = _make_simple_net()
        engine = CarbonAccountingEngine()
        pp.runpp(net)
        result = engine.compute_system_emissions(net)
        # Coal gen at ~50 MW * 950 = 47500 kg; gas gen at ~20 MW * 450 = 9000 kg
        # Total ~56500 kg
        assert 40000 < result["total_kg_co2"] < 80000


# ---------------------------------------------------------------------------
# Average Carbon Emission (ACE)
# ---------------------------------------------------------------------------


class TestAverageCarbonEmission:
    """Validation of ACE computation."""

    def test_positive_ace(self) -> None:
        import pandapower as pp

        net = _make_simple_net()
        engine = CarbonAccountingEngine()
        pp.runpp(net)
        ace = engine.calculate_average_carbon_emission(net)
        assert ace > 0
        # ACE = total_kg / total_load_mw
        # Should be between coal (950) and gas (450), weighted by dispatch
        assert 400 < ace < 1000

    def test_zero_load(self) -> None:
        import pandapower as pp

        net = pp.create_empty_network()
        pp.create_bus(net, vn_kv=110)
        pp.create_gen(net, 0, p_mw=0, max_p_mw=10, name="Coal", slack=True)
        pp.create_load(net, 0, p_mw=0)
        engine = CarbonAccountingEngine()
        pp.runpp(net)
        ace = engine.calculate_average_carbon_emission(net)
        assert ace == 0.0


# ---------------------------------------------------------------------------
# Locational Marginal Carbon Emission (LMCE)
# ---------------------------------------------------------------------------


class TestLocationalMarginalCarbonEmission:
    """Validation of LMCE computation."""

    def test_positive_lmce(self) -> None:
        net = _make_simple_net()
        engine = CarbonAccountingEngine()
        lmce = engine.calculate_locational_marginal_carbon_emission(
            net, target_bus=1, delta_mw=1.0
        )
        # Adding load should increase emissions
        assert lmce >= 0

    def test_lmce_at_bus_with_load(self) -> None:
        net = _make_simple_net()
        engine = CarbonAccountingEngine()
        lmce = engine.calculate_locational_marginal_carbon_emission(
            net, target_bus=2, delta_mw=1.0
        )
        assert lmce >= 0

    def test_lmce_no_load_returns_zero(self) -> None:
        import pandapower as pp

        net = pp.create_empty_network()
        b0 = pp.create_bus(net, vn_kv=110)
        b1 = pp.create_bus(net, vn_kv=110)
        pp.create_line(net, b0, b1, length_km=10, std_type="149-AL1/24-ST1A 110.0")
        pp.create_gen(net, b0, p_mw=0, max_p_mw=50, name="Coal", slack=True)
        pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=30)
        engine = CarbonAccountingEngine()
        lmce = engine.calculate_locational_marginal_carbon_emission(
            net, target_bus=1, delta_mw=1.0
        )
        assert lmce == 0.0

    def test_larger_perturbation(self) -> None:
        net = _make_simple_net()
        engine = CarbonAccountingEngine()
        lmce_small = engine.calculate_locational_marginal_carbon_emission(
            net, target_bus=1, delta_mw=0.5
        )
        net2 = _make_simple_net()
        lmce_large = engine.calculate_locational_marginal_carbon_emission(
            net2, target_bus=1, delta_mw=5.0
        )
        # Both should be in similar range
        assert abs(lmce_small - lmce_large) < 200


# ---------------------------------------------------------------------------
# Multi-period simulation
# ---------------------------------------------------------------------------


class TestMultiPeriodSimulation:
    """Validation of multi-period emissions simulation."""

    def test_baseline_vs_resilient(self) -> None:
        baseline = _make_simple_net()
        resilient = _make_resilient_net()
        engine = CarbonAccountingEngine()

        load_profile = np.array([70.0, 80.0, 60.0, 90.0, 50.0])
        result = engine.simulate_multi_period(baseline, resilient, load_profile)

        assert "cumulative_avoided_tonne" in result
        assert "baseline_emissions_tonne" in result
        assert "resilient_emissions_tonne" in result
        assert "avoided_emissions_tonne" in result
        assert "baseline_ace" in result
        assert "resilient_ace" in result
        assert "converged" in result
        assert len(result["baseline_emissions_tonne"]) == 5

    def test_cumulative_avoided_is_sum(self) -> None:
        baseline = _make_simple_net()
        resilient = _make_resilient_net()
        engine = CarbonAccountingEngine()

        load_profile = np.array([70.0, 80.0, 60.0])
        result = engine.simulate_multi_period(baseline, resilient, load_profile)

        expected_sum = float(np.sum(result["avoided_emissions_tonne"]))
        assert np.isclose(result["cumulative_avoided_tonne"], expected_sum)

    def test_avoided_non_negative(self) -> None:
        baseline = _make_simple_net()
        resilient = _make_resilient_net()
        engine = CarbonAccountingEngine()

        load_profile = np.array([70.0, 80.0, 60.0])
        result = engine.simulate_multi_period(baseline, resilient, load_profile)

        # Resilient should not emit more than baseline
        for i in range(len(load_profile)):
            if result["converged"][i]:
                assert result["avoided_emissions_tonne"][i] >= -1e-6

    def test_ace_values(self) -> None:
        baseline = _make_simple_net()
        resilient = _make_resilient_net()
        engine = CarbonAccountingEngine()

        load_profile = np.array([70.0])
        result = engine.simulate_multi_period(baseline, resilient, load_profile)

        if result["converged"][0]:
            assert result["baseline_ace"][0] > 0
            assert result["resilient_ace"][0] > 0
