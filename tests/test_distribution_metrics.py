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

"""Unit tests for ``resilient_blackout.grid.distribution_metrics``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandapower")

from resilient_blackout.grid.distribution_metrics import (
    IEEEMetricCalculator,
    MicrogridIslandEvaluator,
)


def _make_simple_net() -> Any:
    """Build a minimal pandapower network for testing."""
    pytest.importorskip("pandapower")
    import pandapower as pp

    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=0.4, name="source")
    b1 = pp.create_bus(net, vn_kv=0.4, name="mid")
    b2 = pp.create_bus(net, vn_kv=0.4, name="load_a")
    b3 = pp.create_bus(net, vn_kv=0.4, name="load_b")

    pp.create_ext_grid(net, bus=b0, vm_pu=1.0)
    pp.create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, std_type="NAYY 4x50 SE")
    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=1.0, std_type="NAYY 4x50 SE")
    pp.create_line(net, from_bus=b1, to_bus=b3, length_km=1.0, std_type="NAYY 4x50 SE")

    pp.create_load(net, bus=b2, p_mw=0.5, q_mvar=0.1)
    pp.create_load(net, bus=b3, p_mw=0.3, q_mvar=0.05)

    net.bus["customers"] = [0, 0, 100, 80]
    return net


def _make_outage_events() -> pd.DataFrame:
    """Return a small DataFrame of outage events."""
    return pd.DataFrame(
        {
            "bus": [2, 2, 3, 3, 2],
            "start_h": [0.0, 24.0, 0.0, 48.0, 72.0],
            "end_h": [2.0, 26.0, 4.0, 52.0, 73.0],
        }
    )


@pytest.mark.pandapower
class TestIEEEMetricCalculator:
    """Validation of IEEE 1366 reliability indices."""

    def test_saifi(self) -> None:
        net = _make_simple_net()
        calc = IEEEMetricCalculator(net)
        events = _make_outage_events()
        saifi = calc.calculate_saifi(events)
        # 5 sustained events, 180 total customers
        expected = 5 / 180
        assert np.isclose(saifi, expected, rtol=1e-6)

    def test_saidi(self) -> None:
        net = _make_simple_net()
        calc = IEEEMetricCalculator(net)
        events = _make_outage_events()
        saidi = calc.calculate_saidi(events)
        # durations: 2, 2, 4, 4, 1 hours
        # customer-hours: 2*100 + 2*100 + 4*80 + 4*80 + 1*100 = 200+200+320+320+100 = 1140
        expected = 1140 / 180
        assert np.isclose(saidi, expected, rtol=1e-6)

    def test_caidi(self) -> None:
        net = _make_simple_net()
        calc = IEEEMetricCalculator(net)
        events = _make_outage_events()
        caidi = calc.calculate_caidi(events)
        expected = (1140 / 180) / (5 / 180)
        assert np.isclose(caidi, expected, rtol=1e-6)

    def test_caidi_zero_saifi(self) -> None:
        net = _make_simple_net()
        calc = IEEEMetricCalculator(net)
        empty = pd.DataFrame({"bus": [], "start_h": [], "end_h": []})
        assert calc.calculate_caidi(empty) == 0.0

    def test_cemi_n(self) -> None:
        net = _make_simple_net()
        calc = IEEEMetricCalculator(net)
        events = _make_outage_events()
        # bus 2 has 3 interruptions, bus 3 has 2; n=5 -> none exceed
        assert calc.calculate_cemi_n(events, n=5) == 0.0
        # n=1 -> bus 2 (3 > 1) and bus 3 (2 > 1) both exceed -> 2/4 buses
        assert np.isclose(calc.calculate_cemi_n(events, n=1), 2 / 4, rtol=1e-6)

    def test_meds(self) -> None:
        net = _make_simple_net()
        calc = IEEEMetricCalculator(net)
        events = _make_outage_events()
        # Day 0: SAIDI = (2*100 + 4*80)/180 = 520/180 ≈ 2.89 < 4
        # Day 1: SAIDI = (2*100)/180 = 200/180 ≈ 1.11 < 4
        # Day 2: SAIDI = (4*80)/180 = 320/180 ≈ 1.78 < 4
        # Day 3: SAIDI = (1*100)/180 = 100/180 ≈ 0.56 < 4
        assert calc.calculate_meds(events) == 0

    def test_meds_above_threshold(self) -> None:
        net = _make_simple_net()
        calc = IEEEMetricCalculator(net, med_threshold_h=1.0)
        events = _make_outage_events()
        # Day 0: 2.89 > 1.0, Day 1: 1.11 > 1.0, Day 2: 1.78 > 1.0, Day 3: 0.56 < 1.0
        assert calc.calculate_meds(events) == 3

    def test_reliability_summary(self) -> None:
        net = _make_simple_net()
        calc = IEEEMetricCalculator(net)
        events = _make_outage_events()
        summary = calc.reliability_summary(events)
        assert "saifi" in summary.columns
        assert "saidi" in summary.columns
        assert "caidi" in summary.columns
        assert "cemi_5" in summary.columns
        assert "meds" in summary.columns
        assert len(summary) == 1

    def test_bus_customers_override(self) -> None:
        net = _make_simple_net()
        override = {2: 50, 3: 40}
        calc = IEEEMetricCalculator(net, bus_customers=override)
        events = _make_outage_events()
        saifi = calc.calculate_saifi(events)
        assert np.isclose(saifi, 5 / 90, rtol=1e-6)

    def test_repr(self) -> None:
        net = _make_simple_net()
        calc = IEEEMetricCalculator(net)
        assert "IEEEMetricCalculator" in repr(calc)


@pytest.mark.pandapower
class TestMicrogridIslandEvaluator:
    """Validation of microgrid islanding evaluator."""

    def test_find_downstream_buses(self) -> None:
        net = _make_simple_net()
        specs = {
            2: {"e_max_mwh": 1.0, "p_max_mw": 0.5, "eta": 0.95, "soc_min": 0.2},
        }
        evaluator = MicrogridIslandEvaluator(net, specs)
        # Line 1 goes from bus 1 to bus 2; opening it isolates bus 2
        downstream = evaluator._find_downstream_buses(1)
        assert 2 in downstream
        assert 0 not in downstream

    def test_has_active_der(self) -> None:
        net = _make_simple_net()
        specs = {
            2: {"e_max_mwh": 1.0, "p_max_mw": 0.5, "eta": 0.95, "soc_min": 0.2},
        }
        evaluator = MicrogridIslandEvaluator(net, specs)
        has_der, energy, power = evaluator._has_active_der([2])
        assert has_der is True
        assert np.isclose(energy, 1.0 * 0.8, rtol=1e-6)
        assert power == 0.5

    def test_has_active_der_no_spec(self) -> None:
        net = _make_simple_net()
        evaluator = MicrogridIslandEvaluator(net, {})
        has_der, energy, power = evaluator._has_active_der([2])
        assert has_der is False
        assert energy == 0.0

    def test_calculate_islanded_duration(self) -> None:
        net = _make_simple_net()
        evaluator = MicrogridIslandEvaluator(net, {})
        result = evaluator._calculate_islanded_duration(4.0, 0.5, 1.0)
        # battery hours = 1.0 / 0.5 = 2.0; islanded = 4.0 - 2.0 = 2.0
        assert result == 2.0

    def test_calculate_islanded_duration_zero_load(self) -> None:
        net = _make_simple_net()
        evaluator = MicrogridIslandEvaluator(net, {})
        assert evaluator._calculate_islanded_duration(4.0, 0.0, 1.0) == 0.0

    def test_downstream_load(self) -> None:
        net = _make_simple_net()
        evaluator = MicrogridIslandEvaluator(net, {})
        load = evaluator._downstream_load([2, 3])
        assert load == 0.8  # 0.5 + 0.3

    def test_evaluate_islanding(self) -> None:
        net = _make_simple_net()
        specs = {
            2: {"e_max_mwh": 1.0, "p_max_mw": 0.5, "eta": 0.95, "soc_min": 0.2},
        }
        evaluator = MicrogridIslandEvaluator(net, specs)
        result = evaluator.evaluate_islanding(1, outage_h=4.0)
        assert result["has_der"] is True
        assert result["downstream_load_mw"] == 0.5
        assert result["islanded_duration_h"] == 2.0
        assert result["reduction_h"] == 2.0

    def test_evaluate_islanding_no_der(self) -> None:
        net = _make_simple_net()
        evaluator = MicrogridIslandEvaluator(net, {})
        result = evaluator.evaluate_islanding(1, outage_h=4.0)
        assert result["has_der"] is False
        assert result["islanded_duration_h"] == 4.0
        assert result["reduction_h"] == 0.0

    def test_evaluate_all_lines(self) -> None:
        net = _make_simple_net()
        specs = {
            2: {"e_max_mwh": 1.0, "p_max_mw": 0.5, "eta": 0.95, "soc_min": 0.2},
        }
        evaluator = MicrogridIslandEvaluator(net, specs)
        df = evaluator.evaluate_all_lines(outage_h=4.0)
        assert isinstance(df, pd.DataFrame)
        assert "has_der" in df.columns
        assert "reduction_h" in df.columns
        assert len(df) == len(net.line)

    def test_comparative_summary(self) -> None:
        net = _make_simple_net()
        calc = IEEEMetricCalculator(net)
        baseline = _make_outage_events()
        # Islanded events: reduce durations at bus 2 by 1 hour each
        islanded = baseline.copy()
        islanded.loc[islanded["bus"] == 2, "end_h"] -= 1.0

        summary = MicrogridIslandEvaluator.comparative_summary(
            baseline, islanded, calc
        )
        assert len(summary) == 2
        assert "scenario" in summary.columns
        assert summary.iloc[0]["scenario"] == "baseline"
        assert summary.iloc[1]["scenario"] == "islanded"
        # SAIDI should be lower for islanded
        assert summary.iloc[1]["saidi"] < summary.iloc[0]["saidi"]

    def test_repr(self) -> None:
        net = _make_simple_net()
        evaluator = MicrogridIslandEvaluator(net, {})
        assert "MicrogridIslandEvaluator" in repr(evaluator)
