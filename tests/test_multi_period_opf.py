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

"""Unit tests for ``resilient_blackout.grid.multi_period_opf``."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandapower")

from resilient_blackout.grid.multi_period_opf import MultiPeriodOPFScheduler


def _make_test_net():
    import pandapower as pp

    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=0.4, name="Bus_0")
    b1 = pp.create_bus(net, vn_kv=0.4, name="Bus_1")
    b2 = pp.create_bus(net, vn_kv=0.4, name="Bus_2")
    pp.create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, std_type="NAYY 4x50 SE")
    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=1.0, std_type="NAYY 4x50 SE")
    pp.create_load(net, bus=b0, p_mw=5.0, q_mvar=1.0, name="Load_0")
    pp.create_load(net, bus=b1, p_mw=3.0, q_mvar=0.5, name="Load_1")
    pp.create_gen(net, bus=b2, p_mw=2.0, vm_pu=1.0, max_p_mw=10.0, min_p_mw=0.0, name="Gen_0")
    pp.create_ext_grid(net, bus=b2, vm_pu=1.0, name="Slack")
    return net


class TestMultiPeriodOPFSchedulerInit:
    def test_default_construction(self) -> None:
        sched = MultiPeriodOPFScheduler()
        assert sched.horizon_steps == 24
        assert sched.dt_hours == 1.0
        assert sched.voll * 1000 == pytest.approx(10_000)
        assert sched.max_ramp_pu == 0.3

    def test_custom_parameters(self) -> None:
        sched = MultiPeriodOPFScheduler(horizon_steps=4, dt_hours=0.5, voll_usd_per_mwh=5000.0)
        assert sched.horizon_steps == 4
        assert sched.dt_hours == 0.5
        assert sched.voll * 1000 == pytest.approx(5000)

    def test_invalid_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="horizon_steps"):
            MultiPeriodOPFScheduler(horizon_steps=0)

    def test_invalid_dt_raises(self) -> None:
        with pytest.raises(ValueError, match="dt_hours"):
            MultiPeriodOPFScheduler(dt_hours=-1.0)


class TestBuildSchedule:
    def test_basic_schedule(self) -> None:
        net = _make_test_net()
        T = 4
        load_profile = np.full((T, 2), [[5.0, 3.0]], dtype=np.float64)
        sched = MultiPeriodOPFScheduler(horizon_steps=T, dt_hours=1.0)
        result = sched.build_schedule(net, load_profile)

        assert result["status"] == 0
        assert result["gen_schedule"].shape == (T, 1)
        assert result["shed_per_bus"].shape == (T, 2)
        assert result["objective"] >= 0.0

    def test_schedule_with_battery(self) -> None:
        net = _make_test_net()
        T = 4
        load_profile = np.full((T, 2), [[5.0, 3.0]], dtype=np.float64)
        storage = [
            {
                "bus": 0,
                "p_max_mw": 2.0,
                "e_max_mwh": 4.0,
                "e_min_mwh": 0.5,
                "e_init_mwh": 2.0,
                "eta_in": 0.95,
                "eta_out": 0.95,
            }
        ]
        sched = MultiPeriodOPFScheduler(horizon_steps=T, dt_hours=1.0)
        result = sched.build_schedule(net, load_profile, storage_specs=storage)

        assert result["status"] == 0
        assert len(result["battery_schedule"]) == 1
        batt = result["battery_schedule"][0]
        assert "e_mwh" in batt
        assert len(batt["e_mwh"]) == T
        # SOC should respect bounds
        assert all(e >= 0.5 for e in batt["e_mwh"])
        assert all(e <= 4.0 for e in batt["e_mwh"])

    def test_ramp_constraints_limit_change(self) -> None:
        net = _make_test_net()
        net.gen.at[0, "max_p_mw"] = 10.0
        T = 3
        # Step load to force generator to ramp up
        load_profile = np.array([[3.0, 2.0], [6.0, 4.0], [9.0, 6.0]], dtype=np.float64)
        sched = MultiPeriodOPFScheduler(horizon_steps=T, dt_hours=1.0, max_ramp_pu_per_step=0.2)
        result = sched.build_schedule(net, load_profile)

        assert result["status"] == 0
        gen = result["gen_schedule"].ravel()
        for t in range(1, T):
            ramp_limit = 10.0 * 0.2
            assert abs(gen[t] - gen[t - 1]) <= ramp_limit + 1e-3

    def test_shed_penalty_dominates(self) -> None:
        # Very high VOLL should make LP prefer generation over shed
        net = _make_test_net()
        T = 2
        load_profile = np.full((T, 2), [[5.0, 3.0]], dtype=np.float64)
        sched = MultiPeriodOPFScheduler(horizon_steps=T, dt_hours=1.0, voll_usd_per_mwh=1e6)
        result = sched.build_schedule(net, load_profile)

        assert result["status"] == 0
        total_shed = float(np.sum(result["shed_per_bus"]))
        # With enough generation capacity, shed should be near zero
        total_gen = float(np.sum(result["gen_schedule"]))
        total_load = float(np.sum(load_profile))
        if total_gen >= total_load - 1e-3:
            assert total_shed <= 1e-2

    def test_load_profile_shape_validation(self) -> None:
        net = _make_test_net()
        sched = MultiPeriodOPFScheduler(horizon_steps=3, dt_hours=1.0)
        bad_profile = np.array([[1.0], [2.0]])  # wrong rows
        with pytest.raises(ValueError, match="rows"):
            sched.build_schedule(net, bad_profile)


class TestRollingHorizon:
    def test_rolling_schedule_length(self) -> None:
        net = _make_test_net()
        n_total = 10
        T = 4
        load_profiles = np.full((n_total, 2), [[5.0, 3.0]], dtype=np.float64)
        sched = MultiPeriodOPFScheduler(horizon_steps=T, dt_hours=1.0)
        df = sched.rolling_horizon(net, load_profiles, window_steps=T, step_size=1)

        # Should produce n_total - window_steps + 1 records with step_size=1
        assert len(df) == n_total - T + 1
        assert "hour" in df.columns
        assert "total_gen_mw" in df.columns
        assert "total_battery_soc_mwh" in df.columns
        assert "total_shed_mw" in df.columns

    def test_step_size_two(self) -> None:
        net = _make_test_net()
        n_total = 8
        T = 4
        load_profiles = np.full((n_total, 2), [[5.0, 3.0]], dtype=np.float64)
        sched = MultiPeriodOPFScheduler(horizon_steps=T, dt_hours=1.0)
        df = sched.rolling_horizon(net, load_profiles, window_steps=T, step_size=2)

        expected_records = (n_total - T) // 2 + 1
        assert len(df) == expected_records


class TestApplyToNet:
    def test_apply_mutates_gen(self) -> None:
        net = _make_test_net()
        T = 2
        load_profile = np.full((T, 2), [[5.0, 3.0]], dtype=np.float64)
        sched = MultiPeriodOPFScheduler(horizon_steps=T, dt_hours=1.0)
        result = sched.build_schedule(net, load_profile)
        original_p = float(net.gen.at[0, "p_mw"])
        sched.apply_to_net(net, result, timestep=0)
        assert float(net.gen.at[0, "p_mw"]) != original_p


class TestRepr:
    def test_repr(self) -> None:
        sched = MultiPeriodOPFScheduler(horizon_steps=4, dt_hours=0.5, voll_usd_per_mwh=5000)
        r = repr(sched)
        assert "MultiPeriodOPFScheduler" in r
        assert "horizon=4" in r
        assert "dt=0.5h" in r
        assert "voll=5000" in r
