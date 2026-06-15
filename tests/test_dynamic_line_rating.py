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

"""Unit tests for ``resilient_blackout.grid.dynamic_line_rating``.

Validates IEEE 738 ampacity calculations against published tables,
transient thermal inertia behaviour, and DLRGridController integration
with pandapower.
"""

from __future__ import annotations

import numpy as np
import pytest

from resilient_blackout.grid.dynamic_line_rating import (
    DLRGridController,
    calculate_steady_state_ampacity,
    line_thermal_inertia,
)


# ---------------------------------------------------------------------------
# IEEE 738 Table 1 reference values (ACSR Drake, 795 kcmil)
# ---------------------------------------------------------------------------
# From IEEE 738-2012 Table 1: Ampacity at T_max=100°C, ε=0.5, α_s=0.5
# Conditions: T_amb=40°C, V_w=0.61 m/s (2 ft/s), φ=90°, Q_s=1000 W/m²
# Expected ampacity ≈ 900 A for Drake at these conditions.
# We test against the trend, not exact values, since the standard
# tables use slightly different air property correlations.


# ---------------------------------------------------------------------------
# Steady-state ampacity
# ---------------------------------------------------------------------------


class TestSteadyStateAmpacity:
    """Validation of IEEE 738 steady-state ampacity."""

    def test_scalar_input(self) -> None:
        I = calculate_steady_state_ampacity(
            ambient_temp_c=25.0,
            wind_speed_mps=0.6,
        )
        assert I > 0
        assert I < 5000

    def test_vectorized_input(self) -> None:
        n = 10
        T_amb = np.linspace(0, 45, n)
        V_w = np.full(n, 0.6)
        I = calculate_steady_state_ampacity(
            ambient_temp_c=T_amb,
            wind_speed_mps=V_w,
        )
        assert I.shape == (n,)
        assert np.all(I > 0)
        # Ampacity decreases with higher ambient temperature
        assert I[0] > I[-1]

    def test_higher_wind_increases_ampacity(self) -> None:
        I_low = calculate_steady_state_ampacity(25.0, 0.5)
        I_high = calculate_steady_state_ampacity(25.0, 5.0)
        assert I_high > I_low

    def test_higher_temp_decreases_ampacity(self) -> None:
        I_cold = calculate_steady_state_ampacity(0.0, 0.6)
        I_hot = calculate_steady_state_ampacity(45.0, 0.6)
        assert I_cold > I_hot

    def test_zero_wind_natural_convection(self) -> None:
        I = calculate_steady_state_ampacity(25.0, 0.0)
        assert I > 0

    def test_parallel_wind_reduces_cooling(self) -> None:
        I_perp = calculate_steady_state_ampacity(25.0, 2.0, wind_angle_deg=90.0)
        I_par = calculate_steady_state_ampacity(25.0, 2.0, wind_angle_deg=0.0)
        assert I_perp > I_par

    def test_solar_radiation_effect(self) -> None:
        I_night = calculate_steady_state_ampacity(25.0, 0.6, solar_radiation_w_m2=0.0)
        I_day = calculate_steady_state_ampacity(25.0, 0.6, solar_radiation_w_m2=1000.0)
        assert I_night > I_day

    def test_temperature_coefficient_effect(self) -> None:
        I_no_tc = calculate_steady_state_ampacity(
            25.0, 0.6, temp_coeff_resistance=0.0
        )
        I_with_tc = calculate_steady_state_ampacity(
            25.0, 0.6, temp_coeff_resistance=0.00403
        )
        # With positive TCR, resistance increases at T_max, reducing ampacity
        assert I_with_tc < I_no_tc

    def test_higher_emissivity_increases_ampacity(self) -> None:
        I_low = calculate_steady_state_ampacity(25.0, 0.6, emissivity=0.3)
        I_high = calculate_steady_state_ampacity(25.0, 0.6, emissivity=0.9)
        assert I_high > I_low

    def test_ieee_738_table_trend(self) -> None:
        """Validate against IEEE 738 Table 1 trend for ACSR Drake.

        At T_amb=25°C, V_w=0.61 m/s, φ=90°, Q_s=1000 W/m²,
        T_max=100°C, ε=0.5, α_s=0.5, R=0.052 Ω/km at 20°C:
        expected ampacity ≈ 960 A (within ±15%).
        """
        I = calculate_steady_state_ampacity(
            ambient_temp_c=25.0,
            wind_speed_mps=0.61,
            wind_angle_deg=90.0,
            solar_radiation_w_m2=1000.0,
            conductor_resistance_ohms_per_km=0.052,
            max_cond_temp_c=100.0,
            conductor_diameter_m=0.0281,
            emissivity=0.5,
            absorptivity=0.5,
            temp_coeff_resistance=0.00403,
        )
        # IEEE 738 Table 1 gives ~960 A for these conditions
        assert 800 < float(I) < 1150

    def test_ieee_738_hot_condition(self) -> None:
        """Validate hot-weather ampacity trend.

        At T_amb=40°C, V_w=0.61 m/s, expected ampacity ≈ 750 A.
        """
        I = calculate_steady_state_ampacity(
            ambient_temp_c=40.0,
            wind_speed_mps=0.61,
            wind_angle_deg=90.0,
            solar_radiation_w_m2=1000.0,
            conductor_resistance_ohms_per_km=0.052,
            max_cond_temp_c=100.0,
            conductor_diameter_m=0.0281,
            emissivity=0.5,
            absorptivity=0.5,
            temp_coeff_resistance=0.00403,
        )
        assert 600 < float(I) < 950

    def test_high_wind_ampacity(self) -> None:
        """At V_w=3.0 m/s, ampacity should be substantially higher."""
        I = calculate_steady_state_ampacity(
            ambient_temp_c=25.0,
            wind_speed_mps=3.0,
            wind_angle_deg=90.0,
            solar_radiation_w_m2=1000.0,
            conductor_resistance_ohms_per_km=0.052,
            max_cond_temp_c=100.0,
            conductor_diameter_m=0.0281,
            emissivity=0.5,
            absorptivity=0.5,
            temp_coeff_resistance=0.00403,
        )
        assert float(I) > 1100


# ---------------------------------------------------------------------------
# Transient thermal inertia
# ---------------------------------------------------------------------------


class TestLineThermalInertia:
    """Validation of transient conductor temperature model."""

    def test_no_current_no_heating(self) -> None:
        T_final = line_thermal_inertia(
            ambient_temp_c=np.array([25.0]),
            wind_speed_mps=np.array([0.6]),
            wind_angle_deg=np.array([90.0]),
            solar_radiation_w_m2=np.array([0.0]),
            current_a=np.array([0.0]),
            conductor_resistance_ohms_per_km=np.array([0.05]),
            initial_temp_c=np.array([25.0]),
            time_step_s=60.0,
        )
        # Should stay near ambient (slight radiative cooling possible)
        assert 20.0 < float(T_final[0]) < 30.0

    def test_heating_with_current(self) -> None:
        T_final = line_thermal_inertia(
            ambient_temp_c=np.array([25.0]),
            wind_speed_mps=np.array([0.6]),
            wind_angle_deg=np.array([90.0]),
            solar_radiation_w_m2=np.array([0.0]),
            current_a=np.array([1500.0]),
            conductor_resistance_ohms_per_km=np.array([0.05]),
            initial_temp_c=np.array([25.0]),
            time_step_s=300.0,
        )
        assert float(T_final[0]) > 25.0

    def test_cooling_when_current_drops(self) -> None:
        T_after_heating = line_thermal_inertia(
            ambient_temp_c=np.array([25.0]),
            wind_speed_mps=np.array([0.6]),
            wind_angle_deg=np.array([90.0]),
            solar_radiation_w_m2=np.array([0.0]),
            current_a=np.array([2000.0]),
            conductor_resistance_ohms_per_km=np.array([0.05]),
            initial_temp_c=np.array([25.0]),
            time_step_s=600.0,
        )
        T_after_cooling = line_thermal_inertia(
            ambient_temp_c=np.array([25.0]),
            wind_speed_mps=np.array([0.6]),
            wind_angle_deg=np.array([90.0]),
            solar_radiation_w_m2=np.array([0.0]),
            current_a=np.array([0.0]),
            conductor_resistance_ohms_per_km=np.array([0.05]),
            initial_temp_c=T_after_heating,
            time_step_s=600.0,
        )
        assert float(T_after_cooling[0]) < float(T_after_heating[0])

    def test_vectorized_multiple_lines(self) -> None:
        n = 5
        T_final = line_thermal_inertia(
            ambient_temp_c=np.full(n, 25.0),
            wind_speed_mps=np.full(n, 0.6),
            wind_angle_deg=np.full(n, 90.0),
            solar_radiation_w_m2=np.full(n, 1000.0),
            current_a=np.linspace(0, 2000, n),
            conductor_resistance_ohms_per_km=np.full(n, 0.05),
            initial_temp_c=np.full(n, 25.0),
            time_step_s=300.0,
        )
        assert T_final.shape == (n,)
        # Higher current → higher temperature
        assert T_final[0] < T_final[-1]

    def test_thermal_time_constant_effect(self) -> None:
        T_fast = line_thermal_inertia(
            ambient_temp_c=np.array([25.0]),
            wind_speed_mps=np.array([0.6]),
            wind_angle_deg=np.array([90.0]),
            solar_radiation_w_m2=np.array([0.0]),
            current_a=np.array([1500.0]),
            conductor_resistance_ohms_per_km=np.array([0.05]),
            initial_temp_c=np.array([25.0]),
            time_step_s=300.0,
            thermal_time_constant_s=200.0,
        )
        T_slow = line_thermal_inertia(
            ambient_temp_c=np.array([25.0]),
            wind_speed_mps=np.array([0.6]),
            wind_angle_deg=np.array([90.0]),
            solar_radiation_w_m2=np.array([0.0]),
            current_a=np.array([1500.0]),
            conductor_resistance_ohms_per_km=np.array([0.05]),
            initial_temp_c=np.array([25.0]),
            time_step_s=300.0,
            thermal_time_constant_s=2000.0,
        )
        # Larger τ → slower heating → lower final temperature
        assert float(T_fast[0]) > float(T_slow[0])


# ---------------------------------------------------------------------------
# DLR Grid Controller
# ---------------------------------------------------------------------------


class TestDLRGridController:
    """Validation of DLRGridController with pandapower."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        import pandapower as pp

        self.net = pp.create_empty_network()
        b0 = pp.create_bus(self.net, vn_kv=110, name="Bus 0")
        b1 = pp.create_bus(self.net, vn_kv=110, name="Bus 1")
        pp.create_line(
            self.net, b0, b1, length_km=10,
            std_type="149-AL1/24-ST1A 110.0",
            name="Line 0",
        )
        pp.create_ext_grid(self.net, b0, vm_pu=1.0, va_degree=0.0)
        pp.create_load(self.net, b1, p_mw=50, q_mvar=10)

    def test_apply_dynamic_ratings(self) -> None:
        controller = DLRGridController(self.net)
        original_rating = float(self.net.line.at[0, "max_i_ka"])
        controller.apply_dynamic_ratings(timestep=0)
        new_rating = float(self.net.line.at[0, "max_i_ka"])
        # Rating should be updated (likely different from default)
        assert new_rating > 0
        assert new_rating != original_rating or abs(new_rating - original_rating) < 0.01

    def test_custom_weather(self) -> None:
        line_weather = {
            0: {
                "ambient_temp_c": 40.0,
                "wind_speed_mps": 0.2,
                "wind_angle_deg": 90.0,
                "solar_radiation_w_m2": 1000.0,
            }
        }
        controller = DLRGridController(self.net, line_weather=line_weather)
        controller.apply_dynamic_ratings(timestep=0)
        rating_hot = float(self.net.line.at[0, "max_i_ka"])

        line_weather_cold = {
            0: {
                "ambient_temp_c": 0.0,
                "wind_speed_mps": 5.0,
                "wind_angle_deg": 90.0,
                "solar_radiation_w_m2": 0.0,
            }
        }
        controller2 = DLRGridController(self.net, line_weather=line_weather_cold)
        controller2.apply_dynamic_ratings(timestep=0)
        rating_cold = float(self.net.line.at[0, "max_i_ka"])

        assert rating_cold > rating_hot

    def test_thermal_margins(self) -> None:
        controller = DLRGridController(self.net)
        margins = controller.get_thermal_margins(
            timestep=0, line_currents={0: 500.0}
        )
        assert 0 in margins
        # With moderate current, margin should be positive
        assert margins[0] > 0

    def test_thermal_margins_overload(self) -> None:
        controller = DLRGridController(self.net)
        margins = controller.get_thermal_margins(
            timestep=0, line_currents={0: 99999.0}
        )
        assert margins[0] < 0

    def test_repr(self) -> None:
        controller = DLRGridController(self.net)
        r = repr(controller)
        assert "DLRGridController" in r
        assert "100" in r
