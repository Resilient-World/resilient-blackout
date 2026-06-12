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

"""Unit tests for ``resilient_blackout.climate.ice_accretion``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resilient_blackout.climate.ice_accretion import MakkonenIcer


def _make_icing_weather(n_hours: int = 24, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic icing weather data.

    Parameters
    ----------
    n_hours : int
    seed : int

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2026-01-01 00:00", periods=n_hours, freq="h")

    T = rng.uniform(-10.0, 2.0, size=n_hours)
    V = rng.uniform(2.0, 15.0, size=n_hours)
    LWC = rng.uniform(0.1, 1.5, size=n_hours)

    return pd.DataFrame(
        {
            "timestamp": hours,
            "temperature_c": T,
            "wind_speed_mps": V,
            "liquid_water_content_g_m3": LWC,
        }
    )


class TestMakkonenIcerInit:
    """Validation of constructor and parameter handling."""

    def test_default_construction(self) -> None:
        """Default parameters should produce a valid instance."""
        icer = MakkonenIcer()
        assert icer.conductor_diameter_m == 0.0281
        assert icer.ice_density == 917.0
        assert icer.alpha_2 == 1.0

    def test_glaze_ice_type(self) -> None:
        """Glaze ice type should set density to 917 kg/m³."""
        icer = MakkonenIcer(ice_type="glaze")
        assert icer.ice_density == 917.0

    def test_rime_ice_type(self) -> None:
        """Rime ice type should set density to 500 kg/m³."""
        icer = MakkonenIcer(ice_type="rime")
        assert icer.ice_density == 500.0

    def test_custom_density_override(self) -> None:
        """Explicit density should override ice_type."""
        icer = MakkonenIcer(ice_type="glaze", ice_density_kg_m3=800.0)
        assert icer.ice_density == 800.0

    def test_invalid_ice_type_raises(self) -> None:
        """Unknown ice_type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown ice_type"):
            MakkonenIcer(ice_type="slush")

    def test_negative_diameter_raises(self) -> None:
        """Negative conductor diameter should raise ValueError."""
        with pytest.raises(ValueError, match="conductor_diameter_m"):
            MakkonenIcer(conductor_diameter_m=-0.01)

    def test_zero_span_raises(self) -> None:
        """Zero span length should raise ValueError."""
        with pytest.raises(ValueError, match="span_length_m"):
            MakkonenIcer(span_length_m=0.0)

    def test_invalid_sticking_efficiency_raises(self) -> None:
        """Sticking efficiency outside (0, 1] should raise."""
        with pytest.raises(ValueError, match="sticking_efficiency"):
            MakkonenIcer(sticking_efficiency=1.5)


class TestCollisionEfficiency:
    """Validation of α₁ collision efficiency computation."""

    def test_bounds(self) -> None:
        """α₁ must be in [0, 1]."""
        V = np.array([5.0, 10.0, 15.0, 20.0])
        D = np.array([0.03, 0.04, 0.05, 0.06])
        alpha = MakkonenIcer._collision_efficiency(20e-6, V, D)
        assert np.all(alpha >= 0.0)
        assert np.all(alpha <= 1.0)

    def test_increases_with_wind_speed(self) -> None:
        """Higher wind speed → higher collision efficiency."""
        V_low = np.array([2.0])
        V_high = np.array([20.0])
        D = np.array([0.03])
        alpha_low = MakkonenIcer._collision_efficiency(20e-6, V_low, D)
        alpha_high = MakkonenIcer._collision_efficiency(20e-6, V_high, D)
        assert alpha_high[0] > alpha_low[0]

    def test_decreases_with_diameter(self) -> None:
        """Larger diameter → lower collision efficiency."""
        V = np.array([10.0])
        D_small = np.array([0.01])
        D_large = np.array([0.10])
        alpha_small = MakkonenIcer._collision_efficiency(20e-6, V, D_small)
        alpha_large = MakkonenIcer._collision_efficiency(20e-6, V, D_large)
        assert alpha_small[0] > alpha_large[0]

    def test_zero_wind_speed(self) -> None:
        """Zero wind speed should give near-zero efficiency."""
        V = np.array([0.01])
        D = np.array([0.03])
        alpha = MakkonenIcer._collision_efficiency(20e-6, V, D)
        assert alpha[0] < 0.1


class TestAccretionEfficiency:
    """Validation of α₃ accretion efficiency via heat balance."""

    def test_bounds(self) -> None:
        """α₃ must be in [0, 1]."""
        T = np.array([-5.0, -2.0, -10.0])
        V = np.array([5.0, 10.0, 8.0])
        D = np.array([0.03, 0.04, 0.05])
        LWC = np.array([0.5, 1.0, 0.3])
        alpha_1 = np.array([0.8, 0.7, 0.6])
        alpha_3 = MakkonenIcer._accretion_efficiency(T, V, D, LWC, alpha_1)
        assert np.all(alpha_3 >= 0.0)
        assert np.all(alpha_3 <= 1.0)

    def test_colder_more_freezing(self) -> None:
        """Colder temperatures should allow higher α₃."""
        T_cold = np.array([-15.0])
        T_warm = np.array([-0.5])
        V = np.array([5.0])
        D = np.array([0.03])
        LWC = np.array([0.5])
        alpha_1 = np.array([0.8])
        alpha_cold = MakkonenIcer._accretion_efficiency(T_cold, V, D, LWC, alpha_1)
        alpha_warm = MakkonenIcer._accretion_efficiency(T_warm, V, D, LWC, alpha_1)
        assert alpha_cold[0] >= alpha_warm[0]

    def test_high_lwc_limits_freezing(self) -> None:
        """Very high LWC may reduce α₃ (more water to freeze)."""
        T = np.array([-2.0])
        V = np.array([5.0])
        D = np.array([0.03])
        alpha_1 = np.array([0.8])
        alpha_low = MakkonenIcer._accretion_efficiency(
            T, V, D, np.array([0.1]), alpha_1
        )
        alpha_high = MakkonenIcer._accretion_efficiency(
            T, V, D, np.array([3.0]), alpha_1
        )
        assert alpha_high[0] <= alpha_low[0] + _EPS


class TestRunIntegration:
    """Integration tests for the full time-series simulation."""

    def test_run_returns_dataframe(self) -> None:
        """run() should return a DataFrame with expected columns."""
        icer = MakkonenIcer()
        weather = _make_icing_weather(24)
        result = icer.run(weather)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 24
        assert "ice_mass_kg_per_m" in result.columns
        assert "diameter_m" in result.columns
        assert "alpha_1" in result.columns
        assert "alpha_3" in result.columns
        assert "ice_thickness_m" in result.columns

    def test_ice_mass_monotonic(self) -> None:
        """Ice mass should be non-decreasing under icing conditions."""
        icer = MakkonenIcer()
        weather = _make_icing_weather(48)
        result = icer.run(weather)

        mass = result["ice_mass_kg_per_m"].values
        assert np.all(np.diff(mass) >= -_EPS), "Ice mass decreased unexpectedly"

    def test_diameter_grows_with_ice(self) -> None:
        """Diameter should increase as ice accumulates."""
        icer = MakkonenIcer()
        weather = _make_icing_weather(24)
        result = icer.run(weather)

        assert result["diameter_m"].iloc[-1] >= icer.conductor_diameter_m

    def test_no_accretion_above_freezing(self) -> None:
        """No ice should form when temperature is well above 0 °C."""
        icer = MakkonenIcer()
        rng = np.random.default_rng(42)
        hours = pd.date_range("2026-07-01 00:00", periods=24, freq="h")
        weather = pd.DataFrame(
            {
                "timestamp": hours,
                "temperature_c": rng.uniform(5.0, 15.0, size=24),
                "wind_speed_mps": rng.uniform(2.0, 10.0, size=24),
                "liquid_water_content_g_m3": rng.uniform(0.1, 1.0, size=24),
            }
        )
        result = icer.run(weather)
        assert result["ice_mass_kg_per_m"].max() < 1e-6

    def test_empty_weather_raises(self) -> None:
        """Empty DataFrame should raise ValueError."""
        icer = MakkonenIcer()
        with pytest.raises(ValueError, match="empty"):
            icer.run(pd.DataFrame())

    def test_missing_column_raises(self) -> None:
        """Missing required column should raise ValueError."""
        icer = MakkonenIcer()
        weather = pd.DataFrame({"timestamp": [0], "temperature_c": [-5.0]})
        with pytest.raises(ValueError, match="Missing required column"):
            icer.run(weather)

    def test_rime_vs_glaze_thickness(self) -> None:
        """Rime ice should produce thicker ice for same mass (lower density)."""
        weather = _make_icing_weather(24, seed=123)

        icer_glaze = MakkonenIcer(ice_type="glaze")
        result_glaze = icer_glaze.run(weather)

        icer_rime = MakkonenIcer(ice_type="rime")
        result_rime = icer_rime.run(weather)

        mass_glaze = result_glaze["ice_mass_kg_per_m"].iloc[-1]
        mass_rime = result_rime["ice_mass_kg_per_m"].iloc[-1]

        thickness_glaze = result_glaze["ice_thickness_m"].iloc[-1]
        thickness_rime = result_rime["ice_thickness_m"].iloc[-1]

        if mass_rime > 0 and mass_glaze > 0:
            assert thickness_rime > thickness_glaze, (
                f"Rime thickness {thickness_rime:.4f} should exceed "
                f"glaze thickness {thickness_glaze:.4f}"
            )


class TestFailureThreshold:
    """Validation of mechanical failure probability."""

    def test_returns_dict_with_keys(self) -> None:
        """Should return a dict with required keys."""
        icer = MakkonenIcer(max_tension_n=50_000.0)
        weather = _make_icing_weather(24)
        icer.run(weather)
        result = icer.calculate_ice_failure_threshold()

        assert "failure_probability" in result
        assert "max_tension_n" in result
        assert "tension_limit_n" in result
        assert "safety_factor" in result
        assert "max_ice_thickness_m" in result
        assert "max_ice_mass_kg_per_m" in result

    def test_failure_with_low_tension_limit(self) -> None:
        """Very low tension limit should trigger failure."""
        icer = MakkonenIcer(max_tension_n=100.0)
        weather = _make_icing_weather(24)
        icer.run(weather)
        result = icer.calculate_ice_failure_threshold()
        assert result["failure_probability"] == 1.0

    def test_no_failure_with_high_tension_limit(self) -> None:
        """Very high tension limit should prevent failure."""
        icer = MakkonenIcer(max_tension_n=1_000_000.0)
        weather = _make_icing_weather(24)
        icer.run(weather)
        result = icer.calculate_ice_failure_threshold()
        assert result["failure_probability"] == 0.0

    def test_gust_wind_increases_tension(self) -> None:
        """Gust wind should increase peak tension."""
        icer = MakkonenIcer(max_tension_n=1_000_000.0)
        weather = _make_icing_weather(24)
        icer.run(weather)

        no_gust = icer.calculate_ice_failure_threshold(gust_wind_speed_mps=0.0)
        with_gust = icer.calculate_ice_failure_threshold(gust_wind_speed_mps=30.0)

        assert with_gust["max_tension_n"] > no_gust["max_tension_n"]

    def test_raises_without_run(self) -> None:
        """Should raise RuntimeError if run() not called first."""
        icer = MakkonenIcer()
        with pytest.raises(RuntimeError, match="Call run()"):
            icer.calculate_ice_failure_threshold()


class TestRepr:
    """Validation of string representation."""

    def test_repr_includes_key_params(self) -> None:
        """__repr__ should include diameter, ice type, span, tension."""
        icer = MakkonenIcer(
            conductor_diameter_m=0.03,
            ice_type="rime",
            span_length_m=250.0,
            max_tension_n=60_000.0,
        )
        r = repr(icer)
        assert "30.0mm" in r
        assert "rime" in r
        assert "250m" in r
        assert "60.0kN" in r


_EPS: float = 1e-12
