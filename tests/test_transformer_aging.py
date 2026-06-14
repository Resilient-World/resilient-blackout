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

"""Unit tests for ``resilient_blackout.grid.transformer_aging``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resilient_blackout.grid.transformer_aging import TransformerThermalModel


class TestTransformerThermalModelInit:
    """Validation of constructor and parameter handling."""

    def test_default_construction(self) -> None:
        model = TransformerThermalModel()
        assert model.rated_power_mva == 100.0
        assert model.delta_theta_to_rated == 55.0
        assert model.delta_theta_w_rated == 15.0
        assert model.r_loss_ratio == 6.0
        assert model.n_oil == 0.8
        assert model.m_winding == 0.8
        assert model.weibull_shape == 5.0
        assert model.weibull_scale_years == 50.0
        assert model.cumulative_aging_pu == 0.0

    def test_custom_parameters(self) -> None:
        model = TransformerThermalModel(
            rated_power_mva=50.0,
            delta_theta_to_rated=45.0,
            delta_theta_w_rated=20.0,
            r_loss_ratio=5.0,
            n_oil=1.0,
            m_winding=1.0,
            weibull_shape=3.0,
            weibull_scale_years=40.0,
        )
        assert model.rated_power_mva == 50.0
        assert model.weibull_shape == 3.0

    def test_invalid_power_raises(self) -> None:
        with pytest.raises(ValueError, match="rated_power_mva"):
            TransformerThermalModel(rated_power_mva=-1.0)

    def test_invalid_r_loss_raises(self) -> None:
        with pytest.raises(ValueError, match="r_loss_ratio"):
            TransformerThermalModel(r_loss_ratio=0.0)

    def test_invalid_n_oil_raises(self) -> None:
        with pytest.raises(ValueError, match="n_oil"):
            TransformerThermalModel(n_oil=-0.1)

    def test_repr(self) -> None:
        model = TransformerThermalModel()
        r = repr(model)
        assert "TransformerThermalModel" in r
        assert "100MVA" in r


class TestTopOilRise:
    """Validation of top-oil temperature rise calculations."""

    def test_no_load(self) -> None:
        model = TransformerThermalModel(delta_theta_to_rated=55.0, r_loss_ratio=6.0, n_oil=0.8)
        rise = model.calculate_top_oil_rise(np.array([0.0]))
        # At no load: ratio = 1/(1+R) = 1/7 ≈ 0.143
        expected = 55.0 * (1.0 / 7.0) ** 0.8
        assert np.isclose(rise[0], expected, rtol=1e-6)

    def test_rated_load(self) -> None:
        model = TransformerThermalModel(delta_theta_to_rated=55.0, r_loss_ratio=6.0, n_oil=0.8)
        rise = model.calculate_top_oil_rise(np.array([1.0]))
        # At rated load: ratio = (1+R)/(1+R) = 1.0
        assert np.isclose(rise[0], 55.0, rtol=1e-6)

    def test_overload(self) -> None:
        model = TransformerThermalModel(delta_theta_to_rated=55.0, r_loss_ratio=6.0, n_oil=0.8)
        rise = model.calculate_top_oil_rise(np.array([1.5]))
        assert rise[0] > 55.0

    def test_vectorized(self) -> None:
        model = TransformerThermalModel()
        x = np.array([0.0, 0.5, 1.0, 1.5])
        rise = model.calculate_top_oil_rise(x)
        assert rise.shape == (4,)
        assert np.all(np.diff(rise) > 0)  # monotonically increasing


class TestWindingRise:
    """Validation of winding hottest-spot rise calculations."""

    def test_no_load(self) -> None:
        model = TransformerThermalModel(delta_theta_w_rated=15.0, m_winding=0.8)
        rise = model.calculate_winding_rise(np.array([0.0]))
        assert np.isclose(rise[0], 0.0, rtol=1e-6)

    def test_rated_load(self) -> None:
        model = TransformerThermalModel(delta_theta_w_rated=15.0, m_winding=0.8)
        rise = model.calculate_winding_rise(np.array([1.0]))
        assert np.isclose(rise[0], 15.0, rtol=1e-6)

    def test_overload(self) -> None:
        model = TransformerThermalModel(delta_theta_w_rated=15.0, m_winding=0.8)
        rise = model.calculate_winding_rise(np.array([1.5]))
        assert rise[0] > 15.0


class TestHottestSpot:
    """Validation of winding hottest-spot temperature."""

    def test_rated_conditions(self) -> None:
        model = TransformerThermalModel(
            delta_theta_to_rated=55.0, delta_theta_w_rated=15.0
        )
        theta_h = model.calculate_hottest_spot(
            np.array([1.0]), np.array([30.0])
        )
        # θ_H = 30 + 55 + 15 = 100°C
        assert np.isclose(theta_h[0], 100.0, rtol=1e-6)

    def test_high_ambient(self) -> None:
        model = TransformerThermalModel(
            delta_theta_to_rated=55.0, delta_theta_w_rated=15.0
        )
        theta_h = model.calculate_hottest_spot(
            np.array([1.0]), np.array([40.0])
        )
        assert np.isclose(theta_h[0], 110.0, rtol=1e-6)

    def test_vectorized(self) -> None:
        model = TransformerThermalModel()
        x = np.array([0.5, 1.0])
        theta_a = np.array([20.0, 30.0])
        theta_h = model.calculate_hottest_spot(x, theta_a)
        assert theta_h.shape == (2,)


class TestAgingFactor:
    """Validation of Arrhenius aging acceleration factor."""

    def test_reference_temperature(self) -> None:
        model = TransformerThermalModel()
        # At 110°C (reference), FAA should be 1.0
        faa = model.hourly_aging_factor(np.array([110.0]))
        assert np.isclose(faa[0], 1.0, rtol=1e-4)

    def test_below_reference(self) -> None:
        model = TransformerThermalModel()
        faa = model.hourly_aging_factor(np.array([80.0]))
        assert faa[0] < 1.0

    def test_above_reference(self) -> None:
        model = TransformerThermalModel()
        faa = model.hourly_aging_factor(np.array([140.0]))
        assert faa[0] > 1.0

    def test_vectorized(self) -> None:
        model = TransformerThermalModel()
        theta = np.array([80.0, 110.0, 140.0])
        faa = model.hourly_aging_factor(theta)
        assert faa.shape == (3,)
        assert faa[0] < faa[1] < faa[2]


class TestAnnualLossOfLife:
    """Validation of cumulative aging and Weibull updates."""

    def test_normal_year(self) -> None:
        model = TransformerThermalModel(installed_year=2020.0)
        n = 8760
        load = pd.Series(np.full(n, 70.0), dtype=np.float64)  # 70% loading
        temp = pd.Series(np.full(n, 25.0), dtype=np.float64)  # mild ambient

        result = model.calculate_annual_loss_of_life(load, temp, current_year=2025.0)

        assert "cumulative_aging_pu" in result
        assert "failure_rate_per_year" in result
        assert "mean_remaining_life_years" in result
        assert result["cumulative_aging_pu"] > 0.0
        assert result["weibull_scale_years"] < 50.0  # scale decreases with aging

    def test_severe_year(self) -> None:
        model = TransformerThermalModel(installed_year=2020.0)
        n = 8760
        load = pd.Series(np.full(n, 120.0), dtype=np.float64)  # 120% overload
        temp = pd.Series(np.full(n, 40.0), dtype=np.float64)  # hot ambient

        result = model.calculate_annual_loss_of_life(load, temp, current_year=2025.0)
        assert result["cumulative_aging_pu"] > 0.0
        # Severe year should cause more aging than normal
        assert result["failure_rate_per_year"] > 0.0

    def test_mismatched_lengths_raises(self) -> None:
        model = TransformerThermalModel()
        load = pd.Series([1.0, 2.0])
        temp = pd.Series([20.0])
        with pytest.raises(ValueError, match="same length"):
            model.calculate_annual_loss_of_life(load, temp)


class TestWeibullFailureRate:
    """Validation of Weibull failure rate calculations."""

    def test_zero_age(self) -> None:
        model = TransformerThermalModel(weibull_shape=5.0, weibull_scale_years=50.0)
        rate = model.weibull_failure_rate(0.0)
        assert rate == 0.0

    def test_mid_life(self) -> None:
        model = TransformerThermalModel(weibull_shape=5.0, weibull_scale_years=50.0)
        rate = model.weibull_failure_rate(25.0)
        assert rate > 0.0

    def test_aged_transformer(self) -> None:
        model = TransformerThermalModel(weibull_shape=5.0, weibull_scale_years=50.0)
        rate_young = model.weibull_failure_rate(10.0, cumulative_aging=0.0)
        rate_old = model.weibull_failure_rate(10.0, cumulative_aging=2.0)
        # More aging → higher failure rate
        assert rate_old > rate_young

    def test_cumulative_aging_reduces_scale(self) -> None:
        model = TransformerThermalModel(weibull_shape=5.0, weibull_scale_years=50.0)
        model.cumulative_aging_pu = 1.0  # 100% life consumed
        rate = model.weibull_failure_rate(10.0)
        # Scale halved → higher rate
        assert rate > 0.0
