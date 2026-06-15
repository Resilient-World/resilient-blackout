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

"""Unit tests for ``resilient_blackout.climate.wildfire``."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from resilient_blackout.climate.wildfire import WildfireRiskEngine


def _make_line(x1: float, y1: float, x2: float, y2: float) -> LineString:
    return LineString([(x1, y1), (x2, y2)])


def _make_fire_polygon(
    cx: float, cy: float, radius: float = 1000.0
) -> Polygon:
    return Point(cx, cy).buffer(radius)


class TestInit:
    """Validation of constructor and parameter handling."""

    def test_default_construction(self) -> None:
        engine = WildfireRiskEngine()
        assert engine.lambda_0 == 1e-5
        assert engine.gamma == 0.01
        assert engine.delta_smoke == 0.5
        assert engine.pm25_threshold == 150.0
        assert engine.I_max0 == 1000.0

    def test_negative_base_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="base_failure_rate"):
            WildfireRiskEngine(base_failure_rate=-0.1)

    def test_negative_gamma_raises(self) -> None:
        with pytest.raises(ValueError, match="vulnerability_gamma"):
            WildfireRiskEngine(vulnerability_gamma=-0.01)

    def test_smoke_derating_out_of_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match="smoke_derating_coefficient"):
            WildfireRiskEngine(smoke_derating_coefficient=1.5)

    def test_zero_pm25_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="pm25_threshold_ug_m3"):
            WildfireRiskEngine(pm25_threshold_ug_m3=0.0)

    def test_negative_ampacity_raises(self) -> None:
        with pytest.raises(ValueError, match="default_ampacity_a"):
            WildfireRiskEngine(default_ampacity_a=-100.0)


class TestMinimumDistance:
    """Validation of Shapely distance computation."""

    def test_line_outside_fire(self) -> None:
        line = _make_line(0, 0, 100, 0)
        fire = _make_fire_polygon(500, 0, 100)
        d = WildfireRiskEngine._minimum_distance(line, fire)
        assert d > 0

    def test_line_intersects_fire(self) -> None:
        line = _make_line(0, 0, 2000, 0)
        fire = _make_fire_polygon(1000, 0, 500)
        d = WildfireRiskEngine._minimum_distance(line, fire)
        assert d == 0.0

    def test_line_inside_fire(self) -> None:
        line = _make_line(1000, 10, 1000, -10)
        fire = _make_fire_polygon(1000, 0, 500)
        d = WildfireRiskEngine._minimum_distance(line, fire)
        assert d == 0.0

    def test_multi_polygon(self) -> None:
        line = _make_line(0, 0, 100, 0)
        p1 = _make_fire_polygon(300, 0, 50)
        p2 = _make_fire_polygon(500, 0, 50)
        fire = MultiPolygon([p1, p2])
        d = WildfireRiskEngine._minimum_distance(line, fire)
        assert d > 0


class TestDynamicFailureRate:
    """Validation of λ_a(t) computation."""

    def test_increases_with_temperature(self) -> None:
        engine = WildfireRiskEngine()
        rate_cool = engine._dynamic_failure_rate(500, 300, 100)
        rate_hot = engine._dynamic_failure_rate(1000, 300, 100)
        assert rate_hot > rate_cool

    def test_decreases_with_distance(self) -> None:
        engine = WildfireRiskEngine()
        rate_near = engine._dynamic_failure_rate(1000, 300, 10)
        rate_far = engine._dynamic_failure_rate(1000, 300, 1000)
        assert rate_near > rate_far

    def test_zero_distance_clamped(self) -> None:
        engine = WildfireRiskEngine()
        rate = engine._dynamic_failure_rate(1000, 300, 0.0)
        assert rate > 0
        assert not np.isinf(rate)

    def test_no_delta_t_gives_base_rate(self) -> None:
        engine = WildfireRiskEngine()
        rate = engine._dynamic_failure_rate(300, 300, 100)
        assert rate == engine.lambda_0


class TestTripProbability:
    """Validation of Poisson trip probability."""

    def test_bounds(self) -> None:
        p = WildfireRiskEngine._trip_probability(1e-4, 3600)
        assert 0.0 <= p <= 1.0

    def test_zero_rate_gives_zero_probability(self) -> None:
        p = WildfireRiskEngine._trip_probability(0.0, 3600)
        assert p == 0.0

    def test_high_rate_gives_high_probability(self) -> None:
        p = WildfireRiskEngine._trip_probability(1.0, 3600)
        assert p > 0.99

    def test_increases_with_dt(self) -> None:
        p_short = WildfireRiskEngine._trip_probability(1e-4, 60)
        p_long = WildfireRiskEngine._trip_probability(1e-4, 86400)
        assert p_long > p_short


class TestDerateAmpacity:
    """Validation of smoke ampacity derating."""

    def test_no_smoke_no_derating(self) -> None:
        engine = WildfireRiskEngine()
        I = engine._derate_ampacity(0.0)
        assert I == engine.I_max0

    def test_smoke_reduces_ampacity(self) -> None:
        engine = WildfireRiskEngine(smoke_derating_coefficient=0.5)
        I = engine._derate_ampacity(300.0)
        assert I < engine.I_max0

    def test_derating_never_negative(self) -> None:
        engine = WildfireRiskEngine(smoke_derating_coefficient=1.0)
        I = engine._derate_ampacity(10000.0)
        assert I >= 0.0

    def test_derating_never_exceeds_max(self) -> None:
        engine = WildfireRiskEngine()
        I = engine._derate_ampacity(-50.0)
        assert I <= engine.I_max0


class TestEvaluateLine:
    """Validation of single-line evaluation."""

    def test_returns_expected_keys(self) -> None:
        engine = WildfireRiskEngine()
        line = _make_line(0, 0, 100, 0)
        fire = _make_fire_polygon(500, 0, 500)
        result = engine.evaluate_line(line, fire)

        assert "distance_m" in result
        assert "failure_rate_per_s" in result
        assert "trip_probability" in result
        assert "derated_ampacity_a" in result
        assert "intersects_fire" in result

    def test_intersecting_line_flags_true(self) -> None:
        engine = WildfireRiskEngine()
        line = _make_line(0, 0, 2000, 0)
        fire = _make_fire_polygon(1000, 0, 500)
        result = engine.evaluate_line(line, fire)
        assert result["intersects_fire"] is True

    def test_non_intersecting_line_flags_false(self) -> None:
        engine = WildfireRiskEngine()
        line = _make_line(0, 0, 100, 0)
        fire = _make_fire_polygon(5000, 0, 500)
        result = engine.evaluate_line(line, fire)
        assert result["intersects_fire"] is False

    def test_metadata_overrides_defaults(self) -> None:
        engine = WildfireRiskEngine(default_flame_temp_k=800)
        line = _make_line(0, 0, 100, 0)
        fire = _make_fire_polygon(500, 0, 500)
        result_default = engine.evaluate_line(line, fire)
        result_meta = engine.evaluate_line(
            line, fire,
            fire_metadata={"flame_temperature_k": 1200, "pm25_ug_m3": 300},
        )
        assert result_meta["failure_rate_per_s"] > result_default["failure_rate_per_s"]
        assert result_meta["derated_ampacity_a"] < result_default["derated_ampacity_a"]


class TestEvaluateNetwork:
    """Validation of network-level batch evaluation."""

    def test_returns_geodataframe(self) -> None:
        engine = WildfireRiskEngine()
        lines = gpd.GeoDataFrame(
            {"line_id": [0, 1]},
            geometry=[_make_line(0, 0, 100, 0), _make_line(0, 100, 100, 100)],
            crs="EPSG:3857",
        )
        fire = _make_fire_polygon(500, 0, 500)
        result = engine.evaluate_network(lines, fire)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2
        assert "distance_m" in result.columns
        assert "trip_probability" in result.columns
        assert "derated_ampacity_a" in result.columns

    def test_empty_gdf_returns_empty(self) -> None:
        engine = WildfireRiskEngine()
        lines = gpd.GeoDataFrame({"line_id": []}, geometry=[], crs="EPSG:3857")
        fire = _make_fire_polygon(500, 0, 500)
        result = engine.evaluate_network(lines, fire)
        assert len(result) == 0

    def test_per_line_ambient_temperature(self) -> None:
        engine = WildfireRiskEngine()
        lines = gpd.GeoDataFrame(
            {"line_id": [0, 1]},
            geometry=[_make_line(0, 0, 100, 0), _make_line(0, 100, 100, 100)],
            crs="EPSG:3857",
        )
        fire = _make_fire_polygon(500, 0, 500)
        result = engine.evaluate_network(
            lines, fire, T_ambient=np.array([300, 310])
        )
        assert result["failure_rate_per_s"].iloc[1] > result["failure_rate_per_s"].iloc[0]


class TestEvaluateTimeseries:
    """Validation of time-series evaluation."""

    def test_returns_dataframe(self) -> None:
        engine = WildfireRiskEngine()
        line = _make_line(0, 0, 100, 0)
        fires = [
            _make_fire_polygon(500, 0, 500),
            _make_fire_polygon(200, 0, 500),
            _make_fire_polygon(50, 0, 500),
        ]
        result = engine.evaluate_timeseries(line, fires)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "timestep" in result.columns
        assert "trip_probability" in result.columns

    def test_approaching_fire_increases_risk(self) -> None:
        engine = WildfireRiskEngine()
        line = _make_line(0, 0, 100, 0)
        fires = [
            _make_fire_polygon(1000, 0, 500),
            _make_fire_polygon(500, 0, 500),
            _make_fire_polygon(100, 0, 500),
        ]
        result = engine.evaluate_timeseries(line, fires)
        probs = result["trip_probability"].values
        assert probs[2] > probs[0]


class TestRepr:
    """Validation of string representation."""

    def test_repr_includes_key_params(self) -> None:
        engine = WildfireRiskEngine(
            base_failure_rate=2e-5,
            vulnerability_gamma=0.02,
            smoke_derating_coefficient=0.3,
            default_ampacity_a=800,
        )
        r = repr(engine)
        assert "2.0e-05" in r
        assert "0.020" in r
        assert "0.30" in r
        assert "800A" in r
