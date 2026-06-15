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

"""Unit tests for ``resilient_blackout.climate.flooding``."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from resilient_blackout.climate.flooding import SubstationFlooder


def _make_substations_gdf(n: int = 3) -> gpd.GeoDataFrame:
    """Create a simple GeoDataFrame of substation points.

    Parameters
    ----------
    n : int

    Returns
    -------
    gpd.GeoDataFrame
    """
    points = [Point(i * 1000, i * 1000) for i in range(n)]
    return gpd.GeoDataFrame(
        {
            "substation_id": [f"sub_{i}" for i in range(n)],
            "ffe_m": [0.3, 0.5, 0.2],
            "levee_height_m": [0.0, 1.0, 0.0],
            "pump_rate_mps": [0.0, 0.0, 0.001],
            "geometry": points,
        },
        crs="EPSG:3857",
    )


class TestInit:
    """Validation of constructor and parameter handling."""

    def test_default_construction(self) -> None:
        flooder = SubstationFlooder()
        assert flooder.gamma == 2.0
        assert flooder.default_ffe_m == 0.3
        assert flooder.default_levee_height_m == 0.0
        assert flooder.default_pump_rate_mps == 0.0

    def test_custom_parameters(self) -> None:
        flooder = SubstationFlooder(
            gamma=3.0,
            default_ffe_m=0.5,
            default_levee_height_m=1.0,
            default_pump_rate_m_per_s=0.01,
        )
        assert flooder.gamma == 3.0
        assert flooder.default_ffe_m == 0.5
        assert flooder.default_levee_height_m == 1.0
        assert flooder.default_pump_rate_mps == 0.01

    def test_negative_gamma_raises(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            SubstationFlooder(gamma=-1.0)

    def test_negative_ffe_raises(self) -> None:
        with pytest.raises(ValueError, match="default_ffe_m"):
            SubstationFlooder(default_ffe_m=-0.1)

    def test_negative_levee_raises(self) -> None:
        with pytest.raises(ValueError, match="default_levee_height_m"):
            SubstationFlooder(default_levee_height_m=-0.5)

    def test_negative_pump_raises(self) -> None:
        with pytest.raises(ValueError, match="default_pump_rate_m_per_s"):
            SubstationFlooder(default_pump_rate_m_per_s=-0.001)

    def test_zero_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="default_flood_duration_s"):
            SubstationFlooder(default_flood_duration_s=0.0)


class TestEffectiveDepth:
    """Validation of pump-adjusted effective depth."""

    def test_no_pump_no_change(self) -> None:
        d_raw = np.array([1.0, 2.0, 3.0])
        pump = np.zeros(3)
        d_eff = SubstationFlooder._effective_depth(d_raw, pump, 3600)
        np.testing.assert_array_almost_equal(d_eff, d_raw)

    def test_pump_reduces_depth(self) -> None:
        d_raw = np.array([1.0])
        pump = np.array([0.0001])
        d_eff = SubstationFlooder._effective_depth(d_raw, pump, 3600)
        assert d_eff[0] < d_raw[0]

    def test_pump_cannot_go_negative(self) -> None:
        d_raw = np.array([0.1])
        pump = np.array([0.001])
        d_eff = SubstationFlooder._effective_depth(d_raw, pump, 3600)
        assert d_eff[0] == 0.0

    def test_pump_removes_all_water(self) -> None:
        d_raw = np.array([0.36])
        pump = np.array([0.0001])
        d_eff = SubstationFlooder._effective_depth(d_raw, pump, 3600)
        assert d_eff[0] == 0.0


class TestFailureProbability:
    """Validation of log-logistic failure probability."""

    def test_bounds(self) -> None:
        d = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        ffe = np.full(5, 0.3)
        levee = np.zeros(5)
        p = SubstationFlooder._failure_probability(d, ffe, levee, 2.0)
        assert np.all(p >= 0.0)
        assert np.all(p <= 1.0)

    def test_monotonic_increasing(self) -> None:
        d = np.linspace(0, 5, 100)
        ffe = np.full(100, 0.3)
        levee = np.zeros(100)
        p = SubstationFlooder._failure_probability(d, ffe, levee, 2.0)
        assert np.all(np.diff(p) >= -_EPS)

    def test_zero_depth_low_probability(self) -> None:
        d = np.array([0.0])
        ffe = np.array([0.3])
        levee = np.array([0.0])
        p = SubstationFlooder._failure_probability(d, ffe, levee, 2.0)
        assert p[0] < 0.5

    def test_high_depth_high_probability(self) -> None:
        d = np.array([5.0])
        ffe = np.array([0.3])
        levee = np.array([0.0])
        p = SubstationFlooder._failure_probability(d, ffe, levee, 2.0)
        assert p[0] > 0.99

    def test_at_ffe_gives_half(self) -> None:
        d = np.array([0.3])
        ffe = np.array([0.3])
        levee = np.array([0.0])
        p = SubstationFlooder._failure_probability(d, ffe, levee, 2.0)
        assert abs(p[0] - 0.5) < 0.01

    def test_levee_shifts_curve(self) -> None:
        d = np.array([1.3])
        ffe = np.array([0.3])
        levee_no = np.array([0.0])
        levee_yes = np.array([1.0])
        p_no = SubstationFlooder._failure_probability(d, ffe, levee_no, 2.0)
        p_yes = SubstationFlooder._failure_probability(d, ffe, levee_yes, 2.0)
        assert p_yes[0] < p_no[0]

    def test_gamma_controls_steepness(self) -> None:
        d = np.linspace(0, 2, 100)
        ffe = np.full(100, 0.3)
        levee = np.zeros(100)
        p_low = SubstationFlooder._failure_probability(d, ffe, levee, 0.5)
        p_high = SubstationFlooder._failure_probability(d, ffe, levee, 5.0)
        assert np.std(p_high) > np.std(p_low)


class TestEvaluateSubstation:
    """Validation of single-substation evaluation."""

    def test_returns_expected_keys(self) -> None:
        flooder = SubstationFlooder()
        result = flooder.evaluate_substation(1.0)
        assert "raw_depth_m" in result
        assert "effective_depth_m" in result
        assert "failure_probability" in result
        assert "operational" in result
        assert "ffe_m" in result
        assert "levee_height_m" in result

    def test_deep_flood_not_operational(self) -> None:
        flooder = SubstationFlooder()
        result = flooder.evaluate_substation(5.0)
        assert result["operational"] is False

    def test_shallow_flood_operational(self) -> None:
        flooder = SubstationFlooder()
        result = flooder.evaluate_substation(0.0)
        assert result["operational"] is True

    def test_override_defaults(self) -> None:
        flooder = SubstationFlooder(default_ffe_m=0.3)
        result = flooder.evaluate_substation(1.0, ffe_m=1.0)
        assert result["ffe_m"] == 1.0
        assert result["failure_probability"] < 0.5


class TestEvaluateSubstations:
    """Validation of batch substation evaluation."""

    def test_returns_geodataframe(self) -> None:
        flooder = SubstationFlooder()
        gdf = _make_substations_gdf(3)
        depths = np.array([0.5, 1.5, 0.1])
        result = flooder.evaluate_substations(gdf, depths)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 3
        assert "raw_depth_m" in result.columns
        assert "effective_depth_m" in result.columns
        assert "failure_probability" in result.columns
        assert "operational" in result.columns

    def test_empty_gdf_returns_empty(self) -> None:
        flooder = SubstationFlooder()
        gdf = gpd.GeoDataFrame({"substation_id": []}, geometry=[], crs="EPSG:3857")
        result = flooder.evaluate_substations(gdf, np.array([]))
        assert len(result) == 0

    def test_dict_depths(self) -> None:
        flooder = SubstationFlooder()
        gdf = _make_substations_gdf(3)
        depths = {"sub_0": 0.5, "sub_1": 1.5, "sub_2": 0.1}
        result = flooder.evaluate_substations(gdf, depths)
        assert len(result) == 3
        assert result["failure_probability"].iloc[1] > result["failure_probability"].iloc[2]

    def test_length_mismatch_raises(self) -> None:
        flooder = SubstationFlooder()
        gdf = _make_substations_gdf(3)
        with pytest.raises(ValueError, match="flood_depths_m"):
            flooder.evaluate_substations(gdf, np.array([0.5, 1.5]))

    def test_levee_protects(self) -> None:
        flooder = SubstationFlooder()
        gdf = _make_substations_gdf(3)
        depths = np.array([1.5, 1.5, 1.5])
        result = flooder.evaluate_substations(gdf, depths)
        assert result["failure_probability"].iloc[1] < result["failure_probability"].iloc[0]

    def test_pump_mitigates(self) -> None:
        flooder = SubstationFlooder()
        gdf = _make_substations_gdf(3)
        depths = np.array([0.5, 0.5, 0.5])
        result = flooder.evaluate_substations(gdf, depths)
        assert result["effective_depth_m"].iloc[2] < result["effective_depth_m"].iloc[0]


class TestEvaluateFloodImpact:
    """Validation of geospatial flood impact interface."""

    def test_with_array_source(self) -> None:
        flooder = SubstationFlooder()
        gdf = _make_substations_gdf(3)
        depths = np.array([0.5, 1.5, 0.1])
        result = flooder.evaluate_flood_impact(gdf, depths)
        assert len(result) == 3
        assert "failure_probability" in result.columns

    def test_with_dict_source(self) -> None:
        flooder = SubstationFlooder()
        gdf = _make_substations_gdf(3)
        depths = {"sub_0": 0.5, "sub_1": 1.5, "sub_2": 0.1}
        result = flooder.evaluate_flood_impact(gdf, depths)
        assert len(result) == 3

    def test_invalid_source_type_raises(self) -> None:
        flooder = SubstationFlooder()
        gdf = _make_substations_gdf(3)
        with pytest.raises(TypeError, match="flood_source"):
            flooder.evaluate_flood_impact(gdf, 12345)

    def test_unrecognised_file_extension_raises(self) -> None:
        flooder = SubstationFlooder()
        gdf = _make_substations_gdf(3)
        with pytest.raises(ValueError, match="Unrecognised flood source"):
            flooder.evaluate_flood_impact(gdf, "depths.csv")


class TestEvaluateTimeseries:
    """Validation of time-series evaluation."""

    def test_returns_dataframe(self) -> None:
        flooder = SubstationFlooder()
        depths = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        result = flooder.evaluate_timeseries("sub_0", depths)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "timestep" in result.columns
        assert "failure_probability" in result.columns
        assert "operational" in result.columns

    def test_rising_flood_increases_risk(self) -> None:
        flooder = SubstationFlooder()
        depths = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        result = flooder.evaluate_timeseries("sub_0", depths)
        probs = result["failure_probability"].values
        assert probs[-1] > probs[0]


class TestRepr:
    """Validation of string representation."""

    def test_repr_includes_key_params(self) -> None:
        flooder = SubstationFlooder(
            gamma=3.0,
            default_ffe_m=0.5,
            default_levee_height_m=1.2,
            default_pump_rate_m_per_s=0.01,
        )
        r = repr(flooder)
        assert "3.00" in r
        assert "0.50m" in r
        assert "1.20m" in r
        assert "0.0100m/s" in r


_EPS: float = 1e-12
