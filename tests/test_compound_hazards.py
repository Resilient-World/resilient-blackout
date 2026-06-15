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

"""Unit tests for ``resilient_blackout.climate.compound_hazards``."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from resilient_blackout.climate.compound_hazards import CompoundHazardEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wind_data() -> np.ndarray:
    """Synthetic wind speed data (m/s)."""
    rng = np.random.default_rng(42)
    return rng.weibull(a=2.5, size=500) * 15.0


@pytest.fixture
def temp_data() -> np.ndarray:
    """Synthetic temperature data (°C), correlated with wind."""
    rng = np.random.default_rng(42)
    base = rng.normal(25, 8, 500)
    return base


@pytest.fixture
def engine() -> CompoundHazardEngine:
    return CompoundHazardEngine(copula_type="auto")


@pytest.fixture
def fitted_engine(engine: CompoundHazardEngine, wind_data: np.ndarray, temp_data: np.ndarray) -> CompoundHazardEngine:
    engine.fit_copula(wind_data, temp_data)
    return engine


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestInit:
    """Validation of constructor."""

    def test_default(self) -> None:
        e = CompoundHazardEngine()
        assert e.copula_type == "auto"
        assert e.fitted_copula is None

    def test_gumbel(self) -> None:
        e = CompoundHazardEngine(copula_type="gumbel")
        assert e.copula_type == "gumbel"

    def test_clayton(self) -> None:
        e = CompoundHazardEngine(copula_type="clayton")
        assert e.copula_type == "clayton"

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="copula_type"):
            CompoundHazardEngine(copula_type="frank")  # type: ignore[arg-type]

    def test_repr(self) -> None:
        e = CompoundHazardEngine()
        assert "CompoundHazardEngine" in repr(e)
        assert "unfitted" in repr(e)


# ---------------------------------------------------------------------------
# Copula fitting
# ---------------------------------------------------------------------------


class TestCopulaFitting:
    """Validation of copula fitting."""

    def test_fit_returns_dict(self, engine: CompoundHazardEngine, wind_data: np.ndarray, temp_data: np.ndarray) -> None:
        result = engine.fit_copula(wind_data, temp_data)
        assert "copula_type" in result
        assert "theta" in result
        assert "tau" in result
        assert "marginals" in result

    def test_fit_stores_result(self, engine: CompoundHazardEngine, wind_data: np.ndarray, temp_data: np.ndarray) -> None:
        engine.fit_copula(wind_data, temp_data)
        assert engine.fitted_copula is not None

    def test_theta_positive(self, fitted_engine: CompoundHazardEngine) -> None:
        assert fitted_engine.fitted_copula["theta"] > 0

    def test_tau_in_range(self, fitted_engine: CompoundHazardEngine) -> None:
        tau = fitted_engine.fitted_copula["tau"]
        assert -1.0 <= tau <= 1.0

    def test_mismatched_length_raises(self, engine: CompoundHazardEngine) -> None:
        with pytest.raises(ValueError, match="same length"):
            engine.fit_copula(np.array([1.0, 2.0]), np.array([1.0]))

    def test_too_few_samples_raises(self, engine: CompoundHazardEngine) -> None:
        with pytest.raises(ValueError, match="10 samples"):
            engine.fit_copula(np.array([1.0]), np.array([1.0]))

    def test_auto_selects_family(self, engine: CompoundHazardEngine, wind_data: np.ndarray, temp_data: np.ndarray) -> None:
        result = engine.fit_copula(wind_data, temp_data)
        assert result["copula_type"] in ("gumbel", "clayton")


# ---------------------------------------------------------------------------
# Joint sampling
# ---------------------------------------------------------------------------


class TestJointSampling:
    """Validation of copula sampling."""

    def test_sample_shape(self, fitted_engine: CompoundHazardEngine) -> None:
        a, b = fitted_engine.sample_joint(100)
        assert a.shape == (100,)
        assert b.shape == (100,)

    def test_sample_without_fit_raises(self, engine: CompoundHazardEngine) -> None:
        with pytest.raises(RuntimeError, match="No copula fitted"):
            engine.sample_joint(10)

    def test_sample_reproducible(self, fitted_engine: CompoundHazardEngine) -> None:
        a1, b1 = fitted_engine.sample_joint(50, seed=42)
        a2, b2 = fitted_engine.sample_joint(50, seed=42)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(b1, b2)


# ---------------------------------------------------------------------------
# Joint exceedance probability
# ---------------------------------------------------------------------------


class TestJointExceedance:
    """Validation of joint exceedance probability."""

    def test_probability_in_range(self, fitted_engine: CompoundHazardEngine) -> None:
        p = fitted_engine.joint_exceedance_probability(20.0, 30.0)
        assert 0.0 <= p <= 1.0

    def test_without_fit_raises(self, engine: CompoundHazardEngine) -> None:
        with pytest.raises(RuntimeError, match="No copula fitted"):
            engine.joint_exceedance_probability(10.0, 20.0)


# ---------------------------------------------------------------------------
# Conditional ignition probability
# ---------------------------------------------------------------------------


class TestConditionalIgnition:
    """Validation of logistic ignition model."""

    def test_scalar_input(self) -> None:
        p = CompoundHazardEngine.conditional_ignition_probability(
            wind_speed_ms=np.array([30.0]),
            dryness_index=np.array([0.8]),
        )
        assert 0.0 < float(p[0]) < 1.0

    def test_vectorized(self) -> None:
        v = np.array([0.0, 15.0, 30.0, 45.0])
        d = np.array([0.0, 0.3, 0.6, 1.0])
        p = CompoundHazardEngine.conditional_ignition_probability(v, d)
        assert p.shape == (4,)
        assert np.all(p >= 0.0)
        assert np.all(p <= 1.0)

    def test_monotonic_in_wind(self) -> None:
        v = np.array([10.0, 20.0, 30.0, 40.0])
        d = np.full(4, 0.5)
        p = CompoundHazardEngine.conditional_ignition_probability(v, d)
        assert np.all(np.diff(p) >= 0)

    def test_monotonic_in_dryness(self) -> None:
        v = np.full(4, 25.0)
        d = np.array([0.0, 0.3, 0.6, 1.0])
        p = CompoundHazardEngine.conditional_ignition_probability(v, d)
        assert np.all(np.diff(p) >= 0)

    def test_custom_beta(self) -> None:
        p = CompoundHazardEngine.conditional_ignition_probability(
            wind_speed_ms=np.array([30.0]),
            dryness_index=np.array([0.8]),
            beta=np.array([-2.0, 0.05, 1.0]),
        )
        assert 0.0 < float(p[0]) < 1.0


# ---------------------------------------------------------------------------
# Vulnerability modification
# ---------------------------------------------------------------------------


class TestVulnerabilityModification:
    """Validation of fragility curve modification."""

    def test_wind_damage(self) -> None:
        base = {"mean": 30.0, "std": 5.0}
        mod = CompoundHazardEngine.modify_vulnerability(base, 40.0, "wind_damage")
        assert mod["mean"] < base["mean"]
        assert mod["std"] > base["std"]

    def test_flood_weakening(self) -> None:
        base = {"mean": 30.0, "std": 5.0}
        mod = CompoundHazardEngine.modify_vulnerability(base, 2.0, "flood_weakening")
        assert mod["mean"] < base["mean"]

    def test_thermal_stress(self) -> None:
        base = {"mean": 30.0, "std": 5.0}
        mod = CompoundHazardEngine.modify_vulnerability(base, 45.0, "thermal_stress")
        assert mod["mean"] < base["mean"]

    def test_wildfire_risk(self) -> None:
        base = {"mean": 30.0, "std": 5.0}
        mod = CompoundHazardEngine.modify_vulnerability(base, 30.0, "wildfire_risk")
        assert mod["mean"] < base["mean"]

    def test_below_threshold_no_change(self) -> None:
        base = {"mean": 30.0, "std": 5.0}
        mod = CompoundHazardEngine.modify_vulnerability(base, 10.0, "wind_damage")
        assert mod["mean"] == pytest.approx(base["mean"])
        assert mod["std"] == pytest.approx(base["std"])


# ---------------------------------------------------------------------------
# Temporal compound evaluation
# ---------------------------------------------------------------------------


class TestTemporalCompound:
    """Validation of temporal compound evaluation."""

    @pytest.fixture
    def primary_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"intensity": [10.0, 20.0]},
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:4326",
        )

    @pytest.fixture
    def secondary_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"intensity": [5.0, 15.0]},
            geometry=[Point(0.5, 0.5), Point(1.5, 1.5)],
            crs="EPSG:4326",
        )

    def test_returns_geodataframe(
        self, engine: CompoundHazardEngine, primary_gdf: gpd.GeoDataFrame, secondary_gdf: gpd.GeoDataFrame
    ) -> None:
        result = engine.evaluate_temporal_compound(primary_gdf, secondary_gdf)
        assert isinstance(result, gpd.GeoDataFrame)
        assert "compound_intensity" in result.columns

    def test_compound_intensity_positive(
        self, engine: CompoundHazardEngine, primary_gdf: gpd.GeoDataFrame, secondary_gdf: gpd.GeoDataFrame
    ) -> None:
        result = engine.evaluate_temporal_compound(primary_gdf, secondary_gdf)
        assert np.all(result["compound_intensity"] >= 0)

    def test_missing_intensity_raises(
        self, engine: CompoundHazardEngine, primary_gdf: gpd.GeoDataFrame
    ) -> None:
        bad = gpd.GeoDataFrame({"x": [1.0]}, geometry=[Point(0, 0)], crs="EPSG:4326")
        with pytest.raises(ValueError, match="intensity"):
            engine.evaluate_temporal_compound(primary_gdf, bad)


# ---------------------------------------------------------------------------
# Hazard layer merging
# ---------------------------------------------------------------------------


class TestMergeHazardLayers:
    """Validation of hazard layer merging."""

    @pytest.fixture
    def layer1(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"intensity": [10.0, 20.0]},
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:4326",
        )

    @pytest.fixture
    def layer2(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"intensity": [5.0, 15.0]},
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:4326",
        )

    def test_merge_two_layers(self, layer1: gpd.GeoDataFrame, layer2: gpd.GeoDataFrame) -> None:
        result = CompoundHazardEngine.merge_hazard_layers([layer1, layer2])
        assert isinstance(result, gpd.GeoDataFrame)
        assert "compound_intensity" in result.columns

    def test_weighted_merge(self, layer1: gpd.GeoDataFrame, layer2: gpd.GeoDataFrame) -> None:
        result = CompoundHazardEngine.merge_hazard_layers(
            [layer1, layer2], weights=[0.8, 0.2]
        )
        assert len(result) == 2

    def test_empty_layers_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            CompoundHazardEngine.merge_hazard_layers([])

    def test_mismatched_weights_raises(self, layer1: gpd.GeoDataFrame, layer2: gpd.GeoDataFrame) -> None:
        with pytest.raises(ValueError, match="weights"):
            CompoundHazardEngine.merge_hazard_layers(
                [layer1, layer2], weights=[1.0]
            )
