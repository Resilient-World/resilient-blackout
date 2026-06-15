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

"""Unit tests for ``resilient_blackout.ml.weather_failure``."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandapower")

from resilient_blackout.ml.weather_failure import WeatherFailurePredictor


def _make_synthetic_data(
    n_samples: int = 200,
    n_assets: int = 20,
    random_state: int = 42,
) -> tuple:
    """Generate synthetic outage and weather DataFrames."""
    rng = np.random.default_rng(random_state)

    assets = [f"asset_{i}" for i in range(n_assets)]
    weather = pd.DataFrame(
        {
            "asset_id": rng.choice(assets, size=n_samples),
            "wind_speed_ms": rng.normal(10, 5, size=n_samples),
            "precip_rate_mmh": rng.exponential(2, size=n_samples),
            "temperature_c": rng.normal(20, 10, size=n_samples),
            "humidity_pct": rng.uniform(30, 100, size=n_samples),
            "lightning_flashes_km2": rng.exponential(0.5, size=n_samples),
        }
    )

    # Make higher wind + lightning correlate with failures
    p_failure = 1.0 / (
        1.0
        + np.exp(
            -(
                0.1 * weather["wind_speed_ms"]
                + 0.5 * weather["lightning_flashes_km2"]
                - 2.0
            )
        )
    )
    failed = (rng.random(n_samples) < p_failure).astype(int)

    outages = pd.DataFrame(
        {
            "asset_id": weather["asset_id"],
            "failed": failed,
        }
    )

    return outages, weather


class TestWeatherFailurePredictorInit:
    """Validation of constructor and parameter handling."""

    def test_default_construction(self) -> None:
        predictor = WeatherFailurePredictor()
        assert predictor.model_type == "logistic"
        assert predictor.class_weight == "balanced"
        assert predictor.asset_type is None
        assert predictor.pipeline_ is None

    def test_random_forest_construction(self) -> None:
        predictor = WeatherFailurePredictor(model_type="random_forest")
        assert predictor.model_type == "random_forest"

    def test_invalid_model_type_raises(self) -> None:
        with pytest.raises(ValueError, match="model_type"):
            WeatherFailurePredictor(model_type="xgboost")

    def test_invalid_asset_type_raises(self) -> None:
        with pytest.raises(ValueError, match="asset_type"):
            WeatherFailurePredictor(asset_type="substation")

    def test_repr_unfitted(self) -> None:
        predictor = WeatherFailurePredictor()
        r = repr(predictor)
        assert "unfitted" in r
        assert "logistic" in r


class TestFitVulnerabilityModel:
    """Validation of training pipeline."""

    def test_fit_returns_metrics(self) -> None:
        outages, weather = _make_synthetic_data(n_samples=200)
        predictor = WeatherFailurePredictor(random_state=42)
        metrics = predictor.fit_vulnerability_model(outages, weather)

        assert "auc" in metrics
        assert "log_loss" in metrics
        assert 0.0 <= metrics["auc"] <= 1.0
        assert metrics["log_loss"] >= 0.0
        assert predictor.pipeline_ is not None
        assert predictor.val_metrics_ is not None

    def test_fit_random_forest(self) -> None:
        outages, weather = _make_synthetic_data(n_samples=200)
        predictor = WeatherFailurePredictor(model_type="random_forest", random_state=42)
        metrics = predictor.fit_vulnerability_model(outages, weather)
        assert "auc" in metrics

    def test_fit_empty_data_raises(self) -> None:
        predictor = WeatherFailurePredictor()
        empty = pd.DataFrame({"asset_id": [], "failed": []})
        weather = pd.DataFrame(
            {
                "asset_id": [],
                "wind_speed_ms": [],
                "precip_rate_mmh": [],
                "temperature_c": [],
                "humidity_pct": [],
                "lightning_flashes_km2": [],
            }
        )
        with pytest.raises(ValueError, match="empty"):
            predictor.fit_vulnerability_model(empty, weather)

    def test_fit_single_class_raises(self) -> None:
        outages, weather = _make_synthetic_data(n_samples=50)
        outages["failed"] = 0
        predictor = WeatherFailurePredictor(random_state=42)
        with pytest.raises(ValueError, match="one class"):
            predictor.fit_vulnerability_model(outages, weather)


class TestPredictFailureProbabilities:
    """Validation of runtime prediction interface."""

    def test_predict_returns_probabilities(self) -> None:
        outages, weather = _make_synthetic_data(n_samples=200)
        predictor = WeatherFailurePredictor(random_state=42)
        predictor.fit_vulnerability_model(outages, weather)

        test_weather = weather.head(20).copy()
        probs = predictor.predict_failure_probabilities(test_weather)

        assert isinstance(probs, dict)
        assert len(probs) == 20
        for p in probs.values():
            assert 0.0 <= p <= 1.0

    def test_predict_without_fit_raises(self) -> None:
        predictor = WeatherFailurePredictor()
        with pytest.raises(RuntimeError, match="fitted"):
            predictor.predict_failure_probabilities(pd.DataFrame())

    def test_predict_missing_columns_raises(self) -> None:
        outages, weather = _make_synthetic_data(n_samples=200)
        predictor = WeatherFailurePredictor(random_state=42)
        predictor.fit_vulnerability_model(outages, weather)

        bad_weather = pd.DataFrame({"asset_id": ["a1"], "wind_speed_ms": [10.0]})
        with pytest.raises(ValueError, match="missing"):
            predictor.predict_failure_probabilities(bad_weather)


class TestApplyToNetwork:
    """Validation of pandapower integration."""

    def test_apply_to_lines(self) -> None:
        pytest.importorskip("pandapower")
        import pandapower as pp

        outages, weather = _make_synthetic_data(n_samples=200)
        predictor = WeatherFailurePredictor(random_state=42)
        predictor.fit_vulnerability_model(outages, weather)

        net = pp.create_empty_network()
        b0 = pp.create_bus(net, vn_kv=0.4)
        b1 = pp.create_bus(net, vn_kv=0.4)
        pp.create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, std_type="NAYY 4x50 SE")

        # Create weather row for line_0
        w = pd.DataFrame(
            {
                "asset_id": ["line_0"],
                "wind_speed_ms": [50.0],  # extreme weather → high probability
                "precip_rate_mmh": [10.0],
                "temperature_c": [35.0],
                "humidity_pct": [90.0],
                "lightning_flashes_km2": [5.0],
            }
        )

        rng = np.random.default_rng(0)
        # With deterministic rng and extreme weather, we should trip at least once over many trials
        tripped_count = 0
        for _ in range(50):
            net_copy = pp.create_empty_network()
            b0_c = pp.create_bus(net_copy, vn_kv=0.4)
            b1_c = pp.create_bus(net_copy, vn_kv=0.4)
            pp.create_line(
                net_copy, from_bus=b0_c, to_bus=b1_c, length_km=1.0, std_type="NAYY 4x50 SE"
            )
            tripped = predictor.apply_to_network(net_copy, w, asset_type="line", rng=rng)
            if "line_0" in tripped:
                tripped_count += 1

        # With extreme weather the probability should be high, so expect some trips
        assert tripped_count > 0

    def test_apply_unsupported_asset_type_raises(self) -> None:
        pytest.importorskip("pandapower")
        import pandapower as pp

        outages, weather = _make_synthetic_data(n_samples=200)
        predictor = WeatherFailurePredictor(random_state=42)
        predictor.fit_vulnerability_model(outages, weather)

        net = pp.create_empty_network()
        with pytest.raises(ValueError, match="Unsupported"):
            predictor.apply_to_network(net, pd.DataFrame(), asset_type="bus")


class TestSaveLoad:
    """Validation of model persistence."""

    def test_save_load_roundtrip(self) -> None:
        outages, weather = _make_synthetic_data(n_samples=200)
        predictor = WeatherFailurePredictor(random_state=42)
        predictor.fit_vulnerability_model(outages, weather)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            predictor.save_model(path)
            loaded = WeatherFailurePredictor.load_model(path)

            assert loaded.model_type == predictor.model_type
            assert loaded.feature_cols == predictor.feature_cols
            assert loaded.pipeline_ is not None

            # Predictions should match
            test_weather = weather.head(5).copy()
            original_probs = predictor.predict_failure_probabilities(test_weather)
            loaded_probs = loaded.predict_failure_probabilities(test_weather)
            assert original_probs == loaded_probs

    def test_save_unfitted_raises(self) -> None:
        predictor = WeatherFailurePredictor()
        with pytest.raises(RuntimeError, match="unfitted"):
            predictor.save_model("/tmp/test.joblib")


class TestRepr:
    """Validation of string representation."""

    def test_repr_fitted(self) -> None:
        outages, weather = _make_synthetic_data(n_samples=200)
        predictor = WeatherFailurePredictor(random_state=42)
        predictor.fit_vulnerability_model(outages, weather)
        r = repr(predictor)
        assert "fitted" in r
