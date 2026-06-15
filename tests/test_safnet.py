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

"""Unit tests for ``resilient_blackout.ml.safnet``."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from resilient_blackout.ml.safnet import SAFNetPredictor


# ---------------------------------------------------------------------------
# Synthetic training data
# ---------------------------------------------------------------------------


def _make_synthetic_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic structural attribute data with known patterns."""
    rng = np.random.default_rng(seed)

    # Geospatial features
    dtm_elevation = rng.normal(50, 15, n)
    slope = np.abs(rng.normal(2, 3, n))
    aspect = rng.uniform(0, 360, n)
    roughness = np.abs(rng.normal(0.5, 0.3, n))
    dist_to_water = rng.exponential(200, n)
    twi = rng.normal(6, 2, n)

    # Demographic features
    median_income = rng.lognormal(10.8, 0.4, n)
    pop_density = rng.lognormal(6, 1, n)
    pct_pre1980 = rng.beta(3, 2, n)
    pct_post2000 = rng.beta(2, 3, n)
    pct_owner = rng.beta(5, 3, n)

    # Asset features
    voltage_kv = rng.choice([11, 33, 66, 132], n)
    age_years = rng.exponential(25, n)
    material_steel = rng.binomial(1, 0.4, n).astype(float)
    material_concrete = rng.binomial(1, 0.35, n).astype(float)
    material_wood = rng.binomial(1, 0.25, n).astype(float)
    eq_type_transformer = rng.binomial(1, 0.5, n).astype(float)
    eq_type_switchgear = rng.binomial(1, 0.3, n).astype(float)

    # Targets (with plausible relationships)
    ffe_m = (
        3.0
        + 0.8 * (dtm_elevation - 50) / 15
        + 0.3 * (median_income - np.exp(10.8)) / 20000
        - 0.2 * (dist_to_water / 200)
        + 0.1 * (voltage_kv / 132)
        + rng.normal(0, 0.5, n)
    )

    foundation_type = np.where(
        ffe_m > 4.0,
        np.where(rng.random(n) < 0.6, "pile", "pier"),
        np.where(rng.random(n) < 0.5, "slab", "unknown"),
    )

    has_basement = (
        (ffe_m < 3.5) & (rng.random(n) < 0.7)
    ).astype(float)

    return pd.DataFrame({
        "dtm_elevation_m": dtm_elevation,
        "slope_deg": slope,
        "aspect_deg": aspect,
        "roughness_m": roughness,
        "dist_to_water_m": dist_to_water,
        "twi": twi,
        "median_income_usd": median_income,
        "pop_density_per_km2": pop_density,
        "pct_built_pre1980": pct_pre1980,
        "pct_built_post2000": pct_post2000,
        "pct_owner_occupied": pct_owner,
        "voltage_kv": voltage_kv,
        "age_years": age_years,
        "material_steel": material_steel,
        "material_concrete": material_concrete,
        "material_wood": material_wood,
        "equipment_type_transformer": eq_type_transformer,
        "equipment_type_switchgear": eq_type_switchgear,
        "ffe_m": ffe_m,
        "foundation_type": foundation_type,
        "has_basement": has_basement,
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def training_data() -> pd.DataFrame:
    return _make_synthetic_data(300)


@pytest.fixture
def predictor() -> SAFNetPredictor:
    return SAFNetPredictor()


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestInit:
    """Validation of constructor."""

    def test_default(self) -> None:
        p = SAFNetPredictor()
        assert len(p.geo_features) == 6
        assert len(p.demo_features) == 5
        assert len(p.asset_features) == 7
        assert not p._is_fitted

    def test_custom_features(self) -> None:
        p = SAFNetPredictor(
            geo_features=["elevation"],
            demo_features=["income"],
            asset_features=["voltage"],
        )
        assert p.geo_features == ["elevation"]
        assert p.demo_features == ["income"]
        assert p.asset_features == ["voltage"]

    def test_repr(self) -> None:
        p = SAFNetPredictor()
        r = repr(p)
        assert "SAFNetPredictor" in r
        assert "unfitted" in r


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestTraining:
    """Validation of model training."""

    def test_train_returns_result(self, predictor: SAFNetPredictor, training_data: pd.DataFrame) -> None:
        result = predictor.train_safnet_model(
            training_data, epochs=30, n_folds=2, patience=10,
        )
        assert "model" in result
        assert "cv_results" in result
        assert "feature_importance" in result
        assert "history" in result
        assert len(result["cv_results"]) == 2

    def test_model_is_fitted_after_training(self, predictor: SAFNetPredictor, training_data: pd.DataFrame) -> None:
        predictor.train_safnet_model(training_data, epochs=20, n_folds=2, patience=5)
        assert predictor._is_fitted
        assert predictor.model is not None

    def test_feature_importance_keys(self, predictor: SAFNetPredictor, training_data: pd.DataFrame) -> None:
        result = predictor.train_safnet_model(
            training_data, epochs=20, n_folds=2, patience=5,
        )
        imp = result["feature_importance"]
        assert "dtm_elevation_m" in imp
        assert "median_income_usd" in imp
        assert "voltage_kv" in imp
        for v in imp.values():
            assert "ffe_importance" in v
            assert "foundation_importance" in v
            assert "basement_importance" in v


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestPrediction:
    """Validation of prediction."""

    def test_predict_returns_augmented_df(
        self, predictor: SAFNetPredictor, training_data: pd.DataFrame
    ) -> None:
        predictor.train_safnet_model(training_data, epochs=20, n_folds=2, patience=5)
        test_df = training_data.head(20).copy()
        result = predictor.predict(test_df)

        assert "predicted_ffe_m" in result.columns
        assert "predicted_foundation_type" in result.columns
        assert "predicted_has_basement_prob" in result.columns
        assert "predicted_has_basement" in result.columns
        assert "foundation_confidence" in result.columns

    def test_ffe_predictions_plausible(
        self, predictor: SAFNetPredictor, training_data: pd.DataFrame
    ) -> None:
        predictor.train_safnet_model(training_data, epochs=20, n_folds=2, patience=5)
        result = predictor.predict(training_data.head(20))
        # FFE should be in a reasonable range (0–10 m)
        assert np.all(result["predicted_ffe_m"] >= 0)
        assert np.all(result["predicted_ffe_m"] <= 15)

    def test_basement_prob_in_range(
        self, predictor: SAFNetPredictor, training_data: pd.DataFrame
    ) -> None:
        predictor.train_safnet_model(training_data, epochs=20, n_folds=2, patience=5)
        result = predictor.predict(training_data.head(20))
        assert np.all(result["predicted_has_basement_prob"] >= 0)
        assert np.all(result["predicted_has_basement_prob"] <= 1)

    def test_foundation_types_valid(
        self, predictor: SAFNetPredictor, training_data: pd.DataFrame
    ) -> None:
        predictor.train_safnet_model(training_data, epochs=20, n_folds=2, patience=5)
        result = predictor.predict(training_data.head(20))
        valid_types = {"slab", "pier", "pile", "basement_wall", "unknown"}
        assert set(result["predicted_foundation_type"].unique()).issubset(valid_types)

    def test_predict_before_training_raises(self, predictor: SAFNetPredictor, training_data: pd.DataFrame) -> None:
        with pytest.raises(RuntimeError, match="not trained"):
            predictor.predict(training_data.head(5))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    """Validation of save/load roundtrip."""

    def test_save_load_roundtrip(
        self, predictor: SAFNetPredictor, training_data: pd.DataFrame
    ) -> None:
        predictor.train_safnet_model(training_data, epochs=20, n_folds=2, patience=5)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name

        try:
            predictor.save(tmp_path)
            loaded = SAFNetPredictor.load(tmp_path)

            assert loaded._is_fitted
            assert loaded.geo_features == predictor.geo_features
            assert loaded.demo_features == predictor.demo_features
            assert loaded.asset_features == predictor.asset_features

            # Predictions should match
            test_df = training_data.head(10)
            pred_orig = predictor.predict(test_df)
            pred_loaded = loaded.predict(test_df)

            np.testing.assert_allclose(
                pred_orig["predicted_ffe_m"].values,
                pred_loaded["predicted_ffe_m"].values,
                rtol=1e-5,
            )
        finally:
            Path(tmp_path).unlink()

    def test_save_before_training_raises(self, predictor: SAFNetPredictor) -> None:
        with pytest.raises(RuntimeError, match="No model"):
            predictor.save("/tmp/test.pt")
