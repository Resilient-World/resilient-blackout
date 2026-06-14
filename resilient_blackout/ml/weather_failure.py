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

"""
Machine-learning weather-driven vulnerability predictor.

Provides ``WeatherFailurePredictor``, a scikit-learn binary classifier
that maps dynamic weather vectors (wind speed, precipitation, temperature,
humidity, lightning flash density) to instantaneous component failure
probabilities.  Includes a pandapower integration layer for updating
``in_service`` flags during Monte Carlo simulations.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_EPS: float = 1e-12
_DEFAULT_FEATURES: List[str] = [
    "wind_speed_ms",
    "precip_rate_mmh",
    "temperature_c",
    "humidity_pct",
    "lightning_flashes_km2",
]


# ---------------------------------------------------------------------------
# WeatherFailurePredictor
# ---------------------------------------------------------------------------


class WeatherFailurePredictor:
    """Binary classifier for weather-driven grid asset failure.

    Fits a ``StandardScaler`` + ``LogisticRegression`` or
    ``RandomForestClassifier`` pipeline on historical outage records and
    weather observations, then predicts per-asset instantaneous failure
    probabilities.

    Parameters
    ----------
    feature_cols : list of str
        Weather feature column names.  Defaults to the five canonical
        features.
    model_type : str
        ``"logistic"`` or ``"random_forest"``.  Default ``"logistic"``.
    class_weight : str or dict or None
        Passed to the underlying classifier.  Default ``"balanced"``.
    asset_type : str or None
        Asset type this model targets (``"line"``, ``"gen"``,
        ``"trafo"``, or ``None`` for all).  Default ``None``.
    use_smote : bool
        If ``True``, apply SMOTE oversampling to the training set
        to handle class imbalance.  Requires ``imbalanced-learn``.
        Default ``False``.
    asset_id_format : str
        Format string for mapping pandapower indices to asset IDs
        in :meth:`apply_to_network`.  Must contain ``{type}`` and
        ``{idx}`` placeholders.  Default ``"{type}_{idx}"``.
    random_state : int
        Seed for reproducible train/test splits and forest sampling.

    Attributes
    ----------
    feature_cols : list of str
    model_type : str
    class_weight : str or dict or None
    asset_type : str or None
    pipeline_ : sklearn.Pipeline or None
        Fitted after :meth:`fit_vulnerability_model`.
    val_metrics_ : dict or None
        Validation AUC and log-loss.
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        model_type: str = "logistic",
        class_weight: Optional[str] = "balanced",
        asset_type: Optional[str] = None,
        use_smote: bool = False,
        asset_id_format: str = "{type}_{idx}",
        random_state: int = 42,
    ) -> None:
        if model_type not in {"logistic", "random_forest"}:
            raise ValueError(
                f"model_type must be 'logistic' or 'random_forest', got {model_type}"
            )
        if asset_type is not None and asset_type not in {"line", "gen", "trafo"}:
            raise ValueError(
                f"asset_type must be 'line', 'gen', 'trafo', or None, got {asset_type}"
            )
        if "{type}" not in asset_id_format or "{idx}" not in asset_id_format:
            raise ValueError(
                f"asset_id_format must contain {{type}} and {{idx}} placeholders, "
                f"got {asset_id_format}"
            )

        self.feature_cols: List[str] = list(feature_cols or _DEFAULT_FEATURES)
        self.model_type = model_type
        self.class_weight = class_weight
        self.asset_type = asset_type
        self.use_smote = use_smote
        self.asset_id_format = asset_id_format
        self.random_state = random_state

        self.pipeline_: Optional[Pipeline] = None
        self.val_metrics_: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit_vulnerability_model(
        self,
        historical_outages: pd.DataFrame,
        weather_features: pd.DataFrame,
        test_size: float = 0.2,
    ) -> Dict[str, float]:
        """Fit the vulnerability model to historical data.

        ``historical_outages`` must contain at least the columns
        ``asset_id`` (str) and ``failed`` (bool or int 0/1).
        ``weather_features`` must contain ``asset_id`` and all
        ``feature_cols``.

        Parameters
        ----------
        historical_outages : pd.DataFrame
        weather_features : pd.DataFrame
        test_size : float
            Fraction of data held out for validation.  Default 0.2.

        Returns
        -------
        dict
            Validation metrics ``{"auc": float, "log_loss": float}``.
        """
        df = self._merge_data(historical_outages, weather_features)
        if df.empty:
            raise ValueError("Merged training data is empty after joining outages and weather.")

        X = df[self.feature_cols].values.astype(np.float64)
        y = df["failed"].values.astype(np.int32)

        if len(np.unique(y)) < 2:
            raise ValueError(
                f"Target variable has only one class ({np.unique(y)}). Cannot fit classifier."
            )

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Apply SMOTE if requested
        if self.use_smote:
            try:
                from imblearn.over_sampling import SMOTE
            except ImportError:
                raise ImportError(
                    "imbalanced-learn is required for SMOTE. Install with: pip install imbalanced-learn"
                )
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info("SMOTE applied — training samples: %d", len(y_train))

        clf = self._build_classifier()
        self.pipeline_ = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        self.pipeline_.fit(X_train, y_train)

        # Validation metrics
        val_proba = self.pipeline_.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_proba)
        ll = log_loss(y_val, val_proba, eps=_EPS)
        self.val_metrics_ = {"auc": float(auc), "log_loss": float(ll)}
        logger.info(
            "WeatherFailurePredictor fitted — val_auc=%.4f, val_logloss=%.4f",
            auc, ll,
        )
        return self.val_metrics_

    def _merge_data(
        self,
        historical_outages: pd.DataFrame,
        weather_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Join outages and weather on asset_id."""
        required = {"asset_id", "failed"}
        missing = required - set(historical_outages.columns)
        if missing:
            raise ValueError(f"historical_outages missing columns: {missing}")

        missing_w = {"asset_id"} | set(self.feature_cols)
        missing_w -= set(weather_features.columns)
        if missing_w:
            raise ValueError(f"weather_features missing columns: {missing_w}")

        df = historical_outages.merge(
            weather_features[["asset_id"] + self.feature_cols],
            on="asset_id",
            how="inner",
        )

        if self.asset_type is not None and "asset_type" in df.columns:
            df = df[df["asset_type"] == self.asset_type]

        # Ensure failed is integer 0/1
        df["failed"] = df["failed"].astype(int)
        return df

    def _build_classifier(self) -> Any:
        """Instantiate the sklearn classifier."""
        if self.model_type == "logistic":
            return LogisticRegression(
                max_iter=1000,
                class_weight=self.class_weight,
                random_state=self.random_state,
                solver="lbfgs",
            )
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_failure_probabilities(
        self,
        current_weather: pd.DataFrame,
    ) -> Dict[str, float]:
        """Predict instantaneous failure probability for each asset.

        .. math::
            P_f(a, t) = \sigma(\mathbf{w}^T \mathbf{X}_a(t) + b)

        Parameters
        ----------
        current_weather : pd.DataFrame
            Must contain ``asset_id`` and all ``feature_cols``.

        Returns
        -------
        dict
            ``{asset_id: probability}``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if self.pipeline_ is None:
            raise RuntimeError("Model has not been fitted. Call fit_vulnerability_model() first.")

        missing = {"asset_id"} | set(self.feature_cols)
        missing -= set(current_weather.columns)
        if missing:
            raise ValueError(f"current_weather missing columns: {missing}")

        X = current_weather[self.feature_cols].values.astype(np.float64)
        proba = self.pipeline_.predict_proba(X)[:, 1]

        return {
            str(aid): float(p)
            for aid, p in zip(current_weather["asset_id"].values, proba)
        }

    # ------------------------------------------------------------------
    # Pandapower integration
    # ------------------------------------------------------------------

    def apply_to_network(
        self,
        net: Any,
        weather_df: pd.DataFrame,
        asset_type: str = "line",
        rng: Optional[np.random.Generator] = None,
    ) -> List[str]:
        """Update ``in_service`` status based on predicted failure probabilities.

        For each asset of the requested type, samples a Bernoulli trial
        using the predicted failure probability.  If the trial succeeds,
        the asset is marked ``in_service=False``.

        Parameters
        ----------
        net : pandapowerNet
        weather_df : pd.DataFrame
            Current weather per asset.
        asset_type : str
            ``"line"``, ``"gen"``, or ``"trafo"``.
        rng : np.random.Generator or None
            Random generator for reproducible sampling.

        Returns
        -------
        list of str
            Asset IDs that were tripped.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Filter weather to the requested asset type if the column exists
        w = weather_df.copy()
        if "asset_type" in w.columns:
            w = w[w["asset_type"] == asset_type]

        probs = self.predict_failure_probabilities(w)
        tripped: List[str] = []

        if asset_type == "line":
            for idx in net.line.index:
                aid = self.asset_id_format.format(type=asset_type, idx=idx)
                if aid in probs and rng.random() < probs[aid]:
                    net.line.at[idx, "in_service"] = False
                    tripped.append(aid)
        elif asset_type == "gen":
            for idx in net.gen.index:
                aid = self.asset_id_format.format(type=asset_type, idx=idx)
                if aid in probs and rng.random() < probs[aid]:
                    net.gen.at[idx, "in_service"] = False
                    tripped.append(aid)
        elif asset_type == "trafo":
            for idx in net.trafo.index:
                aid = self.asset_id_format.format(type=asset_type, idx=idx)
                if aid in probs and rng.random() < probs[aid]:
                    net.trafo.at[idx, "in_service"] = False
                    tripped.append(aid)
        else:
            raise ValueError(f"Unsupported asset_type: {asset_type}")

        if tripped:
            logger.info(
                "WeatherFailurePredictor tripped %d %s(s).", len(tripped), asset_type
            )
        return tripped

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> Dict[str, float]:
        """Return per-feature importance scores.

        For LogisticRegression, returns absolute coefficient magnitudes.
        For RandomForest, returns the built-in feature importances.

        Returns
        -------
        dict
            ``{feature_name: importance}`` sorted descending.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if self.pipeline_ is None:
            raise RuntimeError("Model has not been fitted. Call fit_vulnerability_model() first.")

        clf = self.pipeline_.named_steps["clf"]
        if self.model_type == "logistic":
            importances = np.abs(clf.coef_[0])
        else:
            importances = clf.feature_importances_

        pairs = sorted(
            zip(self.feature_cols, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        return {feat: float(val) for feat, val in pairs}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        """Serialize the fitted model to disk.

        Parameters
        ----------
        path : str
            File path (``.joblib`` or ``.pkl``).
        """
        if self.pipeline_ is None:
            raise RuntimeError("Cannot save an unfitted model.")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "pipeline": self.pipeline_,
            "feature_cols": self.feature_cols,
            "model_type": self.model_type,
            "class_weight": self.class_weight,
            "asset_type": self.asset_type,
            "use_smote": self.use_smote,
            "asset_id_format": self.asset_id_format,
            "val_metrics": self.val_metrics_,
        }
        joblib.dump(payload, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load_model(cls, path: str) -> "WeatherFailurePredictor":
        """Deserialize a previously saved model.

        Parameters
        ----------
        path : str

        Returns
        -------
        WeatherFailurePredictor
        """
        payload = joblib.load(path)
        instance = cls(
            feature_cols=payload["feature_cols"],
            model_type=payload["model_type"],
            class_weight=payload["class_weight"],
            asset_type=payload["asset_type"],
            use_smote=payload.get("use_smote", False),
            asset_id_format=payload.get("asset_id_format", "{type}_{idx}"),
        )
        instance.pipeline_ = payload["pipeline"]
        instance.val_metrics_ = payload.get("val_metrics")
        logger.info("Model loaded from %s", path)
        return instance

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self.pipeline_ is not None else "unfitted"
        return (
            f"WeatherFailurePredictor(model={self.model_type}, "
            f"features={len(self.feature_cols)}, status={status})"
        )
