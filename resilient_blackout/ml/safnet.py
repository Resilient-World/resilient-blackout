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

"""
Multi-modal attribute predictor for grid asset structural properties.

Provides ``SAFNetPredictor`` (Structural Attribute Fusion Network), a
PyTorch-based multi-source data fusion engine that predicts critical
structural attributes of grid assets:

* **First Floor Elevation (FFE)** — continuous regression target in
  metres above local ground.
* **Foundation type** — categorical classification (slab, pier, pile,
  basement-wall, unknown).
* **Basement presence** — binary classification.

The model fuses three heterogeneous input streams:

1. **Geospatial** — elevations extracted from Digital Terrain Models
   (DTM), slope, aspect, and local roughness.
2. **Demographic** — neighbourhood census indicators (median income,
   population density, housing age percentiles).
3. **Asset specifications** — voltage class, installation age,
   structural material, and equipment type.

The predicted FFE values integrate directly with flood depth-damage
models (e.g. ``SubstationFlooder``), replacing uniform default
assumptions with geolocated, predicted structural baselines.

All dependencies (PyTorch, scikit-learn, NumPy, Pandas) are
permissively licensed (BSD / PSF).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FOUNDATION_TYPES: List[str] = ["slab", "pier", "pile", "basement_wall", "unknown"]
_NUM_FOUNDATION_CLASSES: int = len(_FOUNDATION_TYPES)

# Default feature dimensions per input stream
_DEFAULT_GEO_FEATURES: List[str] = [
    "dtm_elevation_m", "slope_deg", "aspect_deg", "roughness_m",
    "dist_to_water_m", "twi",
]
_DEFAULT_DEMO_FEATURES: List[str] = [
    "median_income_usd", "pop_density_per_km2", "pct_built_pre1980",
    "pct_built_post2000", "pct_owner_occupied",
]
_DEFAULT_ASSET_FEATURES: List[str] = [
    "voltage_kv", "age_years", "material_steel", "material_concrete",
    "material_wood", "equipment_type_transformer", "equipment_type_switchgear",
]


# ---------------------------------------------------------------------------
# SAFNet model
# ---------------------------------------------------------------------------


class _SAFNet(nn.Module):
    """Structural Attribute Fusion Network.

    Three-branch encoder with shared fusion trunk and dual prediction
    heads (regression for FFE, classification for foundation/basement).

    Parameters
    ----------
    geo_dim : int
        Number of geospatial input features.
    demo_dim : int
        Number of demographic input features.
    asset_dim : int
        Number of asset specification input features.
    hidden_dim : int
        Hidden dimension for each branch encoder.  Default 64.
    fusion_dim : int
        Dimension of the fused joint embedding.  Default 128.
    dropout : float
        Dropout rate.  Default 0.2.
    """

    def __init__(
        self,
        geo_dim: int,
        demo_dim: int,
        asset_dim: int,
        hidden_dim: int = 64,
        fusion_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Branch encoders
        self.geo_encoder = nn.Sequential(
            nn.Linear(geo_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.demo_encoder = nn.Sequential(
            nn.Linear(demo_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.asset_encoder = nn.Sequential(
            nn.Linear(asset_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        total_encoded = hidden_dim * 3

        # Fusion trunk
        self.fusion = nn.Sequential(
            nn.Linear(total_encoded, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Prediction heads
        self.ffe_head = nn.Linear(fusion_dim // 2, 1)  # regression
        self.foundation_head = nn.Linear(fusion_dim // 2, _NUM_FOUNDATION_CLASSES)
        self.basement_head = nn.Linear(fusion_dim // 2, 1)  # binary classification

    def forward(
        self,
        geo: torch.Tensor,
        demo: torch.Tensor,
        asset: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        geo : torch.Tensor
            Shape ``(batch, geo_dim)``.
        demo : torch.Tensor
            Shape ``(batch, demo_dim)``.
        asset : torch.Tensor
            Shape ``(batch, asset_dim)``.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            ``(ffe_pred, foundation_logits, basement_logits)``.
        """
        g = self.geo_encoder(geo)
        d = self.demo_encoder(demo)
        a = self.asset_encoder(asset)

        fused = torch.cat([g, d, a], dim=1)
        embedding = self.fusion(fused)

        ffe = self.ffe_head(embedding)
        foundation = self.foundation_head(embedding)
        basement = self.basement_head(embedding)

        return ffe, foundation, basement


# ---------------------------------------------------------------------------
# SAFNetPredictor
# ---------------------------------------------------------------------------


class SAFNetPredictor:
    """Multi-modal structural attribute predictor.

    Fuses geospatial, demographic, and asset-specification data to
    predict First Floor Elevation (FFE), foundation type, and basement
    presence for grid assets.

    Parameters
    ----------
    geo_features : list of str or None
        Column names for geospatial features.  Defaults to standard DTM
        derivatives.
    demo_features : list of str or None
        Column names for demographic features.
    asset_features : list of str or None
        Column names for asset specification features.
    hidden_dim : int
        Branch encoder hidden dimension.  Default 64.
    fusion_dim : int
        Fusion embedding dimension.  Default 128.
    dropout : float
        Dropout rate.  Default 0.2.
    device : str or None
        Torch device.  Defaults to ``"cuda"`` if available.

    Attributes
    ----------
    model : _SAFNet or None
    scaler_geo : StandardScaler
    scaler_demo : StandardScaler
    scaler_asset : StandardScaler
    foundation_encoder : LabelEncoder
    ffe_mean : float
    ffe_std : float
    """

    def __init__(
        self,
        geo_features: Optional[List[str]] = None,
        demo_features: Optional[List[str]] = None,
        asset_features: Optional[List[str]] = None,
        hidden_dim: int = 64,
        fusion_dim: int = 128,
        dropout: float = 0.2,
        device: Optional[str] = None,
    ) -> None:
        self.geo_features = geo_features or list(_DEFAULT_GEO_FEATURES)
        self.demo_features = demo_features or list(_DEFAULT_DEMO_FEATURES)
        self.asset_features = asset_features or list(_DEFAULT_ASSET_FEATURES)
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[_SAFNet] = None
        self.scaler_geo = StandardScaler()
        self.scaler_demo = StandardScaler()
        self.scaler_asset = StandardScaler()
        self.foundation_encoder = LabelEncoder()
        self.ffe_mean: float = 0.0
        self.ffe_std: float = 1.0

        self._is_fitted = False

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(
        self,
        df: pd.DataFrame,
        fit_scalers: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract and normalize feature blocks and targets.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain all feature columns plus ``ffe_m``,
            ``foundation_type``, and ``has_basement``.
        fit_scalers : bool
            If ``True``, fit scalers and encoders.  Otherwise use
            existing.

        Returns
        -------
        tuple
            ``(geo, demo, asset, ffe, foundation, basement)`` as
            numpy arrays.
        """
        # Features
        geo_raw = df[self.geo_features].values.astype(np.float64)
        demo_raw = df[self.demo_features].values.astype(np.float64)
        asset_raw = df[self.asset_features].values.astype(np.float64)

        if fit_scalers:
            geo = self.scaler_geo.fit_transform(geo_raw)
            demo = self.scaler_demo.fit_transform(demo_raw)
            asset = self.scaler_asset.fit_transform(asset_raw)
        else:
            geo = self.scaler_geo.transform(geo_raw)
            demo = self.scaler_demo.transform(demo_raw)
            asset = self.scaler_asset.transform(asset_raw)

        # Targets
        ffe_raw = df["ffe_m"].values.astype(np.float64)
        if fit_scalers:
            self.ffe_mean = float(np.mean(ffe_raw))
            self.ffe_std = float(np.std(ffe_raw)) if np.std(ffe_raw) > 1e-6 else 1.0
        ffe = (ffe_raw - self.ffe_mean) / self.ffe_std

        foundation_raw = df["foundation_type"].values.astype(str)
        if fit_scalers:
            foundation = self.foundation_encoder.fit_transform(foundation_raw)
        else:
            foundation = self.foundation_encoder.transform(foundation_raw)

        basement = df["has_basement"].values.astype(np.float32)

        return (
            geo.astype(np.float32),
            demo.astype(np.float32),
            asset.astype(np.float32),
            ffe.astype(np.float32),
            foundation.astype(np.int64),
            basement.astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_safnet_model(
        self,
        training_data: pd.DataFrame,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        val_split: float = 0.2,
        weight_decay: float = 1e-5,
        n_folds: int = 5,
        patience: int = 25,
    ) -> Dict[str, Any]:
        """Train the SAFNet fusion network with k-fold cross-validation.

        Parameters
        ----------
        training_data : pd.DataFrame
            Must contain all feature columns plus ``ffe_m``,
            ``foundation_type``, and ``has_basement``.
        epochs : int
            Maximum training epochs per fold.  Default 200.
        batch_size : int
            Mini-batch size.  Default 64.
        lr : float
            Initial learning rate.  Default 1e-3.
        val_split : float
            Fraction of training data reserved for validation within
            each fold.  Default 0.2.
        weight_decay : float
            L2 regularization strength.  Default 1e-5.
        n_folds : int
            Number of cross-validation folds.  Default 5.
        patience : int
            Early stopping patience in epochs.  Default 25.

        Returns
        -------
        dict
            ``{"model": _SAFNet, "cv_results": list of dict,
            "feature_importance": dict, "history": dict}``.
        """
        df = training_data.copy()
        geo, demo, asset, ffe, foundation, basement = self._prepare_data(df, fit_scalers=True)

        n_samples = len(df)
        geo_dim = len(self.geo_features)
        demo_dim = len(self.demo_features)
        asset_dim = len(self.asset_features)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_results: List[Dict[str, float]] = []
        best_model: Optional[_SAFNet] = None
        best_val_loss = float("inf")

        for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_samples))):
            # Further split train into train/val
            n_train = len(train_idx)
            n_val = int(n_train * val_split)
            rng = np.random.default_rng(fold)
            perm = rng.permutation(train_idx)
            val_idx = perm[:n_val]
            train_idx_fold = perm[n_val:]

            X_geo_train = torch.tensor(geo[train_idx_fold], dtype=torch.float32)
            X_demo_train = torch.tensor(demo[train_idx_fold], dtype=torch.float32)
            X_asset_train = torch.tensor(asset[train_idx_fold], dtype=torch.float32)
            y_ffe_train = torch.tensor(ffe[train_idx_fold], dtype=torch.float32).unsqueeze(1)
            y_found_train = torch.tensor(foundation[train_idx_fold], dtype=torch.long)
            y_base_train = torch.tensor(basement[train_idx_fold], dtype=torch.float32).unsqueeze(1)

            X_geo_val = torch.tensor(geo[val_idx], dtype=torch.float32)
            X_demo_val = torch.tensor(demo[val_idx], dtype=torch.float32)
            X_asset_val = torch.tensor(asset[val_idx], dtype=torch.float32)
            y_ffe_val = torch.tensor(ffe[val_idx], dtype=torch.float32).unsqueeze(1)
            y_found_val = torch.tensor(foundation[val_idx], dtype=torch.long)
            y_base_val = torch.tensor(basement[val_idx], dtype=torch.float32).unsqueeze(1)

            train_ds = TensorDataset(
                X_geo_train, X_demo_train, X_asset_train,
                y_ffe_train, y_found_train, y_base_train,
            )
            val_ds = TensorDataset(
                X_geo_val, X_demo_val, X_asset_val,
                y_ffe_val, y_found_val, y_base_val,
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size)

            model = _SAFNet(
                geo_dim=geo_dim,
                demo_dim=demo_dim,
                asset_dim=asset_dim,
                hidden_dim=self.hidden_dim,
                fusion_dim=self.fusion_dim,
                dropout=self.dropout,
            ).to(self.device)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=patience // 2,
            )

            mse_loss = nn.MSELoss()
            ce_loss = nn.CrossEntropyLoss()
            bce_loss = nn.BCEWithLogitsLoss()

            best_fold_val = float("inf")
            patience_counter = 0

            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                for batch in train_loader:
                    g_b, d_b, a_b, ffe_b, found_b, base_b = (
                        b.to(self.device) for b in batch
                    )
                    optimizer.zero_grad()
                    pred_ffe, pred_found, pred_base = model(g_b, d_b, a_b)

                    loss = (
                        mse_loss(pred_ffe, ffe_b)
                        + ce_loss(pred_found, found_b)
                        + bce_loss(pred_base, base_b)
                    )
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * len(g_b)
                train_loss /= len(train_ds)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        g_b, d_b, a_b, ffe_b, found_b, base_b = (
                            b.to(self.device) for b in batch
                        )
                        pred_ffe, pred_found, pred_base = model(g_b, d_b, a_b)
                        loss = (
                            mse_loss(pred_ffe, ffe_b)
                            + ce_loss(pred_found, found_b)
                            + bce_loss(pred_base, base_b)
                        )
                        val_loss += loss.item() * len(g_b)
                val_loss /= len(val_ds)

                scheduler.step(val_loss)

                if val_loss < best_fold_val - 1e-6:
                    best_fold_val = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

            cv_results.append({
                "fold": fold,
                "train_loss": train_loss,
                "val_loss": best_fold_val,
            })

            if best_fold_val < best_val_loss:
                best_val_loss = best_fold_val
                best_model = model

            logger.info(
                "Fold %d/%d — val_loss=%.4f", fold + 1, n_folds, best_fold_val,
            )

        self.model = best_model
        self._is_fitted = True

        # Feature importance via permutation on the full dataset
        importance = self._compute_feature_importance(
            geo, demo, asset, ffe, foundation, basement,
        )

        return {
            "model": self.model,
            "cv_results": cv_results,
            "feature_importance": importance,
            "history": {"cv_val_losses": [r["val_loss"] for r in cv_results]},
        }

    def _compute_feature_importance(
        self,
        geo: np.ndarray,
        demo: np.ndarray,
        asset: np.ndarray,
        ffe: np.ndarray,
        foundation: np.ndarray,
        basement: np.ndarray,
        n_permutations: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """Permutation-based feature importance.

        Parameters
        ----------
        geo, demo, asset, ffe, foundation, basement : np.ndarray
        n_permutations : int

        Returns
        -------
        dict
            ``{feature_name: {"ffe_importance": float,
            "foundation_importance": float, "basement_importance": float}}``.
        """
        if self.model is None:
            return {}

        self.model.eval()
        device = self.device

        g_t = torch.tensor(geo, dtype=torch.float32, device=device)
        d_t = torch.tensor(demo, dtype=torch.float32, device=device)
        a_t = torch.tensor(asset, dtype=torch.float32, device=device)

        with torch.no_grad():
            base_ffe, base_found, base_base = self.model(g_t, d_t, a_t)
            base_ffe_loss = nn.MSELoss()(base_ffe.squeeze(), torch.tensor(ffe, device=device)).item()
            base_found_loss = nn.CrossEntropyLoss()(
                base_found, torch.tensor(foundation, device=device)
            ).item()
            base_base_loss = nn.BCEWithLogitsLoss()(
                base_base.squeeze(), torch.tensor(basement, device=device)
            ).item()

        importance: Dict[str, Dict[str, float]] = {}
        all_features = self.geo_features + self.demo_features + self.asset_features
        all_data = np.concatenate([geo, demo, asset], axis=1)

        for i, fname in enumerate(all_features):
            ffe_deltas: List[float] = []
            found_deltas: List[float] = []
            base_deltas: List[float] = []

            for _ in range(n_permutations):
                permuted = all_data.copy()
                np.random.shuffle(permuted[:, i])
                p_g = permuted[:, : len(self.geo_features)]
                p_d = permuted[:, len(self.geo_features):len(self.geo_features) + len(self.demo_features)]
                p_a = permuted[:, len(self.geo_features) + len(self.demo_features):]

                pg_t = torch.tensor(p_g, dtype=torch.float32, device=device)
                pd_t = torch.tensor(p_d, dtype=torch.float32, device=device)
                pa_t = torch.tensor(p_a, dtype=torch.float32, device=device)

                with torch.no_grad():
                    p_ffe, p_found, p_base = self.model(pg_t, pd_t, pa_t)
                    ffe_deltas.append(
                        nn.MSELoss()(p_ffe.squeeze(), torch.tensor(ffe, device=device)).item()
                        - base_ffe_loss
                    )
                    found_deltas.append(
                        nn.CrossEntropyLoss()(p_found, torch.tensor(foundation, device=device)).item()
                        - base_found_loss
                    )
                    base_deltas.append(
                        nn.BCEWithLogitsLoss()(p_base.squeeze(), torch.tensor(basement, device=device)).item()
                        - base_base_loss
                    )

            importance[fname] = {
                "ffe_importance": float(np.mean(ffe_deltas)),
                "foundation_importance": float(np.mean(found_deltas)),
                "basement_importance": float(np.mean(base_deltas)),
            }

        return importance

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Predict structural attributes for new assets.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain all configured feature columns.

        Returns
        -------
        pd.DataFrame
            Original data augmented with columns:

            - ``predicted_ffe_m`` — predicted FFE in metres.
            - ``predicted_foundation_type`` — predicted foundation class.
            - ``predicted_has_basement`` — basement probability.
            - ``foundation_confidence`` — softmax confidence.
        """
        if self.model is None or not self._is_fitted:
            raise RuntimeError("Model not trained. Call train_safnet_model() first.")

        geo, demo, asset, _, _, _ = self._prepare_data(data, fit_scalers=False)

        g_t = torch.tensor(geo, dtype=torch.float32, device=self.device)
        d_t = torch.tensor(demo, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(asset, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            pred_ffe, pred_found, pred_base = self.model(g_t, d_t, a_t)

        ffe_norm = pred_ffe.squeeze().cpu().numpy()
        ffe_m = ffe_norm * self.ffe_std + self.ffe_mean

        found_logits = pred_found.cpu().numpy()
        found_probs = np.exp(found_logits) / np.exp(found_logits).sum(axis=1, keepdims=True)
        found_idx = np.argmax(found_probs, axis=1)
        found_labels = self.foundation_encoder.inverse_transform(found_idx)
        found_conf = np.max(found_probs, axis=1)

        base_logits = pred_base.squeeze().cpu().numpy()
        base_prob = 1.0 / (1.0 + np.exp(-base_logits))

        result = data.copy()
        result["predicted_ffe_m"] = ffe_m
        result["predicted_foundation_type"] = found_labels
        result["foundation_confidence"] = found_conf
        result["predicted_has_basement_prob"] = base_prob
        result["predicted_has_basement"] = (base_prob >= 0.5).astype(int)

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model, scalers, and metadata to disk.

        Parameters
        ----------
        path : str
            File path (``.pt`` extension recommended).
        """
        import os

        if self.model is None:
            raise RuntimeError("No model to save.")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        payload = {
            "model_state": self.model.state_dict(),
            "geo_features": self.geo_features,
            "demo_features": self.demo_features,
            "asset_features": self.asset_features,
            "hidden_dim": self.hidden_dim,
            "fusion_dim": self.fusion_dim,
            "dropout": self.dropout,
            "scaler_geo_mean": self.scaler_geo.mean_.tolist(),
            "scaler_geo_scale": self.scaler_geo.scale_.tolist(),
            "scaler_demo_mean": self.scaler_demo.mean_.tolist(),
            "scaler_demo_scale": self.scaler_demo.scale_.tolist(),
            "scaler_asset_mean": self.scaler_asset.mean_.tolist(),
            "scaler_asset_scale": self.scaler_asset.scale_.tolist(),
            "foundation_classes": self.foundation_encoder.classes_.tolist(),
            "ffe_mean": self.ffe_mean,
            "ffe_std": self.ffe_std,
        }
        torch.save(payload, path)
        logger.info("SAFNetPredictor saved to %s", path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "SAFNetPredictor":
        """Load a previously saved predictor.

        Parameters
        ----------
        path : str
        device : str or None

        Returns
        -------
        SAFNetPredictor
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)

        predictor = cls(
            geo_features=payload["geo_features"],
            demo_features=payload["demo_features"],
            asset_features=payload["asset_features"],
            hidden_dim=payload["hidden_dim"],
            fusion_dim=payload["fusion_dim"],
            dropout=payload["dropout"],
            device=device,
        )

        predictor.scaler_geo.mean_ = np.array(payload["scaler_geo_mean"])
        predictor.scaler_geo.scale_ = np.array(payload["scaler_geo_scale"])
        predictor.scaler_demo.mean_ = np.array(payload["scaler_demo_mean"])
        predictor.scaler_demo.scale_ = np.array(payload["scaler_demo_scale"])
        predictor.scaler_asset.mean_ = np.array(payload["scaler_asset_mean"])
        predictor.scaler_asset.scale_ = np.array(payload["scaler_asset_scale"])
        predictor.foundation_encoder.classes_ = np.array(payload["foundation_classes"])
        predictor.ffe_mean = payload["ffe_mean"]
        predictor.ffe_std = payload["ffe_std"]

        geo_dim = len(predictor.geo_features)
        demo_dim = len(predictor.demo_features)
        asset_dim = len(predictor.asset_features)

        model = _SAFNet(
            geo_dim=geo_dim,
            demo_dim=demo_dim,
            asset_dim=asset_dim,
            hidden_dim=predictor.hidden_dim,
            fusion_dim=predictor.fusion_dim,
            dropout=predictor.dropout,
        )
        model.load_state_dict(payload["model_state"])
        model.to(predictor.device)
        model.eval()
        predictor.model = model
        predictor._is_fitted = True

        logger.info("SAFNetPredictor loaded from %s", path)
        return predictor

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        state = "fitted" if self._is_fitted else "unfitted"
        return (
            f"SAFNetPredictor(geo={len(self.geo_features)}, "
            f"demo={len(self.demo_features)}, "
            f"asset={len(self.asset_features)}, "
            f"state={state})"
        )
