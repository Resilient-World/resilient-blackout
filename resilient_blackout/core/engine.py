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
Baseline impact evaluation engine.

Provides the ``ImpactEngine`` class that orchestrates hazard-to-asset
mapping, damage computation, and financial loss accumulation for a
``RiskScenario``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import geopandas as gpd
import pandas as pd

from resilient_blackout.core.base import Asset, HazardEvent, RiskScenario


class ImpactEngine:
    """Orchestrates physical risk evaluation for a ``RiskScenario``.

    The engine performs spatial matching between hazard-event centroids
    and asset geometries, applies vulnerability (damage) functions, and
    accumulates financial losses across all assets and events.

    This is a baseline implementation with placeholder methods; concrete
    hazard-to-asset mapping and damage-function logic should be provided
    by subclasses or injected callables in production use.
    """

    def __init__(self) -> None:
        self._last_result: pd.DataFrame | None = None

    @property
    def last_result(self) -> pd.DataFrame | None:
        """The DataFrame produced by the most recent call to :meth:`evaluate`."""
        return self._last_result

    def map_hazards_to_assets(self, scenario: RiskScenario) -> Dict[str, List[Tuple[Asset, float]]]:
        """Spatially join hazard centroids to asset geometries.

        For each hazard event, performs a nearest-neighbour or
        intersection-based spatial join between the event's centroid
        ``GeoDataFrame`` and the scenario's asset geometries.  Returns a
        mapping from event ID to a list of ``(asset, intensity)`` pairs.

        Parameters
        ----------
        scenario : RiskScenario
            The scenario containing assets and hazard events to join.

        Returns
        -------
        dict
            Keys are ``HazardEvent.event_id`` strings; values are lists
            of ``(Asset, float)`` tuples where the float is the hazard
            intensity at the asset location.

        Notes
        -----
        The current implementation is a placeholder that assigns every
        asset the maximum intensity found across all hazard centroids.
        Subclasses should override this method with a proper spatial
        join (e.g., ``gpd.sjoin_nearest`` or raster sampling).
        """
        mapping: Dict[str, List[Tuple[Asset, float]]] = {}

        for hazard in scenario.hazards:
            max_intensity: float = float(hazard.centroids["intensity"].max())
            pairs: List[Tuple[Asset, float]] = [
                (asset, max_intensity) for asset in scenario.assets
            ]
            mapping[hazard.event_id] = pairs

        return mapping

    def compute_asset_loss(
        self,
        asset: Asset,
        hazard_intensity: float,
    ) -> float:
        """Compute the financial loss for a single asset given a hazard intensity.

        Applies a damage (vulnerability) function to convert hazard
        intensity into a mean damage ratio, then multiplies by the
        asset's replacement value.

        Parameters
        ----------
        asset : Asset
            The exposed asset.
        hazard_intensity : float
            Hazard intensity value at the asset location (units depend
            on the hazard type).

        Returns
        -------
        float
            Estimated loss in USD.

        Notes
        -----
        This placeholder returns zero for all inputs.  In production,
        this method should look up ``asset.impact_function_id`` to
        select the appropriate vulnerability curve and interpolate the
        mean damage ratio.
        """
        _ = asset, hazard_intensity
        return 0.0

    def evaluate(self, scenario: RiskScenario) -> pd.DataFrame:
        """Run the full evaluation pipeline for a risk scenario.

        Executes three steps:
        1. Map hazard centroids to assets via :meth:`map_hazards_to_assets`.
        2. Compute per-asset losses via :meth:`compute_asset_loss`.
        3. Aggregate results into a structured ``DataFrame``.

        Parameters
        ----------
        scenario : RiskScenario
            The scenario to evaluate.

        Returns
        -------
        pandas.DataFrame
            A table with columns: ``event_id``, ``asset_id``,
            ``hazard_intensity``, ``loss_usd``, ``hazard_type``,
            ``frequency``.
        """
        hazard_asset_map = self.map_hazards_to_assets(scenario)

        rows: List[Dict[str, Any]] = []
        for hazard in scenario.hazards:
            pairs = hazard_asset_map.get(hazard.event_id, [])
            for asset, intensity in pairs:
                loss = self.compute_asset_loss(asset, intensity)
                rows.append(
                    {
                        "event_id": hazard.event_id,
                        "asset_id": asset.asset_id,
                        "hazard_intensity": intensity,
                        "loss_usd": loss,
                        "hazard_type": hazard.hazard_type,
                        "frequency": hazard.frequency,
                    }
                )

        result = pd.DataFrame(rows)
        if result.empty:
            result = pd.DataFrame(
                columns=[
                    "event_id",
                    "asset_id",
                    "hazard_intensity",
                    "loss_usd",
                    "hazard_type",
                    "frequency",
                ]
            )

        self._last_result = result
        return result
