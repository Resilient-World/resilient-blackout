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
Geospatial hazard-to-asset mapping utilities.

Provides vectorized spatial-join functions that match asset locations
to the nearest hazard-footprint centroid and return interpolated
intensity values.
"""

from __future__ import annotations

from typing import Dict, List

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from resilient_blackout.core.base import Asset, HazardEvent


def _assets_to_geodataframe(assets: List[Asset]) -> gpd.GeoDataFrame:
    """Convert a list of ``Asset`` objects into a ``GeoDataFrame``.

    Parameters
    ----------
    assets : list of Asset
        Assets to convert.

    Returns
    -------
    gpd.GeoDataFrame
        With columns ``asset_id`` and ``geometry``.
    """
    records = [
        {"asset_id": a.asset_id, "geometry": a.geom}
        for a in assets
    ]
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return gdf


def map_hazard_to_assets(
    assets: List[Asset],
    hazard: HazardEvent,
) -> Dict[str, float]:
    """Map each asset to the nearest hazard centroid and return its intensity.

    Uses ``geopandas.sjoin_nearest`` for a single-pass vectorized spatial
    join.  Each asset is matched to the closest centroid in the hazard
    footprint, and the corresponding ``intensity`` value is returned.

    Parameters
    ----------
    assets : list of Asset
        Exposed infrastructure elements.
    hazard : HazardEvent
        A hazard event whose ``centroids`` GeoDataFrame contains
        ``geometry`` and ``intensity`` columns.

    Returns
    -------
    dict
        Mapping from ``asset_id`` to the hazard intensity at that
        asset's location.  Assets that cannot be matched (e.g., no
        centroids within a reasonable distance) are omitted.

    Notes
    -----
    Both the asset geometries and hazard centroids are assumed to be in
    a projected CRS (metres) for accurate distance calculations.  If the
    hazard centroids use a geographic CRS (degrees), the caller should
    reproject before calling this function.
    """
    if not assets:
        return {}

    asset_gdf = _assets_to_geodataframe(assets)
    hazard_gdf = hazard.centroids.copy()

    if hazard_gdf.crs is None:
        hazard_gdf = hazard_gdf.set_crs("EPSG:4326")

    if asset_gdf.crs != hazard_gdf.crs:
        asset_gdf = asset_gdf.to_crs(hazard_gdf.crs)

    joined: gpd.GeoDataFrame = gpd.sjoin_nearest(
        asset_gdf,
        hazard_gdf[["geometry", "intensity"]],
        how="left",
        distance_col="distance_m",
    )

    result: Dict[str, float] = {}
    for _, row in joined.iterrows():
        aid: str = row["asset_id"]
        intensity = row.get("intensity")
        if pd.notna(intensity):
            result[aid] = float(intensity)

    return result
