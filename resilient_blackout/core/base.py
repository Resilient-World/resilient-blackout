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
Core domain classes for grid physical risk modeling.

Defines the primary data structures — Asset, HazardEvent, and RiskScenario —
that form the backbone of the clean-room risk evaluation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import geopandas as gpd
import pandapower as pp
from shapely.geometry import LineString, Point

GeomType = Union[Point, LineString]


@dataclass
class Asset:
    """A geolocated exposure element within the electrical grid.

    Represents a single infrastructure component — such as a substation,
    transmission tower, or line segment — that may be exposed to natural
    hazards.  Each asset carries a replacement value and an optional
    reference to a vulnerability (impact) function.

    Attributes
    ----------
    asset_id : str
        Unique identifier for the asset.
    name : str
        Human-readable label.
    geom : Point or LineString
        Shapely geometry representing the asset's spatial footprint.
        Use ``Point`` for nodes (substations, towers) and ``LineString``
        for linear features (transmission corridors).
    value_usd : float
        Replacement or insured value in US dollars.
    original_properties : dict
        Arbitrary key-value metadata carried through from the source
        dataset (e.g., voltage level, material, age).
    impact_function_id : str or None
        Identifier linking this asset to a vulnerability curve.
        ``None`` indicates no damage function is assigned.
    """

    asset_id: str
    name: str
    geom: GeomType
    value_usd: float
    original_properties: Dict[str, Any] = field(default_factory=dict)
    impact_function_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.geom, (Point, LineString)):
            raise TypeError(
                f"Asset.geom must be a Point or LineString, got {type(self.geom).__name__}"
            )
        if self.value_usd < 0:
            raise ValueError(f"Asset.value_usd must be non-negative, got {self.value_usd}")


@dataclass
class HazardEvent:
    """A spatial representation of a single natural-hazard occurrence.

    Hazard events are defined by a set of geographic centroids, each
    carrying an intensity value (e.g., wind speed in m/s, flood depth in
    metres).  The event also records its annual frequency of occurrence
    and the physical units of the intensity measure.

    Attributes
    ----------
    event_id : str
        Unique identifier for the event.
    name : str
        Human-readable label (e.g., "Cyclone Freddy").
    hazard_type : str
        Short code for the peril type.  Common values: ``"TC"`` (tropical
        cyclone), ``"WF"`` (windstorm / extratropical), ``"EQ"``
        (earthquake), ``"FL"`` (flood).
    frequency : float
        Annual rate of occurrence (events per year).  Used for
        annualised loss calculations.
    centroids : geopandas.GeoDataFrame
        Spatial point locations with at minimum a ``geometry`` column
        (Point) and an ``intensity`` column holding the hazard magnitude.
    units : str
        Physical units of the intensity values (e.g., ``"m/s"``,
        ``"m"``, ``"g"``).
    """

    event_id: str
    name: str
    hazard_type: str
    frequency: float
    centroids: gpd.GeoDataFrame
    units: str

    _REQUIRED_COLUMNS: frozenset[str] = field(
        default=frozenset({"geometry", "intensity"}), init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.frequency <= 0:
            raise ValueError(
                f"HazardEvent.frequency must be positive, got {self.frequency}"
            )
        missing = self._REQUIRED_COLUMNS - set(self.centroids.columns)
        if missing:
            raise ValueError(
                f"HazardEvent.centroids is missing required columns: {missing}"
            )
        if not isinstance(self.centroids, gpd.GeoDataFrame):
            raise TypeError(
                f"HazardEvent.centroids must be a GeoDataFrame, got {type(self.centroids).__name__}"
            )


class RiskScenario:
    """An evaluation container coupling a grid network, assets, and hazards.

    A ``RiskScenario`` bundles the three inputs required to run a
    physical risk assessment: the electrical network topology, the set
    of exposed assets, and the collection of hazard events to evaluate
    against.

    Parameters
    ----------
    grid : pandapower.auxiliary.pandapowerNet
        A pandapower network representing the electrical grid.
    assets : list of Asset
        Exposed infrastructure elements.
    hazards : list of HazardEvent
        Natural-hazard events to simulate.

    Attributes
    ----------
    grid : pandapowerNet
    assets : list of Asset
    hazards : list of HazardEvent
    """

    def __init__(
        self,
        grid: pp.auxiliary.pandapowerNet,
        assets: List[Asset],
        hazards: List[HazardEvent],
    ) -> None:
        if not assets:
            raise ValueError("RiskScenario.assets must not be empty")
        if not hazards:
            raise ValueError("RiskScenario.hazards must not be empty")

        self.grid: pp.auxiliary.pandapowerNet = grid
        self.assets: List[Asset] = assets
        self.hazards: List[HazardEvent] = hazards

    def __repr__(self) -> str:
        return (
            f"RiskScenario(n_assets={len(self.assets)}, "
            f"n_hazards={len(self.hazards)}, "
            f"grid_buses={len(self.grid.bus)})"
        )
