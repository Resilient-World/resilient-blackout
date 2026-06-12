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
Spatial repair crew dispatch and routing optimizer.

Provides ``MultiCrewRestorationRouter``, a heuristic CVRP-TW solver that
dispatches multiple repair crews to restore damaged grid assets.  The
router respects skill-matching and material-capacity constraints, avoids
blocked road edges, and uses greedy nearest-neighbour construction
followed by 2-opt local search.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RepairCrew:
    """A repair crew with depot, speed, skills, and material capacity.

    Attributes
    ----------
    crew_id : str
        Unique identifier.
    depot_node : str or int
        Home depot node in the road network.
    speed_kmh : float
        Travel speed in km/h.
    skills : set of str
        Repair specialisations (e.g. ``{"Vegetation clearing"}``).
    material_capacity : dict of str -> float
        Available repair materials and their quantities.
    """

    crew_id: str
    depot_node: Any
    speed_kmh: float = 40.0
    skills: set = field(default_factory=set)
    material_capacity: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.speed_kmh <= 0:
            raise ValueError(f"speed_kmh must be positive, got {self.speed_kmh}")


@dataclass
class DamagedAsset:
    """A damaged grid asset requiring repair.

    Attributes
    ----------
    asset_id : str
        Unique identifier.
    node : str or int
        Road-network node closest to the asset.
    repair_type : str
        Required skill (e.g. ``"Transformer replacement"``).
    required_materials : dict of str -> float
        Materials needed for the repair.
    repair_duration_h : float
        Repair time in hours.
    failure_time_h : float
        Hour when the asset failed (used for penalty calculation).
    """

    asset_id: str
    node: Any
    repair_type: str
    required_materials: Dict[str, float] = field(default_factory=dict)
    repair_duration_h: float = 2.0
    failure_time_h: float = 0.0


# ---------------------------------------------------------------------------
# Route container
# ---------------------------------------------------------------------------


@dataclass
class _Route:
    """Internal route representation for a single crew."""

    crew_id: str
    sequence: List[Any]  # node sequence including depot
    asset_ids: List[str]
    arrival_times: List[float]  # hours at each stop
    travel_time_h: float = 0.0
    total_repair_h: float = 0.0


# ---------------------------------------------------------------------------
# MultiCrewRestorationRouter
# ---------------------------------------------------------------------------


class MultiCrewRestorationRouter:
    """Heuristic CVRP-TW solver for grid restoration crew dispatch.

    Parameters
    ----------
    road_graph : nx.DiGraph or nx.Graph
        Transportation network.  Edges must have ``travel_time_min``
        (float) and ``status`` (``"passable"`` or ``"blocked"``).
    crews : list of RepairCrew
    penalty_theta : float
        Weight on restoration delay penalty
        :math:`\\theta \\sum (T_{\\text{restored}} - T_{\\text{failed}})`.
        Default 1.0.

    Attributes
    ----------
    road_graph : nx.Graph
    crews : list of RepairCrew
    penalty_theta : float
    result_ : dict or None
        Populated after :meth:`solve`.
    """

    def __init__(
        self,
        road_graph: nx.Graph,
        crews: List[RepairCrew],
        penalty_theta: float = 1.0,
    ) -> None:
        if not crews:
            raise ValueError("crews list must not be empty")
        if penalty_theta < 0:
            raise ValueError(
                f"penalty_theta must be non-negative, got {penalty_theta}"
            )

        self.road_graph = road_graph
        self.crews = list(crews)
        self.penalty_theta = float(penalty_theta)
        self.result_: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Edge / path helpers
    # ------------------------------------------------------------------

    def _passable_subgraph(self) -> nx.Graph:
        """Return the road subgraph with only passable edges."""
        edges = [
            (u, v)
            for u, v, data in self.road_graph.edges(data=True)
            if data.get("status", "passable") != "blocked"
        ]
        if isinstance(self.road_graph, nx.DiGraph):
            return self.road_graph.edge_subgraph(edges).copy()
        return self.road_graph.edge_subgraph(edges).copy()

    def _travel_time(
        self,
        origin: Any,
        destination: Any,
        speed_kmh: float,
    ) -> float:
        """Compute travel time between two nodes in hours.

        Uses the shortest path on the passable subgraph, scaled by
        crew speed if the graph edge weights are in distance units.
        If ``travel_time_min`` is present on edges, uses it directly.
        """
        sub = self._passable_subgraph()
        if origin not in sub or destination not in sub:
            return float("inf")

        try:
            if any(
                "travel_time_min" in d for _, _, d in sub.edges(data=True)
            ):
                # Use precomputed travel time directly
                path = nx.shortest_path(
                    sub, origin, destination, weight="travel_time_min"
                )
                total_min = 0.0
                for u, v in zip(path[:-1], path[1:]):
                    total_min += sub[u][v].get("travel_time_min", 0.0)
                return total_min / 60.0
            else:
                # Use distance / speed
                dist_km = nx.shortest_path_length(
                    sub, origin, destination, weight="length_km"
                )
                return dist_km / max(speed_kmh, _EPS)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")

    def _can_visit(
        self,
        crew: RepairCrew,
        asset: DamagedAsset,
        remaining_materials: Dict[str, float],
    ) -> bool:
        """Check if a crew can service an asset.

        Requires skill match and sufficient remaining materials.
        """
        if asset.repair_type not in crew.skills:
            return False
        for mat, qty in asset.required_materials.items():
            if remaining_materials.get(mat, 0.0) < qty - _EPS:
                return False
        return True

    # ------------------------------------------------------------------
    # Distance matrix
    # ------------------------------------------------------------------

    def _build_distance_matrix(
        self,
        assets: List[DamagedAsset],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build travel-time matrices from depots and between assets.

        Returns
        -------
        depot_to_asset : np.ndarray
            Shape ``(n_crews, n_assets)`` in hours.
        asset_to_asset : np.ndarray
            Shape ``(n_assets, n_assets)`` in hours.
        """
        n_crews = len(self.crews)
        n_assets = len(assets)

        depot_to_asset = np.full((n_crews, n_assets), float("inf"))
        for c_idx, crew in enumerate(self.crews):
            for a_idx, asset in enumerate(assets):
                depot_to_asset[c_idx, a_idx] = self._travel_time(
                    crew.depot_node, asset.node, crew.speed_kmh
                )

        asset_to_asset = np.full((n_assets, n_assets), float("inf"))
        for i, a_i in enumerate(assets):
            for j, a_j in enumerate(assets):
                if i != j:
                    asset_to_asset[i, j] = self._travel_time(
                        a_i.node, a_j.node, 40.0  # nominal speed
                    )

        return depot_to_asset, asset_to_asset

    # ------------------------------------------------------------------
    # Greedy nearest-neighbor construction
    # ------------------------------------------------------------------

    def _greedy_nearest_neighbor(
        self,
        assets: List[DamagedAsset],
    ) -> List[_Route]:
        """Construct initial routes via greedy nearest-neighbour.

        Each crew starts at its depot and iteratively visits the
        closest feasible unassigned asset until no more assets can
        be reached.

        Parameters
        ----------
        assets : list of DamagedAsset

        Returns
        -------
        list of _Route
        """
        n_assets = len(assets)
        if n_assets == 0:
            return []

        depot_to_asset, asset_to_asset = self._build_distance_matrix(assets)

        assigned = set()
        routes: List[_Route] = []

        for c_idx, crew in enumerate(self.crews):
            remaining_materials = dict(crew.material_capacity)
            sequence = [crew.depot_node]
            asset_ids: List[str] = []
            arrival_times: List[float] = [0.0]
            current_time = 0.0
            current_node = crew.depot_node
            current_idx = -1  # depot index

            while len(assigned) < n_assets:
                best_idx = -1
                best_cost = float("inf")

                for a_idx in range(n_assets):
                    if a_idx in assigned:
                        continue
                    asset = assets[a_idx]
                    if not self._can_visit(crew, asset, remaining_materials):
                        continue

                    if current_idx == -1:
                        travel = depot_to_asset[c_idx, a_idx]
                    else:
                        travel = asset_to_asset[current_idx, a_idx]

                    if travel == float("inf"):
                        continue

                    # Objective: travel time + penalty for delay
                    arrival = current_time + travel
                    delay_penalty = self.penalty_theta * max(
                        0.0, arrival - asset.failure_time_h
                    )
                    cost = travel + delay_penalty

                    if cost < best_cost:
                        best_cost = cost
                        best_idx = a_idx

                if best_idx == -1:
                    break

                asset = assets[best_idx]
                if current_idx == -1:
                    travel = depot_to_asset[c_idx, best_idx]
                else:
                    travel = asset_to_asset[current_idx, best_idx]

                current_time += travel + asset.repair_duration_h
                current_node = asset.node
                current_idx = best_idx

                sequence.append(current_node)
                asset_ids.append(asset.asset_id)
                arrival_times.append(current_time - asset.repair_duration_h)
                assigned.add(best_idx)

                for mat, qty in asset.required_materials.items():
                    remaining_materials[mat] = remaining_materials.get(mat, 0.0) - qty

            # Compute total travel time
            total_travel = 0.0
            for i in range(len(sequence) - 1):
                if i == 0:
                    total_travel += depot_to_asset[c_idx, assets.index(next(
                        a for a in assets if a.node == sequence[1]
                    ))]
                else:
                    a_i = next(a for a in assets if a.node == sequence[i])
                    a_j = next(a for a in assets if a.node == sequence[i + 1])
                    idx_i = assets.index(a_i)
                    idx_j = assets.index(a_j)
                    total_travel += asset_to_asset[idx_i, idx_j]

            total_repair = sum(
                a.repair_duration_h for a in assets if a.asset_id in asset_ids
            )

            routes.append(
                _Route(
                    crew_id=crew.crew_id,
                    sequence=sequence,
                    asset_ids=asset_ids,
                    arrival_times=arrival_times,
                    travel_time_h=total_travel,
                    total_repair_h=total_repair,
                )
            )

        return routes

    # ------------------------------------------------------------------
    # 2-opt local search
    # ------------------------------------------------------------------

    def _route_cost(
        self,
        route: _Route,
        assets: List[DamagedAsset],
        asset_to_asset: np.ndarray,
        depot_to_asset: np.ndarray,
        c_idx: int,
    ) -> float:
        """Compute total cost (travel + penalty) of a route."""
        if len(route.sequence) <= 1:
            return 0.0

        cost = 0.0
        current_time = 0.0

        for i in range(len(route.sequence) - 1):
            if i == 0:
                a_j = next(a for a in assets if a.node == route.sequence[1])
                j_idx = assets.index(a_j)
                travel = depot_to_asset[c_idx, j_idx]
            else:
                a_i = next(a for a in assets if a.node == route.sequence[i])
                a_j = next(a for a in assets if a.node == route.sequence[i + 1])
                idx_i = assets.index(a_i)
                idx_j = assets.index(a_j)
                travel = asset_to_asset[idx_i, idx_j]

            current_time += travel
            if i > 0:
                asset = next(a for a in assets if a.node == route.sequence[i])
                cost += self.penalty_theta * max(
                    0.0, current_time - asset.failure_time_h
                )
                current_time += asset.repair_duration_h
            cost += travel

        return cost

    def _two_opt(
        self,
        route: _Route,
        assets: List[DamagedAsset],
        asset_to_asset: np.ndarray,
        depot_to_asset: np.ndarray,
        c_idx: int,
    ) -> _Route:
        """Apply 2-opt local search to a single route.

        Iteratively reverses segments to reduce total travel cost.

        Parameters
        ----------
        route : _Route
        assets : list of DamagedAsset
        asset_to_asset : np.ndarray
        depot_to_asset : np.ndarray
        c_idx : int
            Crew index.

        Returns
        -------
        _Route
            Improved (or unchanged) route.
        """
        if len(route.sequence) <= 2:
            return route

        best_cost = self._route_cost(route, assets, asset_to_asset, depot_to_asset, c_idx)
        improved = True
        max_iter = 100
        iteration = 0

        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            n = len(route.sequence)

            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    new_sequence = (
                        route.sequence[:i]
                        + route.sequence[i:j][::-1]
                        + route.sequence[j:]
                    )
                    new_asset_ids = [
                        next(a.asset_id for a in assets if a.node == node)
                        for node in new_sequence[1:]
                    ]
                    new_arrival_times = self._compute_arrival_times(
                        new_sequence, assets, depot_to_asset, asset_to_asset, c_idx
                    )

                    new_route = _Route(
                        crew_id=route.crew_id,
                        sequence=new_sequence,
                        asset_ids=new_asset_ids,
                        arrival_times=new_arrival_times,
                        travel_time_h=0.0,
                        total_repair_h=route.total_repair_h,
                    )

                    new_cost = self._route_cost(
                        new_route, assets, asset_to_asset, depot_to_asset, c_idx
                    )

                    if new_cost < best_cost - _EPS:
                        route = new_route
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break

        return route

    def _compute_arrival_times(
        self,
        sequence: List[Any],
        assets: List[DamagedAsset],
        depot_to_asset: np.ndarray,
        asset_to_asset: np.ndarray,
        c_idx: int,
    ) -> List[float]:
        """Compute arrival times for a node sequence."""
        if len(sequence) <= 1:
            return [0.0]

        arrival_times = [0.0]
        current_time = 0.0

        for i in range(1, len(sequence)):
            if i == 1:
                a_j = next(a for a in assets if a.node == sequence[1])
                j_idx = assets.index(a_j)
                travel = depot_to_asset[c_idx, j_idx]
            else:
                a_i = next(a for a in assets if a.node == sequence[i - 1])
                a_j = next(a for a in assets if a.node == sequence[i])
                idx_i = assets.index(a_i)
                idx_j = assets.index(a_j)
                travel = asset_to_asset[idx_i, idx_j]

            current_time += travel
            arrival_times.append(current_time)
            asset = next(a for a in assets if a.node == sequence[i])
            current_time += asset.repair_duration_h

        return arrival_times

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        assets: List[DamagedAsset],
        max_iterations: int = 50,
    ) -> Dict[str, Any]:
        """Solve the crew dispatch problem.

        Parameters
        ----------
        assets : list of DamagedAsset
            Damaged assets to repair.
        max_iterations : int
            Maximum 2-opt iterations.  Default 50.

        Returns
        -------
        dict
            Keys:

            - ``routes`` (list of dict) — per-crew route details.
            - ``total_travel_time_h`` (float).
            - ``total_penalty_h`` (float) — sum of restoration delays.
            - ``unassigned_assets`` (list of str) — assets that could
              not be reached by any crew.
            - ``restoration_schedule`` (dict) — asset_id →
              restored_hour.
        """
        if not assets:
            return {
                "routes": [],
                "total_travel_time_h": 0.0,
                "total_penalty_h": 0.0,
                "unassigned_assets": [],
                "restoration_schedule": {},
            }

        # Greedy construction
        routes = self._greedy_nearest_neighbor(assets)

        # 2-opt improvement
        depot_to_asset, asset_to_asset = self._build_distance_matrix(assets)
        improved_routes: List[_Route] = []
        for c_idx, route in enumerate(routes):
            improved = self._two_opt(
                route, assets, asset_to_asset, depot_to_asset, c_idx
            )
            improved_routes.append(improved)

        routes = improved_routes

        # Compute metrics
        total_travel = sum(r.travel_time_h for r in routes)
        total_penalty = 0.0
        restoration_schedule: Dict[str, float] = {}

        assigned_assets = set()
        for route in routes:
            for i, asset_id in enumerate(route.asset_ids):
                assigned_assets.add(asset_id)
                arrival = route.arrival_times[i + 1]
                asset = next(a for a in assets if a.asset_id == asset_id)
                restored = arrival + asset.repair_duration_h
                restoration_schedule[asset_id] = restored
                total_penalty += self.penalty_theta * max(
                    0.0, arrival - asset.failure_time_h
                )

        unassigned = [a.asset_id for a in assets if a.asset_id not in assigned_assets]

        route_dicts = [
            {
                "crew_id": r.crew_id,
                "sequence": r.sequence,
                "asset_ids": r.asset_ids,
                "arrival_times_h": r.arrival_times,
                "travel_time_h": r.travel_time_h,
                "total_repair_h": r.total_repair_h,
            }
            for r in routes
        ]

        self.result_ = {
            "routes": route_dicts,
            "total_travel_time_h": total_travel,
            "total_penalty_h": total_penalty,
            "unassigned_assets": unassigned,
            "restoration_schedule": restoration_schedule,
        }
        return self.result_

    # ------------------------------------------------------------------
    # Hourly restoration state
    # ------------------------------------------------------------------

    def restoration_timeseries(
        self,
        assets: List[DamagedAsset],
        max_hour: int = 24,
    ) -> pd.DataFrame:
        """Return hourly restoration state of physical assets.

        Parameters
        ----------
        assets : list of DamagedAsset
        max_hour : int
            Maximum simulation hour.  Default 24.

        Returns
        -------
        pd.DataFrame
            Rows = hours, columns = asset_ids, values = boolean
            (``True`` = operational).

        Raises
        ------
        RuntimeError
            If :meth:`solve` has not been called.
        """
        if self.result_ is None:
            raise RuntimeError("Call solve() before restoration_timeseries().")

        hours = np.arange(max_hour + 1)
        records: List[Dict[str, Any]] = []

        for h in hours:
            row: Dict[str, Any] = {"hour": int(h)}
            for asset in assets:
                restored_at = self.result_["restoration_schedule"].get(
                    asset.asset_id, float("inf")
                )
                row[asset.asset_id] = h >= restored_at
            records.append(row)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MultiCrewRestorationRouter(crews={len(self.crews)}, "
            f"theta={self.penalty_theta})"
        )
