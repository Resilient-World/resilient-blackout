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
Dynamic spatial routing and crew restoration engine.

Provides ``CrewRestorationRouter`` for simulating post-event
restoration times based on geospatial network connectivity, hazard
exclusion zones, and shortest-path crew dispatch routing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from resilient_blackout.core.base import Asset

logger = logging.getLogger(__name__)

_ISOLATION_PENALTY: float = 10.0
_EPS: float = 1e-10


class CrewRestorationRouter:
    """Geospatial crew dispatch and restoration time calculator.

    Builds a road network graph, marks segments blocked by hazard
    zones, computes shortest passable routes from depots to failed
    assets, and dynamically adjusts restoration durations.

    Parameters
    ----------
    depots : list of tuple
        ``[(name, Point), ...]`` — maintenance depot locations.
    travel_speed_kmh : float
        Average crew travel speed in km/h.  Default 40.
    base_restore_hours : float
        Baseline repair time in hours.  Default 4.0.
    repair_complexity_hours : float
        Fixed per-asset repair time adder.  Default 2.0.

    Attributes
    ----------
    depots : list
    travel_speed_kmh : float
    base_restore_hours : float
    repair_complexity_hours : float
    graph : nx.Graph or None
    """

    def __init__(
        self,
        depots: List[Tuple[str, Point]],
        travel_speed_kmh: float = 40.0,
        base_restore_hours: float = 4.0,
        repair_complexity_hours: float = 2.0,
    ) -> None:
        if travel_speed_kmh <= 0:
            raise ValueError("travel_speed_kmh must be positive")
        if base_restore_hours < 0:
            raise ValueError("base_restore_hours must be non-negative")

        self.depots = depots
        self.travel_speed_kmh = travel_speed_kmh
        self.base_restore_hours = base_restore_hours
        self.repair_complexity_hours = repair_complexity_hours
        self.graph: Optional[nx.Graph] = None

    # ------------------------------------------------------------------
    # Road graph construction
    # ------------------------------------------------------------------

    def build_road_graph(
        self,
        assets: List[Asset],
        place_name: Optional[str] = None,
    ) -> nx.Graph:
        """Build a NetworkX road network graph.

        Uses OSMnx if available and ``place_name`` is provided;
        otherwise builds a synthetic graph via Delaunay triangulation
        of asset and depot locations.

        Parameters
        ----------
        assets : list of Asset
        place_name : str or None
            Optional place name for OSMnx download.

        Returns
        -------
        nx.Graph
            Graph with ``geometry`` and ``weight_km`` edge attributes.
        """
        if place_name is not None:
            try:
                import osmnx as ox
                G = ox.graph_from_place(place_name, network_type="drive")
                G = nx.Graph(G)
                for u, v, data in G.edges(data=True):
                    if "length" in data:
                        data["weight_km"] = data["length"] / 1000.0
                    else:
                        data["weight_km"] = 1.0
                self.graph = G
                logger.info("Built road graph from OSM: %s (%d nodes)", place_name, G.number_of_nodes())
                return G
            except ImportError:
                logger.info("osmnx not installed; falling back to synthetic graph.")
            except Exception as exc:
                logger.warning("OSMnx download failed: %s; falling back to synthetic graph.", exc)

        points: List[Tuple[float, float]] = []
        point_ids: List[str] = []

        for name, pt in self.depots:
            points.append((pt.x, pt.y))
            point_ids.append(f"depot:{name}")

        for asset in assets:
            centroid = asset.geom.centroid
            points.append((centroid.x, centroid.y))
            point_ids.append(asset.asset_id)

        G = nx.Graph()
        for pid, (x, y) in zip(point_ids, points):
            G.add_node(pid, x=x, y=y, geometry=Point(x, y))

        from scipy.spatial import Delaunay
        pts_array = np.array(points)
        tri = Delaunay(pts_array)

        edges_seen: set = set()
        for simplex in tri.simplices:
            for i in range(3):
                u_idx, v_idx = simplex[i], simplex[(i + 1) % 3]
                if u_idx > v_idx:
                    u_idx, v_idx = v_idx, u_idx
                if (u_idx, v_idx) in edges_seen:
                    continue
                edges_seen.add((u_idx, v_idx))

                u_id, v_id = point_ids[u_idx], point_ids[v_idx]
                u_pt = Point(points[u_idx][0], points[u_idx][1])
                v_pt = Point(points[v_idx][0], points[v_idx][1])

                dist_km = self._haversine_distance(u_pt, v_pt)
                line = LineString([u_pt, v_pt])

                G.add_edge(u_id, v_id, weight_km=dist_km, geometry=line)

        self.graph = G
        logger.info("Built synthetic road graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
        return G

    # ------------------------------------------------------------------
    # Hazard blockage
    # ------------------------------------------------------------------

    def mark_hazard_blockages(
        self,
        G: nx.Graph,
        hazard_zones: List[Polygon],
    ) -> Tuple[nx.Graph, List[Tuple[str, str]]]:
        """Mark road segments blocked by hazard zones.

        Parameters
        ----------
        G : nx.Graph
        hazard_zones : list of Polygon

        Returns
        -------
        tuple of (nx.Graph, list)
            Blocked graph and list of ``(u, v)`` blocked edges.
        """
        if not hazard_zones:
            return G.copy(), []

        hazard_union = unary_union(hazard_zones)
        G_blocked = G.copy()
        blocked: List[Tuple[str, str]] = []

        for u, v, data in G.edges(data=True):
            geom = data.get("geometry")
            if geom is None:
                u_pt = Point(G.nodes[u].get("x", 0), G.nodes[u].get("y", 0))
                v_pt = Point(G.nodes[v].get("x", 0), G.nodes[v].get("y", 0))
                geom = LineString([u_pt, v_pt])

            if geom.intersects(hazard_union):
                G_blocked.remove_edge(u, v)
                blocked.append((u, v))

        logger.info(
            "Marked %d/%d edges as blocked by hazard zones.",
            len(blocked), G.number_of_edges(),
        )
        return G_blocked, blocked

    # ------------------------------------------------------------------
    # Depot lookup
    # ------------------------------------------------------------------

    def find_nearest_depot(
        self, asset_geom: Point
    ) -> Tuple[str, float]:
        """Find the nearest maintenance depot to an asset.

        Parameters
        ----------
        asset_geom : Point

        Returns
        -------
        tuple of (str, float)
            Depot name and distance in km.
        """
        best_name = ""
        best_dist = float("inf")

        for name, pt in self.depots:
            dist = self._haversine_distance(asset_geom, pt)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        return best_name, best_dist

    # ------------------------------------------------------------------
    # Restoration time
    # ------------------------------------------------------------------

    def compute_restoration_time(
        self,
        asset: Asset,
        G_blocked: nx.Graph,
    ) -> Dict[str, Any]:
        """Compute restoration time for a single failed asset.

        Parameters
        ----------
        asset : Asset
        G_blocked : nx.Graph

        Returns
        -------
        dict
            ``{"restore_hours": float, "route_km": float,
            "isolated": bool, "path": list, "depot": str}``.
        """
        asset_geom = asset.geom.centroid
        depot_name, direct_dist = self.find_nearest_depot(asset_geom)
        depot_node = f"depot:{depot_name}"

        if depot_node not in G_blocked or asset.asset_id not in G_blocked:
            penalty = self.base_restore_hours * _ISOLATION_PENALTY
            return {
                "restore_hours": penalty + self.repair_complexity_hours,
                "route_km": float("inf"),
                "isolated": True,
                "path": [],
                "depot": depot_name,
            }

        try:
            path = nx.shortest_path(
                G_blocked, source=depot_node, target=asset.asset_id, weight="weight_km"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            penalty = self.base_restore_hours * _ISOLATION_PENALTY
            return {
                "restore_hours": penalty + self.repair_complexity_hours,
                "route_km": float("inf"),
                "isolated": True,
                "path": [],
                "depot": depot_name,
            }

        route_km = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = G_blocked.get_edge_data(u, v)
            if edge_data:
                route_km += float(edge_data.get("weight_km", 0.0))

        travel_hours = 2.0 * route_km / self.travel_speed_kmh
        restore_hours = self.base_restore_hours + travel_hours + self.repair_complexity_hours

        return {
            "restore_hours": restore_hours,
            "route_km": route_km,
            "isolated": False,
            "path": path,
            "depot": depot_name,
        }

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def compute_restoration_times_batch(
        self,
        assets: List[Asset],
        hazard_zones: Optional[List[Polygon]] = None,
        place_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute restoration times for a batch of failed assets.

        Parameters
        ----------
        assets : list of Asset
        hazard_zones : list of Polygon or None
        place_name : str or None

        Returns
        -------
        dict
            ``{"restore_hours": np.ndarray, "per_asset": dict,
            "n_isolated": int}``.
        """
        G = self.build_road_graph(assets, place_name)
        G_blocked, blocked = self.mark_hazard_blockages(G, hazard_zones or [])

        n = len(assets)
        restore_hours = np.empty(n, dtype=np.float64)
        per_asset: Dict[str, Any] = {}
        n_isolated = 0

        for i, asset in enumerate(assets):
            result = self.compute_restoration_time(asset, G_blocked)
            restore_hours[i] = result["restore_hours"]
            per_asset[asset.asset_id] = result
            if result["isolated"]:
                n_isolated += 1

        logger.info(
            "Batch restoration: %d assets, %d isolated, mean=%.1f h",
            n, n_isolated, float(np.mean(restore_hours)),
        )

        return {
            "restore_hours": restore_hours,
            "per_asset": per_asset,
            "n_isolated": n_isolated,
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_routes(
        self,
        ax: Any,
        G: Optional[nx.Graph] = None,
        blocked_edges: Optional[List[Tuple[str, str]]] = None,
        paths: Optional[List[List[str]]] = None,
        assets: Optional[List[Asset]] = None,
    ) -> Any:
        """Plot road network, blockages, and crew dispatch paths.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        G : nx.Graph or None
            Uses ``self.graph`` if None.
        blocked_edges : list of tuple or None
        paths : list of list of str or None
        assets : list of Asset or None

        Returns
        -------
        matplotlib.axes.Axes
        """
        graph = G or self.graph
        if graph is None:
            logger.warning("No graph to plot.")
            return ax

        blocked_set = set(blocked_edges or [])

        for u, v, data in graph.edges(data=True):
            geom = data.get("geometry")
            if geom is not None:
                x, y = geom.xy
                color = "red" if (u, v) in blocked_set or (v, u) in blocked_set else "lightgrey"
                ax.plot(x, y, color=color, linewidth=0.8, alpha=0.6)

        if paths:
            for path in paths:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge_data = graph.get_edge_data(u, v)
                    if edge_data and "geometry" in edge_data:
                        x, y = edge_data["geometry"].xy
                        ax.plot(x, y, color="green", linewidth=2.0, alpha=0.9)

        for name, pt in self.depots:
            ax.plot(pt.x, pt.y, "s", color="orange", markersize=8, label="Depot" if name == self.depots[0][0] else "")

        if assets:
            for asset in assets:
                c = asset.geom.centroid
                ax.plot(c.x, c.y, "o", color="blue", markersize=4, alpha=0.7)

        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        return ax

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _haversine_distance(pt1: Point, pt2: Point) -> float:
        """Great-circle distance between two points in km.

        Parameters
        ----------
        pt1, pt2 : Point

        Returns
        -------
        float
        """
        from math import asin, cos, radians, sin, sqrt

        lon1, lat1 = radians(pt1.x), radians(pt1.y)
        lon2, lat2 = radians(pt2.x), radians(pt2.y)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 6371.0 * 2.0 * asin(sqrt(a))
