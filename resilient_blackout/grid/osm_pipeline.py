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
Automated grid construction from OpenStreetMap data.

Provides ``OSMGridBuilder`` which queries the Overpass API for power
infrastructure, reconstructs a topologically consistent network, and returns a
solvable ``pandapowerNet``.
"""

from __future__ import annotations

import json
import logging
import math
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

logger = logging.getLogger(__name__)

_DEFAULT_SNAP: float = 50.0  # metres
_DEFAULT_TIMEOUT: int = 60  # seconds
_OVERPASS_URL: str = "https://overpass-api.de/api/interpreter"

# ---------------------------------------------------------------------------
# Voltage / conductor lookup
# ---------------------------------------------------------------------------

_VoltageEntry = Dict[str, float]
_VOLTAGE_LOOKUP: Dict[str, _VoltageEntry] = {
    "765": {"r_ohm_per_km": 0.015, "x_ohm_per_km": 0.275, "max_i_ka": 3.5, "c_nf_per_km": 14.0},
    "500": {"r_ohm_per_km": 0.020, "x_ohm_per_km": 0.280, "max_i_ka": 3.0, "c_nf_per_km": 13.0},
    "400": {"r_ohm_per_km": 0.022, "x_ohm_per_km": 0.285, "max_i_ka": 2.5, "c_nf_per_km": 12.5},
    "345": {"r_ohm_per_km": 0.025, "x_ohm_per_km": 0.290, "max_i_ka": 2.2, "c_nf_per_km": 12.0},
    "230": {"r_ohm_per_km": 0.030, "x_ohm_per_km": 0.300, "max_i_ka": 1.8, "c_nf_per_km": 11.5},
    "220": {"r_ohm_per_km": 0.032, "x_ohm_per_km": 0.305, "max_i_ka": 1.7, "c_nf_per_km": 11.0},
    "138": {"r_ohm_per_km": 0.050, "x_ohm_per_km": 0.350, "max_i_ka": 1.2, "c_nf_per_km": 10.0},
    "132": {"r_ohm_per_km": 0.055, "x_ohm_per_km": 0.355, "max_i_ka": 1.1, "c_nf_per_km": 9.5},
    "115": {"r_ohm_per_km": 0.060, "x_ohm_per_km": 0.360, "max_i_ka": 1.0, "c_nf_per_km": 9.0},
    "110": {"r_ohm_per_km": 0.065, "x_ohm_per_km": 0.370, "max_i_ka": 0.9, "c_nf_per_km": 8.5},
    "69":  {"r_ohm_per_km": 0.120, "x_ohm_per_km": 0.400, "max_i_ka": 0.7, "c_nf_per_km": 8.0},
    "35":  {"r_ohm_per_km": 0.180, "x_ohm_per_km": 0.380, "max_i_ka": 0.5, "c_nf_per_km": 9.0},
    "22":  {"r_ohm_per_km": 0.250, "x_ohm_per_km": 0.390, "max_i_ka": 0.4, "c_nf_per_km": 9.5},
    "20":  {"r_ohm_per_km": 0.260, "x_ohm_per_km": 0.395, "max_i_ka": 0.4, "c_nf_per_km": 9.5},
    "15":  {"r_ohm_per_km": 0.280, "x_ohm_per_km": 0.400, "max_i_ka": 0.35, "c_nf_per_km": 10.0},
    "13":  {"r_ohm_per_km": 0.300, "x_ohm_per_km": 0.410, "max_i_ka": 0.3, "c_nf_per_km": 10.0},
    "12":  {"r_ohm_per_km": 0.320, "x_ohm_per_km": 0.420, "max_i_ka": 0.3, "c_nf_per_km": 10.5},
    "11":  {"r_ohm_per_km": 0.340, "x_ohm_per_km": 0.430, "max_i_ka": 0.25, "c_nf_per_km": 11.0},
}

# Common voltage synonyms / rounding targets (V → lookup key in kV)
_VOLTAGE_ALIASES: Dict[int, int] = {
    110000: 110, 115000: 115, 132000: 132, 138000: 138,
    220000: 220, 230000: 230, 345000: 345, 380000: 400,
    400000: 400, 500000: 500, 765000: 765, 69000: 69,
    35000: 35, 34500: 35, 22000: 22, 20000: 20,
    15000: 15, 13800: 13, 13200: 13, 12700: 13,
    12470: 12, 11000: 11,
}


def _resolve_voltage_kv(raw_v: Any, default_kv: float = 12.47) -> Tuple[float, _VoltageEntry]:
    """Map raw OSM voltage tag to nearest standard class."""
    try:
        v_int = int(str(raw_v).replace(" ", "").replace("V", "").replace("kV", "000").replace(".", ""))
    except (ValueError, TypeError):
        return default_kv, _VOLTAGE_LOOKUP.get(str(int(default_kv)), _VOLTAGE_LOOKUP["12"])

    if v_int in _VOLTAGE_ALIASES:
        key = str(_VOLTAGE_ALIASES[v_int])
        return int(key), _VOLTAGE_LOOKUP[key]

    kv = v_int / 1000.0
    nearest = min(_VOLTAGE_LOOKUP.keys(), key=lambda k: abs(float(k) - kv))
    return float(nearest), _VOLTAGE_LOOKUP[nearest]


class OSMGridBuilder:
    """Construct a solvable pandapower network from OpenStreetMap data.

    Queries the Overpass API for substations, generators, and power lines
    within a bounding box, then reconstructs a topologically and
    electrically coherent ``pandapowerNet``.

    Parameters
    ----------
    snap_threshold_m : float
        Maximum distance (metres) for snapping a floating line endpoint
        to the nearest bus/node.  Default 50.
    default_voltage_kv : float
        Voltage to use when OSM tags are absent.  Default 12.47.
    overpass_url : str
        Overpass API endpoint.  Default public instance.
    timeout_s : int
        HTTP timeout for Overpass queries.  Default 60.
    """

    def __init__(
        self,
        snap_threshold_m: float = _DEFAULT_SNAP,
        default_voltage_kv: float = 12.47,
        overpass_url: str = _OVERPASS_URL,
        timeout_s: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self.snap_threshold_m = snap_threshold_m
        self.default_voltage_kv = default_voltage_kv
        self.overpass_url = overpass_url
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------
    # Overpass query
    # ------------------------------------------------------------------

    def query_overpass(self, bbox: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """Fetch OSM power elements within a bounding box.

        Parameters
        ----------
        bbox : tuple
            ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84.

        Returns
        -------
        dict
            Parsed Overpass JSON response.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        query = f"""
        [out:json][timeout:60];
        (
          node["power"="substation"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["power"="substation"]({min_lat},{min_lon},{max_lat},{max_lon});
          node["power"="generator"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["power"="line"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["power"="cable"]({min_lat},{min_lon},{max_lat},{max_lon});
          node["power"="line"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """
        data = query.encode("utf-8")
        req = urllib.request.Request(
            self.overpass_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_osm_elements(
        self, osm_json: Dict[str, Any]
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Convert raw Overpass JSON into GeoDataFrames.

        Returns
        -------
        tuple
            ``(substations, generators, lines)``.
        """
        elements = osm_json.get("elements", [])
        nodes: Dict[int, Dict[str, Any]] = {}
        ways: Dict[int, Dict[str, Any]] = {}
        for el in elements:
            if el.get("type") == "node":
                nodes[el["id"]] = el
            elif el.get("type") == "way":
                ways[el["id"]] = el

        node_records: List[Dict[str, Any]] = []
        for nid, el in nodes.items():
            tags = el.get("tags", {})
            lat = el.get("lat")
            lon = el.get("lon")
            if lat is None or lon is None:
                continue
            power = tags.get("power", "")
            node_records.append({
                "osm_id": nid, "lat": lat, "lon": lon,
                "geometry": Point(lon, lat), "power": power,
                "name": tags.get("name", f"node_{nid}"),
                "voltage": tags.get("voltage", ""),
            })
        nodes_gdf = gpd.GeoDataFrame(node_records, crs="EPSG:4326")

        substation_nodes = nodes_gdf[nodes_gdf["power"] == "substation"].copy()
        substation_way_nodes: List[int] = []
        for wid, way in ways.items():
            tags = way.get("tags", {})
            if tags.get("power") == "substation":
                substation_way_nodes.extend(way.get("nodes", []))
        if substation_way_nodes:
            sw = nodes_gdf[nodes_gdf["osm_id"].isin(substation_way_nodes)].copy()
            sw["power"] = "substation"
            substation_nodes = pd.concat([substation_nodes, sw], ignore_index=True)
            substation_nodes = substation_nodes.drop_duplicates(subset=["osm_id"])

        gen_nodes = nodes_gdf[nodes_gdf["power"] == "generator"].copy()

        line_records: List[Dict[str, Any]] = []
        for wid, way in ways.items():
            tags = way.get("tags", {})
            if tags.get("power") not in ("line", "cable"):
                continue
            nds = way.get("nodes", [])
            if len(nds) < 2:
                continue
            coords = [(nodes[nid]["lon"], nodes[nid]["lat"]) for nid in nds if nid in nodes]
            if len(coords) < 2:
                continue
            line_records.append({
                "osm_id": wid, "geometry": LineString(coords),
                "voltage": tags.get("voltage", ""),
                "cables": tags.get("cables", ""),
                "name": tags.get("name", f"line_{wid}"),
                "nodes": nds,
            })
        lines_gdf = gpd.GeoDataFrame(line_records, crs="EPSG:4326")

        return substation_nodes, gen_nodes, lines_gdf

    # ------------------------------------------------------------------
    # Snapping
    # ------------------------------------------------------------------

    def snap_line_endpoints(
        self, lines_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Snap line endpoints to the nearest node within threshold.

        Adds ``from_node`` and ``to_node`` columns to ``lines_gdf``.
        """
        if lines_gdf.empty:
            return lines_gdf

        centroid = nodes_gdf.unary_union.centroid if not nodes_gdf.empty else Point(0, 0)
        utm_zone = int((centroid.x + 180) / 6) + 1
        hemisphere = "north" if centroid.y >= 0 else "south"
        epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
        utm_crs = f"EPSG:{epsg}"

        nodes_proj = nodes_gdf.to_crs(utm_crs)
        lines_proj = lines_gdf.to_crs(utm_crs)

        from_nodes: List[Optional[int]] = []
        to_nodes: List[Optional[int]] = []
        for _idx, row in lines_proj.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                from_nodes.append(None)
                to_nodes.append(None)
                continue
            start = Point(geom.coords[0])
            end = Point(geom.coords[-1])
            ds = nodes_proj.geometry.distance(start)
            de = nodes_proj.geometry.distance(end)
            i_s = ds.idxmin()
            i_e = de.idxmin()
            from_nodes.append(int(nodes_gdf.at[i_s, "osm_id"]) if ds[i_s] <= self.snap_threshold_m else None)
            to_nodes.append(int(nodes_gdf.at[i_e, "osm_id"]) if de[i_e] <= self.snap_threshold_m else None)

        lines_gdf["from_node"] = from_nodes
        lines_gdf["to_node"] = to_nodes
        return lines_gdf

    # ------------------------------------------------------------------
    # Voltage inference
    # ------------------------------------------------------------------

    def infer_voltage_and_properties(self, lines_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add ``vn_kv`` and electrical parameters."""
        if lines_gdf.empty:
            return lines_gdf
        vn_kvs, rs, xs, max_is, cs = [], [], [], [], []
        for _idx, row in lines_gdf.iterrows():
            vn, props = _resolve_voltage_kv(row.get("voltage", ""), self.default_voltage_kv)
            vn_kvs.append(vn)
            rs.append(props["r_ohm_per_km"])
            xs.append(props["x_ohm_per_km"])
            max_is.append(props["max_i_ka"])
            cs.append(props["c_nf_per_km"])
        lines_gdf["vn_kv"] = vn_kvs
        lines_gdf["r_ohm_per_km"] = rs
        lines_gdf["x_ohm_per_km"] = xs
        lines_gdf["max_i_ka"] = max_is
        lines_gdf["c_nf_per_km"] = cs
        return lines_gdf

    # ------------------------------------------------------------------
    # Transformer inference
    # ------------------------------------------------------------------

    def identify_transformers(
        self, nodes_gdf: gpd.GeoDataFrame, lines_gdf: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """Identify buses where HV and LV lines meet (≥2× voltage ratio)."""
        if lines_gdf.empty or nodes_gdf.empty:
            return pd.DataFrame()
        node_voltages: Dict[int, List[float]] = {}
        for _idx, row in lines_gdf.iterrows():
            vn = row["vn_kv"]
            for nid in (row.get("from_node"), row.get("to_node")):
                if nid is not None:
                    node_voltages.setdefault(int(nid), []).append(vn)
        records: List[Dict[str, Any]] = []
        for nid, voltages in node_voltages.items():
            uniq = sorted(set(voltages), reverse=True)
            if len(uniq) >= 2 and uniq[0] >= 2 * uniq[-1]:
                records.append({"node_id": nid, "hv_vn_kv": uniq[0], "lv_vn_kv": uniq[-1]})
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # pandapower build
    # ------------------------------------------------------------------

    def build_pandapower_net(
        self,
        nodes_gdf: gpd.GeoDataFrame,
        generators_gdf: gpd.GeoDataFrame,
        lines_gdf: gpd.GeoDataFrame,
        transformers_df: pd.DataFrame,
    ) -> Any:
        """Assemble a pandapower network from parsed elements.

        Returns
        -------
        pandapowerNet
        """
        import pandapower as pp

        net = pp.create_empty_network(name="osm_reconstructed")
        node_to_bus: Dict[int, int] = {}
        bus_voltages: Dict[int, float] = {}

        # Determine bus voltage from connected lines
        node_vn: Dict[int, float] = {}
        for _idx, row in lines_gdf.iterrows():
            vn = row["vn_kv"]
            for nid in (row.get("from_node"), row.get("to_node")):
                if nid is not None:
                    node_vn[int(nid)] = max(node_vn.get(int(nid), 0.0), vn)

        # Create buses
        for _idx, row in nodes_gdf.iterrows():
            nid = int(row["osm_id"])
            vn = node_vn.get(nid, self.default_voltage_kv)
            name = str(row.get("name", f"bus_{nid}"))
            bus_idx = pp.create_bus(net, vn_kv=vn, name=name,
                                    geodata=(float(row["lon"]), float(row["lat"])), type="b")
            node_to_bus[nid] = bus_idx
            bus_voltages[bus_idx] = vn

        # Create lines
        for _idx, row in lines_gdf.iterrows():
            fn = row.get("from_node")
            tn = row.get("to_node")
            if fn is None or tn is None:
                continue
            from_bus = node_to_bus.get(int(fn))
            to_bus = node_to_bus.get(int(tn))
            if from_bus is None or to_bus is None:
                continue
            geom = row["geometry"]
            if geom is not None and not geom.is_empty:
                c = geom.centroid
                utm_zone = int((c.x + 180) / 6) + 1
                epsg = 32600 + utm_zone if c.y >= 0 else 32700 + utm_zone
                try:
                    length_km = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(
                        f"EPSG:{epsg}").iloc[0].length / 1000.0
                except Exception:
                    length_km = 1.0
            else:
                length_km = 1.0
            pp.create_line_from_parameters(
                net, from_bus=from_bus, to_bus=to_bus,
                length_km=max(length_km, 0.01),
                r_ohm_per_km=row["r_ohm_per_km"],
                x_ohm_per_km=row["x_ohm_per_km"],
                c_nf_per_km=row["c_nf_per_km"],
                max_i_ka=row["max_i_ka"],
                name=str(row.get("name", f"line_{row['osm_id']}")),
            )

        # Create transformers
        for _idx, row in transformers_df.iterrows():
            nid = int(row["node_id"])
            bus_idx = node_to_bus.get(nid)
            if bus_idx is None:
                continue
            hv_vn = row["hv_vn_kv"]
            lv_vn = row["lv_vn_kv"]
            lv_bus = pp.create_bus(net, vn_kv=lv_vn,
                                   name=f"{net.bus.at[bus_idx, 'name']}_lv", type="b")
            # Move LV-connected lines to new bus
            for lidx in net.line.index:
                for col, threshold in [("from_bus", lv_vn * 1.5), ("to_bus", lv_vn * 1.5)]:
                    if net.line.at[lidx, col] == bus_idx and bus_voltages.get(bus_idx, 0) <= threshold:
                        net.line.at[lidx, col] = lv_bus
            # Transformer parameters (per-unit on HV base)
            pp.create_transformer_from_parameters(
                net, hv_bus=bus_idx, lv_bus=lv_bus,
                sn_mva=min(hv_vn, lv_vn) * 10.0,
                vn_hv_kv=hv_vn, vn_lv_kv=lv_vn,
                vk_percent=10.0, vkr_percent=0.5,
                pfe_kw=1.0, i0_percent=0.1,
            )

        # Generators
        for _idx, row in generators_gdf.iterrows():
            nid = int(row["osm_id"])
            bus_idx = node_to_bus.get(nid)
            if bus_idx is None:
                continue
            pp.create_gen(net, bus=bus_idx, p_mw=5.0, vm_pu=1.0,
                          name=str(row.get("name", f"gen_{nid}")))

        # External grid: choose highest-voltage bus with most lines
        bus_degree: Dict[int, int] = {}
        for lidx in net.line.index:
            bus_degree[net.line.at[lidx, "from_bus"]] = bus_degree.get(net.line.at[lidx, "from_bus"], 0) + 1
            bus_degree[net.line.at[lidx, "to_bus"]] = bus_degree.get(net.line.at[lidx, "to_bus"], 0) + 1
        slack_candidates = sorted(
            net.bus.index,
            key=lambda b: (-bus_voltages.get(b, 0), -bus_degree.get(b, 0)),
        )
        if slack_candidates:
            pp.create_ext_grid(net, bus=slack_candidates[0], vm_pu=1.0,
                               name=f"slack_{slack_candidates[0]}")

        # Loads: proportional to line degree
        total_gen_mw = len(generators_gdf) * 5.0 if not generators_gdf.empty else 10.0
        total_degree = sum(bus_degree.get(b, 0) for b in net.bus.index)
        for bidx in net.bus.index:
            if bidx == slack_candidates[0] if slack_candidates else True:
                continue
            degree = bus_degree.get(bidx, 0)
            if total_degree > 0:
                p = total_gen_mw * 0.8 * degree / total_degree
            else:
                p = total_gen_mw * 0.8 / max(len(net.bus), 1)
            if p > 0:
                pp.create_load(net, bus=bidx, p_mw=p, q_mvar=p * 0.3,
                               name=f"load_{bidx}")

        return net

    # ------------------------------------------------------------------
    # Validation / progressive relaxation
    # ------------------------------------------------------------------

    def validate_and_solve(self, net: Any) -> Dict[str, Any]:
        """Run AC power flow with progressive convergence fallback.

        Returns
        -------
        dict
            ``converged``, ``solver_used``, ``vm_pu``, ``loading_percent``,
            ``total_losses_mw``.
        """
        import pandapower as pp

        strategies = [
            ("nr", lambda: (pp.runpp(net, algorithm="nr", init="auto"), None)[0]),
            ("bfsw", lambda: (pp.runpp(net, algorithm="bfsw", init="dc", max_iteration=50), None)[0]),
            ("dc", lambda: (pp.rundcpp(net), None)[0]),
        ]

        for label, strategy in strategies:
            try:
                strategy()
                result = {
                    "converged": True,
                    "solver_used": label,
                    "vm_pu": list(net.res_bus["vm_pu"].values) if hasattr(net, "res_bus") else [],
                    "loading_percent": list(net.res_line["loading_percent"].values)
                    if hasattr(net, "res_line") else [],
                    "total_losses_mw": float(
                        net.res_line["pl_mw"].sum()
                    ) if hasattr(net, "res_line") else 0.0,
                }
                logger.info("OSM power flow converged with %s solver.", label)
                return result
            except (pp.LoadflowNotConverged, Exception) as exc:
                logger.warning("Solver %s failed: %s", label, exc)
                continue

        # Final fallback: reduce loads and retry
        logger.warning("All standard solvers failed — reducing loads by 10%%.")
        for lidx in net.load.index:
            net.load.at[lidx, "p_mw"] *= 0.9
            net.load.at[lidx, "q_mvar"] *= 0.9
        try:
            pp.runpp(net, algorithm="nr", init="dc")
            return {
                "converged": True,
                "solver_used": "nr_relaxed_loads",
                "vm_pu": list(net.res_bus["vm_pu"].values) if hasattr(net, "res_bus") else [],
                "loading_percent": list(net.res_line["loading_percent"].values)
                if hasattr(net, "res_line") else [],
                "total_losses_mw": float(net.res_line["pl_mw"].sum()) if hasattr(net, "res_line") else 0.0,
            }
        except (pp.LoadflowNotConverged, Exception):
            pass

        return {"converged": False, "solver_used": "none", "vm_pu": [], "loading_percent": [], "total_losses_mw": 0.0}

    # ------------------------------------------------------------------
    # High-level convenience
    # ------------------------------------------------------------------

    def build_from_bbox(
        self, bbox: Tuple[float, float, float, float]
    ) -> Tuple[Any, Dict[str, Any]]:
        """End-to-end pipeline: query → parse → build → solve.

        Parameters
        ----------
        bbox : tuple
            ``(min_lon, min_lat, max_lon, max_lat)``.

        Returns
        -------
        tuple
            ``(pandapowerNet, convergence_result)``.
        """
        osm_json = self.query_overpass(bbox)
        substations, generators, lines = self.parse_osm_elements(osm_json)

        # Merge all nodes for snapping
        all_nodes = pd.concat([substations, generators], ignore_index=True)
        if all_nodes.empty:
            all_nodes = gpd.GeoDataFrame(
                columns=["osm_id", "lat", "lon", "geometry", "power", "name", "voltage"],
                crs="EPSG:4326",
            )

        lines = self.snap_line_endpoints(lines, all_nodes)
        lines = self.infer_voltage_and_properties(lines)
        transformers = self.identify_transformers(all_nodes, lines)
        net = self.build_pandapower_net(all_nodes, generators, lines, transformers)
        result = self.validate_and_solve(net)
        return net, result

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"OSMGridBuilder(snap_threshold_m={self.snap_threshold_m}, "
            f"default_voltage_kv={self.default_voltage_kv})"
        )
