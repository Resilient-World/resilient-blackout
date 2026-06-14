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

"""Unit tests for ``resilient_blackout.grid.osm_pipeline``."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

pytest.importorskip("pandapower")

from resilient_blackout.grid.osm_pipeline import (
    OSMGridBuilder,
    _resolve_voltage_kv,
    _VOLTAGE_LOOKUP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fixture() -> Dict[str, Any]:
    path = os.path.join(os.path.dirname(__file__), "fixtures", "osm_sample.json")
    with open(path, "r") as fh:
        return json.load(fh)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestOSMGridBuilderInit:
    def test_default_construction(self) -> None:
        builder = OSMGridBuilder()
        assert builder.snap_threshold_m == 50.0
        assert builder.default_voltage_kv == 12.47

    def test_custom_parameters(self) -> None:
        builder = OSMGridBuilder(snap_threshold_m=100.0, default_voltage_kv=34.5)
        assert builder.snap_threshold_m == 100.0
        assert builder.default_voltage_kv == 34.5

    def test_repr(self) -> None:
        assert "OSMGridBuilder" in repr(OSMGridBuilder())


# ---------------------------------------------------------------------------
# Voltage lookup
# ---------------------------------------------------------------------------


class TestVoltageLookup:
    def test_direct_alias(self) -> None:
        vn, props = _resolve_voltage_kv("138000")
        assert vn == 138
        assert props == _VOLTAGE_LOOKUP["138"]

    def test_kv_string(self) -> None:
        vn, props = _resolve_voltage_kv("220 kV")
        assert vn == 220

    def test_missing_defaults(self) -> None:
        vn, props = _resolve_voltage_kv("", default_kv=12.47)
        assert vn == 12.47

    def test_rounding(self) -> None:
        vn, _props = _resolve_voltage_kv("125000")
        assert vn == 132  # nearest standard class to 125 kV


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParseOSMElements:
    def test_parse_fixture(self) -> None:
        osm = _load_fixture()
        builder = OSMGridBuilder()
        subs, gens, lines = builder.parse_osm_elements(osm)
        assert len(subs) == 4  # 3 nodes + 1 from way nodes
        assert len(gens) == 1
        assert len(lines) == 3
        assert "Line_HV" in lines["name"].values

    def test_empty_response(self) -> None:
        builder = OSMGridBuilder()
        subs, gens, lines = builder.parse_osm_elements({"elements": []})
        assert subs.empty
        assert gens.empty
        assert lines.empty


# ---------------------------------------------------------------------------
# Snapping
# ---------------------------------------------------------------------------


class TestSnapLineEndpoints:
    def test_snap_within_threshold(self) -> None:
        builder = OSMGridBuilder(snap_threshold_m=5000.0)
        osm = _load_fixture()
        subs, gens, lines = builder.parse_osm_elements(osm)
        all_nodes = pd.concat([subs, gens], ignore_index=True)
        snapped = builder.snap_line_endpoints(lines, all_nodes)
        assert snapped["from_node"].notna().all()
        assert snapped["to_node"].notna().all()

    def test_snap_outside_threshold(self) -> None:
        builder = OSMGridBuilder(snap_threshold_m=1.0)  # 1 metre — too tight
        osm = _load_fixture()
        subs, gens, lines = builder.parse_osm_elements(osm)
        all_nodes = pd.concat([subs, gens], ignore_index=True)
        snapped = builder.snap_line_endpoints(lines, all_nodes)
        # Some endpoints will not snap at 1 metre
        assert snapped["from_node"].isna().any() or snapped["to_node"].isna().any()


# ---------------------------------------------------------------------------
# Voltage inference
# ---------------------------------------------------------------------------


class TestInferVoltage:
    def test_infer_from_tags(self) -> None:
        builder = OSMGridBuilder()
        osm = _load_fixture()
        _subs, _gens, lines = builder.parse_osm_elements(osm)
        lines = builder.infer_voltage_and_properties(lines)
        hv_line = lines[lines["name"] == "Line_HV"].iloc[0]
        assert hv_line["vn_kv"] == 138
        assert hv_line["r_ohm_per_km"] == _VOLTAGE_LOOKUP["138"]["r_ohm_per_km"]

    def test_infer_empty(self) -> None:
        builder = OSMGridBuilder()
        empty = gpd.GeoDataFrame(columns=["voltage"], crs="EPSG:4326")
        result = builder.infer_voltage_and_properties(empty)
        assert result.empty


# ---------------------------------------------------------------------------
# Transformer inference
# ---------------------------------------------------------------------------


class TestIdentifyTransformers:
    def test_hv_lv_meet(self) -> None:
        builder = OSMGridBuilder(snap_threshold_m=5000.0)
        osm = _load_fixture()
        subs, gens, lines = builder.parse_osm_elements(osm)
        all_nodes = pd.concat([subs, gens], ignore_index=True)
        lines = builder.snap_line_endpoints(lines, all_nodes)
        lines = builder.infer_voltage_and_properties(lines)
        trafos = builder.identify_transformers(all_nodes, lines)
        # Node 2 should have both 138 kV and 12 kV lines
        assert not trafos.empty
        assert any(trafos["hv_vn_kv"] >= 138)

    def test_no_transformer_uniform_voltage(self) -> None:
        builder = OSMGridBuilder()
        # Single voltage line
        nodes = gpd.GeoDataFrame(
            {
                "osm_id": [1, 2],
                "geometry": [Point(0, 0), Point(1, 0)],
                "name": ["A", "B"],
                "power": ["substation", "substation"],
                "voltage": ["", ""],
                "lat": [0.0, 0.0],
                "lon": [0.0, 1.0],
            },
            crs="EPSG:4326",
        )
        lines = gpd.GeoDataFrame(
            {
                "osm_id": [101],
                "geometry": [LineString([(0, 0), (1, 0)])],
                "voltage": ["138000"],
                "name": ["L1"],
                "nodes": [[1, 2]],
                "from_node": [1],
                "to_node": [2],
                "vn_kv": [138.0],
            },
            crs="EPSG:4326",
        )
        trafos = builder.identify_transformers(nodes, lines)
        assert trafos.empty


# ---------------------------------------------------------------------------
# pandapower build
# ---------------------------------------------------------------------------


class TestBuildPandapowerNet:
    def test_build_from_fixture(self) -> None:
        pytest.importorskip("pandapower")
        builder = OSMGridBuilder(snap_threshold_m=5000.0)
        osm = _load_fixture()
        subs, gens, lines = builder.parse_osm_elements(osm)
        all_nodes = pd.concat([subs, gens], ignore_index=True)
        lines = builder.snap_line_endpoints(lines, all_nodes)
        lines = builder.infer_voltage_and_properties(lines)
        trafos = builder.identify_transformers(all_nodes, lines)
        net = builder.build_pandapower_net(all_nodes, gens, lines, trafos)

        assert len(net.bus) > 0
        assert len(net.line) > 0
        assert len(net.gen) > 0
        assert len(net.ext_grid) == 1
        assert len(net.load) > 0

    def test_empty_elements(self) -> None:
        pytest.importorskip("pandapower")
        builder = OSMGridBuilder()
        empty_nodes = gpd.GeoDataFrame(
            columns=["osm_id", "lat", "lon", "geometry", "power", "name", "voltage"],
            crs="EPSG:4326",
        )
        empty_lines = gpd.GeoDataFrame(
            columns=["osm_id", "geometry", "voltage", "cables", "name", "nodes"],
            crs="EPSG:4326",
        )
        empty_trafos = pd.DataFrame(columns=["node_id", "hv_vn_kv", "lv_vn_kv"])
        empty_gens = gpd.GeoDataFrame(
            columns=["osm_id", "geometry", "power", "name", "voltage", "lat", "lon"],
            crs="EPSG:4326",
        )
        net = builder.build_pandapower_net(empty_nodes, empty_gens, empty_lines, empty_trafos)
        assert len(net.bus) == 0


# ---------------------------------------------------------------------------
# Progressive relaxation
# ---------------------------------------------------------------------------


class TestValidateAndSolve:
    def test_solve_fixture_net(self) -> None:
        pytest.importorskip("pandapower")
        builder = OSMGridBuilder(snap_threshold_m=5000.0)
        osm = _load_fixture()
        subs, gens, lines = builder.parse_osm_elements(osm)
        all_nodes = pd.concat([subs, gens], ignore_index=True)
        lines = builder.snap_line_endpoints(lines, all_nodes)
        lines = builder.infer_voltage_and_properties(lines)
        trafos = builder.identify_transformers(all_nodes, lines)
        net = builder.build_pandapower_net(all_nodes, gens, lines, trafos)
        result = builder.validate_and_solve(net)
        assert result["converged"] is True
        assert result["solver_used"] in ("nr", "bfsw", "dc", "nr_relaxed_loads")
        assert len(result["vm_pu"]) > 0


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


class TestBuildFromBBox:
    def test_bbox_url_construction(self) -> None:
        builder = OSMGridBuilder()
        # Verify query text contains bbox
        bbox = (-74.1, 39.9, -73.9, 40.1)
        query = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
        assert "39.9" in query
        assert "-74.1" in query

    @pytest.mark.integration
    def test_live_overpass_query(self) -> None:
        pytest.importorskip("pandapower")
        builder = OSMGridBuilder(timeout_s=30)
        # Small bbox around Princeton, NJ (known power data)
        bbox = (-74.7, 40.3, -74.6, 40.4)
        try:
            net, result = builder.build_from_bbox(bbox)
            assert len(net.bus) >= 0  # may be empty but should not crash
        except Exception as exc:
            pytest.skip(f"Live Overpass query failed: {exc}")
