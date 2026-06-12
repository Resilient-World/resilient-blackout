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

"""Unit tests for ``resilient_blackout.grid.crew_routing``."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from resilient_blackout.grid.crew_routing import (
    DamagedAsset,
    MultiCrewRestorationRouter,
    RepairCrew,
)


def _make_grid_graph() -> nx.DiGraph:
    """Create a simple directed road network."""
    G = nx.DiGraph()
    edges = [
        ("depot_A", "A", {"travel_time_min": 10.0, "status": "passable"}),
        ("depot_A", "B", {"travel_time_min": 20.0, "status": "passable"}),
        ("depot_B", "C", {"travel_time_min": 15.0, "status": "passable"}),
        ("depot_B", "D", {"travel_time_min": 25.0, "status": "passable"}),
        ("A", "B", {"travel_time_min": 10.0, "status": "passable"}),
        ("A", "C", {"travel_time_min": 30.0, "status": "passable"}),
        ("B", "D", {"travel_time_min": 15.0, "status": "passable"}),
        ("C", "D", {"travel_time_min": 10.0, "status": "passable"}),
        ("A", "blocked", {"travel_time_min": 5.0, "status": "blocked"}),
        ("blocked", "B", {"travel_time_min": 5.0, "status": "blocked"}),
    ]
    G.add_edges_from(edges)
    return G


def _make_crews() -> list:
    return [
        RepairCrew(
            crew_id="crew_1",
            depot_node="depot_A",
            speed_kmh=40.0,
            skills={"Vegetation clearing", "Pole replacement"},
            material_capacity={"wood_poles": 5, "wire_spools": 2},
        ),
        RepairCrew(
            crew_id="crew_2",
            depot_node="depot_B",
            speed_kmh=40.0,
            skills={"Transformer replacement"},
            material_capacity={"transformers": 3},
        ),
    ]


def _make_assets() -> list:
    return [
        DamagedAsset(
            asset_id="line_1",
            node="A",
            repair_type="Vegetation clearing",
            required_materials={"wood_poles": 2},
            repair_duration_h=1.0,
            failure_time_h=0.0,
        ),
        DamagedAsset(
            asset_id="line_2",
            node="B",
            repair_type="Pole replacement",
            required_materials={"wood_poles": 3, "wire_spools": 1},
            repair_duration_h=2.0,
            failure_time_h=1.0,
        ),
        DamagedAsset(
            asset_id="sub_1",
            node="C",
            repair_type="Transformer replacement",
            required_materials={"transformers": 1},
            repair_duration_h=3.0,
            failure_time_h=0.0,
        ),
        DamagedAsset(
            asset_id="sub_2",
            node="D",
            repair_type="Transformer replacement",
            required_materials={"transformers": 2},
            repair_duration_h=2.0,
            failure_time_h=2.0,
        ),
    ]


class TestRepairCrew:
    """Validation of RepairCrew dataclass."""

    def test_valid_construction(self) -> None:
        c = RepairCrew(crew_id="c1", depot_node="depot")
        assert c.crew_id == "c1"
        assert c.speed_kmh == 40.0
        assert c.skills == set()

    def test_negative_speed_raises(self) -> None:
        with pytest.raises(ValueError, match="speed_kmh"):
            RepairCrew(crew_id="c1", depot_node="depot", speed_kmh=-10)


class TestDamagedAsset:
    """Validation of DamagedAsset dataclass."""

    def test_valid_construction(self) -> None:
        a = DamagedAsset(asset_id="a1", node="A", repair_type="test")
        assert a.asset_id == "a1"
        assert a.repair_duration_h == 2.0


class TestMultiCrewRestorationRouterInit:
    """Validation of router construction."""

    def test_valid_construction(self) -> None:
        G = _make_grid_graph()
        crews = _make_crews()
        router = MultiCrewRestorationRouter(G, crews)
        assert len(router.crews) == 2
        assert router.penalty_theta == 1.0

    def test_empty_crews_raises(self) -> None:
        G = _make_grid_graph()
        with pytest.raises(ValueError, match="crews"):
            MultiCrewRestorationRouter(G, [])

    def test_negative_theta_raises(self) -> None:
        G = _make_grid_graph()
        crews = _make_crews()
        with pytest.raises(ValueError, match="penalty_theta"):
            MultiCrewRestorationRouter(G, crews, penalty_theta=-1.0)


class TestPassableSubgraph:
    """Validation of blocked edge filtering."""

    def test_blocked_edges_excluded(self) -> None:
        G = _make_grid_graph()
        crews = _make_crews()
        router = MultiCrewRestorationRouter(G, crews)
        sub = router._passable_subgraph()
        assert not sub.has_edge("A", "blocked")
        assert sub.has_edge("A", "B")


class TestSkillMatching:
    """Validation of skill and material constraints."""

    def test_matching_skill(self) -> None:
        G = _make_grid_graph()
        crews = _make_crews()
        router = MultiCrewRestorationRouter(G, crews)
        crew = crews[0]
        asset = DamagedAsset(
            asset_id="test",
            node="A",
            repair_type="Vegetation clearing",
        )
        assert router._can_visit(crew, asset, crew.material_capacity)

    def test_non_matching_skill(self) -> None:
        G = _make_grid_graph()
        crews = _make_crews()
        router = MultiCrewRestorationRouter(G, crews)
        crew = crews[0]
        asset = DamagedAsset(
            asset_id="test",
            node="A",
            repair_type="Transformer replacement",
        )
        assert not router._can_visit(crew, asset, crew.material_capacity)

    def test_insufficient_materials(self) -> None:
        G = _make_grid_graph()
        crews = _make_crews()
        router = MultiCrewRestorationRouter(G, crews)
        crew = crews[0]
        asset = DamagedAsset(
            asset_id="test",
            node="A",
            repair_type="Vegetation clearing",
            required_materials={"wood_poles": 10},
        )
        assert not router._can_visit(crew, asset, crew.material_capacity)


class TestSolve:
    """Validation of solve and route quality."""

    def test_solve_returns_dict(self) -> None:
        G = _make_grid_graph()
        crews = _make_crews()
        assets = _make_assets()
        router = MultiCrewRestorationRouter(G, crews)
        result = router.solve(assets)

        assert isinstance(result, dict)
        assert "routes" in result
        assert "total_travel_time_h" in result
        assert "restoration_schedule" in result

    def test_all_assets_assigned_or_unassigned(self) -> None:
        G = _make_grid_graph()
        crews = _make_crews()
        assets = _make_assets()
        router = MultiCrewRestorationRouter(G, crews)
        result = router.solve(assets)

        assigned = set()
        for route in result["routes"]:
            assigned.update(route["asset_ids"])
        unassigned = set(result["unassigned_assets"])
        all_assets = {a.asset_id for a in assets}
        assert assigned | unassigned == all_assets

    def test_skill_constraint_respected(self) -> None:
        G = _make_grid_graph()
        crews = _make_crews()
        assets = _make_assets()
        router = MultiCrewRestorationRouter(G, crews)
        result = router.solve(assets)

        for route in result["routes"]:
            crew = next(c for c in crews if c.crew_id == route["crew_id"])
            for aid in route["asset_ids"]:
                asset = next(a for a in assets if a.asset_id == aid)
                assert asset.repair_type in crew.skills

    def test_travel_time_finite(self) -> None:
        G = _make_grid_graph()
        crews = _make_crews()
        assets = _make_assets()
        router = MultiCrewRestorationRouter(G, crews)
        result = router.solve(assets)
        assert result["total_travel_time_h"] < float("inf")
        assert result["total_travel_time_h"] >= 0.0

    def test_empty_assets(self) -> None:
        G = _make_grid_graph()
        crews = _make_crews()
        router = MultiCrewRestorationRouter(G, crews)
        result = router.solve([])
        assert result["total_travel_time_h"] == 0.0
        assert len(result["routes"]) == 0


class TestRestorationTimeseries:
    """Validation of hourly restoration state output."""

    def test_returns_dataframe(self) -> None:
        G = _make_grid_graph()
        crews = _make_crews()
        assets = _make_assets()
        router = MultiCrewRestorationRouter(G, crews)
        router.solve(assets)
        ts = router.restoration_timeseries(assets, max_hour=12)

        assert isinstance(ts, pd.DataFrame)
        assert "hour" in ts.columns
        for a in assets:
            assert a.asset_id in ts.columns

    def test_all_false_at_hour_zero(self) -> None:
        G = _make_grid_graph()
        crews = _make_crews()
        assets = _make_assets()
        router = MultiCrewRestorationRouter(G, crews)
        router.solve(assets)
        ts = router.restoration_timeseries(assets, max_hour=12)

        for a in assets:
            assert ts[a.asset_id].iloc[0] is False

    def test_some_true_by_end(self) -> None:
        G = _make_grid_graph()
        crews = _make_crews()
        assets = _make_assets()
        router = MultiCrewRestorationRouter(G, crews)
        router.solve(assets)
        ts = router.restoration_timeseries(assets, max_hour=24)

        assert ts.iloc[-1].drop("hour").any()

    def test_raises_without_solve(self) -> None:
        G = _make_grid_graph()
        crews = _make_crews()
        assets = _make_assets()
        router = MultiCrewRestorationRouter(G, crews)
        with pytest.raises(RuntimeError, match="Call solve()"):
            router.restoration_timeseries(assets)


class TestTwoOpt:
    """Validation of 2-opt local search."""

    def test_reduces_or_preserves_cost(self) -> None:
        G = _make_grid_graph()
        crews = _make_crews()
        assets = _make_assets()
        router = MultiCrewRestorationRouter(G, crews)

        # Get initial greedy routes
        routes = router._greedy_nearest_neighbor(assets)
        depot_to_asset, asset_to_asset = router._build_distance_matrix(assets)

        for c_idx, route in enumerate(routes):
            if len(route.sequence) > 2:
                improved = router._two_opt(
                    route, assets, asset_to_asset, depot_to_asset, c_idx
                )
                assert (
                    router._route_cost(improved, assets, asset_to_asset, depot_to_asset, c_idx)
                    <= router._route_cost(route, assets, asset_to_asset, depot_to_asset, c_idx)
                    + _EPS
                )


class TestRepr:
    """Validation of string representation."""

    def test_repr_includes_params(self) -> None:
        G = _make_grid_graph()
        crews = _make_crews()
        router = MultiCrewRestorationRouter(G, crews, penalty_theta=2.5)
        r = repr(router)
        assert "crews=2" in r
        assert "theta=2.5" in r


_EPS: float = 1e-12
