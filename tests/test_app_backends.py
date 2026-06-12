# Copyright (c) 2026, Resilient World
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for resilient_blackout.app backends (no heavy deps)."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("pandapower")

from resilient_blackout.app.demo_data import (
    create_demo_cascade_history,
    create_demo_hazard,
    create_demo_load_profile,
    create_demo_rrs_report,
)


class TestDemoData:
    def test_demo_hazard(self) -> None:
        h = create_demo_hazard()
        assert h["type"] == "Feature"
        assert h["properties"]["hazard_type"] == "wildfire"
        coords = h["geometry"]["coordinates"][0]
        assert len(coords) == 5  # closed polygon

    def test_demo_load_profile_shape(self) -> None:
        prof = create_demo_load_profile(24)
        assert prof.shape == (24, 4)
        assert prof.dtype == np.float64
        assert np.all(prof >= 0)

    def test_demo_cascade_history(self) -> None:
        hist = create_demo_cascade_history()
        assert len(hist) == 4
        assert hist[0]["iteration"] == 0
        assert len(hist[-1]["tripped_lines"]) > 0

    def test_demo_rrs_report(self) -> None:
        rep = create_demo_rrs_report()
        assert rep["resilience_of_the_project"]["grade"] == "A+"
        assert rep["key_performance_indicators"]["bcr"] == pytest.approx(2.35)


class TestHazardBackend:
    def test_from_geojson(self) -> None:
        pytest.importorskip("shapely")
        from resilient_blackout.app.backends import HazardBackend

        feature = create_demo_hazard()
        backend = HazardBackend(feature)
        poly = backend.get_polygon_coordinates()
        assert poly is not None
        assert len(poly) == 5

    def test_intersects_bus(self) -> None:
        pytest.importorskip("shapely")
        from resilient_blackout.app.backends import HazardBackend

        backend = HazardBackend(create_demo_hazard())
        # A point inside the demo polygon (~40.71, -73.95)
        assert backend.intersects_bus(40.71, -73.95)
        # A point outside
        assert not backend.intersects_bus(40.0, -75.0)


class TestScorecardBackend:
    def test_kpis(self) -> None:
        from resilient_blackout.app.backends import ScorecardBackend

        rep = create_demo_rrs_report()
        backend = ScorecardBackend(rep)
        kpis = backend.get_kpis()
        assert "NPV ($)" in kpis
        assert kpis["BCR"] == pytest.approx(2.35)
        assert backend.get_grade() == "A+"
        assert backend.get_community_score() == pytest.approx(87.5)

    def test_to_dataframe(self) -> None:
        pytest.importorskip("pandas")
        from resilient_blackout.app.backends import ScorecardBackend

        rep = create_demo_rrs_report()
        df = ScorecardBackend(rep).to_dataframe()
        assert not df.empty
        assert "metric" in df.columns
        assert "value" in df.columns


class TestCascadeAnimatorBackend:
    def test_frames(self) -> None:
        pytest.importorskip("pandapower")
        from resilient_blackout.app.backends import CascadeAnimatorBackend, GridBackend
        from resilient_blackout.app.demo_data import create_demo_grid

        net = create_demo_grid()
        grid = GridBackend(net)
        history = create_demo_cascade_history()
        animator = CascadeAnimatorBackend(grid, history)

        state = animator.frame_at(0)
        assert state["lines"]
        assert len(state["islands"]) == 1

        state2 = animator.frame_at(3)
        assert len(state2["tripped_lines"]) > 0
        assert len(state2["islands"]) > 1

        state_bad = animator.frame_at(100)
        assert state_bad["lines"] == {}
