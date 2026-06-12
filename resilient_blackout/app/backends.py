# Copyright (c) 2026, Resilient World
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Data-processing backends for the Streamlit dashboard.

Separates heavy lifting (grid loading, cascade simulation, RRS scorecard
generation) from Streamlit UI components so that the dashboard code
remains readable and the backends are unit-testable.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class GridBackend:
    """Load and prepare grid models for visualisation.

    Parameters
    ----------
    net : pandapowerNet
    """

    def __init__(self, net: Any) -> None:
        self.net = net
        self._bus_coords: Optional[Dict[int, Tuple[float, float]]] = None
        self._line_coords: Optional[Dict[int, List[Tuple[float, float]]]] = None

    @classmethod
    def from_file(cls, filepath: str) -> "GridBackend":
        """Load a grid from an Excel or JSON file.

        Parameters
        ----------
        filepath : str
            Path to pandapower-compatible file.

        Returns
        -------
        GridBackend
        """
        import pandapower as pp

        if filepath.endswith(".xlsx"):
            net = pp.from_excel(filepath)
        else:
            net = pp.from_json(filepath)
        return cls(net)

    def get_bus_coordinates(self) -> Dict[int, Tuple[float, float]]:
        """Return ``{bus_index: (lat, lon)}`` from geodata or name-based fallback.

        Returns
        -------
        dict
        """
        if self._bus_coords is not None:
            return self._bus_coords

        coords: Dict[int, Tuple[float, float]] = {}
        for idx, row in self.net.bus.iterrows():
            geo = row.get("geodata")
            if isinstance(geo, (tuple, list)) and len(geo) >= 2:
                coords[int(idx)] = (float(geo[0]), float(geo[1]))
            elif "lat" in self.net.bus.columns and "lon" in self.net.bus.columns:
                coords[int(idx)] = (float(row["lat"]), float(row["lon"]))
            else:
                # Deterministic pseudo-geography from bus index
                coords[int(idx)] = (40.7 + int(idx) * 0.01, -74.0 - int(idx) * 0.01)
        self._bus_coords = coords
        return coords

    def get_line_coordinates(self) -> Dict[int, List[Tuple[float, float]]]:
        """Return ``{line_index: [(lat, lon), ...]}`` for each line.

        Returns
        -------
        dict
        """
        if self._line_coords is not None:
            return self._line_coords

        bus_coords = self.get_bus_coordinates()
        lines: Dict[int, List[Tuple[float, float]]] = {}
        for idx, row in self.net.line.iterrows():
            if not row.get("in_service", True):
                continue
            fbus = int(row["from_bus"])
            tbus = int(row["to_bus"])
            fcoord = bus_coords.get(fbus)
            tcoord = bus_coords.get(tbus)
            if fcoord and tcoord:
                lines[int(idx)] = [fcoord, tcoord]
        self._line_coords = lines
        return lines

    def run_power_flow(self) -> bool:
        """Run AC power flow and populate ``res_line``.

        Returns
        -------
        bool
            ``True`` if converged.
        """
        import pandapower as pp

        try:
            pp.runpp(self.net, suppress_warnings=True)
            return True
        except Exception as exc:
            logger.warning("Power flow failed: %s", exc)
            return False

    def get_line_loading(self) -> Dict[int, float]:
        """Return ``{line_index: loading_percent}`` after a successful PF.

        Returns
        -------
        dict
        """
        if "res_line" not in self.net or self.net.res_line is None:
            return {}
        loading = {}
        for idx in self.net.res_line.index:
            val = self.net.res_line.at[idx, "loading_percent"]
            loading[int(idx)] = float(val) if not np.isnan(val) else 0.0
        return loading

    def get_tripped_lines_from_cascade(self, cascade_result: Dict[str, Any]) -> List[int]:
        """Extract line indices that tripped during a cascade.

        Parameters
        ----------
        cascade_result : dict
            Output from ``CascadingSimulator.simulate_cascade``.

        Returns
        -------
        list of int
        """
        return [int(i) for i in cascade_result.get("tripped_lines", [])]


class HazardBackend:
    """Parse and project GIS hazard footprints.

    Parameters
    ----------
    geojson_feature : dict
        A GeoJSON Feature dict.
    """

    def __init__(self, geojson_feature: Dict[str, Any]) -> None:
        self.feature = geojson_feature

    @classmethod
    def from_file(cls, filepath: str) -> "HazardBackend":
        """Load a GeoJSON file (single Feature or FeatureCollection).

        Parameters
        ----------
        filepath : str

        Returns
        -------
        HazardBackend
        """
        with open(filepath, "r") as fh:
            data = json.load(fh)
        if data.get("type") == "FeatureCollection":
            feature = data["features"][0]
        else:
            feature = data
        return cls(feature)

    def get_polygon_coordinates(self) -> Optional[List[Tuple[float, float]]]:
        """Return exterior polygon ring as list of (lat, lon) tuples.

        Returns
        -------
        list of tuple or None
        """
        geom = self.feature.get("geometry", {})
        if geom.get("type") != "Polygon":
            return None
        coords = geom.get("coordinates", [[]])[0]
        return [(float(c[1]), float(c[0])) for c in coords]

    def intersects_bus(self, bus_lat: float, bus_lon: float) -> bool:
        """Check whether a point lies inside the hazard polygon.

        Parameters
        ----------
        bus_lat : float
        bus_lon : float

        Returns
        -------
        bool
        """
        try:
            from shapely.geometry import Point, shape
        except ImportError:
            return False
        geom = self.feature.get("geometry")
        if geom is None:
            return False
        poly = shape(geom)
        return poly.contains(Point(bus_lon, bus_lat))


class CascadeAnimatorBackend:
    """Replay a cascade iteration log and annotate network state."""

    def __init__(
        self,
        grid_backend: GridBackend,
        cascade_history: List[Dict[str, Any]],
    ) -> None:
        self.grid = grid_backend
        self.history = cascade_history

    def frame_at(self, iteration: int) -> Dict[str, Any]:
        """Return annotated network state for a given iteration.

        Parameters
        ----------
        iteration : int
            0-based cascade step.

        Returns
        -------
        dict
            ``{"lines": {idx: {"loading": float, "tripped": bool}},
            "islands": list of list of int}``.
        """
        if iteration < 0 or iteration >= len(self.history):
            return {"lines": {}, "islands": []}

        frame = self.history[iteration]
        tripped = set(frame.get("tripped_lines", []))
        loading = frame.get("loading_percent", [])

        line_info: Dict[int, Dict[str, Any]] = {}
        for li in self.grid.net.line.index:
            line_info[int(li)] = {
                "loading": loading[int(li)] if int(li) < len(loading) else 0.0,
                "tripped": int(li) in tripped,
            }

        return {
            "lines": line_info,
            "islands": frame.get("islands", []),
            "tripped_lines": sorted(tripped),
        }


class ScorecardBackend:
    """Compute and format RRS resilience scorecards.

    Parameters
    ----------
    rrs_report : dict
        Output from ``RRSReportGenerator.generate_report``.
    """

    def __init__(self, rrs_report: Dict[str, Any]) -> None:
        self.report = rrs_report

    def get_kpis(self) -> Dict[str, float]:
        """Return flat KPI dictionary.

        Returns
        -------
        dict
        """
        kpis = self.report.get("key_performance_indicators", {})
        return {
            "NPV ($)": float(kpis.get("npv_usd", 0.0)),
            "BCR": float(kpis.get("bcr", 0.0)),
            "IRR (%)": (float(kpis.get("irr", 0.0)) * 100) if kpis.get("irr") is not None else 0.0,
            "Avoided EENS (MWh)": float(kpis.get("avoided_eens_mwh", 0.0)),
            "Avoided Loss ($)": float(kpis.get("avoided_loss_usd", 0.0)),
        }

    def get_grade(self) -> str:
        """Return project survival grade (e.g. ``"A+"``)."""
        return str(self.report.get("resilience_of_the_project", {}).get("grade", "N/A"))

    def get_community_score(self) -> float:
        """Return community impact score (0–100)."""
        return float(
            self.report.get("resilience_through_the_project", {}).get("community_impact_score", 0.0)
        )

    def to_dataframe(self) -> "pd.DataFrame":
        """Return a tidy DataFrame of all metrics for Plotly charts.

        Returns
        -------
        pd.DataFrame
        """
        import pandas as pd

        rows: List[Dict[str, Any]] = []
        for key, val in self.get_kpis().items():
            rows.append({"metric": key, "value": val})
        rows.append({"metric": "Community Impact Score", "value": self.get_community_score()})
        return pd.DataFrame(rows)


class SimulationRunner:
    """Lightweight wrapper to run backend simulations with progress hooks.

    Parameters
    ----------
    grid_backend : GridBackend
    """

    def __init__(self, grid_backend: GridBackend) -> None:
        self.grid = grid_backend

    def run_cascade(
        self, initial_failed_assets: List[str], **sim_kwargs: Any
    ) -> Dict[str, Any]:
        """Run a cascading simulation and return the result dict.

        Parameters
        ----------
        initial_failed_assets : list of str
        **sim_kwargs
            Passed to ``CascadingSimulator`` constructor.

        Returns
        -------
        dict
        """
        from resilient_blackout.grid.cascade import CascadingSimulator
        from resilient_blackout.grid.network import GridModel

        grid_model = GridModel(self.grid.net, {})
        sim = CascadingSimulator(grid_model, **sim_kwargs)
        return sim.simulate_cascade(initial_failed_assets)

    def run_opf_schedule(
        self,
        load_profile: np.ndarray,
        storage_specs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run a multi-period OPF schedule.

        Parameters
        ----------
        load_profile : np.ndarray
        storage_specs : list of dict or None

        Returns
        -------
        dict
        """
        from resilient_blackout.grid.multi_period_opf import MultiPeriodOPFScheduler

        T = load_profile.shape[0]
        scheduler = MultiPeriodOPFScheduler(horizon_steps=T, dt_hours=1.0)
        return scheduler.build_schedule(self.grid.net, load_profile, storage_specs=storage_specs)

    def generate_rrs_report(
        self,
        avoided_loss_result: Dict[str, Any],
        project_name: str = "Dashboard Project",
    ) -> Dict[str, Any]:
        """Generate an RRS scorecard from an avoided-loss result.

        Parameters
        ----------
        avoided_loss_result : dict
        project_name : str

        Returns
        -------
        dict
        """
        from resilient_blackout.reporting.rrs_scorecard import RRSReportGenerator

        gen = RRSReportGenerator(project_name=project_name)
        return gen.generate_report(avoided_loss_result)
