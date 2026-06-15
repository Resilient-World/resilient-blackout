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
Electrical network integration via pandapower.

Provides the ``GridModel`` class for parsing open transmission grid models,
running steady-state power flow with progressive convergence fallback, and
dynamically degrading or disconnecting assets to simulate physical hazard
impacts.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

import pandapower as pp
from pandapower.auxiliary import pandapowerNet

logger = logging.getLogger(__name__)


class _BusMappingEntry(TypedDict):
    """Structure of a single entry in ``GridModel.bus_mapping``."""

    type: str
    index: int


class GridModel:
    """Wraps a pandapower network with asset mapping and power-flow control.

    Parses open transmission grid models (MATPOWER ``.m``, pandapower
    ``.json``, ``.xlsx``), maintains a bidirectional mapping between
    logical asset IDs and pandapower element indices, and provides
    steady-state power flow with progressive convergence fallback.

    Parameters
    ----------
    net : pandapowerNet
        A pandapower network object.
    bus_mapping : dict
        Mapping from logical ``asset_id`` strings to pandapower element
        descriptors of the form ``{"type": "bus"|"line"|"trafo", "index": int}``.

    Attributes
    ----------
    net : pandapowerNet
    bus_mapping : dict
    """

    _SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".m", ".json", ".xlsx", ".xls"})

    def __init__(
        self,
        net: pandapowerNet,
        bus_mapping: Optional[Dict[str, _BusMappingEntry]] = None,
    ) -> None:
        self.net: pandapowerNet = net
        self.bus_mapping: Dict[str, _BusMappingEntry] = bus_mapping or {}

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> GridModel:
        """Parse an open transmission grid model from a file.

        Auto-detects the format based on the file extension:

        - ``.m`` — MATPOWER case file (via ``pp.converter.from_mpc``).
        - ``.json`` — pandapower JSON export (via ``pp.from_json``).
        - ``.xlsx`` / ``.xls`` — pandapower Excel export (via ``pp.from_excel``).

        After loading, builds a ``bus_mapping`` from bus names and line
        names present in the network.

        Parameters
        ----------
        filepath : str or Path
            Path to the grid model file.

        Returns
        -------
        GridModel
            Initialised model with the parsed network and auto-generated
            bus mapping.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        ValueError
            If the file extension is not supported.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Grid model file not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in cls._SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension '{suffix}'. "
                f"Supported: {sorted(cls._SUPPORTED_EXTENSIONS)}"
            )

        if suffix == ".m":
            net = pp.converter.from_mpc(str(path))
        elif suffix == ".json":
            net = pp.from_json(str(path))
        else:
            net = pp.from_excel(str(path))

        mapping = cls._build_bus_mapping(net)
        return cls(net, mapping)

    @classmethod
    def from_openstreetmap_data(cls, filepath: Union[str, Path]) -> GridModel:
        """Parse an open transmission model (alias for :meth:`from_file`).

        This is a convenience alias for ``from_file``.  For live
        OpenStreetMap extraction see :meth:`from_osm_bbox`.

        Parameters
        ----------
        filepath : str or Path
            Path to the grid model file.

        Returns
        -------
        GridModel
        """
        return cls.from_file(filepath)

    @classmethod
    def from_osm_bbox(
        cls,
        bbox: Tuple[float, float, float, float],
        snap_threshold_m: float = 50.0,
        default_voltage_kv: float = 12.47,
    ) -> GridModel:
        """Build a grid model from an OpenStreetMap bounding box.

        Queries the Overpass API, reconstructs topology, and runs
        progressive AC power flow validation.

        Parameters
        ----------
        bbox : tuple of float
            ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84.
        snap_threshold_m : float
            Spatial snapping threshold for line endpoints.  Default 50.
        default_voltage_kv : float
            Fallback voltage when OSM tags are absent.  Default 12.47.

        Returns
        -------
        GridModel
        """
        from resilient_blackout.grid.osm_pipeline import OSMGridBuilder

        builder = OSMGridBuilder(
            snap_threshold_m=snap_threshold_m,
            default_voltage_kv=default_voltage_kv,
        )
        net, _result = builder.build_from_bbox(bbox)
        mapping = cls._build_bus_mapping(net)
        return cls(net, mapping)

    @staticmethod
    def _build_bus_mapping(net: pandapowerNet) -> Dict[str, _BusMappingEntry]:
        """Auto-generate a bus mapping from network element names.

        Parameters
        ----------
        net : pandapowerNet
            The loaded network.

        Returns
        -------
        dict
            Mapping from name strings to ``{"type": ..., "index": ...}``.
        """
        mapping: Dict[str, _BusMappingEntry] = {}

        for idx in net.bus.index:
            name = net.bus.at[idx, "name"]
            if name and isinstance(name, str):
                mapping[name] = {"type": "bus", "index": int(idx)}

        for idx in net.line.index:
            name = net.line.at[idx, "name"]
            if name and isinstance(name, str):
                mapping[name] = {"type": "line", "index": int(idx)}

        if hasattr(net, "trafo") and len(net.trafo) > 0:
            for idx in net.trafo.index:
                name = net.trafo.at[idx, "name"]
                if name and isinstance(name, str):
                    mapping[name] = {"type": "trafo", "index": int(idx)}

        return mapping

    # ------------------------------------------------------------------
    # Power flow
    # ------------------------------------------------------------------

    def run_baseline_power_flow(self) -> Dict[str, Any]:
        """Execute steady-state power flow with progressive convergence fallback.

        Attempts to solve the AC power flow in the following order:

        1. **Newton-Raphson** (default ``pp.runpp``).
        2. **Relaxed Newton-Raphson** — increases ``max_iteration`` to 30,
           enables ``enforce_q_lims``, and switches to the
           ``'bfsw'`` algorithm.
        3. **DC power flow** — ``pp.rundcpp`` as a guaranteed-convergent
           linear approximation.

        Each fallback step emits a warning via the ``logging`` module.

        Returns
        -------
        dict
            Keys:

            - ``converged`` (bool) — whether any solver succeeded.
            - ``solver_used`` (str) — ``"nr"``, ``"bfsw"``, or ``"dc"``.
            - ``vm_pu`` (list of float) — per-unit voltage magnitudes at
              each bus.
            - ``loading_percent`` (list of float) — line loading
              percentages.
            - ``total_losses_mw`` (float) — total active power losses in
              MW (0 for DC).
        """
        strategies = [
            ("nr", self._try_nr_default),
            ("bfsw", self._try_nr_relaxed),
            ("dc", self._try_dc),
        ]

        for label, strategy in strategies:
            try:
                result = strategy()
                result["converged"] = True
                result["solver_used"] = label
                return result
            except pp.LoadflowNotConverged:
                logger.warning(
                    "Power flow solver '%s' did not converge; trying next fallback.", label
                )

        return {
            "converged": False,
            "solver_used": "none",
            "vm_pu": [],
            "loading_percent": [],
            "total_losses_mw": 0.0,
        }

    def _try_nr_default(self) -> Dict[str, Any]:
        """Attempt default Newton-Raphson AC power flow."""
        pp.runpp(self.net, numba=False)
        return self._extract_results()

    def _try_nr_relaxed(self) -> Dict[str, Any]:
        """Attempt relaxed Newton-Raphson with BFSW algorithm."""
        pp.runpp(
            self.net,
            algorithm="bfsw",
            max_iteration=30,
            enforce_q_lims=True,
            numba=False,
        )
        return self._extract_results()

    def _try_dc(self) -> Dict[str, Any]:
        """Attempt DC power flow (linear approximation)."""
        pp.rundcpp(self.net)
        return self._extract_results(dc=True)

    def _extract_results(self, dc: bool = False) -> Dict[str, Any]:
        """Extract standard result metrics from the solved network.

        Parameters
        ----------
        dc : bool
            If ``True``, treats the network as a DC solution (no reactive
            power, losses are zero).

        Returns
        -------
        dict
            With keys ``vm_pu``, ``loading_percent``, ``total_losses_mw``.
        """
        vm_pu: List[float] = self.net.res_bus.vm_pu.to_list()

        loading: List[float] = []
        if hasattr(self.net.res_line, "loading_percent"):
            loading = self.net.res_line.loading_percent.to_list()

        total_losses_mw: float = 0.0
        if not dc and hasattr(self.net.res_line, "pl_mw"):
            total_losses_mw = float(self.net.res_line.pl_mw.sum())

        return {
            "vm_pu": vm_pu,
            "loading_percent": loading,
            "total_losses_mw": total_losses_mw,
        }

    # ------------------------------------------------------------------
    # Dynamic asset modification
    # ------------------------------------------------------------------

    def degrade_line_capacity(self, line_id: int, derating_factor: float) -> None:
        """Dynamically reduce the thermal limit of a transmission line.

        Multiplies ``max_i_ka`` by *derating_factor* to model ambient
        heat stress, wildfire risk deratings, or conductor ageing.

        Parameters
        ----------
        line_id : int
            Pandapower line index.
        derating_factor : float
            Multiplicative factor in (0, 1].  A value of 0.8 reduces
            capacity to 80 % of the original rating.

        Raises
        ------
        ValueError
            If *derating_factor* is not in (0, 1].
        KeyError
            If *line_id* does not exist in the network.
        """
        if not (0 < derating_factor <= 1):
            raise ValueError(
                f"derating_factor must be in (0, 1], got {derating_factor}"
            )
        if line_id not in self.net.line.index:
            raise KeyError(f"Line index {line_id} not found in network")

        self.net.line.at[line_id, "max_i_ka"] *= derating_factor
        logger.info(
            "Line %d max_i_ka derated by factor %.3f → %.6f kA",
            line_id,
            derating_factor,
            self.net.line.at[line_id, "max_i_ka"],
        )

    def disconnect_assets(self, failed_assets: List[str]) -> int:
        """Set ``in_service=False`` for assets that have physically failed.

        Looks up each asset ID in ``self.bus_mapping`` and sets the
        ``in_service`` flag to ``False`` on the corresponding pandapower
        element (bus, line, or transformer).

        Parameters
        ----------
        failed_assets : list of str
            Asset IDs to disconnect.

        Returns
        -------
        int
            Number of assets successfully disconnected.

        Notes
        -----
        Assets not found in ``bus_mapping`` are silently skipped (a
        warning is logged).
        """
        count = 0
        for asset_id in failed_assets:
            entry = self.bus_mapping.get(asset_id)
            if entry is None:
                logger.warning(
                    "Asset '%s' not found in bus_mapping; skipping disconnection.",
                    asset_id,
                )
                continue

            element_type = entry["type"]
            idx = entry["index"]

            if element_type == "bus":
                self.net.bus.at[idx, "in_service"] = False
            elif element_type == "line":
                self.net.line.at[idx, "in_service"] = False
            elif element_type == "trafo":
                self.net.trafo.at[idx, "in_service"] = False
            else:
                logger.warning("Unknown element type '%s' for asset '%s'", element_type, asset_id)
                continue

            count += 1
            logger.info("Disconnected asset '%s' (%s[%d])", asset_id, element_type, idx)

        return count

    def __repr__(self) -> str:
        n_buses = len(self.net.bus)
        n_lines = len(self.net.line)
        n_mapped = len(self.bus_mapping)
        return (
            f"GridModel(buses={n_buses}, lines={n_lines}, "
            f"mapped_assets={n_mapped})"
        )
