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
Cascading failure and optimal load-shedding simulator.

Implements the ``CascadingSimulator`` class that models overload-driven
line and transformer trips propagating through an electrical network,
followed by optimal or proportional load shedding — corresponding to the
"Absorb" dimension of the M-A-R-C (Mitigation, Adaptation, Resilience,
Coping) framework.

Optimised for Monte Carlo loops: works on a deep copy of the network,
uses vectorised overload scans, and batches Bernoulli trials.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from resilient_blackout.grid.network import GridModel

logger = logging.getLogger(__name__)


class CascadingSimulator:
    """Simulates overload-driven cascading failures and load shedding.

    Starting from an initial set of physically failed assets, iteratively
    solves power flow, detects overloaded lines and transformers, applies
    probabilistic tripping, and — when generation shortfalls occur —
    performs optimal or proportional load shedding.

    Parameters
    ----------
    grid_model : GridModel
        The grid model to simulate on.  The original network is never
        mutated; each call to :meth:`simulate_cascade` operates on a
        deep copy.
    tolerance_factor : float
        Loading threshold above which a line or transformer trips with
        certainty.  Between 100 % and *tolerance_factor*, the trip
        probability scales linearly from 0 to 1.  Default 1.2 (120 %).
    max_iterations : int
        Maximum number of cascade propagation rounds before forced
        termination.  Default 50.
    rng : numpy.random.Generator or None
        Random number generator for Bernoulli trials.  If ``None``, a
        new ``default_rng`` is created.  Pass a seeded generator for
        reproducible Monte Carlo runs.

    Attributes
    ----------
    grid_model : GridModel
    tolerance_factor : float
    max_iterations : int
    rng : numpy.random.Generator
    """

    def __init__(
        self,
        grid_model: GridModel,
        tolerance_factor: float = 1.2,
        max_iterations: int = 50,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if tolerance_factor <= 1.0:
            raise ValueError(
                f"tolerance_factor must be > 1.0, got {tolerance_factor}"
            )
        if max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {max_iterations}"
            )

        self.grid_model: GridModel = grid_model
        self.tolerance_factor: float = tolerance_factor
        self.max_iterations: int = max_iterations
        self.rng: np.random.Generator = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_cascade(
        self, initial_failed_assets: List[str]
    ) -> Dict[str, Any]:
        """Run the full cascading-failure simulation.

        Parameters
        ----------
        initial_failed_assets : list of str
            Asset IDs that are initially tripped (e.g., from physical
            hazard damage).

        Returns
        -------
        dict
            Keys:

            - ``converged`` (bool) — whether the cascade terminated
              naturally (no further overloads) rather than hitting
              ``max_iterations``.
            - ``iterations`` (int) — number of cascade rounds executed.
            - ``tripped_lines`` (list of int) — pandapower line indices
              tripped during the cascade (excluding initial failures).
            - ``tripped_trafos`` (list of int) — pandapower transformer
              indices tripped.
            - ``islands`` (list of list of int) — bus indices in each
              detected island after the cascade.
            - ``total_load_shed_mw`` (float) — total load shed across
              all islands (MW).
            - ``final_loading`` (list of float) — final line loading
              percentages.
            - ``shed_per_bus`` (dict) — bus index → MW shed.
        """
        import pandapower as pp

        net = copy.deepcopy(self.grid_model.net)
        mapping = dict(self.grid_model.bus_mapping)

        tripped_lines: List[int] = []
        tripped_trafos: List[int] = []

        # Step a: apply initial trips
        self._apply_initial_trips(net, mapping, initial_failed_assets)

        # Steps b–f: cascade loop
        for iteration in range(1, self.max_iterations + 1):
            islands = self._detect_islands(net)

            # Step c: solve power flow per island
            self._solve_islands(net, islands)

            # Step d: scan overloads
            overloaded_lines, overloaded_trafos = self._scan_overloads(net)

            if not overloaded_lines and not overloaded_trafos:
                # Cascade naturally terminated
                return self._build_result(
                    net=net,
                    converged=True,
                    iterations=iteration,
                    tripped_lines=tripped_lines,
                    tripped_trafos=tripped_trafos,
                    islands=islands,
                )

            # Step e: Bernoulli trials
            new_line_trips, new_trafo_trips = self._trip_overloaded(
                net, overloaded_lines, overloaded_trafos
            )

            tripped_lines.extend(new_line_trips)
            tripped_trafos.extend(new_trafo_trips)

        # Max iterations reached without convergence
        islands = self._detect_islands(net)
        return self._build_result(
            net=net,
            converged=False,
            iterations=self.max_iterations,
            tripped_lines=tripped_lines,
            tripped_trafos=tripped_trafos,
            islands=islands,
        )

    # ------------------------------------------------------------------
    # Step a: initial trips
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_initial_trips(
        net: Any,
        mapping: Dict[str, Any],
        failed_assets: List[str],
    ) -> None:
        """Set ``in_service=False`` for initially failed assets.

        Parameters
        ----------
        net : pandapowerNet
            The (copied) network.
        mapping : dict
            Asset ID → ``{"type": ..., "index": ...}``.
        failed_assets : list of str
            Asset IDs to disconnect.
        """
        for asset_id in failed_assets:
            entry = mapping.get(asset_id)
            if entry is None:
                logger.warning("Initial asset '%s' not in mapping; skipped.", asset_id)
                continue
            el_type = entry["type"]
            idx = entry["index"]
            if el_type == "bus":
                net.bus.at[idx, "in_service"] = False
            elif el_type == "line":
                net.line.at[idx, "in_service"] = False
            elif el_type == "trafo":
                net.trafo.at[idx, "in_service"] = False

    # ------------------------------------------------------------------
    # Step b: island detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_islands(net: Any) -> List[List[int]]:
        """Find electrically isolated subgraphs (islands).

        Uses pandapower's topology helper to identify connected
        components in the bus–line graph, considering only in-service
        elements.

        Parameters
        ----------
        net : pandapowerNet
            The network.

        Returns
        -------
        list of list of int
            Each inner list contains the bus indices belonging to one
            island.
        """
        import pandapower as pp

        try:
            pp.topology.noseparate_isolated_areas(net)
            components = pp.topology.connected_components(
                net, respect_switches=False, respect_in_service=True
            )
            return [list(comp) for comp in components]
        except Exception:
            logger.warning("Island detection via pandapower failed; treating as single island.")
            active_buses = net.bus[net.bus.in_service].index.to_list()
            return [active_buses] if active_buses else []

    # ------------------------------------------------------------------
    # Step c: per-island power flow
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_islands(net: Any, islands: List[List[int]]) -> None:
        """Solve power flow for each island independently.

        Tries Newton-Raphson first; falls back to DC on failure.

        Parameters
        ----------
        net : pandapowerNet
        islands : list of list of int
            Bus groups to solve.
        """
        import pandapower as pp

        for bus_group in islands:
            if len(bus_group) == 0:
                continue
            try:
                pp.runpp(net, numba=False, bus_estimation=bus_group)
            except pp.LoadflowNotConverged:
                logger.debug("NR failed for island %s; trying DC.", bus_group[:3])
                try:
                    pp.rundcpp(net)
                except Exception:
                    logger.warning("DC power flow also failed for island %s.", bus_group[:3])

    # ------------------------------------------------------------------
    # Step d: overload scan
    # ------------------------------------------------------------------

    def _scan_overloads(self, net: Any) -> Tuple[List[int], List[int]]:
        """Identify lines and transformers exceeding 100 % loading.

        Parameters
        ----------
        net : pandapowerNet
            Solved network.

        Returns
        -------
        tuple of (list of int, list of int)
            Indices of overloaded lines and transformers.
        """
        overloaded_lines: List[int] = []
        overloaded_trafos: List[int] = []

        if hasattr(net, "res_line") and len(net.res_line) > 0:
            loading = net.res_line.loading_percent.values
            in_service = net.line.in_service.values
            mask = (loading > 100.0) & in_service
            overloaded_lines = net.line.index[mask].tolist()

        if hasattr(net, "res_trafo") and len(net.res_trafo) > 0:
            loading = net.res_trafo.loading_percent.values
            in_service = net.trafo.in_service.values
            mask = (loading > 100.0) & in_service
            overloaded_trafos = net.trafo.index[mask].tolist()

        return overloaded_lines, overloaded_trafos

    # ------------------------------------------------------------------
    # Step e: Bernoulli trip trials
    # ------------------------------------------------------------------

    def _trip_overloaded(
        self,
        net: Any,
        overloaded_lines: List[int],
        overloaded_trafos: List[int],
    ) -> Tuple[List[int], List[int]]:
        """Execute probabilistic tripping of overloaded elements.

        Trip probability is 1.0 if loading ≥ ``tolerance_factor``,
        otherwise scales linearly from 0 at 100 % to 1 at
        ``tolerance_factor``.

        Parameters
        ----------
        net : pandapowerNet
        overloaded_lines : list of int
        overloaded_trafos : list of int

        Returns
        -------
        tuple of (list of int, list of int)
            Indices of elements that were tripped.
        """
        tripped_lines = self._trip_elements(
            net, "line", overloaded_lines, net.res_line.loading_percent
        )
        tripped_trafos: List[int] = []
        if overloaded_trafos and hasattr(net, "res_trafo"):
            tripped_trafos = self._trip_elements(
                net, "trafo", overloaded_trafos, net.res_trafo.loading_percent
            )
        return tripped_lines, tripped_trafos

    def _trip_elements(
        self,
        net: Any,
        element_type: str,
        candidates: List[int],
        loading_series: pd.Series,
    ) -> List[int]:
        """Apply Bernoulli trials to a list of candidate elements.

        Parameters
        ----------
        net : pandapowerNet
        element_type : str
            ``"line"`` or ``"trafo"``.
        candidates : list of int
            Indices to evaluate.
        loading_series : pd.Series
            Loading percentages indexed by element index.

        Returns
        -------
        list of int
            Indices that were tripped.
        """
        if not candidates:
            return []

        loadings = loading_series.loc[candidates].values / 100.0
        tf = self.tolerance_factor

        probs = np.where(
            loadings >= tf,
            1.0,
            (loadings - 1.0) / (tf - 1.0),
        )
        probs = np.clip(probs, 0.0, 1.0)

        trials = self.rng.random(len(candidates))
        trip_mask = trials < probs

        tripped: List[int] = []
        table = getattr(net, element_type)
        for i, should_trip in enumerate(trip_mask):
            if should_trip:
                idx = candidates[i]
                table.at[idx, "in_service"] = False
                tripped.append(idx)
                logger.debug(
                    "Tripped %s[%d] (loading=%.1f%%, prob=%.3f)",
                    element_type,
                    idx,
                    loadings[i] * 100,
                    probs[i],
                )

        return tripped

    # ------------------------------------------------------------------
    # Step g: load shedding
    # ------------------------------------------------------------------

    def _shed_load(self, net: Any, islands: List[List[int]]) -> Tuple[float, Dict[int, float]]:
        """Perform optimal or proportional load shedding for each island.

        For each island with a generation deficit, attempts pandapower's
        OPF first; falls back to proportional shedding if OPF is
        unavailable or fails.

        Parameters
        ----------
        net : pandapowerNet
        islands : list of list of int

        Returns
        -------
        tuple of (float, dict)
            Total MW shed and per-bus shed dictionary.
        """
        total_shed = 0.0
        shed_per_bus: Dict[int, float] = {}

        for bus_group in islands:
            if not bus_group:
                continue
            deficit = self._compute_island_deficit(net, bus_group)
            if deficit <= 0:
                continue

            shed = self._opf_shed(net, bus_group, deficit)
            if shed is None:
                shed = self._proportional_shed(net, bus_group, deficit)

            total_shed += sum(shed.values())
            shed_per_bus.update(shed)

        return total_shed, shed_per_bus

    @staticmethod
    def _compute_island_deficit(net: Any, bus_group: List[int]) -> float:
        """Compute generation shortfall for an island in MW.

        Parameters
        ----------
        net : pandapowerNet
        bus_group : list of int

        Returns
        -------
        float
            Positive value means deficit (MW); ≤ 0 means sufficient
            generation.
        """
        total_gen = 0.0
        total_load = 0.0

        for bid in bus_group:
            gen_mask = net.gen.bus == bid
            total_gen += net.gen.loc[gen_mask & net.gen.in_service, "p_mw"].sum()

            sgen_mask = net.sgen.bus == bid
            total_gen += net.sgen.loc[sgen_mask & net.sgen.in_service, "p_mw"].sum()

            load_mask = net.load.bus == bid
            total_load += net.load.loc[load_mask & net.load.in_service, "p_mw"].sum()

        ext_grid_mask = net.ext_grid.bus.isin(bus_group)
        total_gen += net.ext_grid.loc[ext_grid_mask & net.ext_grid.in_service, "max_p_mw"].sum()

        return max(0.0, total_load - total_gen)

    @staticmethod
    def _opf_shed(
        net: Any, bus_group: List[int], deficit: float
    ) -> Optional[Dict[int, float]]:
        """Attempt optimal load shedding via pandapower OPF.

        Parameters
        ----------
        net : pandapowerNet
        bus_group : list of int
        deficit : float
            Required shed amount in MW.

        Returns
        -------
        dict or None
            Bus index → MW shed, or ``None`` if OPF failed.
        """
        import pandapower as pp

        try:
            for bid in bus_group:
                load_mask = net.load.bus == bid
                net.load.loc[load_mask & net.load.in_service, "controllable"] = True
                net.load.loc[load_mask & net.load.in_service, "max_p_mw"] = net.load.loc[
                    load_mask & net.load.in_service, "p_mw"
                ]
                net.load.loc[load_mask & net.load.in_service, "min_p_mw"] = 0.0
                net.load.loc[load_mask & net.load.in_service, "cost_per_mw"] = 10000.0

            pp.runopp(net, numba=False)

            shed: Dict[int, float] = {}
            for bid in bus_group:
                load_mask = net.load.bus == bid
                original = net.load.loc[load_mask & net.load.in_service, "max_p_mw"].sum()
                actual = net.res_load.loc[load_mask & net.load.in_service, "p_mw"].sum()
                bus_shed = original - actual
                if bus_shed > 0.01:
                    shed[bid] = bus_shed

            return shed if shed else None
        except Exception:
            logger.debug("OPF load shedding failed; falling back to proportional.")
            return None

    @staticmethod
    def _proportional_shed(
        net: Any, bus_group: List[int], deficit: float
    ) -> Dict[int, float]:
        """Shed load proportionally across all buses in an island.

        Parameters
        ----------
        net : pandapowerNet
        bus_group : list of int
        deficit : float
            Required shed amount in MW.

        Returns
        -------
        dict
            Bus index → MW shed.
        """
        total_load = 0.0
        bus_loads: Dict[int, float] = {}

        for bid in bus_group:
            load_mask = net.load.bus == bid
            bl = net.load.loc[load_mask & net.load.in_service, "p_mw"].sum()
            if bl > 0:
                bus_loads[bid] = bl
                total_load += bl

        if total_load <= 0:
            return {}

        shed: Dict[int, float] = {}
        remaining = deficit
        for bid, bl in sorted(bus_loads.items()):
            fraction = bl / total_load
            bus_shed = min(bl, remaining * fraction)
            shed[bid] = bus_shed
            remaining -= bus_shed

        return shed

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------

    def _build_result(
        self,
        net: Any,
        converged: bool,
        iterations: int,
        tripped_lines: List[int],
        tripped_trafos: List[int],
        islands: List[List[int]],
    ) -> Dict[str, Any]:
        """Assemble the final result dictionary, including load shedding.

        Parameters
        ----------
        net : pandapowerNet
        converged : bool
        iterations : int
        tripped_lines : list of int
        tripped_trafos : list of int
        islands : list of list of int

        Returns
        -------
        dict
        """
        total_shed, shed_per_bus = self._shed_load(net, islands)

        final_loading: List[float] = []
        if hasattr(net, "res_line") and len(net.res_line) > 0:
            final_loading = net.res_line.loading_percent.to_list()

        return {
            "converged": converged,
            "iterations": iterations,
            "tripped_lines": tripped_lines,
            "tripped_trafos": tripped_trafos,
            "islands": islands,
            "total_load_shed_mw": total_shed,
            "final_loading": final_loading,
            "shed_per_bus": shed_per_bus,
        }
