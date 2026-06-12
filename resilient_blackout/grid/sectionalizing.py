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
Optimized overload sectionalizing and line-switching module.

Provides ``GridSectionalizer`` for identifying optimal controlled
switching actions to isolate faulted sections and minimise system-wide
load loss during propagating cascades, using spectral bisection,
minimum-cut graph partitioning, and Frobenius-norm susceptance
optimisation.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandapower as pp
from scipy.sparse.linalg import eigsh

from resilient_blackout.grid.network import GridModel

logger = logging.getLogger(__name__)

_EPS: float = 1e-10


class GridSectionalizer:
    """Controlled islanding via graph partitioning and susceptance optimisation.

    Identifies optimal breaker operations to isolate overloaded sections
    while minimising the Frobenius distance between original and
    post-switching bus susceptance matrices.

    Parameters
    ----------
    grid_model : GridModel
        The grid model to sectionalize.
    overload_threshold_pct : float
        Lines exceeding this loading percentage trigger sectionalizing.
        Default 100.0.
    min_island_size : int
        Minimum number of buses per island.  Default 2.

    Attributes
    ----------
    grid_model : GridModel
    overload_threshold_pct : float
    min_island_size : int
    """

    def __init__(
        self,
        grid_model: GridModel,
        overload_threshold_pct: float = 100.0,
        min_island_size: int = 2,
    ) -> None:
        if overload_threshold_pct <= 0:
            raise ValueError("overload_threshold_pct must be positive")
        if min_island_size < 1:
            raise ValueError("min_island_size must be >= 1")

        self.grid_model = grid_model
        self.overload_threshold_pct = overload_threshold_pct
        self.min_island_size = min_island_size

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_weighted_graph(self, net: Any) -> Tuple[nx.Graph, Dict[int, int], Dict[int, int]]:
        """Build weighted NetworkX graph from active pandapower network.

        Edge weights are the inverse of the loading margin, so heavily
        loaded lines are cheaper to cut.

        Parameters
        ----------
        net : pandapowerNet

        Returns
        -------
        tuple of (nx.Graph, dict, dict)
            Graph, bus index → node id mapping, edge index → (u, v) mapping.
        """
        G = nx.Graph()
        bus_map: Dict[int, int] = {}
        line_map: Dict[int, Tuple[int, int]] = {}

        for i, bid in enumerate(net.bus.index):
            if net.bus.at[bid, "in_service"]:
                G.add_node(i, bus=bid)
                bus_map[bid] = i

        for lidx in net.line.index:
            if not net.line.at[lidx, "in_service"]:
                continue
            fb = int(net.line.at[lidx, "from_bus"])
            tb = int(net.line.at[lidx, "to_bus"])
            if fb not in bus_map or tb not in bus_map:
                continue

            u, v = bus_map[fb], bus_map[tb]
            loading = 50.0
            if hasattr(net, "res_line") and lidx in net.res_line.index:
                loading = float(net.res_line.at[lidx, "loading_percent"])
            margin = max(_EPS, abs(100.0 - loading))
            weight = 1.0 / margin

            G.add_edge(u, v, weight=weight, line=lidx)
            line_map[lidx] = (u, v)

        return G, bus_map, line_map

    # ------------------------------------------------------------------
    # Overload cluster detection
    # ------------------------------------------------------------------

    def detect_overload_clusters(self, net: Any) -> List[Set[int]]:
        """Find connected components of overloaded lines.

        Parameters
        ----------
        net : pandapowerNet

        Returns
        -------
        list of set of int
            Each set contains line indices forming a connected overload
            cluster.
        """
        G_over = nx.Graph()

        for lidx in net.line.index:
            if not net.line.at[lidx, "in_service"]:
                continue
            if hasattr(net, "res_line") and lidx in net.res_line.index:
                loading = float(net.res_line.at[lidx, "loading_percent"])
                if loading > self.overload_threshold_pct:
                    fb = int(net.line.at[lidx, "from_bus"])
                    tb = int(net.line.at[lidx, "to_bus"])
                    G_over.add_edge(fb, tb, line=lidx)

        clusters: List[Set[int]] = []
        for comp in nx.connected_components(G_over):
            lines = set()
            for u, v, data in G_over.subgraph(comp).edges(data=True):
                lines.add(data["line"])
            if lines:
                clusters.append(lines)

        return clusters

    # ------------------------------------------------------------------
    # Spectral bisection
    # ------------------------------------------------------------------

    def spectral_bisection(
        self, G: nx.Graph, overloaded_edges: Set[int]
    ) -> Optional[Tuple[List[int], List[int], List[int]]]:
        """Partition graph using Fiedler vector of weighted Laplacian.

        Parameters
        ----------
        G : nx.Graph
        overloaded_edges : set of int

        Returns
        -------
        tuple or None
            ``(island_a, island_b, cut_edges)`` or ``None`` if
            bisection fails.
        """
        if G.number_of_nodes() < 2:
            return None

        n = G.number_of_nodes()
        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        L = nx.laplacian_matrix(G, weight="weight").astype(np.float64)

        try:
            _, eigenvectors = eigsh(L, k=2, which="SM")
            fiedler = eigenvectors[:, 1]
        except Exception:
            logger.warning("Spectral bisection eigen-solve failed.")
            return None

        island_a = [nodes[i] for i in range(n) if fiedler[i] >= 0]
        island_b = [nodes[i] for i in range(n) if fiedler[i] < 0]

        if len(island_a) < self.min_island_size or len(island_b) < self.min_island_size:
            return None

        cut_edges: List[int] = []
        for u, v, data in G.edges(data=True):
            if (u in island_a and v in island_b) or (u in island_b and v in island_a):
                cut_edges.append(data["line"])

        return island_a, island_b, cut_edges

    # ------------------------------------------------------------------
    # Minimum-cut islanding
    # ------------------------------------------------------------------

    def minimum_cut_islanding(
        self, G: nx.Graph, overloaded_edges: Set[int]
    ) -> Optional[Tuple[List[int], List[int], List[int]]]:
        """Find minimum-weight cut separating overloaded region.

        Uses Stoer-Wagner minimum cut algorithm.

        Parameters
        ----------
        G : nx.Graph
        overloaded_edges : set of int

        Returns
        -------
        tuple or None
            ``(island_a, island_b, cut_edges)`` or ``None``.
        """
        if G.number_of_nodes() < 2:
            return None

        try:
            cut_value, partition = nx.stoer_wagner(G, weight="weight")
        except Exception:
            logger.warning("Stoer-Wagner minimum cut failed.")
            return None

        island_a = list(partition[0])
        island_b = list(partition[1])

        if len(island_a) < self.min_island_size or len(island_b) < self.min_island_size:
            return None

        cut_edges: List[int] = []
        for u, v, data in G.edges(data=True):
            if (u in island_a and v in island_b) or (u in island_b and v in island_a):
                cut_edges.append(data["line"])

        return island_a, island_b, cut_edges

    # ------------------------------------------------------------------
    # Susceptance matrix
    # ------------------------------------------------------------------

    @staticmethod
    def compute_susceptance_matrix(net: Any) -> np.ndarray:
        """Extract imaginary part of the bus admittance matrix.

        Parameters
        ----------
        net : pandapowerNet

        Returns
        -------
        np.ndarray
            B matrix of shape ``(n_buses, n_buses)``.
        """
        pp.rundcpp(net)
        Ybus = net._ppc["internal"]["Ybus"].todense()
        return np.asarray(Ybus.imag, dtype=np.float64)

    # ------------------------------------------------------------------
    # Switching optimisation
    # ------------------------------------------------------------------

    def optimize_switching(
        self,
        net: Any,
        candidate_cuts: List[Tuple[List[int], List[int], List[int]]],
    ) -> Optional[Dict[str, Any]]:
        """Evaluate candidate cuts and select the best one.

        Chooses the cut that minimises the Frobenius distance between
        original and post-switching susceptance matrices, subject to
        generation-demand balance.

        Parameters
        ----------
        net : pandapowerNet
        candidate_cuts : list of tuple

        Returns
        -------
        dict or None
            ``{"islands": list, "cut_edges": list,
            "frobenius_distance": float}``.
        """
        try:
            B_orig = self.compute_susceptance_matrix(net)
        except Exception:
            logger.warning("Could not compute original susceptance matrix.")
            return None

        best: Optional[Dict[str, Any]] = None
        best_dist = float("inf")

        for island_a, island_b, cut_edges in candidate_cuts:
            test_net = copy.deepcopy(net)
            for lidx in cut_edges:
                if lidx in test_net.line.index:
                    test_net.line.at[lidx, "in_service"] = False

            try:
                B_new = self.compute_susceptance_matrix(test_net)
            except Exception:
                continue

            diff = B_orig - B_new
            frob_dist = float(np.sum(diff * diff))

            balanced = self._check_generation_balance(test_net, [island_a, island_b])
            if not balanced:
                continue

            if frob_dist < best_dist:
                best_dist = frob_dist
                best = {
                    "islands": [island_a, island_b],
                    "cut_edges": cut_edges,
                    "frobenius_distance": frob_dist,
                }

        return best

    @staticmethod
    def _check_generation_balance(
        net: Any, islands: List[List[int]]
    ) -> bool:
        """Check generation-demand balance for each island.

        Parameters
        ----------
        net : pandapowerNet
        islands : list of list of int

        Returns
        -------
        bool
        """
        for island in islands:
            total_gen = 0.0
            total_load = 0.0

            for node in island:
                bid = net.bus.index[node] if node in net.bus.index else node
                gen_mask = (net.gen.bus == bid) & net.gen.in_service
                total_gen += net.gen.loc[gen_mask, "p_mw"].sum()

                sgen_mask = (net.sgen.bus == bid) & net.sgen.in_service
                total_gen += net.sgen.loc[sgen_mask, "p_mw"].sum()

                load_mask = (net.load.bus == bid) & net.load.in_service
                total_load += net.load.loc[load_mask, "p_mw"].sum()

            if total_gen < total_load * 0.1:
                return False

        return True

    # ------------------------------------------------------------------
    # Main sectionalizing entry point
    # ------------------------------------------------------------------

    def sectionalize(
        self,
        net: Any,
        overloaded_lines: List[int],
    ) -> Dict[str, Any]:
        """Execute controlled islanding to isolate overloaded sections.

        Parameters
        ----------
        net : pandapowerNet
        overloaded_lines : list of int

        Returns
        -------
        dict
            ``{"switches_to_open": list, "islands": list,
            "frobenius_distance": float, "load_shed_mw": float,
            "success": bool}``.
        """
        G, bus_map, line_map = self.build_weighted_graph(net)
        overloaded_set = set(overloaded_lines)

        candidates: List[Tuple[List[int], List[int], List[int]]] = []

        result = self.spectral_bisection(G, overloaded_set)
        if result is not None:
            candidates.append(result)

        if not candidates:
            result = self.minimum_cut_islanding(G, overloaded_set)
            if result is not None:
                candidates.append(result)

        if not candidates:
            logger.warning("No valid cut found for sectionalizing.")
            return {
                "switches_to_open": [],
                "islands": [],
                "frobenius_distance": float("inf"),
                "load_shed_mw": 0.0,
                "success": False,
            }

        best = self.optimize_switching(net, candidates)
        if best is None:
            return {
                "switches_to_open": [],
                "islands": [],
                "frobenius_distance": float("inf"),
                "load_shed_mw": 0.0,
                "success": False,
            }

        test_net = copy.deepcopy(net)
        for lidx in best["cut_edges"]:
            if lidx in test_net.line.index:
                test_net.line.at[lidx, "in_service"] = False

        load_shed = 0.0
        try:
            pp.runpp(test_net)
            if hasattr(test_net, "res_bus"):
                for bid in test_net.res_bus.index:
                    if test_net.res_bus.at[bid, "vm_pu"] < 0.85:
                        load_mask = (test_net.load.bus == bid) & test_net.load.in_service
                        load_shed += test_net.load.loc[load_mask, "p_mw"].sum()
        except pp.LoadflowNotConverged:
            logger.warning("Power flow did not converge for sectionalized network.")
            try:
                pp.rundcpp(test_net)
            except Exception:
                pass

        logger.info(
            "Sectionalizing complete: %d lines opened, frob_dist=%.4f, shed=%.2f MW",
            len(best["cut_edges"]), best["frobenius_distance"], load_shed,
        )

        return {
            "switches_to_open": best["cut_edges"],
            "islands": best["islands"],
            "frobenius_distance": best["frobenius_distance"],
            "load_shed_mw": load_shed,
            "success": True,
        }

    # ------------------------------------------------------------------
    # Absorb resilience factor
    # ------------------------------------------------------------------

    def compute_absorb_factor(
        self,
        net: Any,
        overloaded_lines: List[int],
        baseline_shed_mw: float,
    ) -> Dict[str, Any]:
        """Compute the Absorb resilience factor.

        Compares controlled load shed from sectionalizing against the
        uncontrolled cascading baseline.

        Parameters
        ----------
        net : pandapowerNet
        overloaded_lines : list of int
        baseline_shed_mw : float
            Load shed from uncontrolled cascade.

        Returns
        -------
        dict
            ``{"absorb_factor": float, "controlled_shed_mw": float,
            "baseline_shed_mw": float}``.
        """
        result = self.sectionalize(net, overloaded_lines)
        controlled_shed = result["load_shed_mw"]

        if baseline_shed_mw > _EPS:
            absorb = 1.0 - controlled_shed / baseline_shed_mw
        else:
            absorb = 1.0 if controlled_shed < _EPS else 0.0

        absorb = max(0.0, min(1.0, absorb))

        return {
            "absorb_factor": absorb,
            "controlled_shed_mw": controlled_shed,
            "baseline_shed_mw": baseline_shed_mw,
        }
