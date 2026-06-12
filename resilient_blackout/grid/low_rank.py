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
Ultra-fast topological-change flow engine.

Provides the ``LowRankFlowEngine`` class that uses the Sherman-Morrison-
Woodbury formula for low-rank updates of Power Transfer Distribution
Factors (PTDF) and Line Outage Distribution Factors (LODF) to model
line disconnections and generator trips without refactorizing the bus
admittance matrix.

Integrates as a pre-screening loop in ``CascadingSimulator`` to bypass
full AC/DC power flow checks when line loading margins are below
critical thresholds.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from resilient_blackout.grid.network import GridModel

logger = logging.getLogger(__name__)

_EPS: float = 1e-12


class LowRankFlowEngine:
    """Cached PTDF/LODF engine with low-rank contingency updates.

    Builds and caches the baseline PTDF and LODF matrices from a
    pandapower network, then uses the Sherman-Morrison-Woodbury
    formula to perform rank-1 updates when lines are disconnected —
    avoiding full refactorization of the bus admittance matrix.

    Parameters
    ----------
    grid_model : GridModel
        The grid model to build distribution factors from.

    Attributes
    ----------
    grid_model : GridModel
    n_buses : int
    n_lines : int
    PTDF : np.ndarray
        ``(n_lines, n_buses)`` dense PTDF matrix.
    LODF : np.ndarray
        ``(n_lines, n_lines)`` dense LODF matrix.
    B_bus : scipy.sparse.csc_matrix
        Sparse bus susceptance matrix.
    line_reactance : np.ndarray
        Per-unit reactance for each line.
    line_ratings : np.ndarray
        ``max_i_ka`` converted to MVA at 1.0 pu voltage.
    bus_from : np.ndarray
        From-bus index per line.
    bus_to : np.ndarray
        To-bus index per line.
    ref_bus : int
        Reference (slack) bus index.
    """

    def __init__(self, grid_model: GridModel) -> None:
        self.grid_model = grid_model
        net = grid_model.net

        self.n_buses = len(net.bus)
        self.n_lines = len(net.line)

        self.bus_from = net.line.from_bus.values.astype(np.int32)
        self.bus_to = net.line.to_bus.values.astype(np.int32)

        self.line_reactance = net.line.x_ohm_per_km.values.copy()
        if np.any(self.line_reactance <= 0):
            self.line_reactance = np.maximum(self.line_reactance, _EPS)

        self.line_ratings = net.line.max_i_ka.values.copy() * 1.0

        self.ref_bus = self._find_ref_bus(net)

        self.B_bus = self._build_B_bus()
        self.PTDF = self._compute_ptdf()
        self.LODF = self._compute_lodf()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_ref_bus(net: Any) -> int:
        """Identify the reference (slack) bus.

        Parameters
        ----------
        net : pandapowerNet

        Returns
        -------
        int
            Bus index of the first in-service external grid or the
            first bus.
        """
        ext_grid = net.ext_grid[net.ext_grid.in_service]
        if len(ext_grid) > 0:
            return int(ext_grid.iloc[0].bus)
        return int(net.bus.index[0])

    def _build_B_bus(self) -> sparse.csc_matrix:
        """Build the sparse DC bus susceptance matrix.

        Returns
        -------
        scipy.sparse.csc_matrix
            ``(n_buses, n_buses)`` susceptance matrix with the
            reference bus row/column zeroed.
        """
        B = sparse.lil_matrix((self.n_buses, self.n_buses), dtype=np.float64)

        for lidx in range(self.n_lines):
            x = self.line_reactance[lidx]
            if x <= _EPS:
                continue
            b = 1.0 / x
            f = self.bus_from[lidx]
            t = self.bus_to[lidx]
            B[f, f] += b
            B[t, t] += b
            B[f, t] -= b
            B[t, f] -= b

        B[self.ref_bus, :] = 0.0
        B[:, self.ref_bus] = 0.0
        B[self.ref_bus, self.ref_bus] = 1.0

        return B.tocsc()

    def _compute_ptdf(self) -> np.ndarray:
        """Compute the baseline PTDF matrix.

        :math:`PTDF = \\text{diag}(1/x) \\cdot A \\cdot B_{\\text{bus}}^{-1}`

        where :math:`A` is the branch-bus incidence matrix.

        Returns
        -------
        np.ndarray
            ``(n_lines, n_buses)`` dense PTDF.
        """
        PTDF = np.zeros((self.n_lines, self.n_buses), dtype=np.float64)

        for lidx in range(self.n_lines):
            x = self.line_reactance[lidx]
            if x <= _EPS:
                continue
            e = np.zeros(self.n_buses, dtype=np.float64)
            f = self.bus_from[lidx]
            t = self.bus_to[lidx]
            e[f] = 1.0
            e[t] = -1.0
            theta = spsolve(self.B_bus, e)
            PTDF[lidx, :] = theta / x

        return PTDF

    def _compute_lodf(self) -> np.ndarray:
        """Compute the baseline LODF matrix from PTDF.

        :math:`LODF_{l,k} = \\frac{PTDF_{l,f} - PTDF_{l,t}}
        {1 - (PTDF_{k,f} - PTDF_{k,t})}`

        for :math:`l \\neq k`, and :math:`LODF_{k,k} = -1`.

        Returns
        -------
        np.ndarray
            ``(n_lines, n_lines)`` dense LODF.
        """
        LODF = np.zeros((self.n_lines, self.n_lines), dtype=np.float64)

        ptdf_diff = np.zeros(self.n_lines, dtype=np.float64)
        for k in range(self.n_lines):
            ptdf_diff[k] = (
                self.PTDF[k, self.bus_from[k]] - self.PTDF[k, self.bus_to[k]]
            )

        for l in range(self.n_lines):
            ptdf_l_diff = (
                self.PTDF[l, self.bus_from[l]] - self.PTDF[l, self.bus_to[l]]
            )
            for k in range(self.n_lines):
                if l == k:
                    LODF[l, k] = -1.0
                else:
                    denom = 1.0 - ptdf_diff[k]
                    if abs(denom) > _EPS:
                        LODF[l, k] = ptdf_l_diff / denom

        return LODF

    # ------------------------------------------------------------------
    # Branch outage simulation
    # ------------------------------------------------------------------

    def simulate_branch_outage(
        self,
        active_flows: np.ndarray,
        tripped_line_ids: List[int],
    ) -> np.ndarray:
        """Compute post-contingency flows using cached LODF.

        .. math::

            P^{\\text{new}} = P^{\\text{old}}
            + \\text{LODF}[:, \\text{tripped}]
            \\cdot P^{\\text{tripped}}

        Parameters
        ----------
        active_flows : np.ndarray
            Pre-contingency active power flows per line (MW).
        tripped_line_ids : list of int
            Indices of tripped lines.

        Returns
        -------
        np.ndarray
            Post-contingency flows for all lines (MW).
        """
        P = np.asarray(active_flows, dtype=np.float64).copy()
        tripped = [t for t in tripped_line_ids if 0 <= t < self.n_lines]

        if not tripped:
            return P

        P_tripped = P[tripped]
        delta = self.LODF[:, tripped] @ P_tripped
        P_new = P + delta
        P_new[tripped] = 0.0

        return P_new

    # ------------------------------------------------------------------
    # Generator outage simulation
    # ------------------------------------------------------------------

    def simulate_generator_outage(
        self,
        active_flows: np.ndarray,
        failed_gen_id: int,
        gen_droop_factors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Redistribute lost generation across remaining units.

        Uses inertia-matching droop factors if provided; otherwise
        distributes equally across all remaining generators.

        Parameters
        ----------
        active_flows : np.ndarray
            Pre-contingency flows (MW).
        failed_gen_id : int
            Index of the failed generator (pandapower gen index).
        gen_droop_factors : np.ndarray or None
            Droop factors per generator.  If ``None``, equal sharing
            is used.

        Returns
        -------
        np.ndarray
            Post-contingency flows (MW).
        """
        net = self.grid_model.net
        gen = net.gen[net.gen.in_service]

        if failed_gen_id not in gen.index:
            return np.asarray(active_flows, dtype=np.float64).copy()

        lost_p = float(gen.at[failed_gen_id, "p_mw"])
        remaining = gen.index[gen.index != failed_gen_id]

        if len(remaining) == 0:
            return np.asarray(active_flows, dtype=np.float64).copy()

        if gen_droop_factors is not None:
            droop = np.asarray(gen_droop_factors, dtype=np.float64)
            remaining_droop = droop[remaining]
            total_droop = np.sum(remaining_droop)
            if total_droop > _EPS:
                shares = remaining_droop / total_droop
            else:
                shares = np.ones(len(remaining)) / len(remaining)
        else:
            shares = np.ones(len(remaining)) / len(remaining)

        delta_injection = np.zeros(self.n_buses, dtype=np.float64)
        for i, gidx in enumerate(remaining):
            bus = int(gen.at[gidx, "bus"])
            delta_injection[bus] += lost_p * shares[i]

        delta_flows = self.PTDF @ delta_injection
        P = np.asarray(active_flows, dtype=np.float64).copy()
        return P + delta_flows

    # ------------------------------------------------------------------
    # Sherman-Morrison-Woodbury LODF update
    # ------------------------------------------------------------------

    def update_lodf_for_outage(self, tripped_line_id: int) -> np.ndarray:
        """Update LODF after a line outage using rank-1 SMW formula.

        .. math::

            \\text{LODF}' = \\text{LODF}
            - \\frac{\\text{LODF}[:,k] \\cdot \\text{LODF}[k,:]}
            {1 + \\text{LODF}[k,k]}

        where :math:`k` is the tripped line index.

        Parameters
        ----------
        tripped_line_id : int
            Index of the newly tripped line.

        Returns
        -------
        np.ndarray
            Updated ``(n_lines, n_lines)`` LODF matrix.
        """
        k = tripped_line_id
        if k < 0 or k >= self.n_lines:
            return self.LODF.copy()

        col_k = self.LODF[:, k].copy()
        row_k = self.LODF[k, :].copy()
        denom = 1.0 + self.LODF[k, k]

        if abs(denom) > _EPS:
            update = np.outer(col_k, row_k) / denom
            new_lodf = self.LODF - update
        else:
            new_lodf = self.LODF.copy()

        new_lodf[k, :] = 0.0
        new_lodf[:, k] = 0.0
        return new_lodf

    # ------------------------------------------------------------------
    # Overload screening
    # ------------------------------------------------------------------

    def screen_overloads(
        self,
        active_flows: np.ndarray,
        tripped_line_ids: List[int],
        threshold: float = 1.0,
    ) -> List[int]:
        """Fast pre-screening for potential overloads after contingencies.

        Applies LODF to estimate post-contingency flows and returns
        line indices whose loading exceeds ``threshold`` × rating.

        Parameters
        ----------
        active_flows : np.ndarray
            Pre-contingency flows (MW).
        tripped_line_ids : list of int
            Recently tripped line indices.
        threshold : float
            Loading threshold as a fraction of rating.  Default 1.0
            (100 %).

        Returns
        -------
        list of int
            Line indices that need full AC/DC verification.
        """
        P_new = self.simulate_branch_outage(active_flows, tripped_line_ids)
        loading = self.get_loading_percent(P_new)
        candidates = np.where(loading > threshold * 100.0)[0]
        return [int(c) for c in candidates if c not in tripped_line_ids]

    def get_loading_percent(self, flows: np.ndarray) -> np.ndarray:
        """Compute loading percentage for all lines.

        Parameters
        ----------
        flows : np.ndarray
            Active power flows (MW).

        Returns
        -------
        np.ndarray
            Loading in percent (0–∞).
        """
        ratings_mw = self.line_ratings * 1.0
        safe_ratings = np.maximum(ratings_mw, _EPS)
        return np.abs(flows) / safe_ratings * 100.0
