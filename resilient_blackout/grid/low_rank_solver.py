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
Ultra-fast topological-change flow engine with low-rank updates.

Provides ``LowRankFlowEngine``, which caches the baseline PTDF and
LODF matrices from a pandapower network and uses the Sherman-Morrison-
Woodbury (SMW) formula to perform rank-1 updates when lines are
disconnected — avoiding full refactorization of the bus admittance
matrix.

The SMW formula for a rank-1 update to an invertible matrix :math:`A`
is:

.. math::

    (A + u v^T)^{-1} = A^{-1}
    - \\frac{A^{-1} u v^T A^{-1}}{1 + v^T A^{-1} u}

For a line outage between buses :math:`f` and :math:`t` with reactance
:math:`x`, the susceptance matrix update is:

.. math::

    \\Delta B = -\\frac{1}{x} (e_f - e_t)(e_f - e_t)^T

which is a symmetric rank-1 modification.  The SMW formula yields the
updated PTDF in :math:`O(n^2)` instead of :math:`O(n^3)` for a full
re-factorization.

Integrates as a pre-screening loop in ``CascadingSimulator`` to bypass
full AC/DC power flow checks when line loading margins are below
critical thresholds.

Reference
---------
* Wood, A. J., Wollenberg, B. F., & Sheblé, G. B. (2014).  *Power
  Generation, Operation, and Control* (3rd ed.).  Wiley.
* Sherman, J. & Morrison, W. J. (1950).  Adjustment of an inverse
  matrix corresponding to a change in one element of a given matrix.
  *Annals of Mathematical Statistics*, 21(1), 124–127.
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


# ---------------------------------------------------------------------------
# LowRankFlowEngine
# ---------------------------------------------------------------------------


class LowRankFlowEngine:
    """Cached PTDF/LODF engine with SMW low-rank contingency updates.

    Builds and caches the baseline PTDF and LODF matrices from a
    pandapower network.  Uses the Sherman-Morrison-Woodbury formula
    to perform rank-1 updates to the PTDF matrix when lines are
    disconnected, avoiding full refactorization of the bus admittance
    matrix.

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
        Sparse bus susceptance matrix (reference bus grounded).
    B_inv_dense : np.ndarray or None
        Dense inverse of B_bus, cached for SMW updates.  ``None`` if
        the network is too large for dense storage.
    line_reactance : np.ndarray
        Per-unit reactance for each line.
    line_ratings_mw : np.ndarray
        Line thermal ratings in MW (derived from ``max_i_ka``).
    bus_from : np.ndarray
        From-bus index per line.
    bus_to : np.ndarray
        To-bus index per line.
    ref_bus : int
        Reference (slack) bus index.
    active_mask : np.ndarray
        Boolean mask of in-service lines.
    """

    _MAX_DENSE_BUSES: int = 2000  # threshold for dense B_inv caching

    def __init__(self, grid_model: GridModel) -> None:
        self.grid_model = grid_model
        net = grid_model.net

        self.n_buses = len(net.bus)
        self.n_lines = len(net.line)

        self.bus_from = net.line.from_bus.values.astype(np.int32)
        self.bus_to = net.line.to_bus.values.astype(np.int32)

        self.line_reactance = net.line.x_ohm_per_km.values.astype(np.float64).copy()
        zero_x = self.line_reactance <= _EPS
        if np.any(zero_x):
            self.line_reactance[zero_x] = _EPS

        self.line_ratings_mw = net.line.max_i_ka.values.astype(np.float64) * 1.0

        self.active_mask = net.line.in_service.values.astype(bool).copy()

        self.ref_bus = self._find_ref_bus(net)

        self.B_bus = self._build_B_bus()
        self.B_inv_dense: Optional[np.ndarray] = None

        if self.n_buses <= self._MAX_DENSE_BUSES:
            self.B_inv_dense = self._compute_B_inv_dense()

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
        """
        ext_grid = net.ext_grid[net.ext_grid.in_service]
        if len(ext_grid) > 0:
            return int(ext_grid.iloc[0].bus)
        return int(net.bus.index[0])

    def _build_B_bus(self) -> sparse.csc_matrix:
        """Build the sparse DC bus susceptance matrix.

        .. math::

            B_{ij} = -\\frac{1}{x_{ij}}, \\quad
            B_{ii} = \\sum_{k \\neq i} \\frac{1}{x_{ik}}

        The reference bus row and column are zeroed and the diagonal
        entry set to 1.0 for invertibility.

        Returns
        -------
        scipy.sparse.csc_matrix
            ``(n_buses, n_buses)`` sparse susceptance matrix.
        """
        data: List[float] = []
        row_ind: List[int] = []
        col_ind: List[int] = []

        for lidx in range(self.n_lines):
            if not self.active_mask[lidx]:
                continue
            x = self.line_reactance[lidx]
            b = 1.0 / x
            f = int(self.bus_from[lidx])
            t = int(self.bus_to[lidx])

            data.extend([b, b, -b, -b])
            row_ind.extend([f, t, f, t])
            col_ind.extend([f, t, t, f])

        B = sparse.coo_matrix(
            (data, (row_ind, col_ind)),
            shape=(self.n_buses, self.n_buses),
            dtype=np.float64,
        ).tocsc()

        # Ground the reference bus
        B = B.tolil()
        B[self.ref_bus, :] = 0.0
        B[:, self.ref_bus] = 0.0
        B[self.ref_bus, self.ref_bus] = 1.0

        return B.tocsc()

    def _compute_B_inv_dense(self) -> np.ndarray:
        """Compute the dense inverse of B_bus for SMW updates.

        Returns
        -------
        np.ndarray
            ``(n_buses, n_buses)`` dense B^{-1}.
        """
        I = sparse.eye(self.n_buses, format="csc", dtype=np.float64)
        B_inv = spsolve(self.B_bus, I)
        return np.asarray(B_inv.todense() if sparse.issparse(B_inv) else B_inv)

    def _compute_ptdf(self) -> np.ndarray:
        """Compute the baseline PTDF matrix.

        .. math::

            \\text{PTDF} = \\text{diag}(1/x) \\cdot A \\cdot B^{-1}

        where :math:`A` is the branch-bus incidence matrix.

        Returns
        -------
        np.ndarray
            ``(n_lines, n_buses)`` dense PTDF.
        """
        PTDF = np.zeros((self.n_lines, self.n_buses), dtype=np.float64)

        for lidx in range(self.n_lines):
            if not self.active_mask[lidx]:
                continue
            x = self.line_reactance[lidx]
            f = int(self.bus_from[lidx])
            t = int(self.bus_to[lidx])

            e = np.zeros(self.n_buses, dtype=np.float64)
            e[f] = 1.0
            e[t] = -1.0

            theta = spsolve(self.B_bus, e)
            PTDF[lidx, :] = theta / x

        return PTDF

    def _compute_lodf(self) -> np.ndarray:
        """Compute the baseline LODF matrix from PTDF.

        .. math::

            \\text{LODF}_{l,k} =
            \\frac{\\text{PTDF}_{l,f_k} - \\text{PTDF}_{l,t_k}}
            {1 - (\\text{PTDF}_{k,f_k} - \\text{PTDF}_{k,t_k})}

        for :math:`l \\neq k`, and :math:`\\text{LODF}_{k,k} = -1`.

        Returns
        -------
        np.ndarray
            ``(n_lines, n_lines)`` dense LODF.
        """
        LODF = np.zeros((self.n_lines, self.n_lines), dtype=np.float64)

        ptdf_diff = np.zeros(self.n_lines, dtype=np.float64)
        for k in range(self.n_lines):
            if self.active_mask[k]:
                ptdf_diff[k] = (
                    self.PTDF[k, self.bus_from[k]] - self.PTDF[k, self.bus_to[k]]
                )

        for l in range(self.n_lines):
            if not self.active_mask[l]:
                continue
            ptdf_l_diff = (
                self.PTDF[l, self.bus_from[l]] - self.PTDF[l, self.bus_to[l]]
            )
            for k in range(self.n_lines):
                if not self.active_mask[k]:
                    continue
                if l == k:
                    LODF[l, k] = -1.0
                else:
                    denom = 1.0 - ptdf_diff[k]
                    if abs(denom) > _EPS:
                        LODF[l, k] = ptdf_l_diff / denom

        return LODF

    # ------------------------------------------------------------------
    # Sherman-Morrison-Woodbury PTDF update
    # ------------------------------------------------------------------

    def update_ptdf_for_outage(self, tripped_line_id: int) -> np.ndarray:
        """Update PTDF after a line outage using the SMW formula.

        For a line :math:`k` between buses :math:`f` and :math:`t`
        with reactance :math:`x_k`, removing the line modifies the
        susceptance matrix by:

        .. math::

            \\Delta B = -\\frac{1}{x_k} (e_f - e_t)(e_f - e_t)^T

        Let :math:`u = -(1/x_k)(e_f - e_t)` and :math:`v = e_f - e_t`.
        Then :math:`B' = B + u v^T` and by SMW:

        .. math::

            (B')^{-1} = B^{-1}
            - \\frac{B^{-1} u v^T B^{-1}}{1 + v^T B^{-1} u}

        The updated PTDF row for any line :math:`l` is then:

        .. math::

            \\text{PTDF}'_l = \\frac{1}{x_l} (e_{f_l} - e_{t_l})^T (B')^{-1}

        Parameters
        ----------
        tripped_line_id : int
            Index of the tripped line.

        Returns
        -------
        np.ndarray
            Updated ``(n_lines, n_buses)`` PTDF matrix.
        """
        k = tripped_line_id
        if k < 0 or k >= self.n_lines or not self.active_mask[k]:
            return self.PTDF.copy()

        if self.B_inv_dense is None:
            logger.warning(
                "SMW PTDF update skipped: B_inv_dense not cached "
                "(n_buses=%d > %d).",
                self.n_buses, self._MAX_DENSE_BUSES,
            )
            return self.PTDF.copy()

        f = int(self.bus_from[k])
        t = int(self.bus_to[k])
        x_k = self.line_reactance[k]

        # Build u and v vectors
        e_diff = np.zeros(self.n_buses, dtype=np.float64)
        e_diff[f] = 1.0
        e_diff[t] = -1.0

        u = -(1.0 / x_k) * e_diff
        v = e_diff

        # SMW: B_inv_u = B^{-1} @ u
        B_inv_u = self.B_inv_dense @ u

        # vT_B_inv = v^T @ B^{-1}
        vT_B_inv = v @ self.B_inv_dense

        # denominator = 1 + v^T B^{-1} u
        denom = 1.0 + np.dot(v, B_inv_u)

        if abs(denom) < _EPS:
            logger.warning("SMW denominator near zero for line %d; skipping update.", k)
            return self.PTDF.copy()

        # SMW update: B_inv' = B_inv - (B_inv_u @ vT_B_inv) / denom
        B_inv_new = self.B_inv_dense - np.outer(B_inv_u, vT_B_inv) / denom

        # Recompute PTDF rows for active lines
        PTDF_new = self.PTDF.copy()
        for l in range(self.n_lines):
            if not self.active_mask[l] or l == k:
                continue
            x_l = self.line_reactance[l]
            f_l = int(self.bus_from[l])
            t_l = int(self.bus_to[l])
            e_l = np.zeros(self.n_buses, dtype=np.float64)
            e_l[f_l] = 1.0
            e_l[t_l] = -1.0
            PTDF_new[l, :] = (e_l @ B_inv_new) / x_l

        PTDF_new[k, :] = 0.0

        return PTDF_new

    def update_lodf_for_outage(self, tripped_line_id: int) -> np.ndarray:
        """Update LODF after a line outage using rank-1 SMW formula.

        .. math::

            \\text{LODF}' = \\text{LODF}
            - \\frac{\\text{LODF}[:,k] \\cdot \\text{LODF}[k,:]}
            {1 + \\text{LODF}[k,k]}

        where :math:`k` is the tripped line index.  After the update,
        row and column :math:`k` are zeroed.

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
            new_lodf = self.LODF - np.outer(col_k, row_k) / denom
        else:
            new_lodf = self.LODF.copy()

        new_lodf[k, :] = 0.0
        new_lodf[:, k] = 0.0
        return new_lodf

    # ------------------------------------------------------------------
    # Branch outage simulation
    # ------------------------------------------------------------------

    def simulate_branch_outages(
        self,
        active_flows: np.ndarray,
        tripped_line_ids: List[int],
    ) -> np.ndarray:
        """Compute post-contingency branch flows using cached LODF.

        .. math::

            P^{\\text{new}} = P^{\\text{old}}
            + \\text{LODF}[:, \\text{tripped}]
            \\cdot P^{\\text{tripped}}

        The tripped lines' flows are set to zero in the result.

        Parameters
        ----------
        active_flows : np.ndarray
            Pre-contingency active power flows per line in MW,
            shape ``(n_lines,)``.
        tripped_line_ids : list of int
            Indices of tripped lines.

        Returns
        -------
        np.ndarray
            Post-contingency flows for all lines in MW, shape
            ``(n_lines,)``.
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

    def simulate_generator_outage(
        self,
        active_flows: np.ndarray,
        failed_gen_id: int,
        gen_droop_factors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Redistribute lost generation across remaining units via PTDF.

        Parameters
        ----------
        active_flows : np.ndarray
            Pre-contingency flows in MW, shape ``(n_lines,)``.
        failed_gen_id : int
            Pandapower index of the failed generator.
        gen_droop_factors : np.ndarray or None
            Droop factors per generator.  ``None`` uses equal sharing.

        Returns
        -------
        np.ndarray
            Post-contingency flows in MW, shape ``(n_lines,)``.
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
            shares = (
                remaining_droop / total_droop
                if total_droop > _EPS
                else np.ones(len(remaining)) / len(remaining)
            )
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
    # Overload pre-screening
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
        Use this as a pre-filter before running full AC/DC power flow.

        Parameters
        ----------
        active_flows : np.ndarray
            Pre-contingency flows in MW, shape ``(n_lines,)``.
        tripped_line_ids : list of int
            Recently tripped line indices.
        threshold : float
            Loading threshold as a fraction of rating.  Default 1.0
            (100 %).  Set to 0.8 for conservative pre-screening.

        Returns
        -------
        list of int
            Line indices that need full AC/DC verification.
        """
        P_new = self.simulate_branch_outages(active_flows, tripped_line_ids)
        loading = self.get_loading_percent(P_new)
        candidates = np.where(loading > threshold * 100.0)[0]
        return [int(c) for c in candidates if c not in tripped_line_ids]

    def get_loading_percent(self, flows: np.ndarray) -> np.ndarray:
        """Compute loading percentage for all lines.

        Parameters
        ----------
        flows : np.ndarray
            Active power flows in MW, shape ``(n_lines,)``.

        Returns
        -------
        np.ndarray
            Loading in percent (0–∞), shape ``(n_lines,)``.
        """
        safe_ratings = np.maximum(self.line_ratings_mw, _EPS)
        return np.abs(flows) / safe_ratings * 100.0

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_active = int(np.sum(self.active_mask))
        return (
            f"LowRankFlowEngine(buses={self.n_buses}, "
            f"lines={n_active}/{self.n_lines}, "
            f"dense_inv={'cached' if self.B_inv_dense is not None else 'sparse'})"
        )
