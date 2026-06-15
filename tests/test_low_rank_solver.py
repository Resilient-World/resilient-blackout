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
# FOR ANY DIRECT, INDIRECT, INCIDENTIAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Unit tests for ``resilient_blackout.grid.low_rank_solver``."""

from __future__ import annotations

import numpy as np
import pytest

from resilient_blackout.grid.low_rank_solver import LowRankFlowEngine
from resilient_blackout.grid.network import GridModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid_model():
    """Build a 5-bus test network with known topology."""
    import pandapower as pp

    net = pp.create_empty_network()
    buses = [pp.create_bus(net, vn_kv=110) for _ in range(5)]

    pp.create_line(net, buses[0], buses[1], length_km=10, x_ohm_per_km=0.4, max_i_ka=1.0, name="L0")
    pp.create_line(net, buses[1], buses[2], length_km=10, x_ohm_per_km=0.3, max_i_ka=1.0, name="L1")
    pp.create_line(net, buses[2], buses[3], length_km=10, x_ohm_per_km=0.5, max_i_ka=1.0, name="L2")
    pp.create_line(net, buses[3], buses[4], length_km=10, x_ohm_per_km=0.2, max_i_ka=1.0, name="L3")
    pp.create_line(net, buses[0], buses[4], length_km=15, x_ohm_per_km=0.35, max_i_ka=1.0, name="L4")

    pp.create_ext_grid(net, buses[0], vm_pu=1.0, va_degree=0.0)
    pp.create_gen(net, buses[2], p_mw=50, min_p_mw=0, max_p_mw=100, name="G1")
    pp.create_load(net, buses[3], p_mw=80, name="Load1")
    pp.create_load(net, buses[4], p_mw=30, name="Load2")

    return GridModel(net)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid_model():
    return _make_grid_model()


@pytest.fixture
def engine(grid_model):
    return LowRankFlowEngine(grid_model)


# ---------------------------------------------------------------------------
# Constructor and initialization
# ---------------------------------------------------------------------------


class TestInit:
    """Validation of engine initialization."""

    def test_basic_properties(self, engine: LowRankFlowEngine) -> None:
        assert engine.n_buses == 5
        assert engine.n_lines == 5
        assert engine.ref_bus == 0

    def test_B_bus_sparse(self, engine: LowRankFlowEngine) -> None:
        B = engine.B_bus
        assert B.shape == (5, 5)
        assert B[0, 0] == 1.0  # ref bus grounded
        assert B[0, 1] == 0.0  # ref bus row zeroed

    def test_B_inv_dense_cached(self, engine: LowRankFlowEngine) -> None:
        assert engine.B_inv_dense is not None
        assert engine.B_inv_dense.shape == (5, 5)

    def test_PTDF_shape(self, engine: LowRankFlowEngine) -> None:
        assert engine.PTDF.shape == (5, 5)

    def test_LODF_shape(self, engine: LowRankFlowEngine) -> None:
        assert engine.LODF.shape == (5, 5)

    def test_PTDF_row_sums_to_zero(self, engine: LowRankFlowEngine) -> None:
        """Each PTDF row should sum to approximately zero (flow conservation)."""
        for l in range(engine.n_lines):
            if engine.active_mask[l]:
                assert abs(np.sum(engine.PTDF[l, :])) < 1e-10

    def test_LODF_diagonal_is_minus_one(self, engine: LowRankFlowEngine) -> None:
        for k in range(engine.n_lines):
            if engine.active_mask[k]:
                assert engine.LODF[k, k] == pytest.approx(-1.0, abs=1e-10)

    def test_repr(self, engine: LowRankFlowEngine) -> None:
        r = repr(engine)
        assert "LowRankFlowEngine" in r
        assert "5" in r


# ---------------------------------------------------------------------------
# Branch outage simulation
# ---------------------------------------------------------------------------


class TestBranchOutages:
    """Validation of branch outage simulation via LODF."""

    def test_no_outages(self, engine: LowRankFlowEngine) -> None:
        flows = np.array([100.0, 50.0, 30.0, 20.0, 10.0])
        result = engine.simulate_branch_outages(flows, [])
        np.testing.assert_array_equal(result, flows)

    def test_single_outage(self, engine: LowRankFlowEngine) -> None:
        flows = np.array([100.0, 50.0, 30.0, 20.0, 10.0])
        result = engine.simulate_branch_outages(flows, [1])
        # Tripped line flow should be zero
        assert result[1] == 0.0
        # Other lines should have changed
        assert not np.allclose(result[[0, 2, 3, 4]], flows[[0, 2, 3, 4]])

    def test_multiple_outages(self, engine: LowRankFlowEngine) -> None:
        flows = np.array([100.0, 50.0, 30.0, 20.0, 10.0])
        result = engine.simulate_branch_outages(flows, [0, 3])
        assert result[0] == 0.0
        assert result[3] == 0.0

    def test_invalid_line_id_ignored(self, engine: LowRankFlowEngine) -> None:
        flows = np.array([100.0, 50.0, 30.0, 20.0, 10.0])
        result = engine.simulate_branch_outages(flows, [999])
        np.testing.assert_array_equal(result, flows)


# ---------------------------------------------------------------------------
# Generator outage simulation
# ---------------------------------------------------------------------------


class TestGeneratorOutage:
    """Validation of generator outage simulation via PTDF."""

    def test_valid_outage(self, engine: LowRankFlowEngine) -> None:
        flows = np.array([100.0, 50.0, 30.0, 20.0, 10.0])
        result = engine.simulate_generator_outage(flows, failed_gen_id=0)
        assert result.shape == (5,)
        # Flows should change due to redistribution
        assert not np.allclose(result, flows)

    def test_invalid_gen_id(self, engine: LowRankFlowEngine) -> None:
        flows = np.array([100.0, 50.0, 30.0, 20.0, 10.0])
        result = engine.simulate_generator_outage(flows, failed_gen_id=999)
        np.testing.assert_array_equal(result, flows)


# ---------------------------------------------------------------------------
# SMW PTDF update
# ---------------------------------------------------------------------------


class TestSMWPTDFUpdate:
    """Validation of Sherman-Morrison-Woodbury PTDF update."""

    def test_update_reduces_rank(self, engine: LowRankFlowEngine) -> None:
        ptdf_new = engine.update_ptdf_for_outage(1)
        # Row for tripped line should be zeroed
        assert np.allclose(ptdf_new[1, :], 0.0)

    def test_update_preserves_other_rows(self, engine: LowRankFlowEngine) -> None:
        ptdf_new = engine.update_ptdf_for_outage(1)
        # Other rows should still sum to ~0
        for l in [0, 2, 3, 4]:
            assert abs(np.sum(ptdf_new[l, :])) < 1e-10

    def test_update_invalid_line(self, engine: LowRankFlowEngine) -> None:
        ptdf_new = engine.update_ptdf_for_outage(999)
        np.testing.assert_array_equal(ptdf_new, engine.PTDF)

    def test_smw_vs_recompute(self, engine: LowRankFlowEngine) -> None:
        """SMW-updated PTDF should match a full recomputation from scratch."""
        import pandapower as pp

        # Get SMW result
        ptdf_smw = engine.update_ptdf_for_outage(1)

        # Build a new network without line 1 and compute PTDF fresh
        gm2 = _make_grid_model()
        gm2.net.line.at[1, "in_service"] = False
        engine2 = LowRankFlowEngine(gm2)

        # Compare non-tripped rows
        for l in [0, 2, 3, 4]:
            np.testing.assert_allclose(
                ptdf_smw[l, :], engine2.PTDF[l, :],
                atol=1e-8, rtol=1e-6,
            )


# ---------------------------------------------------------------------------
# SMW LODF update
# ---------------------------------------------------------------------------


class TestSMWLODFUpdate:
    """Validation of Sherman-Morrison-Woodbury LODF update."""

    def test_update_zeroes_row_col(self, engine: LowRankFlowEngine) -> None:
        lodf_new = engine.update_lodf_for_outage(2)
        assert np.allclose(lodf_new[2, :], 0.0)
        assert np.allclose(lodf_new[:, 2], 0.0)

    def test_update_invalid_line(self, engine: LowRankFlowEngine) -> None:
        lodf_new = engine.update_lodf_for_outage(-1)
        np.testing.assert_array_equal(lodf_new, engine.LODF)


# ---------------------------------------------------------------------------
# Overload screening
# ---------------------------------------------------------------------------


class TestOverloadScreening:
    """Validation of overload pre-screening."""

    def test_no_overloads(self, engine: LowRankFlowEngine) -> None:
        flows = np.array([0.5, 0.3, 0.2, 0.1, 0.4])
        candidates = engine.screen_overloads(flows, [], threshold=1.0)
        assert candidates == []

    def test_detects_overloads(self, engine: LowRankFlowEngine) -> None:
        flows = np.array([2.0, 0.3, 0.2, 0.1, 0.4])
        candidates = engine.screen_overloads(flows, [], threshold=1.0)
        assert 0 in candidates

    def test_excludes_tripped_lines(self, engine: LowRankFlowEngine) -> None:
        flows = np.array([2.0, 2.0, 0.2, 0.1, 0.4])
        candidates = engine.screen_overloads(flows, [0], threshold=1.0)
        assert 0 not in candidates

    def test_conservative_threshold(self, engine: LowRankFlowEngine) -> None:
        flows = np.array([0.85, 0.3, 0.2, 0.1, 0.4])
        candidates = engine.screen_overloads(flows, [], threshold=0.8)
        assert 0 in candidates


# ---------------------------------------------------------------------------
# Loading percent
# ---------------------------------------------------------------------------


class TestLoadingPercent:
    """Validation of loading percentage computation."""

    def test_below_rating(self, engine: LowRankFlowEngine) -> None:
        loading = engine.get_loading_percent(np.array([0.5, 0.3, 0.2, 0.1, 0.4]))
        assert np.all(loading < 100.0)

    def test_above_rating(self, engine: LowRankFlowEngine) -> None:
        loading = engine.get_loading_percent(np.array([1.5, 0.3, 0.2, 0.1, 0.4]))
        assert loading[0] > 100.0

    def test_negative_flows(self, engine: LowRankFlowEngine) -> None:
        loading = engine.get_loading_percent(np.array([-1.5, -0.3, 0.2, 0.1, 0.4]))
        assert loading[0] > 100.0
