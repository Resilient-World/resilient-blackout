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
Validation harness for the resilient-blackout integration test suite.

Provides pytest fixtures and assertion helpers for benchmark grid
networks (IEEE 24-bus RTS, SimBench-like HV urban), synthetic hazard
footprints, and tolerance-based validation of power flow results,
EENS, and financial metrics against published literature benchmarks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

try:
    import pandapower as pp
    import pandapower.networks as pn

    _HAS_PANDAPOWER = True
except ImportError:  # pragma: no cover
    _HAS_PANDAPOWER = False

from resilient_blackout.core.base import Asset, HazardEvent
from resilient_blackout.core.fragility import ImpactFunction, ImpactFunctionSet
from resilient_blackout.grid.network import GridModel

# ---------------------------------------------------------------------------
# Published IEEE 24-bus RTS benchmark values
# ---------------------------------------------------------------------------

IEEE24_RTS_BENCHMARKS: Dict[str, float] = {
    "total_generation_mw": 3405.0,
    "peak_load_mw": 2850.0,
    "expected_line_loading_max_pct": 85.0,
    "expected_total_losses_mw_max": 55.0,
    "n_buses": 24,
    "n_lines": 38,
    "n_generators": 33,
    "voltage_pu_min": 0.95,
    "voltage_pu_max": 1.05,
}

SIMBENCH_HV_BENCHMARKS: Dict[str, float] = {
    "n_buses": 30,
    "n_lines": 40,
    "voltage_kv": 110.0,
    "total_load_mw_min": 200.0,
    "total_load_mw_max": 600.0,
}

# ---------------------------------------------------------------------------
# Tolerance configuration
# ---------------------------------------------------------------------------

DEFAULT_TOLERANCE: float = 0.01  # 1%


# ---------------------------------------------------------------------------
# Grid fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ieee24_rts_grid() -> GridModel:
    """IEEE 24-bus Reliability Test System as a GridModel.

    Uses pandapower's built-in ``case24_ieee_rts`` network.  If
    pandapower is not installed, the fixture is skipped.

    Returns
    -------
    GridModel
    """
    if not _HAS_PANDAPOWER:
        pytest.skip("pandapower not installed")
    net = pn.case24_ieee_rts()
    return GridModel(net)


@pytest.fixture(scope="session")
def simbench_hv_grid() -> GridModel:
    """Synthetic SimBench-like 110 kV high-voltage urban grid.

    Builds a meshed urban transmission network with 30 buses and
    approximately 40 lines, mimicking the SimBench HV urban topology.

    Returns
    -------
    GridModel
    """
    if not _HAS_PANDAPOWER:
        pytest.skip("pandapower not installed")

    net = pp.create_empty_network()

    n_buses = 30
    bus_indices: List[int] = []
    rng = np.random.default_rng(42)

    for i in range(n_buses):
        bid = pp.create_bus(
            net,
            vn_kv=110.0,
            name=f"HV_bus_{i}",
            geodata=(rng.uniform(0, 20), rng.uniform(0, 20)),
        )
        bus_indices.append(bid)

    pp.create_ext_grid(net, bus=bus_indices[0], vm_pu=1.02, max_p_mw=500.0)

    for i in range(1, 6):
        pp.create_gen(
            net,
            bus=bus_indices[i],
            p_mw=rng.uniform(40, 120),
            vm_pu=1.0,
            max_p_mw=150.0,
            name=f"Gen_{i}",
        )

    for i in range(n_buses):
        pp.create_load(
            net,
            bus=bus_indices[i],
            p_mw=rng.uniform(5, 30),
            q_mvar=rng.uniform(1, 8),
            name=f"Load_{i}",
        )

    line_count = 0
    for i in range(n_buses):
        for j in range(i + 1, n_buses):
            if line_count >= 42:
                break
            dist = np.sqrt(
                (net.bus.at[bus_indices[i], "geodata"].x - net.bus.at[bus_indices[j], "geodata"].x) ** 2
                + (net.bus.at[bus_indices[i], "geodata"].y - net.bus.at[bus_indices[j], "geodata"].y) ** 2
            )
            if dist < 8.0 and rng.random() < 0.25:
                pp.create_line(
                    net,
                    from_bus=bus_indices[i],
                    to_bus=bus_indices[j],
                    length_km=dist * 2.0,
                    std_type="149-AL1/24-ST1A 110.0",
                    name=f"Line_{i}_{j}",
                )
                line_count += 1
        if line_count >= 42:
            break

    return GridModel(net)


# ---------------------------------------------------------------------------
# Hazard fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_wind_hazard() -> HazardEvent:
    """Synthetic extreme wind hazard event.

    A Gaussian wind field with peak speed of 60 m/s centred on the
    grid region, suitable for testing fragility curve evaluation.

    Returns
    -------
    HazardEvent
    """
    return HazardEvent(
        event_id="synthetic_wind_001",
        hazard_type="wind",
        intensity=60.0,
        geometry=None,
        metadata={
            "peak_speed_mps": 60.0,
            "gust_factor": 1.4,
            "description": "Synthetic Category 3 equivalent windstorm",
        },
    )


@pytest.fixture(scope="session")
def synthetic_flood_hazard() -> HazardEvent:
    """Synthetic flood hazard event.

    A radial flood depth field with peak depth of 3.0 m, suitable for
    testing substation inundation fragility.

    Returns
    -------
    HazardEvent
    """
    return HazardEvent(
        event_id="synthetic_flood_001",
        hazard_type="flood",
        intensity=3.0,
        geometry=None,
        metadata={
            "peak_depth_m": 3.0,
            "duration_h": 6.0,
            "description": "Synthetic 100-year pluvial flood",
        },
    )


# ---------------------------------------------------------------------------
# Fragility fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def wind_fragility_set() -> ImpactFunctionSet:
    """Log-normal fragility curves for wind vulnerability.

    Returns
    -------
    ImpactFunctionSet
    """
    transmission_line_wind = ImpactFunction(
        function_id="wind_tline_001",
        hazard_type="wind",
        asset_type="transmission_line",
        intensity_measure="wind_speed_mps",
        median_intensity=45.0,
        beta=0.25,
    )

    substation_wind = ImpactFunction(
        function_id="wind_sub_001",
        hazard_type="wind",
        asset_type="substation",
        intensity_measure="wind_speed_mps",
        median_intensity=55.0,
        beta=0.30,
    )

    tower_wind = ImpactFunction(
        function_id="wind_tower_001",
        hazard_type="wind",
        asset_type="transmission_tower",
        intensity_measure="wind_speed_mps",
        median_intensity=50.0,
        beta=0.20,
    )

    return ImpactFunctionSet(
        functions=[transmission_line_wind, substation_wind, tower_wind]
    )


@pytest.fixture(scope="session")
def flood_fragility_set() -> ImpactFunctionSet:
    """Log-normal fragility curves for flood vulnerability.

    Returns
    -------
    ImpactFunctionSet
    """
    substation_flood = ImpactFunction(
        function_id="flood_sub_001",
        hazard_type="flood",
        asset_type="substation",
        intensity_measure="flood_depth_m",
        median_intensity=1.5,
        beta=0.35,
    )

    return ImpactFunctionSet(functions=[substation_flood])


# ---------------------------------------------------------------------------
# Asset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ieee24_assets(ieee24_rts_grid: GridModel) -> List[Asset]:
    """Generate Asset objects from the IEEE 24-bus RTS grid.

    Parameters
    ----------
    ieee24_rts_grid : GridModel

    Returns
    -------
    list of Asset
    """
    assets: List[Asset] = []
    net = ieee24_rts_grid.net

    for idx in net.bus.index:
        name = net.bus.at[idx, "name"] or f"bus_{idx}"
        assets.append(
            Asset(
                asset_id=name,
                asset_type="substation",
                geometry=None,
                metadata={"bus_index": int(idx), "vn_kv": float(net.bus.at[idx, "vn_kv"])},
            )
        )

    for idx in net.line.index:
        name = net.line.at[idx, "name"] or f"line_{idx}"
        assets.append(
            Asset(
                asset_id=name,
                asset_type="transmission_line",
                geometry=None,
                metadata={
                    "line_index": int(idx),
                    "length_km": float(net.line.at[idx, "length_km"]),
                },
            )
        )

    return assets


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_power_flow_valid(
    result: Dict[str, Any],
    label: str = "",
    voltage_min_pu: float = 0.95,
    voltage_max_pu: float = 1.05,
    max_loading_pct: float = 100.0,
) -> None:
    """Assert that a power flow result is electrically valid.

    Parameters
    ----------
    result : dict
        Output from ``GridModel.run_baseline_power_flow()``.
    label : str
        Optional label for assertion messages.
    voltage_min_pu : float
        Minimum acceptable per-unit voltage.
    voltage_max_pu : float
        Maximum acceptable per-unit voltage.
    max_loading_pct : float
        Maximum acceptable line loading percentage.

    Raises
    ------
    AssertionError
        If any check fails.
    """
    prefix = f"[{label}] " if label else ""

    assert result["converged"], f"{prefix}Power flow did not converge"

    for i, vm in enumerate(result["vm_pu"]):
        assert voltage_min_pu <= vm <= voltage_max_pu, (
            f"{prefix}Bus {i} voltage {vm:.4f} pu outside "
            f"[{voltage_min_pu}, {voltage_max_pu}]"
        )

    for i, loading in enumerate(result["loading_percent"]):
        assert loading <= max_loading_pct, (
            f"{prefix}Line {i} loading {loading:.1f}% exceeds {max_loading_pct}%"
        )


def assert_eens_tolerance(
    computed_eens: float,
    expected_eens: float,
    tolerance: float = DEFAULT_TOLERANCE,
    label: str = "",
) -> None:
    """Assert EENS is within tolerance of expected benchmark.

    Parameters
    ----------
    computed_eens : float
        EENS value computed by the simulation.
    expected_eens : float
        Published benchmark EENS value.
    tolerance : float
        Relative tolerance (e.g., 0.01 = 1%).
    label : str
        Optional label for assertion messages.

    Raises
    ------
    AssertionError
        If the relative error exceeds *tolerance*.
    """
    prefix = f"[{label}] " if label else ""

    if expected_eens < 1e-12:
        if computed_eens < 1e-12:
            return
        raise AssertionError(
            f"{prefix}Expected EENS ≈ 0, got {computed_eens:.4f}"
        )

    rel_error = abs(computed_eens - expected_eens) / expected_eens
    assert rel_error <= tolerance, (
        f"{prefix}EENS {computed_eens:.4f} deviates from benchmark "
        f"{expected_eens:.4f} by {rel_error * 100:.2f}% "
        f"(tolerance: {tolerance * 100:.1f}%)"
    )


def assert_metric_tolerance(
    computed: float,
    expected: float,
    tolerance: float = DEFAULT_TOLERANCE,
    label: str = "",
) -> None:
    """Assert a generic metric is within tolerance of expected value.

    Parameters
    ----------
    computed : float
    expected : float
    tolerance : float
    label : str

    Raises
    ------
    AssertionError
    """
    prefix = f"[{label}] " if label else ""

    if abs(expected) < 1e-12:
        if abs(computed) < 1e-12:
            return
        raise AssertionError(
            f"{prefix}Expected ≈ 0, got {computed:.6f}"
        )

    rel_error = abs(computed - expected) / abs(expected)
    assert rel_error <= tolerance, (
        f"{prefix}{computed:.6f} vs expected {expected:.6f} "
        f"(error: {rel_error * 100:.2f}%, tolerance: {tolerance * 100:.1f}%)"
    )


def assert_financial_metrics_valid(
    npv: float,
    bcr: float,
    label: str = "",
) -> None:
    """Assert financial metrics are economically sensible.

    Parameters
    ----------
    npv : float
        Net Present Value in USD.
    bcr : float
        Benefit-Cost Ratio.
    label : str

    Raises
    ------
    AssertionError
    """
    prefix = f"[{label}] " if label else ""

    assert npv > 0, f"{prefix}NPV must be positive for viable investment, got {npv:.2f}"
    assert bcr > 1.0, f"{prefix}BCR must exceed 1.0 for viable investment, got {bcr:.3f}"
    assert not np.isnan(npv), f"{prefix}NPV is NaN"
    assert not np.isnan(bcr), f"{prefix}BCR is NaN"
    assert not np.isinf(npv), f"{prefix}NPV is infinite"
    assert not np.isinf(bcr), f"{prefix}BCR is infinite"


def assert_rrs_scorecard_valid(scorecard: Dict[str, Any]) -> None:
    """Assert RRS scorecard contains required fields and valid grades.

    Parameters
    ----------
    scorecard : dict
        Output from ``RRSReportGenerator``.

    Raises
    ------
    AssertionError
    """
    assert "overall_grade" in scorecard or "resilience_grade" in scorecard, (
        "Scorecard missing grade field"
    )

    grade = scorecard.get("overall_grade") or scorecard.get("resilience_grade")
    valid_grades = {"AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"}
    if isinstance(grade, str):
        assert grade in valid_grades, f"Invalid RRS grade: {grade}"


def generate_hazard_intensity_map(
    grid_model: GridModel,
    hazard: HazardEvent,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Generate per-asset hazard intensity values for a grid.

    Creates a synthetic spatial intensity map based on the hazard
    event's peak intensity and bus geodata coordinates.

    Parameters
    ----------
    grid_model : GridModel
    hazard : HazardEvent
    rng : np.random.Generator or None

    Returns
    -------
    dict
        Mapping from asset_id to intensity value.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    net = grid_model.net
    intensities: Dict[str, float] = {}

    peak = float(hazard.intensity)

    for idx in net.bus.index:
        name = net.bus.at[idx, "name"] or f"bus_{idx}"
        x = net.bus.at[idx, "geodata"].x if hasattr(net.bus.at[idx, "geodata"], "x") else 0.0
        y = net.bus.at[idx, "geodata"].y if hasattr(net.bus.at[idx, "geodata"], "y") else 0.0
        decay = 1.0 / (1.0 + np.sqrt(x**2 + y**2) / 10.0)
        noise = rng.uniform(0.85, 1.15)
        intensities[name] = peak * decay * noise

    for idx in net.line.index:
        name = net.line.at[idx, "name"] or f"line_{idx}"
        from_bus = int(net.line.at[idx, "from_bus"])
        to_bus = int(net.line.at[idx, "to_bus"])
        from_name = net.bus.at[from_bus, "name"] or f"bus_{from_bus}"
        to_name = net.bus.at[to_bus, "name"] or f"bus_{to_bus}"
        avg_intensity = (intensities.get(from_name, peak * 0.5) + intensities.get(to_name, peak * 0.5)) / 2.0
        intensities[name] = avg_intensity * rng.uniform(0.9, 1.1)

    return intensities
