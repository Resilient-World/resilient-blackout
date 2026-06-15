# Copyright (c) 2026, Resilient World
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Synthetic demo grid and hazard data for the Streamlit dashboard."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def create_demo_grid() -> Any:
    """Build a small 5-bus demo pandapower network.

    Returns
    -------
    pandapowerNet
    """
    try:
        import pandapower as pp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pandapower is required for demo data") from exc

    net = pp.create_empty_network(name="demo_5bus")
    # Buses (lat/lon approximated as geo coordinates)
    b0 = pp.create_bus(net, vn_kv=110.0, name="Sub_A", geodata=(40.7128, -74.0060))
    b1 = pp.create_bus(net, vn_kv=110.0, name="Sub_B", geodata=(40.7300, -73.9350))
    b2 = pp.create_bus(net, vn_kv=110.0, name="Sub_C", geodata=(40.6500, -73.9500))
    b3 = pp.create_bus(net, vn_kv=110.0, name="Sub_D", geodata=(40.6800, -74.0400))
    b4 = pp.create_bus(net, vn_kv=110.0, name="Sub_E", geodata=(40.7500, -74.1000))

    # Lines
    pp.create_line(net, from_bus=b0, to_bus=b1, length_km=10.0, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV", name="L01")
    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=12.0, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV", name="L02")
    pp.create_line(net, from_bus=b2, to_bus=b3, length_km=8.0, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV", name="L03")
    pp.create_line(net, from_bus=b3, to_bus=b4, length_km=15.0, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV", name="L04")
    pp.create_line(net, from_bus=b4, to_bus=b0, length_km=18.0, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV", name="L05")
    pp.create_line(net, from_bus=b1, to_bus=b3, length_km=11.0, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV", name="L06")

    # Generation
    pp.create_gen(net, bus=b0, p_mw=50.0, vm_pu=1.0, max_p_mw=120.0, min_p_mw=0.0, name="Gen_A")
    pp.create_gen(net, bus=b2, p_mw=30.0, vm_pu=1.0, max_p_mw=80.0, min_p_mw=0.0, name="Gen_C")

    # Slack / external grid
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0, name="Slack")

    # Loads
    pp.create_load(net, bus=b1, p_mw=40.0, q_mvar=10.0, name="Load_B")
    pp.create_load(net, bus=b2, p_mw=25.0, q_mvar=6.0, name="Load_C")
    pp.create_load(net, bus=b3, p_mw=35.0, q_mvar=8.0, name="Load_D")
    pp.create_load(net, bus=b4, p_mw=20.0, q_mvar=5.0, name="Load_E")

    # Cost column for OPF / scheduling
    net.gen["cost_per_mwh"] = [45.0, 55.0]

    return net


def create_demo_hazard() -> Dict[str, Any]:
    """Return a synthetic polygon hazard footprint.

    Returns
    -------
    dict
        GeoJSON-like feature with ``type``, ``geometry``, ``properties``.
    """
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-73.98, 40.68],
                [-73.92, 40.68],
                [-73.92, 40.74],
                [-73.98, 40.74],
                [-73.98, 40.68],
            ]],
        },
        "properties": {
            "hazard_type": "wildfire",
            "intensity": 0.85,
            "timestamp": "2026-06-11T12:00:00Z",
        },
    }


def create_demo_load_profile(n_steps: int = 24) -> np.ndarray:
    """Synthetic daily load profile with morning/evening peaks.

    Parameters
    ----------
    n_steps : int
        Number of hourly steps.  Default 24.

    Returns
    -------
    np.ndarray
        Shape ``(n_steps, n_loads)``.  Each column is a load element.
    """
    hours = np.arange(n_steps)
    # Base pattern: morning peak ~8h, evening peak ~19h
    pattern = (
        0.6
        + 0.3 * np.exp(-((hours - 8.0) ** 2) / 8.0)
        + 0.2 * np.exp(-((hours - 19.0) ** 2) / 12.0)
    )
    pattern = pattern / pattern.mean()  # normalise to mean 1.0

    n_loads = 4
    base_loads = np.array([40.0, 25.0, 35.0, 20.0], dtype=np.float64)
    profile = np.outer(pattern, base_loads)
    return profile


def create_demo_cascade_history() -> List[Dict[str, Any]]:
    """Return a synthetic cascade iteration log for the animator.

    Returns
    -------
    list of dict
        Each dict has ``iteration``, ``tripped_lines``, ``islands``,
        ``loading_percent``.
    """
    return [
        {
            "iteration": 0,
            "tripped_lines": [],
            "islands": [[0, 1, 2, 3, 4]],
            "loading_percent": [45.0, 62.0, 55.0, 38.0, 50.0, 70.0],
        },
        {
            "iteration": 1,
            "tripped_lines": [2],  # L03 trips
            "islands": [[0, 1], [2, 3, 4]],
            "loading_percent": [45.0, 62.0, 0.0, 48.0, 55.0, 75.0],
        },
        {
            "iteration": 2,
            "tripped_lines": [2, 5],  # L03 + L06 trip
            "islands": [[0, 1], [2], [3, 4]],
            "loading_percent": [45.0, 62.0, 0.0, 48.0, 55.0, 0.0],
        },
        {
            "iteration": 3,
            "tripped_lines": [2, 5, 3],  # L04 also trips
            "islands": [[0, 1], [2], [3], [4]],
            "loading_percent": [45.0, 62.0, 0.0, 0.0, 0.0, 0.0],
        },
    ]


def create_demo_rrs_report() -> Dict[str, Any]:
    """Return a synthetic RRS scorecard for the dashboard.

    Returns
    -------
    dict
    """
    return {
        "report_metadata": {
            "project_name": "Demo Resilience Upgrade",
            "planning_horizon_years": 20,
            "discount_rate": 0.05,
        },
        "key_performance_indicators": {
            "expected_annual_loss_usd": 1_200_000.0,
            "system_wide_voll_usd_per_mwh": 10_000.0,
            "bcr": 2.35,
            "npv_usd": 4_500_000.0,
            "irr": 0.18,
            "avoided_eens_mwh": 450.0,
            "avoided_loss_usd": 3_200_000.0,
        },
        "resilience_of_the_project": {
            "grade": "A+",
            "npv_cv": 0.08,
            "psi": 0.92,
            "rate_of_return": 0.71,
            "npv": 4_500_000.0,
        },
        "resilience_through_the_project": {
            "cmi_reduction_minutes": 120.0,
            "community_impact_score": 87.5,
            "avoided_supply_chain_loss_usd": 1_800_000.0,
            "emissions_offset_tco2": 180.0,
        },
    }
