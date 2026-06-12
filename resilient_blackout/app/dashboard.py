# Copyright (c) 2026, Resilient World
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive Streamlit dashboard for resilient-blackout.

Run with::

    streamlit run resilient_blackout/app/dashboard.py

Supports scenario selection, spatial map rendering, cascade animation,
and RRS resilience scorecards.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------

_MISSING_DEPS: List[str] = []

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    _MISSING_DEPS.append(f"streamlit: {exc}")
    st = None  # type: ignore[assignment]

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover
    _MISSING_DEPS.append(f"plotly: {exc}")
    px = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]

try:
    import folium
    from streamlit_folium import st_folium
except ImportError as exc:  # pragma: no cover
    _MISSING_DEPS.append(f"folium/streamlit_folium: {exc}")
    folium = None  # type: ignore[assignment]
    st_folium = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    _MISSING_DEPS.append(f"pandas: {exc}")
    pd = None  # type: ignore[assignment]

try:
    import pandapower as pp
except ImportError as exc:  # pragma: no cover
    _MISSING_DEPS.append(f"pandapower: {exc}")
    pp = None  # type: ignore[assignment]

if _MISSING_DEPS:  # pragma: no cover
    print("Missing dashboard dependencies:", "; ".join(_MISSING_DEPS))
    sys.exit(1)

from resilient_blackout.app.backends import (
    CascadeAnimatorBackend,
    GridBackend,
    HazardBackend,
    ScorecardBackend,
    SimulationRunner,
)
from resilient_blackout.app.demo_data import (
    create_demo_cascade_history,
    create_demo_grid,
    create_demo_hazard,
    create_demo_load_profile,
    create_demo_rrs_report,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Resilient Blackout Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _ensure_session_state(key: str, default: Any) -> Any:
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# ---------------------------------------------------------------------------
# Sidebar — Scenario Selector
# ---------------------------------------------------------------------------


def render_sidebar() -> Dict[str, Any]:
    """Render the left sidebar and return user selections."""
    st.sidebar.title("⚡ Scenario Selector")

    # Grid source
    st.sidebar.header("Grid Model")
    grid_source = st.sidebar.radio(
        "Source",
        ["Demo (5-bus)", "Upload file"],
        help="Select a pre-built demo or upload your own pandapower model.",
    )

    net = None
    if grid_source == "Demo (5-bus)":
        net = create_demo_grid()
    else:
        uploaded = st.sidebar.file_uploader("Grid file (.xlsx / .json)", type=["xlsx", "json"])
        if uploaded is not None:
            suffix = Path(uploaded.name).suffix
            tmp_path = Path("/tmp") / uploaded.name
            tmp_path.write_bytes(uploaded.read())
            try:
                if suffix == ".xlsx":
                    net = pp.from_excel(str(tmp_path))
                else:
                    net = pp.from_json(str(tmp_path))
            except Exception as exc:
                st.sidebar.error(f"Failed to load grid: {exc}")

    # Hazard source
    st.sidebar.header("Hazard Footprint")
    hazard_source = st.sidebar.radio(
        "Hazard source",
        ["Demo (wildfire polygon)", "Upload GeoJSON"],
        help="Overlay a hazard perimeter or storm track on the map.",
    )

    hazard_feature = create_demo_hazard()
    if hazard_source == "Upload GeoJSON":
        uploaded_geo = st.sidebar.file_uploader("GeoJSON file", type=["json", "geojson"])
        if uploaded_geo is not None:
            try:
                hazard_feature = json.load(uploaded_geo)
            except Exception as exc:
                st.sidebar.error(f"Failed to parse GeoJSON: {exc}")

    # VoLL parameters
    st.sidebar.header("Customer VoLL")
    voll_res = st.sidebar.number_input("Residential ($/MWh)", value=10_000.0, step=1000.0)
    voll_com = st.sidebar.number_input("Commercial ($/MWh)", value=25_000.0, step=1000.0)
    voll_ind = st.sidebar.number_input("Industrial ($/MWh)", value=50_000.0, step=1000.0)

    # Simulation timesteps
    st.sidebar.header("Simulation")
    n_steps = st.sidebar.slider("Horizon (hours)", min_value=4, max_value=72, value=24, step=4)
    dt = st.sidebar.selectbox("Time step", [0.25, 0.5, 1.0], index=2)

    # Grid state
    st.sidebar.header("Grid State")
    grid_state = st.sidebar.radio("Configuration", ["Baseline", "Hardened"])

    return {
        "net": net,
        "hazard_feature": hazard_feature,
        "voll": {"residential": voll_res, "commercial": voll_com, "industrial": voll_ind},
        "n_steps": n_steps,
        "dt_hours": dt,
        "grid_state": grid_state,
    }


# ---------------------------------------------------------------------------
# Spatial Map
# ---------------------------------------------------------------------------


def render_spatial_map(grid_backend: GridBackend, hazard_backend: HazardBackend) -> None:
    """Render the Folium map with substations, lines, and hazard overlay."""
    st.subheader("🗺️ Network & Hazard Map")

    # Compute loading
    converged = grid_backend.run_power_flow()
    if not converged:
        st.warning("Power flow did not converge — line colours may be inaccurate.")

    loading = grid_backend.get_line_loading()

    # Centre map on mean bus position
    bus_coords = grid_backend.get_bus_coordinates()
    if bus_coords:
        lats = [c[0] for c in bus_coords.values()]
        lons = [c[1] for c in bus_coords.values()]
        centre = (float(np.mean(lats)), float(np.mean(lons)))
    else:
        centre = (40.7128, -74.0060)

    m = folium.Map(location=centre, zoom_start=11, tiles="CartoDB positron")

    # Hazard polygon
    hazard_coords = hazard_backend.get_polygon_coordinates()
    if hazard_coords:
        folium.Polygon(
            locations=hazard_coords,
            color="red",
            weight=2,
            fill=True,
            fill_color="red",
            fill_opacity=0.25,
            tooltip="Hazard footprint",
        ).add_to(m)

    # Lines
    line_coords = grid_backend.get_line_coordinates()
    for li, coords in line_coords.items():
        pct = loading.get(li, 0.0)
        if pct > 100.0:
            color = "red"
        elif pct > 80.0:
            color = "orange"
        else:
            color = "green"
        folium.PolyLine(
            locations=coords,
            color=color,
            weight=3,
            tooltip=f"Line {li}: {pct:.1f}%",
        ).add_to(m)

    # Substations (buses)
    for bi, (lat, lon) in bus_coords.items():
        name = grid_backend.net.bus.at[bi, "name"] if "name" in grid_backend.net.bus.columns else f"Bus {bi}"
        folium.CircleMarker(
            location=(lat, lon),
            radius=6,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.8,
            tooltip=name,
        ).add_to(m)

    st_folium(m, width=700, height=500, returned_objects=[])

    # Legend
    legend_html = """
    <div style="background:white;padding:8px;border-radius:4px;box-shadow:0 0 4px rgba(0,0,0,0.2);font-size:12px;">
    <b>Line loading</b><br>
    <span style="color:green;">●</span> Safe (&lt;80%)<br>
    <span style="color:orange;">●</span> Caution (80–100%)<br>
    <span style="color:red;">●</span> Overloaded (&gt;100%)
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cascade Animator
# ---------------------------------------------------------------------------


def render_cascade_animator(grid_backend: GridBackend) -> None:
    """Render a playback widget for cascade iterations."""
    st.subheader("🔥 Real-Time Cascade Animator")

    # Use demo history or run a lightweight synthetic cascade
    history = create_demo_cascade_history()
    animator = CascadeAnimatorBackend(grid_backend, history)

    n_frames = len(history)
    frame = st.slider("Cascade iteration", min_value=0, max_value=n_frames - 1, value=0)

    state = animator.frame_at(frame)
    tripped = state.get("tripped_lines", [])
    islands = state.get("islands", [])

    col1, col2, col3 = st.columns(3)
    col1.metric("Iteration", frame + 1)
    col2.metric("Tripped lines", len(tripped))
    col3.metric("Islands", len(islands))

    # Mini loading bar chart
    line_info = state.get("lines", {})
    if line_info:
        df = pd.DataFrame(
            [
                {"Line": f"L{li:02d}", "Loading (%)": info["loading"], "Tripped": info["tripped"]}
                for li, info in line_info.items()
            ]
        )
        fig = px.bar(
            df,
            x="Line",
            y="Loading (%)",
            color="Tripped",
            color_discrete_map={True: "red", False: "steelblue"},
            title=f"Line loading at iteration {frame + 1}",
        )
        fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Thermal limit")
        st.plotly_chart(fig, use_container_width=True)

    # Islands table
    if islands:
        st.markdown("**Electrical islands**")
        for i, island in enumerate(islands, start=1):
            st.markdown(f"- Island {i}: buses {island}")


# ---------------------------------------------------------------------------
# Resilience Scorecard
# ---------------------------------------------------------------------------


def render_scorecard() -> None:
    """Render the RRS resilience scorecard."""
    st.subheader("📊 Resilience Scorecard")

    report = create_demo_rrs_report()
    backend = ScorecardBackend(report)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("NPV", f"${backend.get_kpis()['NPV ($)']:,.0f}")
    col2.metric("BCR", f"{backend.get_kpis()['BCR']:.2f}")
    col3.metric("Grade", backend.get_grade())
    col4.metric("Community Score", f"{backend.get_community_score():.1f}")

    # Detailed KPIs
    with st.expander("Detailed KPIs"):
        for key, val in backend.get_kpis().items():
            st.write(f"**{key}**: {val:,.2f}")

    # Plotly bar chart of metrics
    df = backend.to_dataframe()
    fig = px.bar(
        df,
        x="metric",
        y="value",
        color="metric",
        title="Resilience Metrics",
        labels={"value": "Value", "metric": "Metric"},
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# OPF Schedule preview
# ---------------------------------------------------------------------------


def render_opf_schedule(grid_backend: GridBackend, n_steps: int) -> None:
    """Render a simple multi-period OPF schedule preview."""
    st.subheader("⚙️ Multi-Period OPF Schedule")

    load_profile = create_demo_load_profile(n_steps)
    runner = SimulationRunner(grid_backend)

    with st.spinner("Solving OPF schedule …"):
        result = runner.run_opf_schedule(load_profile)

    if result["status"] != 0:
        st.error(f"OPF failed: {result.get('message', 'unknown')}")
        return

    gen = result["gen_schedule"]
    shed = result["shed_per_bus"]
    hours = list(range(n_steps))

    # Generation stack
    df_gen = pd.DataFrame({"Hour": hours, "Total Gen (MW)": np.sum(gen, axis=1)})
    fig_gen = px.area(df_gen, x="Hour", y="Total Gen (MW)", title="Scheduled Generation")
    st.plotly_chart(fig_gen, use_container_width=True)

    # Load shed
    df_shed = pd.DataFrame({"Hour": hours, "Total Shed (MW)": np.sum(shed, axis=1)})
    fig_shed = px.line(df_shed, x="Hour", y="Total Shed (MW)", title="Unserved Load")
    st.plotly_chart(fig_shed, use_container_width=True)

    # Battery SOC if present
    if result["battery_schedule"]:
        for s, sched in result["battery_schedule"].items():
            df_batt = pd.DataFrame({
                "Hour": hours,
                "SOC (MWh)": sched["e_mwh"],
                "Charge (MW)": sched["p_char_mw"],
                "Discharge (MW)": sched["p_disch_mw"],
            })
            fig_batt = px.line(
                df_batt, x="Hour", y=["SOC (MWh)", "Charge (MW)", "Discharge (MW)"],
                title=f"Battery {s} Dispatch"
            )
            st.plotly_chart(fig_batt, use_container_width=True)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------


def main() -> None:
    """Dashboard entry point."""
    st.title("Resilient Blackout Dashboard")
    st.caption("Visual wrapper for the resilient-blackout Python package")

    selections = render_sidebar()
    net = selections["net"]

    if net is None:
        st.info("👈 Upload a grid model in the sidebar to get started, or select the demo.")
        return

    grid_backend = GridBackend(net)
    hazard_backend = HazardBackend(selections["hazard_feature"])

    tab_map, tab_cascade, tab_opf, tab_scorecard = st.tabs(
        ["🗺️ Spatial Map", "🔥 Cascade Animator", "⚙️ OPF Schedule", "📊 Scorecard"]
    )

    with tab_map:
        render_spatial_map(grid_backend, hazard_backend)

    with tab_cascade:
        render_cascade_animator(grid_backend)

    with tab_opf:
        render_opf_schedule(grid_backend, selections["n_steps"])

    with tab_scorecard:
        render_scorecard()


if __name__ == "__main__":
    main()
