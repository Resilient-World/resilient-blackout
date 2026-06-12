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
Distribution reliability indices and microgrid islanding evaluation.

Provides ``IEEEMetricCalculator`` for standard IEEE 1366 reliability
indices (SAIFI, SAIDI, CAIDI, CEMI-n, MEDs) and
``MicrogridIslandEvaluator`` for assessing whether downstream loads can
island behind local DER when upstream lines fail.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# IEEEMetricCalculator
# ---------------------------------------------------------------------------


class IEEEMetricCalculator:
    """Calculator for IEEE 1366 distribution reliability indices.

    Parameters
    ----------
    net : pandapowerNet
        The distribution network.  Used to resolve bus indices and
        default customer counts.
    bus_customers : dict or None
        Optional override mapping ``{bus_index: n_customers}``.
        If ``None``, reads ``net.bus['customers']`` (defaulting to 1).
    med_threshold_h : float
        Major Event Day threshold in hours of SAIDI.  Default 4.0.
    sustained_min_h : float
        Minimum outage duration (hours) to count as a *sustained*
        interruption.  Default 5/60 (5 minutes).

    Attributes
    ----------
    net : pandapowerNet
    bus_customers : dict
    med_threshold_h : float
    sustained_min_h : float
    """

    def __init__(
        self,
        net: Any,
        bus_customers: Optional[Dict[int, int]] = None,
        med_threshold_h: float = 4.0,
        sustained_min_h: float = 5.0 / 60.0,
    ) -> None:
        self.net = net
        self.med_threshold_h = float(med_threshold_h)
        self.sustained_min_h = float(sustained_min_h)

        if bus_customers is not None:
            self.bus_customers = dict(bus_customers)
        else:
            if "customers" in net.bus.columns:
                self.bus_customers = net.bus["customers"].to_dict()
            else:
                self.bus_customers = {b: 1 for b in net.bus.index}

    # ------------------------------------------------------------------
    # SAIFI
    # ------------------------------------------------------------------

    def calculate_saifi(self, outage_events: pd.DataFrame) -> float:
        r"""System Average Interruption Frequency Index.

        .. math::
            \text{SAIFI} = \frac{\sum N_i}{N_{\text{total}}}

        Where :math:`N_i` is the number of customers interrupted by
        event :math:`i`.

        Parameters
        ----------
        outage_events : pd.DataFrame
            Columns: ``bus`` (int), ``start_h`` (float), ``end_h``
            (float), and optionally ``customers`` (int).

        Returns
        -------
        float
        """
        n_total = sum(self.bus_customers.values())
        if n_total == 0:
            return 0.0

        customers_per_event = self._resolve_customers(outage_events)
        sustained = self._sustained_mask(outage_events)
        return float(customers_per_event[sustained].sum()) / n_total

    # ------------------------------------------------------------------
    # SAIDI
    # ------------------------------------------------------------------

    def calculate_saidi(self, outage_events: pd.DataFrame) -> float:
        r"""System Average Interruption Duration Index.

        .. math::
            \text{SAIDI} = \frac{\sum U_i N_i}{N_{\text{total}}}

        Where :math:`U_i` is the duration (hours) of event :math:`i`.

        Parameters
        ----------
        outage_events : pd.DataFrame

        Returns
        -------
        float
        """
        n_total = sum(self.bus_customers.values())
        if n_total == 0:
            return 0.0

        customers_per_event = self._resolve_customers(outage_events)
        durations = outage_events["end_h"] - outage_events["start_h"]
        sustained = self._sustained_mask(outage_events)
        return float((durations[sustained] * customers_per_event[sustained]).sum()) / n_total

    # ------------------------------------------------------------------
    # CAIDI
    # ------------------------------------------------------------------

    def calculate_caidi(self, outage_events: pd.DataFrame) -> float:
        r"""Customer Average Interruption Duration Index.

        .. math::
            \text{CAIDI} = \frac{\text{SAIDI}}{\text{SAIFI}}

        Parameters
        ----------
        outage_events : pd.DataFrame

        Returns
        -------
        float
            Returns ``0.0`` when SAIFI is zero to avoid division by zero.
        """
        saifi = self.calculate_saifi(outage_events)
        if saifi < _EPS:
            return 0.0
        return self.calculate_saidi(outage_events) / saifi

    # ------------------------------------------------------------------
    # CEMI-n
    # ------------------------------------------------------------------

    def calculate_cemi_n(self, outage_events: pd.DataFrame, n: int = 5) -> float:
        r"""Customers Experiencing Multiple Interruptions.

        Fraction of customers that experience more than *n* sustained
        interruptions.

        Parameters
        ----------
        outage_events : pd.DataFrame
        n : int
            Interruption count threshold.  Default 5.

        Returns
        -------
        float
        """
        n_total = sum(self.bus_customers.values())
        if n_total == 0:
            return 0.0

        sustained = self._sustained_mask(outage_events)
        events = outage_events[sustained].copy()
        if events.empty:
            return 0.0

        # Count interruptions per customer
        per_customer = events.groupby("bus").size().reindex(
            self.bus_customers.keys(), fill_value=0
        )
        affected = (per_customer > n).sum()
        return float(affected) / len(self.bus_customers)

    # ------------------------------------------------------------------
    # MEDs
    # ------------------------------------------------------------------

    def calculate_meds(self, outage_events: pd.DataFrame) -> int:
        """Major Event Days.

        Counts the number of unique days where the daily SAIDI exceeds
        ``med_threshold_h``.

        Parameters
        ----------
        outage_events : pd.DataFrame

        Returns
        -------
        int
        """
        if outage_events.empty:
            return 0

        events = outage_events.copy()
        events["day"] = (events["start_h"] // 24).astype(int)
        events["duration_h"] = events["end_h"] - events["start_h"]
        customers = self._resolve_customers(events)
        events["customer_hours"] = events["duration_h"] * customers

        daily_saidi = (
            events.groupby("day")["customer_hours"].sum()
            / sum(self.bus_customers.values())
        )
        return int((daily_saidi > self.med_threshold_h).sum())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def reliability_summary(self, outage_events: pd.DataFrame) -> pd.DataFrame:
        """Compute all reliability indices and return a summary.

        Parameters
        ----------
        outage_events : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with columns ``saifi``, ``saidi``,
            ``caidi``, ``cemi_5``, ``meds``.
        """
        saifi = self.calculate_saifi(outage_events)
        saidi = self.calculate_saidi(outage_events)
        caidi = self.calculate_caidi(outage_events)
        cemi = self.calculate_cemi_n(outage_events, n=5)
        meds = self.calculate_meds(outage_events)

        return pd.DataFrame(
            {
                "saifi": [saifi],
                "saidi": [saidi],
                "caidi": [caidi],
                "cemi_5": [cemi],
                "meds": [meds],
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_customers(self, outage_events: pd.DataFrame) -> pd.Series:
        """Return a Series of customer counts aligned with outage_events."""
        if "customers" in outage_events.columns:
            return outage_events["customers"]
        return outage_events["bus"].map(self.bus_customers).fillna(1).astype(int)

    def _sustained_mask(self, outage_events: pd.DataFrame) -> pd.Series:
        """Boolean mask for outages exceeding the sustained threshold."""
        duration = outage_events["end_h"] - outage_events["start_h"]
        return duration >= self.sustained_min_h

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"IEEEMetricCalculator(buses={len(self.net.bus)}, "
            f"med_threshold_h={self.med_threshold_h})"
        )


# ---------------------------------------------------------------------------
# MicrogridIslandEvaluator
# ---------------------------------------------------------------------------


class MicrogridIslandEvaluator:
    """Evaluate whether downstream loads can island behind local DER.

    When an upstream line fails, the evaluator checks if the isolated
    downstream subgraph contains an active, controllable DER (battery
    or solar-plus-storage) with sufficient energy to sustain critical
    loads for part of the outage duration.

    Parameters
    ----------
    net : pandapowerNet
    storage_specs : dict
        ``{bus_idx: {"e_max_mwh": float, "p_max_mw": float,
        "eta": float, "soc_min": float}}``.
    min_battery_soc : float
        Minimum allowable state of charge (0–1).  Default 0.2.

    Attributes
    ----------
    net : pandapowerNet
    storage_specs : dict
    min_battery_soc : float
    """

    def __init__(
        self,
        net: Any,
        storage_specs: Dict[int, Dict[str, float]],
        min_battery_soc: float = 0.2,
    ) -> None:
        self.net = net
        self.storage_specs = dict(storage_specs)
        self.min_battery_soc = float(min_battery_soc)

    # ------------------------------------------------------------------
    # Downstream discovery
    # ------------------------------------------------------------------

    def _find_downstream_buses(self, failed_line_idx: int) -> List[int]:
        """Return buses isolated when *failed_line_idx* is opened.

        Uses pandapower topology helpers to find the connected
        component on the "to" side of the line (away from the
        swing/ext grid).

        Parameters
        ----------
        failed_line_idx : int
            Index of the failed line in ``net.line``.

        Returns
        -------
        list of int
            Downstream bus indices.
        """
        import pandapower as pp

        line = self.net.line.loc[failed_line_idx]
        from_bus = int(line["from_bus"])
        to_bus = int(line["to_bus"])

        # Temporarily open the line and find connected components
        original_in_service = bool(self.net.line.at[failed_line_idx, "in_service"])
        self.net.line.at[failed_line_idx, "in_service"] = False

        try:
            components = pp.topology.connected_components(
                self.net, respect_switches=False, respect_in_service=True
            )
            # Find the component containing to_bus
            downstream = []
            for comp in components:
                if to_bus in comp:
                    downstream = list(comp)
                    break
        finally:
            self.net.line.at[failed_line_idx, "in_service"] = original_in_service

        return downstream

    # ------------------------------------------------------------------
    # DER availability
    # ------------------------------------------------------------------

    def _has_active_der(self, buses: List[int]) -> Tuple[bool, float, float]:
        """Check if any downstream bus has usable battery capacity.

        Returns
        -------
        has_der : bool
        total_energy_mwh : float
            Usable battery energy (above min SOC).
        total_power_mw : float
            Maximum discharge power.
        """
        total_energy = 0.0
        total_power = 0.0
        has_any = False

        for bus in buses:
            spec = self.storage_specs.get(bus)
            if spec is None:
                continue
            e_max = spec.get("e_max_mwh", 0.0)
            soc_min = spec.get("soc_min", self.min_battery_soc)
            usable = e_max * (1.0 - soc_min)
            if usable > _EPS:
                has_any = True
                total_energy += usable
                total_power += spec.get("p_max_mw", 0.0)

        return has_any, total_energy, total_power

    # ------------------------------------------------------------------
    # Islanded duration
    # ------------------------------------------------------------------

    def _calculate_islanded_duration(
        self,
        outage_h: float,
        load_mw: float,
        battery_mwh: float,
    ) -> float:
        r"""Reduce outage duration based on available battery hours.

        .. math::
            U_{i,\text{islanded}} = \max\left(0, U_i -
            \frac{E_{\text{battery}}}{P_{\text{load},i}}\right)

        Parameters
        ----------
        outage_h : float
            Original outage duration in hours.
        load_mw : float
            Downstream load in MW.
        battery_mwh : float
            Usable battery energy in MWh.

        Returns
        -------
        float
            Reduced outage duration in hours.
        """
        if load_mw < _EPS:
            return 0.0
        battery_hours = battery_mwh / load_mw
        return max(0.0, outage_h - battery_hours)

    # ------------------------------------------------------------------
    # Load aggregation
    # ------------------------------------------------------------------

    def _downstream_load(self, buses: List[int]) -> float:
        """Sum active load at the given buses in MW."""
        total = 0.0
        for bid in buses:
            mask = (self.net.load.bus == bid) & self.net.load.in_service
            total += self.net.load.loc[mask, "p_mw"].sum()
        return total

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_islanding(self, failed_line_idx: int, outage_h: float) -> Dict[str, Any]:
        """Assess islanding potential for a single line failure.

        Parameters
        ----------
        failed_line_idx : int
            Index of the failed line.
        outage_h : float
            Original outage duration without islanding.

        Returns
        -------
        dict
            Keys: ``downstream_buses`` (list), ``downstream_load_mw``
            (float), ``has_der`` (bool), ``battery_mwh`` (float),
            ``battery_hours`` (float), ``islanded_duration_h`` (float),
            ``reduction_h`` (float).
        """
        downstream = self._find_downstream_buses(failed_line_idx)
        load_mw = self._downstream_load(downstream)
        has_der, battery_mwh, battery_power = self._has_active_der(downstream)

        islanded_duration = outage_h
        if has_der:
            islanded_duration = self._calculate_islanded_duration(
                outage_h, load_mw, battery_mwh
            )

        return {
            "downstream_buses": downstream,
            "downstream_load_mw": load_mw,
            "has_der": has_der,
            "battery_mwh": battery_mwh,
            "battery_power_mw": battery_power,
            "battery_hours": battery_mwh / load_mw if load_mw > _EPS else 0.0,
            "islanded_duration_h": islanded_duration,
            "reduction_h": outage_h - islanded_duration,
        }

    def evaluate_all_lines(
        self,
        outage_h: float,
        line_indices: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Evaluate islanding for all (or a subset of) lines.

        Parameters
        ----------
        outage_h : float
            Assumed outage duration for each failed line.
        line_indices : list of int or None
            If ``None``, evaluates all in-service lines.

        Returns
        -------
        pd.DataFrame
            One row per line with islanding metrics.
        """
        if line_indices is None:
            line_indices = self.net.line[self.net.line.in_service].index.tolist()

        records: List[Dict[str, Any]] = []
        for idx in line_indices:
            result = self.evaluate_islanding(idx, outage_h)
            records.append(
                {
                    "line_idx": idx,
                    "from_bus": self.net.line.at[idx, "from_bus"],
                    "to_bus": self.net.line.at[idx, "to_bus"],
                    "n_downstream_buses": len(result["downstream_buses"]),
                    "downstream_load_mw": result["downstream_load_mw"],
                    "has_der": result["has_der"],
                    "battery_mwh": result["battery_mwh"],
                    "battery_hours": result["battery_hours"],
                    "islanded_duration_h": result["islanded_duration_h"],
                    "reduction_h": result["reduction_h"],
                }
            )
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Comparative summary
    # ------------------------------------------------------------------

    @staticmethod
    def comparative_summary(
        baseline_events: pd.DataFrame,
        islanded_events: pd.DataFrame,
        calculator: IEEEMetricCalculator,
    ) -> pd.DataFrame:
        """Compare baseline and islanded reliability metrics.

        Parameters
        ----------
        baseline_events : pd.DataFrame
            Outage events *without* microgrid islanding.
        islanded_events : pd.DataFrame
            Outage events *with* microgrid islanding (reduced durations).
        calculator : IEEEMetricCalculator
            Pre-initialised calculator.

        Returns
        -------
        pd.DataFrame
            Two-row DataFrame with ``scenario`` (baseline / islanded)
            and all reliability indices.
        """
        base = calculator.reliability_summary(baseline_events)
        island = calculator.reliability_summary(islanded_events)

        base["scenario"] = "baseline"
        island["scenario"] = "islanded"

        return pd.concat([base, island], ignore_index=True)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MicrogridIslandEvaluator(storage_buses={len(self.storage_specs)}, "
            f"min_soc={self.min_battery_soc})"
        )
