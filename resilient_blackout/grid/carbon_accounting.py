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

"""
Carbon emissions optimization and tracking module.

Provides ``CarbonAccountingEngine`` for generator-level carbon
profiling, Average Carbon Emission (ACE), Locational Marginal Carbon
Emission (LMCE) via numerical finite differences, and multi-period
simulation comparing baseline versus resilient grid configurations
with cumulative avoided emissions in metric tons.

Reference
---------
* Locational Marginal Carbon Emissions (LMCE): methodology analogous
  to Locational Marginal Pricing (LMP) but applied to CO₂ flows.
* IPCC Guidelines for National Greenhouse Gas Inventories (2006)
  for fuel-based emission factors.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fuel emission factors (kg CO₂-eq per MWh)
# ---------------------------------------------------------------------------

_FUEL_EMISSION_FACTORS: Dict[str, float] = {
    "coal": 950.0,
    "lignite": 1050.0,
    "gas": 450.0,
    "ng": 450.0,
    "natural_gas": 450.0,
    "ccgt": 400.0,
    "ocgt": 550.0,
    "oil": 750.0,
    "diesel": 750.0,
    "petroleum": 750.0,
    "nuclear": 0.0,
    "uranium": 0.0,
    "wind": 0.0,
    "solar": 0.0,
    "pv": 0.0,
    "hydro": 0.0,
    "geo": 0.0,
    "geothermal": 0.0,
    "biomass": 50.0,
    "bio": 50.0,
    "storage": 0.0,
    "battery": 0.0,
}

_FUEL_KEYWORDS: Dict[str, str] = {
    "coal": "coal",
    "lignite": "coal",
    "gas": "gas",
    "ng": "gas",
    "natural": "gas",
    "ccgt": "gas",
    "ocgt": "gas",
    "oil": "oil",
    "diesel": "oil",
    "petroleum": "oil",
    "nuclear": "nuclear",
    "uranium": "nuclear",
    "wind": "wind",
    "solar": "solar",
    "pv": "solar",
    "hydro": "hydro",
    "geo": "geo",
    "geothermal": "geo",
    "biomass": "biomass",
    "bio": "biomass",
    "storage": "storage",
    "battery": "storage",
}

_DEFAULT_CO2_FACTOR: float = 600.0
_DEFAULT_FUEL_TYPE: str = "unknown"
_EPS: float = 1e-10


# ---------------------------------------------------------------------------
# CarbonAccountingEngine
# ---------------------------------------------------------------------------


class CarbonAccountingEngine:
    """Carbon emissions optimization and tracking engine.

    Enriches pandapower generator tables with fuel-type-based CO₂
    emission factors, computes system-wide Average Carbon Emission
    (ACE), Locational Marginal Carbon Emission (LMCE) at individual
    buses via numerical finite differences, and simulates multi-period
    planning horizons to quantify cumulative avoided emissions between
    baseline and resilient grid configurations.

    Parameters
    ----------
    fuel_map : dict or None
        Optional ``{gen_index: {"fuel": str, "co2_factor": float}}``
        mapping.  If ``None``, uses heuristic detection from generator
        names.
    default_co2_factor : float
        Default CO₂ factor in kg/MWh for unrecognised generators.
        Default 600.

    Attributes
    ----------
    fuel_map : dict or None
    default_co2_factor : float
    """

    def __init__(
        self,
        fuel_map: Optional[Dict[int, Dict[str, Any]]] = None,
        default_co2_factor: float = _DEFAULT_CO2_FACTOR,
    ) -> None:
        if default_co2_factor < 0:
            raise ValueError(
                f"default_co2_factor must be non-negative, got {default_co2_factor}"
            )

        self.fuel_map = fuel_map
        self.default_co2_factor = float(default_co2_factor)

    # ------------------------------------------------------------------
    # Generator enrichment
    # ------------------------------------------------------------------

    def enrich_generators(self, net: Any) -> Any:
        """Add ``fuel_type`` and ``co2_kg_per_mwh`` columns to generator tables.

        Mutates ``net.gen`` and ``net.sgen`` in place.  Uses the
        instance ``fuel_map`` if provided, otherwise detects fuel type
        heuristically from generator names.

        Parameters
        ----------
        net : pandapowerNet

        Returns
        -------
        pandapowerNet
            Enriched network (mutated in place).
        """
        for table_name in ["gen", "sgen"]:
            table = getattr(net, table_name)
            if "fuel_type" not in table.columns:
                table["fuel_type"] = _DEFAULT_FUEL_TYPE
            if "co2_kg_per_mwh" not in table.columns:
                table["co2_kg_per_mwh"] = self.default_co2_factor

            for idx in table.index:
                if self.fuel_map and idx in self.fuel_map:
                    info = self.fuel_map[idx]
                    table.at[idx, "fuel_type"] = str(
                        info.get("fuel", _DEFAULT_FUEL_TYPE)
                    )
                    table.at[idx, "co2_kg_per_mwh"] = float(
                        info.get("co2_factor", self.default_co2_factor)
                    )
                else:
                    name = (
                        str(table.at[idx, "name"])
                        if "name" in table.columns
                        else ""
                    )
                    fuel, factor = self._detect_fuel_from_name(name)
                    table.at[idx, "fuel_type"] = fuel
                    table.at[idx, "co2_kg_per_mwh"] = factor

        logger.info(
            "Enriched %d gen + %d sgen with fuel types and CO₂ factors.",
            len(net.gen), len(net.sgen),
        )
        return net

    @staticmethod
    def _detect_fuel_from_name(name: str) -> Tuple[str, float]:
        """Detect fuel type and emission factor from generator name.

        Parameters
        ----------
        name : str

        Returns
        -------
        tuple of (str, float)
        """
        name_lower = name.lower()
        for keyword, fuel_type in sorted(
            _FUEL_KEYWORDS.items(), key=lambda x: -len(x[0])
        ):
            if re.search(rf"\b{re.escape(keyword)}\b", name_lower):
                factor = _FUEL_EMISSION_FACTORS.get(
                    keyword, _DEFAULT_CO2_FACTOR
                )
                return fuel_type, factor
        return _DEFAULT_FUEL_TYPE, _DEFAULT_CO2_FACTOR

    # ------------------------------------------------------------------
    # System emissions
    # ------------------------------------------------------------------

    def compute_system_emissions(self, net: Any) -> Dict[str, Any]:
        """Calculate total system CO₂ emissions from generator dispatch.

        Requires that a power flow or OPF has been run so that
        ``res_gen`` and ``res_sgen`` are populated.

        Parameters
        ----------
        net : pandapowerNet

        Returns
        -------
        dict
            ``{"total_kg_co2": float, "total_tonne_co2": float,
            "per_gen": dict, "per_sgen": dict}``.
        """
        self.enrich_generators(net)

        total_kg = 0.0
        per_gen: Dict[int, float] = {}
        per_sgen: Dict[int, float] = {}

        for idx in net.gen.index:
            if not net.gen.at[idx, "in_service"]:
                continue
            p_mw = 0.0
            if hasattr(net, "res_gen") and idx in net.res_gen.index:
                p_mw = float(net.res_gen.at[idx, "p_mw"])
            co2_factor = float(net.gen.at[idx, "co2_kg_per_mwh"])
            kg = p_mw * co2_factor
            total_kg += kg
            per_gen[idx] = kg

        for idx in net.sgen.index:
            if not net.sgen.at[idx, "in_service"]:
                continue
            p_mw = 0.0
            if hasattr(net, "res_sgen") and idx in net.res_sgen.index:
                p_mw = float(net.res_sgen.at[idx, "p_mw"])
            co2_factor = float(net.sgen.at[idx, "co2_kg_per_mwh"])
            kg = p_mw * co2_factor
            total_kg += kg
            per_sgen[idx] = kg

        return {
            "total_kg_co2": total_kg,
            "total_tonne_co2": total_kg / 1000.0,
            "per_gen": per_gen,
            "per_sgen": per_sgen,
        }

    # ------------------------------------------------------------------
    # Average Carbon Emission (ACE)
    # ------------------------------------------------------------------

    def calculate_average_carbon_emission(self, net: Any) -> float:
        r"""Compute system-wide Average Carbon Emission (ACE) factor.

        .. math::

            \text{ACE} = \frac{\text{Total Carbon Emissions (kg/h)}}
            {\text{Total Load Demand (MWh/h)}}

        Parameters
        ----------
        net : pandapowerNet
            Must have power-flow results populated.

        Returns
        -------
        float
            ACE in kg CO₂ per MWh of load served.
        """
        emissions = self.compute_system_emissions(net)

        total_load = 0.0
        if hasattr(net, "res_load"):
            total_load = float(
                net.res_load.loc[net.load.in_service, "p_mw"].sum()
            )

        if total_load < _EPS:
            return 0.0

        return emissions["total_kg_co2"] / total_load

    # ------------------------------------------------------------------
    # Locational Marginal Carbon Emission (LMCE)
    # ------------------------------------------------------------------

    def calculate_locational_marginal_carbon_emission(
        self,
        net: Any,
        target_bus: int,
        delta_mw: float = 1.0,
    ) -> float:
        r"""Compute LMCE at a single target bus via numerical finite difference.

        .. math::

            \text{LMCE}_i = \frac{dR}{dP_i}
            \approx \frac{R(P_i + \Delta) - R(P_i)}{\Delta}

        where :math:`R` is total system CO₂ emissions and
        :math:`\Delta` is *delta_mw*.

        Runs OPF on the baseline network, then perturbs the load at
        *target_bus* by *delta_mw*, re-runs OPF, and computes the
        marginal change in emissions.

        Parameters
        ----------
        net : pandapowerNet
            Network with polynomial cost data for OPF.
        target_bus : int
            Bus index at which to evaluate LMCE.
        delta_mw : float
            Load perturbation in MW.  Default 1.0.

        Returns
        -------
        float
            LMCE in kg CO₂ per MWh of incremental load at *target_bus*.
            Returns 0.0 if OPF fails or bus has no load.
        """
        import pandapower as pp

        self.enrich_generators(net)

        # Baseline OPF
        try:
            pp.runopp(net)
        except pp.OPFNotConverged:
            logger.warning("Baseline OPF did not converge for LMCE at bus %d.", target_bus)
            return 0.0

        baseline_emissions = self.compute_system_emissions(net)
        baseline_kg = baseline_emissions["total_kg_co2"]

        # Check target bus has load
        load_mask = (net.load.bus == target_bus) & net.load.in_service
        if not load_mask.any():
            logger.warning("Bus %d has no in-service load; LMCE is zero.", target_bus)
            return 0.0

        # Perturbed OPF
        test_net = copy.deepcopy(net)
        test_net.load.loc[load_mask, "p_mw"] += delta_mw

        try:
            pp.runopp(test_net)
        except pp.OPFNotConverged:
            logger.warning("Perturbed OPF did not converge for LMCE at bus %d.", target_bus)
            return 0.0

        perturbed_emissions = self.compute_system_emissions(test_net)
        perturbed_kg = perturbed_emissions["total_kg_co2"]

        return (perturbed_kg - baseline_kg) / delta_mw

    # ------------------------------------------------------------------
    # Multi-period simulation
    # ------------------------------------------------------------------

    def simulate_multi_period(
        self,
        baseline_net: Any,
        resilient_net: Any,
        load_profile_mw: np.ndarray,
        delta_mw: float = 1.0,
    ) -> Dict[str, Any]:
        """Simulate emissions over a multi-period planning horizon.

        Compares a baseline grid configuration against a resilient
        configuration (e.g. with added local energy storage) across
        multiple time steps, computing cumulative avoided carbon
        emissions in metric tons.

        Parameters
        ----------
        baseline_net : pandapowerNet
            Baseline network (e.g. without storage).
        resilient_net : pandapowerNet
            Resilient network (e.g. with storage or DERs added).
        load_profile_mw : np.ndarray
            Total system load per time step in MW.  Shape ``(T,)``.
        delta_mw : float
            Perturbation for LMCE computation at each step.
            Default 1.0.

        Returns
        -------
        dict
            Keys:

            - ``baseline_emissions_tonne`` (np.ndarray) — per-step
              baseline emissions in metric tons.
            - ``resilient_emissions_tonne`` (np.ndarray) — per-step
              resilient emissions in metric tons.
            - ``avoided_emissions_tonne`` (np.ndarray) — per-step
              avoided emissions in metric tons.
            - ``cumulative_avoided_tonne`` (float) — total avoided
              emissions over the horizon.
            - ``baseline_ace`` (np.ndarray) — per-step ACE.
            - ``resilient_ace`` (np.ndarray) — per-step ACE.
            - ``converged`` (np.ndarray) — bool per step.
        """
        import pandapower as pp

        T = len(load_profile_mw)
        load_profile_mw = np.asarray(load_profile_mw, dtype=np.float64)

        baseline_tonne = np.zeros(T, dtype=np.float64)
        resilient_tonne = np.zeros(T, dtype=np.float64)
        avoided_tonne = np.zeros(T, dtype=np.float64)
        baseline_ace = np.zeros(T, dtype=np.float64)
        resilient_ace = np.zeros(T, dtype=np.float64)
        converged = np.zeros(T, dtype=bool)

        for t in range(T):
            total_load = float(load_profile_mw[t])

            # Scale all loads proportionally to match the profile
            _scale_loads(baseline_net, total_load)
            _scale_loads(resilient_net, total_load)

            # Baseline OPF
            try:
                pp.runopp(baseline_net)
                b_emis = self.compute_system_emissions(baseline_net)
                baseline_tonne[t] = b_emis["total_tonne_co2"]
                baseline_ace[t] = self.calculate_average_carbon_emission(baseline_net)
                conv_b = True
            except pp.OPFNotConverged:
                logger.warning("Baseline OPF failed at step %d.", t)
                conv_b = False

            # Resilient OPF
            try:
                pp.runopp(resilient_net)
                r_emis = self.compute_system_emissions(resilient_net)
                resilient_tonne[t] = r_emis["total_tonne_co2"]
                resilient_ace[t] = self.calculate_average_carbon_emission(resilient_net)
                conv_r = True
            except pp.OPFNotConverged:
                logger.warning("Resilient OPF failed at step %d.", t)
                conv_r = False

            converged[t] = conv_b and conv_r

            if converged[t]:
                avoided_tonne[t] = baseline_tonne[t] - resilient_tonne[t]

        cumulative_avoided = float(np.sum(avoided_tonne))

        logger.info(
            "Multi-period simulation: %.2f tonnes CO₂ avoided over %d steps.",
            cumulative_avoided, T,
        )

        return {
            "baseline_emissions_tonne": baseline_tonne,
            "resilient_emissions_tonne": resilient_tonne,
            "avoided_emissions_tonne": avoided_tonne,
            "cumulative_avoided_tonne": cumulative_avoided,
            "baseline_ace": baseline_ace,
            "resilient_ace": resilient_ace,
            "converged": converged,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CarbonAccountingEngine(fuel_map={'set' if self.fuel_map else 'auto'}, "
            f"default_co2={self.default_co2_factor:.0f} kg/MWh)"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scale_loads(net: Any, total_load_mw: float) -> None:
    """Scale all in-service loads proportionally to match *total_load_mw*.

    Parameters
    ----------
    net : pandapowerNet
    total_load_mw : float
    """
    load_mask = net.load.in_service.values.astype(bool)
    if not load_mask.any():
        return

    current_total = float(net.load.loc[load_mask, "p_mw"].sum())
    if current_total < _EPS:
        return

    scale = total_load_mw / current_total
    net.load.loc[load_mask, "p_mw"] *= scale
