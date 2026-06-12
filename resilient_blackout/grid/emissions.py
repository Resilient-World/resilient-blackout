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
Carbon-emissions tracking and optimization layer.

Provides ``CarbonEmissionsTracker`` for generator-level carbon
profiling, Average Carbon Emission (ACE), Locational Marginal Carbon
Emission (LMCE), carbon-tax co-optimized optimal power flow, and
avoided emissions co-benefit calculation.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandapower as pp

from resilient_blackout.grid.network import GridModel

logger = logging.getLogger(__name__)

_FUEL_EMISSION_FACTORS: Dict[str, float] = {
    "coal": 950.0,
    "gas": 450.0,
    "ng": 450.0,
    "natural_gas": 450.0,
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
}

_FUEL_KEYWORDS: Dict[str, str] = {
    "coal": "coal",
    "gas": "gas",
    "ng": "gas",
    "natural": "gas",
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
}

_DEFAULT_CO2_FACTOR: float = 600.0
_DEFAULT_FUEL_TYPE: str = "unknown"
_EPS: float = 1e-10


class CarbonEmissionsTracker:
    """Generator-level carbon profiling and emissions optimization.

    Enriches pandapower generator tables with fuel types and CO₂
    emission factors, computes ACE and LMCE, co-optimizes OPF with
    carbon taxes, and calculates avoided emissions as NPV co-benefits.

    Parameters
    ----------
    grid_model : GridModel
        The grid model to analyze.
    fuel_map : dict or str or None
        Optional ``{gen_index: {"fuel": str, "co2_factor": float}}``
        mapping or path to CSV/JSON file.  If ``None``, uses heuristic
        detection from generator names.
    carbon_tax_usd_per_tonne : float
        Carbon price in USD per tonne CO₂.  Default 0.

    Attributes
    ----------
    grid_model : GridModel
    fuel_map : dict or None
    carbon_tax : float
    """

    def __init__(
        self,
        grid_model: GridModel,
        fuel_map: Optional[Dict[int, Dict[str, Any]]] = None,
        carbon_tax_usd_per_tonne: float = 0.0,
    ) -> None:
        if carbon_tax_usd_per_tonne < 0:
            raise ValueError(
                f"carbon_tax_usd_per_tonne must be non-negative, got {carbon_tax_usd_per_tonne}"
            )

        self.grid_model = grid_model
        self.fuel_map = fuel_map
        self.carbon_tax = carbon_tax_usd_per_tonne

    # ------------------------------------------------------------------
    # Generator enrichment
    # ------------------------------------------------------------------

    def enrich_generators(self, net: Any) -> Any:
        """Add fuel_type and co2_kg_per_mwh columns to generator tables.

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
                table["co2_kg_per_mwh"] = _DEFAULT_CO2_FACTOR

            for idx in table.index:
                if self.fuel_map and idx in self.fuel_map:
                    info = self.fuel_map[idx]
                    table.at[idx, "fuel_type"] = str(info.get("fuel", _DEFAULT_FUEL_TYPE))
                    table.at[idx, "co2_kg_per_mwh"] = float(
                        info.get("co2_factor", _DEFAULT_CO2_FACTOR)
                    )
                else:
                    name = str(table.at[idx, "name"]) if "name" in table.columns else ""
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
                factor = _FUEL_EMISSION_FACTORS.get(keyword, _DEFAULT_CO2_FACTOR)
                return fuel_type, factor
        return _DEFAULT_FUEL_TYPE, _DEFAULT_CO2_FACTOR

    # ------------------------------------------------------------------
    # System emissions
    # ------------------------------------------------------------------

    def compute_system_emissions(self, net: Any) -> Dict[str, Any]:
        """Calculate total system CO₂ emissions.

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

    def compute_average_carbon_emission(self, net: Any) -> float:
        """Compute system-wide Average Carbon Emission.

        .. math::

            \\text{ACE} = \\frac{\\sum P_i \\cdot e_i}{\\sum P_i}

        where :math:`e_i` is the CO₂ emission factor of generator i.

        Parameters
        ----------
        net : pandapowerNet

        Returns
        -------
        float
            ACE in kg CO₂/MWh.
        """
        emissions = self.compute_system_emissions(net)
        total_gen = 0.0

        if hasattr(net, "res_gen"):
            total_gen += net.res_gen.loc[net.gen.in_service, "p_mw"].sum()
        if hasattr(net, "res_sgen"):
            total_gen += net.res_sgen.loc[net.sgen.in_service, "p_mw"].sum()

        if total_gen < _EPS:
            return 0.0

        return emissions["total_kg_co2"] / total_gen

    # ------------------------------------------------------------------
    # Locational Marginal Carbon Emission (LMCE)
    # ------------------------------------------------------------------

    def compute_locational_marginal_carbon_emission(
        self,
        net: Any,
        delta_mw: float = 1.0,
    ) -> Dict[str, Any]:
        """Compute LMCE at each bus via finite differences.

        .. math::

            \\text{LMCE}_i = \\frac{\\Delta R}{\\Delta P_i}

        Parameters
        ----------
        net : pandapowerNet
        delta_mw : float
            Load perturbation in MW.  Default 1.0.

        Returns
        -------
        dict
            ``{"lmce": np.ndarray, "bus_indices": list,
            "baseline_emissions_kg": float}``.
        """
        self.enrich_generators(net)

        try:
            pp.runopp(net)
        except pp.OPFNotConverged:
            logger.warning("Baseline OPF did not converge for LMCE.")
            return {"lmce": np.array([]), "bus_indices": [], "baseline_emissions_kg": 0.0}

        baseline = self.compute_system_emissions(net)
        baseline_kg = baseline["total_kg_co2"]

        n_buses = len(net.bus)
        lmce = np.zeros(n_buses, dtype=np.float64)

        for i, bid in enumerate(net.bus.index):
            if not net.bus.at[bid, "in_service"]:
                continue

            test_net = copy.deepcopy(net)
            load_mask = (test_net.load.bus == bid) & test_net.load.in_service
            if not load_mask.any():
                continue

            test_net.load.loc[load_mask, "p_mw"] += delta_mw

            try:
                pp.runopp(test_net)
            except pp.OPFNotConverged:
                continue

            perturbed = self.compute_system_emissions(test_net)
            lmce[i] = (perturbed["total_kg_co2"] - baseline_kg) / delta_mw

        return {
            "lmce": lmce,
            "bus_indices": list(net.bus.index),
            "baseline_emissions_kg": baseline_kg,
        }

    # ------------------------------------------------------------------
    # Carbon-tax OPF
    # ------------------------------------------------------------------

    def run_carbon_opf(
        self,
        net: Any,
        carbon_tax: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run OPF with carbon tax co-optimization.

        Modifies generator polynomial cost curves to include carbon
        costs: ``cost' = cost + co2_factor × carbon_tax / 1000``
        (converting kg/MWh to tonne/MWh for tax application).

        Parameters
        ----------
        net : pandapowerNet
        carbon_tax : float or None
            Carbon price in USD/tonne.  Uses instance default if None.

        Returns
        -------
        dict
            ``{"opf_converged": bool, "total_cost_usd": float,
            "carbon_cost_usd": float, "emissions": dict}``.
        """
        tax = carbon_tax if carbon_tax is not None else self.carbon_tax
        self.enrich_generators(net)

        if "cost_per_mw" not in net.poly_cost.columns:
            logger.warning("No polynomial cost data; cannot run carbon OPF.")
            return {
                "opf_converged": False,
                "total_cost_usd": 0.0,
                "carbon_cost_usd": 0.0,
                "emissions": {},
            }

        original_costs = net.poly_cost["cp1_eur_per_mw"].copy()

        for idx in net.poly_cost.index:
            et = net.poly_cost.at[idx, "et"]
            element = net.poly_cost.at[idx, "element"]

            if et == "gen" and element in net.gen.index:
                co2_factor = float(net.gen.at[element, "co2_kg_per_mwh"])
                carbon_adder = co2_factor * tax / 1000.0
                net.poly_cost.at[idx, "cp1_eur_per_mw"] += carbon_adder
            elif et == "sgen" and element in net.sgen.index:
                co2_factor = float(net.sgen.at[element, "co2_kg_per_mwh"])
                carbon_adder = co2_factor * tax / 1000.0
                net.poly_cost.at[idx, "cp1_eur_per_mw"] += carbon_adder

        try:
            pp.runopp(net)
            converged = True
        except pp.OPFNotConverged:
            logger.warning("Carbon OPF did not converge.")
            converged = False

        net.poly_cost["cp1_eur_per_mw"] = original_costs

        emissions = self.compute_system_emissions(net) if converged else {}
        carbon_cost = emissions.get("total_tonne_co2", 0.0) * tax

        total_cost = 0.0
        if converged and hasattr(net, "res_cost"):
            total_cost = float(net.res_cost.sum())

        return {
            "opf_converged": converged,
            "total_cost_usd": total_cost,
            "carbon_cost_usd": carbon_cost,
            "emissions": emissions,
        }

    # ------------------------------------------------------------------
    # Avoided emissions
    # ------------------------------------------------------------------

    def compute_avoided_emissions(
        self,
        baseline_net: Any,
        resilient_net: Any,
    ) -> Dict[str, Any]:
        """Compute avoided CO₂ emissions between two configurations.

        Parameters
        ----------
        baseline_net : pandapowerNet
        resilient_net : pandapowerNet

        Returns
        -------
        dict
            ``{"avoided_kg_co2": float, "avoided_tonne_co2": float,
            "baseline_emissions": dict, "resilient_emissions": dict}``.
        """
        baseline = self.compute_system_emissions(baseline_net)
        resilient = self.compute_system_emissions(resilient_net)

        avoided_kg = baseline["total_kg_co2"] - resilient["total_kg_co2"]

        return {
            "avoided_kg_co2": avoided_kg,
            "avoided_tonne_co2": avoided_kg / 1000.0,
            "baseline_emissions": baseline,
            "resilient_emissions": resilient,
        }

    # ------------------------------------------------------------------
    # Carbon NPV co-benefit
    # ------------------------------------------------------------------

    @staticmethod
    def compute_carbon_npv_co_benefit(
        avoided_tonne_co2_per_year: float,
        carbon_price_forecast: List[float],
        planning_horizon: int,
        discount_rate: float,
    ) -> Dict[str, Any]:
        """Compute NPV of avoided carbon emissions.

        Parameters
        ----------
        avoided_tonne_co2_per_year : float
            Annual avoided CO₂ in tonnes.
        carbon_price_forecast : list of float
            Projected carbon price per year (USD/tonne).
        planning_horizon : int
            Number of years.
        discount_rate : float
            Annual discount rate.

        Returns
        -------
        dict
            ``{"carbon_npv_usd": float, "annual_benefit_usd": list}``.
        """
        if len(carbon_price_forecast) < planning_horizon:
            last_price = carbon_price_forecast[-1] if carbon_price_forecast else 0.0
            carbon_price_forecast = list(carbon_price_forecast) + [last_price] * (
                planning_horizon - len(carbon_price_forecast)
            )

        annual_benefit: List[float] = []
        npv = 0.0

        for t in range(planning_horizon):
            benefit = avoided_tonne_co2_per_year * carbon_price_forecast[t]
            annual_benefit.append(benefit)
            npv += benefit / ((1.0 + discount_rate) ** (t + 1))

        return {
            "carbon_npv_usd": npv,
            "annual_benefit_usd": annual_benefit,
        }
