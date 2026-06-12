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
Resilient Blackout — Clean-room grid physical risk model.

Provides geospatial hazard-to-asset impact evaluation and financial loss
accumulation for electrical grid infrastructure exposed to natural perils.
"""

from resilient_blackout.climate.downscaling import QuantileDeltaMapper
from resilient_blackout.core.base import Asset, HazardEvent, RiskScenario
from resilient_blackout.core.degradation import ArrheniusDegradationModel, DynamicFragilityAdjuster
from resilient_blackout.core.economics import AvoidedLossCalculator
from resilient_blackout.core.engine import ImpactEngine
from resilient_blackout.core.fragility import ImpactFunction, ImpactFunctionSet
from resilient_blackout.grid.cascade import CascadingSimulator
from resilient_blackout.grid.network import GridModel
from resilient_blackout.grid.thermal_line import DLRGridController, calculate_dynamic_ampacity
from resilient_blackout.utils.geo import map_hazard_to_assets

__all__ = [
    "ArrheniusDegradationModel",
    "Asset",
    "AvoidedLossCalculator",
    "calculate_dynamic_ampacity",
    "CascadingSimulator",
    "DLRGridController",
    "DynamicFragilityAdjuster",
    "GridModel",
    "HazardEvent",
    "ImpactEngine",
    "ImpactFunction",
    "ImpactFunctionSet",
    "QuantileDeltaMapper",
    "RiskScenario",
    "map_hazard_to_assets",
]
__version__ = "0.1.0"
