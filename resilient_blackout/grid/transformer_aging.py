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
Physical transformer thermal degradation and loss-of-life engine.

Implements IEC 60076-7 and IEEE Std C57.91-1995 models for winding
hottest-spot temperature, Arrhenius insulation aging, and Weibull
age-dependent failure rates.  Provides ``TransformerThermalModel``
for calculating cumulative loss-of-life from hourly load and ambient
temperature profiles.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IEEE C57.91-1995 physical constants
# ---------------------------------------------------------------------------

# Arrhenius constants for 65°C rise insulation (IEEE C57.91-1995 Table 4)
_A_ARRHENIUS: float = -11.269
_B_ARRHENIUS: float = 6328.8

# Reference absolute temperature for aging (383 K = 110°C)
_T_REF_K: float = 383.0

# Default Weibull parameters
_DEFAULT_WEIBULL_SHAPE: float = 5.0
_DEFAULT_WEIBULL_SCALE_YEARS: float = 50.0

# Kelvin offset
_KELVIN: float = 273.15

_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# TransformerThermalModel
# ---------------------------------------------------------------------------


class TransformerThermalModel:
    """IEEE C57.91-1995 transformer thermal aging model.

    Computes winding hottest-spot temperature from loading ratio and
    ambient temperature, then calculates insulation aging via the
    Arrhenius equation and updates a Weibull failure-rate model.

    Parameters
    ----------
    rated_power_mva : float
        Transformer rated power in MVA.
    delta_theta_to_rated : float
        Rated top-oil temperature rise over ambient at rated load (K).
        Default 55.0 (65°C average winding rise class).
    delta_theta_w_rated : float
        Rated winding hottest-spot rise over top-oil at rated load (K).
        Default 15.0.
    r_loss_ratio : float
        Ratio of rated load loss to no-load loss (R = P_LL / P_NL).
        Default 6.0.
    n_oil : float
        Oil exponent for top-oil rise.  Default 0.8 (ONAN cooling).
    m_winding : float
        Winding exponent for conductor rise.  Default 0.8.
    tau_oil_h : float
        Oil time constant in hours.  Default 3.0.
    tau_winding_h : float
        Winding time constant in hours.  Default 0.15.
    weibull_shape : float
        Weibull shape parameter β.  Default 5.0.
    weibull_scale_years : float
        Weibull scale parameter η at zero aging (years).  Default 50.0.
    installed_year : float
        Year the transformer was installed (for age calculation).
        Default 0.0.

    Attributes
    ----------
    rated_power_mva : float
    delta_theta_to_rated : float
    delta_theta_w_rated : float
    r_loss_ratio : float
    n_oil : float
    m_winding : float
    tau_oil_h : float
    tau_winding_h : float
    weibull_shape : float
    weibull_scale_years : float
    installed_year : float
    cumulative_aging_pu : float
        Accumulated aging factor (per-unit of normal life).
    """

    def __init__(
        self,
        rated_power_mva: float = 100.0,
        delta_theta_to_rated: float = 55.0,
        delta_theta_w_rated: float = 15.0,
        r_loss_ratio: float = 6.0,
        n_oil: float = 0.8,
        m_winding: float = 0.8,
        tau_oil_h: float = 3.0,
        tau_winding_h: float = 0.15,
        weibull_shape: float = _DEFAULT_WEIBULL_SHAPE,
        weibull_scale_years: float = _DEFAULT_WEIBULL_SCALE_YEARS,
        installed_year: float = 0.0,
    ) -> None:
        if rated_power_mva <= 0:
            raise ValueError(f"rated_power_mva must be > 0, got {rated_power_mva}")
        if delta_theta_to_rated < 0:
            raise ValueError(f"delta_theta_to_rated must be >= 0, got {delta_theta_to_rated}")
        if delta_theta_w_rated < 0:
            raise ValueError(f"delta_theta_w_rated must be >= 0, got {delta_theta_w_rated}")
        if r_loss_ratio <= 0:
            raise ValueError(f"r_loss_ratio must be > 0, got {r_loss_ratio}")
        if n_oil <= 0:
            raise ValueError(f"n_oil must be > 0, got {n_oil}")
        if m_winding <= 0:
            raise ValueError(f"m_winding must be > 0, got {m_winding}")

        self.rated_power_mva = float(rated_power_mva)
        self.delta_theta_to_rated = float(delta_theta_to_rated)
        self.delta_theta_w_rated = float(delta_theta_w_rated)
        self.r_loss_ratio = float(r_loss_ratio)
        self.n_oil = float(n_oil)
        self.m_winding = float(m_winding)
        self.tau_oil_h = float(tau_oil_h)
        self.tau_winding_h = float(tau_winding_h)
        self.weibull_shape = float(weibull_shape)
        self.weibull_scale_years = float(weibull_scale_years)
        self.installed_year = float(installed_year)

        self.cumulative_aging_pu: float = 0.0

    # ------------------------------------------------------------------
    # Top-oil temperature rise
    # ------------------------------------------------------------------

    def calculate_top_oil_rise(self, x: np.ndarray) -> np.ndarray:
        r"""Top-oil temperature rise over ambient.

        .. math::

            \Delta\theta_{TO} = \Delta\theta_{TO,\text{rated}}
            \cdot \left[\frac{1 + R x^2}{1 + R}\right]^n

        Parameters
        ----------
        x : np.ndarray
            Loading ratio (per-unit of rated power).

        Returns
        -------
        np.ndarray
            Top-oil rise in K.
        """
        x = np.asarray(x, dtype=np.float64)
        ratio = (1.0 + self.r_loss_ratio * x ** 2) / (1.0 + self.r_loss_ratio)
        return self.delta_theta_to_rated * ratio ** self.n_oil

    # ------------------------------------------------------------------
    # Winding hottest-spot rise
    # ------------------------------------------------------------------

    def calculate_winding_rise(self, x: np.ndarray) -> np.ndarray:
        r"""Winding hottest-spot temperature rise over top-oil.

        .. math::

            \Delta\theta_W = \Delta\theta_{W,\text{rated}} \cdot x^{2m}

        Parameters
        ----------
        x : np.ndarray
            Loading ratio (per-unit).

        Returns
        -------
        np.ndarray
            Winding rise in K.
        """
        x = np.asarray(x, dtype=np.float64)
        return self.delta_theta_w_rated * x ** (2.0 * self.m_winding)

    # ------------------------------------------------------------------
    # Hottest-spot temperature
    # ------------------------------------------------------------------

    def calculate_hottest_spot(
        self,
        x: np.ndarray,
        theta_a: np.ndarray,
    ) -> np.ndarray:
        r"""Winding hottest-spot temperature.

        .. math::

            \theta_H = \theta_a + \Delta\theta_{TO} + \Delta\theta_W

        Parameters
        ----------
        x : np.ndarray
            Loading ratio (per-unit).
        theta_a : np.ndarray
            Ambient temperature in °C.

        Returns
        -------
        np.ndarray
            Hottest-spot temperature in °C.
        """
        x = np.asarray(x, dtype=np.float64)
        theta_a = np.asarray(theta_a, dtype=np.float64)
        return theta_a + self.calculate_top_oil_rise(x) + self.calculate_winding_rise(x)

    # ------------------------------------------------------------------
    # Aging factor
    # ------------------------------------------------------------------

    def hourly_aging_factor(self, theta_h: np.ndarray) -> np.ndarray:
        r"""Insulation aging acceleration factor per IEEE C57.91 §7.

        .. math::

            F_{AA} = \exp\left(\frac{B}{T_{\text{ref}}} -
            \frac{B}{\theta_H + 273}\right)

        where :math:`B = 15000` for 65°C rise insulation (simplified)
        or the full Arrhenius form with :math:`A, B` constants.

        Parameters
        ----------
        theta_h : np.ndarray
            Hottest-spot temperature in °C.

        Returns
        -------
        np.ndarray
            Aging acceleration factor (1.0 = normal aging).
        """
        theta_h = np.asarray(theta_h, dtype=np.float64)
        t_abs = theta_h + _KELVIN
        # Full Arrhenius: log(E) = A + B/T_abs
        # FAA = E / E_ref = exp(A + B/T_abs) / exp(A + B/T_ref)
        #     = exp(B*(1/T_ref - 1/T_abs))
        return np.exp(_B_ARRHENIUS * (1.0 / _T_REF_K - 1.0 / np.maximum(t_abs, _EPS)))

    # ------------------------------------------------------------------
    # Cumulative aging
    # ------------------------------------------------------------------

    def calculate_annual_loss_of_life(
        self,
        hourly_load_profile: pd.Series,
        hourly_temp_profile: pd.Series,
        current_year: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute cumulative transformer aging from hourly profiles.

        Integrates the hourly aging acceleration factor over the year
        and updates the Weibull scale parameter to reflect accumulated
        degradation.

        Parameters
        ----------
        hourly_load_profile : pd.Series
            Hourly transformer loading in MVA (8760 values for one year).
        hourly_temp_profile : pd.Series
            Hourly ambient temperature in °C (same length).
        current_year : float or None
            Current calendar year for age calculation.  If ``None``,
            uses ``self.installed_year``.

        Returns
        -------
        dict
            Keys:

            - ``cumulative_aging_pu`` (float) — total aging in per-unit
              of normal life.
            - ``equivalent_aging_hours`` (float) — equivalent hours at
              rated temperature.
            - ``weibull_scale_years`` (float) — updated Weibull scale
              parameter.
            - ``failure_rate_per_year`` (float) — instantaneous failure
              rate at current age.
            - ``mean_remaining_life_years`` (float) — expected remaining
              life.
        """
        n_hours = len(hourly_load_profile)
        if n_hours != len(hourly_temp_profile):
            raise ValueError(
                f"Load and temperature profiles must have same length, "
                f"got {n_hours} vs {len(hourly_temp_profile)}"
            )

        x = hourly_load_profile.values / self.rated_power_mva
        theta_a = hourly_temp_profile.values
        theta_h = self.calculate_hottest_spot(x, theta_a)
        faa = self.hourly_aging_factor(theta_h)

        # Accumulate aging: sum of FAA over hours → per-unit life consumed
        annual_aging = float(np.sum(faa))
        self.cumulative_aging_pu += annual_aging / n_hours  # normalize to per-unit years

        # Equivalent hours at reference temperature
        equiv_hours = annual_aging

        # Update Weibull scale: η = η_0 / (1 + cumulative_aging_pu)
        eta = self.weibull_scale_years / max(1.0 + self.cumulative_aging_pu, _EPS)

        # Current age in years
        cy = current_year if current_year is not None else self.installed_year
        age_years = max(0.0, cy - self.installed_year) + self.cumulative_aging_pu

        # Weibull failure rate: λ(t) = (β/η) * (t/η)^(β-1)
        beta = self.weibull_shape
        if age_years > _EPS and eta > _EPS:
            failure_rate = (beta / eta) * (age_years / eta) ** (beta - 1.0)
        else:
            failure_rate = 0.0

        # Mean remaining life (expected value of Weibull given survived to t)
        import math
        mean_remaining = eta * math.gamma(1.0 + 1.0 / beta) if eta > _EPS else 0.0

        return {
            "cumulative_aging_pu": self.cumulative_aging_pu,
            "equivalent_aging_hours": equiv_hours,
            "weibull_scale_years": eta,
            "failure_rate_per_year": failure_rate,
            "mean_remaining_life_years": mean_remaining,
        }

    # ------------------------------------------------------------------
    # Weibull failure rate (standalone)
    # ------------------------------------------------------------------

    def weibull_failure_rate(
        self,
        t_years: float,
        cumulative_aging: Optional[float] = None,
    ) -> float:
        """Compute instantaneous Weibull failure rate at age *t_years*.

        .. math::

            \lambda(t) = \frac{\beta}{\eta} \left(\frac{t}{\eta}\right)^{\beta-1}

        where :math:`\eta = \eta_0 / (1 + \text{cumulative\_aging})`.

        Parameters
        ----------
        t_years : float
            Effective age in years (including cumulative aging).
        cumulative_aging : float or None
            Cumulative aging in per-unit.  If ``None``, uses
            ``self.cumulative_aging_pu``.

        Returns
        -------
        float
            Failure rate in failures per year.
        """
        aging = cumulative_aging if cumulative_aging is not None else self.cumulative_aging_pu
        eta = self.weibull_scale_years / max(1.0 + aging, _EPS)
        beta = self.weibull_shape
        if t_years <= _EPS or eta <= _EPS:
            return 0.0
        return (beta / eta) * (t_years / eta) ** (beta - 1.0)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TransformerThermalModel(power={self.rated_power_mva:.0f}MVA, "
            f"aging={self.cumulative_aging_pu:.3f}pu, "
            f"η={self.weibull_scale_years:.0f}y, β={self.weibull_shape:.1f})"
        )
