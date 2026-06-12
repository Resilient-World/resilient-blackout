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
Quantile Delta Mapping (QDM) bias correction and downscaling.

Implements the Cannon et al. (2015) univariate QDM methodology using
empirical CDFs with optional parametric tail extrapolation.  Avoids
restrictive GPL dependencies — built purely on NumPy, SciPy, and Pandas.

Supports multiplicative correction for bounded variables (precipitation,
humidity) and additive correction for unbounded variables (temperature),
with explicit dry-day frequency preservation for precipitation fields.
"""

from __future__ import annotations

import logging
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

VariableType = Literal["temperature", "precipitation", "humidity", "generic"]
_EPS = 1e-12


class QuantileDeltaMapper:
    """Univariate Quantile Delta Mapping (QDM) bias corrector.

    Implements the Cannon et al. (2015) methodology:

    1. Compute the CDF probability of each projected value in the model
       projected distribution:
       :math:`\\tau(t) = F_{m,p}[x_{m,p}(t)]`

    2. Compute the non-stationary delta factor between model historical
       and projected quantiles:
       :math:`\\Delta(t) = F_{m,p}^{-1}(\\tau(t)) \\;/\\; F_{m,h}^{-1}(\\tau(t))`
       (multiplicative) or the additive analogue.

    3. Apply the delta to the observed historical inverse CDF:
       :math:`x_{adj}(t) = F_{o,h}^{-1}(\\tau(t)) \\times \\Delta(t)`

    Parameters
    ----------
    O_h : np.ndarray
        1-D array of historical **observed** values.
    M_h : np.ndarray
        1-D array of historical **model-simulated** values (same period).
    M_p : np.ndarray
        1-D array of projected future model-simulated values.
    variable_type : str
        One of ``"temperature"``, ``"precipitation"``, ``"humidity"``,
        or ``"generic"``.  Controls:

        - **Delta mode** — multiplicative for bounded variables
          (precipitation, humidity), additive for unbounded
          (temperature, generic).
        - **Zero handling** — dry-day frequency correction for
          precipitation.
    n_quantiles : int
        Number of quantile bins for the empirical CDF.  Larger values
        give finer resolution at the cost of memory.  Default 100.
    tail_parametric : bool
        If ``True``, fit parametric distributions (GEV for upper tail,
        Gamma for lower tail of precipitation) to extrapolate beyond the
        observed calibration range.  Default ``True``.

    Attributes
    ----------
    O_h_sorted : np.ndarray
    M_h_sorted : np.ndarray
    M_p_sorted : np.ndarray
    probs_oh : np.ndarray
    probs_mh : np.ndarray
    probs_mp : np.ndarray
    multiplicative : bool
    dry_day_correction : bool
    """

    def __init__(
        self,
        O_h: np.ndarray,
        M_h: np.ndarray,
        M_p: np.ndarray,
        variable_type: VariableType = "generic",
        n_quantiles: int = 100,
        tail_parametric: bool = True,
    ) -> None:
        self.O_h = np.asarray(O_h, dtype=np.float64).ravel()
        self.M_h = np.asarray(M_h, dtype=np.float64).ravel()
        self.M_p = np.asarray(M_p, dtype=np.float64).ravel()
        self.variable_type: VariableType = variable_type
        self.n_quantiles = n_quantiles
        self.tail_parametric = tail_parametric

        self.multiplicative = variable_type in ("precipitation", "humidity")
        self.dry_day_correction = variable_type == "precipitation"

        self.O_h_sorted, self.probs_oh = self._build_empirical_cdf(self.O_h)
        self.M_h_sorted, self.probs_mh = self._build_empirical_cdf(self.M_h)
        self.M_p_sorted, self.probs_mp = self._build_empirical_cdf(self.M_p)

        self._oh_dry_freq: float = 0.0
        self._mh_dry_freq: float = 0.0
        if self.dry_day_correction:
            self._oh_dry_freq = float(np.mean(self.O_h < _EPS))
            self._mh_dry_freq = float(np.mean(self.M_h < _EPS))

        self._tail_fits: Dict[str, Any] = {}
        if tail_parametric:
            self._fit_parametric_tails()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map(self) -> np.ndarray:
        """Produce bias-adjusted projected values.

        Returns
        -------
        np.ndarray
            ``x_adj`` with the same shape as ``M_p``.
        """
        x_adj = np.empty_like(self.M_p, dtype=np.float64)

        if self.dry_day_correction:
            x_adj = self._map_precipitation()
        else:
            tau = self._compute_cdf_prob(self.M_p, self.M_p_sorted, self.probs_mp)
            delta = self._compute_delta(tau)
            x_adj = self._apply_adjustment(tau, delta)

        return x_adj

    # ------------------------------------------------------------------
    # CDF construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_empirical_cdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build an empirical CDF from sorted data.

        Uses Weibull plotting positions :math:`(i + 0.5) / n` for
        unbiased quantile estimation.

        Parameters
        ----------
        data : np.ndarray
            1-D array.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            ``(sorted_values, cumulative_probabilities)``.
        """
        n = len(data)
        if n == 0:
            return np.array([0.0]), np.array([0.5])
        sorted_vals = np.sort(data)
        probs = (np.arange(n, dtype=np.float64) + 0.5) / n
        return sorted_vals, probs

    # ------------------------------------------------------------------
    # CDF / inverse CDF with optional parametric tails
    # ------------------------------------------------------------------

    def _inverse_cdf(
        self,
        prob: np.ndarray,
        sorted_vals: np.ndarray,
        probs: np.ndarray,
        tail_key: str = "",
    ) -> np.ndarray:
        """Evaluate the inverse CDF at given probabilities.

        Uses linear interpolation on the empirical CDF.  If parametric
        tails are fitted, extrapolates beyond the empirical range using
        the fitted distribution.

        Parameters
        ----------
        prob : np.ndarray
            Probabilities in [0, 1].
        sorted_vals : np.ndarray
            Sorted empirical values.
        probs : np.ndarray
            Corresponding cumulative probabilities.
        tail_key : str
            Key into ``_tail_fits`` for parametric extrapolation.

        Returns
        -------
        np.ndarray
            Quantile values.
        """
        prob = np.asarray(prob, dtype=np.float64)
        result = np.interp(prob, probs, sorted_vals)

        if self.tail_parametric and tail_key in self._tail_fits:
            fit = self._tail_fits[tail_key]
            lower_prob = probs[0]
            upper_prob = probs[-1]

            lower_mask = prob < lower_prob
            upper_mask = prob > upper_prob

            if np.any(lower_mask) and "lower" in fit:
                result[lower_mask] = fit["lower"].ppf(np.clip(prob[lower_mask], _EPS, lower_prob))
            if np.any(upper_mask) and "upper" in fit:
                result[upper_mask] = fit["upper"].ppf(np.clip(prob[upper_mask], upper_prob, 1 - _EPS))

        return result

    def _compute_cdf_prob(
        self,
        values: np.ndarray,
        sorted_vals: np.ndarray,
        probs: np.ndarray,
    ) -> np.ndarray:
        """Map values to CDF probabilities.

        Parameters
        ----------
        values : np.ndarray
            Input values.
        sorted_vals : np.ndarray
            Sorted empirical values.
        probs : np.ndarray
            Cumulative probabilities.

        Returns
        -------
        np.ndarray
            Probabilities in [0, 1].
        """
        values = np.asarray(values, dtype=np.float64)
        return np.interp(values, sorted_vals, probs)

    # ------------------------------------------------------------------
    # Delta computation
    # ------------------------------------------------------------------

    def _compute_delta(self, tau: np.ndarray) -> np.ndarray:
        """Compute the non-stationary delta factor.

        For multiplicative mode:
        :math:`\\Delta = F_{m,p}^{-1}(\\tau) / F_{m,h}^{-1}(\\tau)`

        For additive mode:
        :math:`\\Delta = F_{m,p}^{-1}(\\tau) - F_{m,h}^{-1}(\\tau)`

        Parameters
        ----------
        tau : np.ndarray
            CDF probabilities of projected values.

        Returns
        -------
        np.ndarray
            Delta factors.
        """
        inv_mp = self._inverse_cdf(tau, self.M_p_sorted, self.probs_mp, tail_key="mp")
        inv_mh = self._inverse_cdf(tau, self.M_h_sorted, self.probs_mh, tail_key="mh")

        if self.multiplicative:
            safe_mh = np.maximum(np.abs(inv_mh), _EPS) * np.sign(inv_mh + _EPS)
            return inv_mp / np.where(np.abs(safe_mh) < _EPS, _EPS, safe_mh)
        else:
            return inv_mp - inv_mh

    # ------------------------------------------------------------------
    # Adjustment application
    # ------------------------------------------------------------------

    def _apply_adjustment(self, tau: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Apply the delta factor to the observed historical inverse CDF.

        Parameters
        ----------
        tau : np.ndarray
            CDF probabilities.
        delta : np.ndarray
            Delta factors.

        Returns
        -------
        np.ndarray
            Bias-adjusted values.
        """
        inv_oh = self._inverse_cdf(tau, self.O_h_sorted, self.probs_oh, tail_key="oh")

        if self.multiplicative:
            return inv_oh * delta
        else:
            return inv_oh + delta

    # ------------------------------------------------------------------
    # Precipitation-specific mapping
    # ------------------------------------------------------------------

    def _map_precipitation(self) -> np.ndarray:
        """Bias-correct precipitation with dry-day frequency adjustment.

        Preserves the observed dry-day fraction by adjusting the
        threshold at which projected values are set to zero.

        Returns
        -------
        np.ndarray
            Bias-adjusted precipitation values.
        """
        x_adj = np.empty_like(self.M_p, dtype=np.float64)

        oh_dry = self._oh_dry_freq
        mh_dry = self._mh_dry_freq

        if oh_dry >= 1.0:
            x_adj.fill(0.0)
            return x_adj

        if mh_dry >= 1.0:
            mh_dry = 0.999

        if oh_dry > 0:
            adjusted_threshold_prob = oh_dry
        else:
            adjusted_threshold_prob = 0.0

        tau = self._compute_cdf_prob(self.M_p, self.M_p_sorted, self.probs_mp)

        dry_mask = tau <= adjusted_threshold_prob
        wet_mask = ~dry_mask

        x_adj[dry_mask] = 0.0

        if np.any(wet_mask):
            tau_wet = tau[wet_mask]
            delta_wet = self._compute_delta(tau_wet)
            x_adj[wet_mask] = self._apply_adjustment(tau_wet, delta_wet)
            x_adj[wet_mask] = np.maximum(x_adj[wet_mask], 0.0)

        return x_adj

    # ------------------------------------------------------------------
    # Parametric tail fitting
    # ------------------------------------------------------------------

    def _fit_parametric_tails(self) -> None:
        """Fit parametric distributions for extrapolation beyond the
        empirical calibration range.

        - **Upper tail**: Generalized Extreme Value (GEV) on the top 10 %
          of values.
        - **Lower tail** (precipitation only): Gamma distribution on
          positive values below the 10th percentile.
        """
        for label, data in [("oh", self.O_h), ("mh", self.M_h), ("mp", self.M_p)]:
            fits: Dict[str, Any] = {}
            n = len(data)
            if n < 20:
                continue

            upper_thresh = np.percentile(data, 90)
            upper_data = data[data >= upper_thresh]
            if len(upper_data) >= 10:
                try:
                    shape, loc, scale = stats.genextreme.fit(upper_data)
                    fits["upper"] = stats.genextreme(shape, loc=loc, scale=scale)
                except Exception:
                    logger.debug("GEV fit failed for %s upper tail.", label)

            if self.variable_type == "precipitation":
                positive = data[data > _EPS]
                if len(positive) >= 10:
                    lower_thresh = np.percentile(positive, 10)
                    lower_data = positive[positive <= lower_thresh]
                    if len(lower_data) >= 5:
                        try:
                            shape, loc, scale = stats.gamma.fit(lower_data, floc=0)
                            fits["lower"] = stats.gamma(shape, loc=loc, scale=scale)
                        except Exception:
                            logger.debug("Gamma fit failed for %s lower tail.", label)

            if fits:
                self._tail_fits[label] = fits
