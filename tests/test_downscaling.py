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

"""Unit tests for ``resilient_blackout.climate.downscaling``."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from resilient_blackout.climate.downscaling import QuantileDeltaMapper


class TestQuantileDeltaMapper:
    """Test suite for the QuantileDeltaMapper class."""

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @staticmethod
    def _make_synthetic_data(
        n: int = 1000,
        base_mean: float = 20.0,
        shift: float = 3.0,
        seed: int = 42,
    ):
        """Generate synthetic temperature-like data."""
        rng = np.random.default_rng(seed)
        O_h = rng.normal(loc=base_mean, scale=5.0, size=n)
        M_h = rng.normal(loc=base_mean + 1.0, scale=5.5, size=n)
        M_p = rng.normal(loc=base_mean + 1.0 + shift, scale=5.5, size=n)
        return O_h, M_h, M_p

    @staticmethod
    def _make_precip_data(
        n: int = 1000,
        dry_fraction: float = 0.3,
        scale: float = 5.0,
        shift_factor: float = 1.3,
        seed: int = 42,
    ):
        """Generate synthetic precipitation-like data with dry days."""
        rng = np.random.default_rng(seed)
        O_h = rng.exponential(scale=scale, size=n)
        O_h[rng.random(n) < dry_fraction] = 0.0

        M_h = rng.exponential(scale=scale * 1.1, size=n)
        M_h[rng.random(n) < dry_fraction * 0.8] = 0.0

        M_p = rng.exponential(scale=scale * 1.1 * shift_factor, size=n)
        M_p[rng.random(n) < dry_fraction * 0.7] = 0.0

        return O_h, M_h, M_p

    # ------------------------------------------------------------------
    # Additive mode (temperature)
    # ------------------------------------------------------------------

    def test_additive_mode_preserves_delta_trend(self):
        """Adjusted values should reflect the projected warming signal."""
        O_h, M_h, M_p = self._make_synthetic_data(n=2000, shift=3.0)
        mapper = QuantileDeltaMapper(O_h, M_h, M_p, variable_type="temperature")
        x_adj = mapper.map()

        assert len(x_adj) == len(M_p)
        assert x_adj.dtype == np.float64

        mean_shift = np.mean(x_adj) - np.mean(O_h)
        assert mean_shift > 0, "Adjusted values should be warmer than observed historical"

    def test_additive_mode_no_nan(self):
        """Output must contain no NaN or infinite values."""
        O_h, M_h, M_p = self._make_synthetic_data(n=500)
        mapper = QuantileDeltaMapper(O_h, M_h, M_p, variable_type="temperature")
        x_adj = mapper.map()

        assert not np.any(np.isnan(x_adj))
        assert not np.any(np.isinf(x_adj))

    def test_additive_mode_preserves_variance_structure(self):
        """Adjusted variance should be comparable to observed variance."""
        O_h, M_h, M_p = self._make_synthetic_data(n=2000, shift=2.0)
        mapper = QuantileDeltaMapper(O_h, M_h, M_p, variable_type="temperature")
        x_adj = mapper.map()

        ratio = np.std(x_adj) / np.std(O_h)
        assert 0.5 < ratio < 2.0, f"Variance ratio {ratio:.2f} outside expected range"

    # ------------------------------------------------------------------
    # Multiplicative mode (precipitation)
    # ------------------------------------------------------------------

    def test_multiplicative_mode_non_negative(self):
        """Precipitation output must be non-negative."""
        O_h, M_h, M_p = self._make_precip_data(n=1000)
        mapper = QuantileDeltaMapper(O_h, M_h, M_p, variable_type="precipitation")
        x_adj = mapper.map()

        assert np.all(x_adj >= 0.0)

    def test_multiplicative_mode_dry_day_preservation(self):
        """Dry-day frequency in adjusted output should approximate observed."""
        O_h, M_h, M_p = self._make_precip_data(n=2000, dry_fraction=0.3)
        mapper = QuantileDeltaMapper(O_h, M_h, M_p, variable_type="precipitation")
        x_adj = mapper.map()

        obs_dry = np.mean(O_h < 1e-12)
        adj_dry = np.mean(x_adj < 1e-12)

        assert abs(adj_dry - obs_dry) < 0.15, (
            f"Dry-day fraction mismatch: obs={obs_dry:.3f}, adj={adj_dry:.3f}"
        )

    def test_multiplicative_mode_intensity_increase(self):
        """Wet-day mean should increase when shift_factor > 1."""
        O_h, M_h, M_p = self._make_precip_data(n=2000, shift_factor=1.5)
        mapper = QuantileDeltaMapper(O_h, M_h, M_p, variable_type="precipitation")
        x_adj = mapper.map()

        wet_oh = O_h[O_h > 1e-12]
        wet_adj = x_adj[x_adj > 1e-12]

        if len(wet_oh) > 0 and len(wet_adj) > 0:
            assert np.mean(wet_adj) > np.mean(wet_oh) * 0.8

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_all_zero_precipitation(self):
        """All-zero observed precipitation should produce all-zero output."""
        O_h = np.zeros(500)
        M_h = np.random.default_rng(0).exponential(scale=3.0, size=500)
        M_p = np.random.default_rng(1).exponential(scale=4.0, size=500)

        mapper = QuantileDeltaMapper(O_h, M_h, M_p, variable_type="precipitation")
        x_adj = mapper.map()

        assert_array_almost_equal(x_adj, np.zeros(500))

    def test_single_value_input(self):
        """Single-value inputs should not crash."""
        O_h = np.array([15.0])
        M_h = np.array([16.0])
        M_p = np.array([18.0])

        mapper = QuantileDeltaMapper(O_h, M_h, M_p, variable_type="temperature")
        x_adj = mapper.map()

        assert len(x_adj) == 1
        assert not np.isnan(x_adj[0])

    def test_identical_historical_and_projected(self):
        """When M_h == M_p, adjusted values should approximate O_h."""
        rng = np.random.default_rng(99)
        O_h = rng.normal(loc=20, scale=4, size=1000)
        M = rng.normal(loc=21, scale=4.5, size=1000)

        mapper = QuantileDeltaMapper(O_h, M, M, variable_type="temperature")
        x_adj = mapper.map()

        assert np.abs(np.mean(x_adj) - np.mean(O_h)) < 1.0

    def test_extreme_projected_values(self):
        """Extreme projected values should not produce NaN."""
        O_h, M_h, M_p = self._make_synthetic_data(n=1000, shift=10.0)
        M_p[0] = 100.0

        mapper = QuantileDeltaMapper(O_h, M_h, M_p, variable_type="temperature")
        x_adj = mapper.map()

        assert not np.any(np.isnan(x_adj))

    # ------------------------------------------------------------------
    # Parametric tails
    # ------------------------------------------------------------------

    def test_parametric_tails_enabled(self):
        """With tail_parametric=True, extreme values should be handled."""
        O_h, M_h, M_p = self._make_synthetic_data(n=500, shift=5.0)
        M_p[0] = np.percentile(M_p, 99.9) * 1.5

        mapper = QuantileDeltaMapper(
            O_h, M_h, M_p, variable_type="temperature", tail_parametric=True
        )
        x_adj = mapper.map()

        assert not np.any(np.isnan(x_adj))
        assert not np.any(np.isinf(x_adj))

    def test_parametric_tails_disabled(self):
        """With tail_parametric=False, should still work."""
        O_h, M_h, M_p = self._make_synthetic_data(n=500, shift=3.0)
        mapper = QuantileDeltaMapper(
            O_h, M_h, M_p, variable_type="temperature", tail_parametric=False
        )
        x_adj = mapper.map()

        assert len(x_adj) == len(M_p)
        assert not np.any(np.isnan(x_adj))

    # ------------------------------------------------------------------
    # Humidity (multiplicative, bounded)
    # ------------------------------------------------------------------

    def test_humidity_bounded(self):
        """Humidity output should respect [0, 100] bounds approximately."""
        rng = np.random.default_rng(7)
        O_h = rng.uniform(30, 90, size=1000)
        M_h = rng.uniform(32, 88, size=1000)
        M_p = rng.uniform(34, 95, size=1000)

        mapper = QuantileDeltaMapper(O_h, M_h, M_p, variable_type="humidity")
        x_adj = mapper.map()

        assert np.all(x_adj >= 0.0)
        assert np.percentile(x_adj, 99) < 120.0
