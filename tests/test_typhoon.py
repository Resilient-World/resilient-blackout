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

"""Unit tests for ``resilient_blackout.climate.typhoon``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resilient_blackout.climate.typhoon import TyphoonWindSimulator


# ---------------------------------------------------------------------------
# Synthetic track helpers
# ---------------------------------------------------------------------------


def _make_track(
    n_points: int = 24,
    start_lon: float = 125.0,
    start_lat: float = 15.0,
    delta_lon: float = 0.3,
    delta_lat: float = 0.25,
    pc: float = 950.0,
    vt: float = 15.0,
) -> pd.DataFrame:
    """Build a synthetic straight-line storm track."""
    hours = np.arange(n_points)
    return pd.DataFrame({
        "time": pd.date_range("2025-09-01 00:00", periods=n_points, freq="h"),
        "lon": start_lon + hours * delta_lon,
        "lat": start_lat + hours * delta_lat,
        "pc": np.full(n_points, pc, dtype=np.float64),
        "vt": np.full(n_points, vt, dtype=np.float64),
    })


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestInit:
    """Validation of constructor and parameter handling."""

    def test_default_construction(self) -> None:
        sim = TyphoonWindSimulator()
        assert sim.rho_air == 1.15
        assert sim.p_ambient_hpa == 1013.25
        assert sim.k_s == 0.9
        assert sim.alpha_shear == 0.143
        assert sim.h_g == 10.0
        assert sim.buffer_radius_km == 200.0

    def test_custom_parameters(self) -> None:
        sim = TyphoonWindSimulator(
            rho_air=1.2,
            p_ambient_hpa=1010.0,
            k_s=0.85,
            alpha_shear=0.2,
            h_g=15.0,
            buffer_radius_km=150.0,
        )
        assert sim.rho_air == 1.2
        assert sim.buffer_radius_km == 150.0

    def test_invalid_rho_raises(self) -> None:
        with pytest.raises(ValueError, match="rho_air"):
            TyphoonWindSimulator(rho_air=-1.0)

    def test_invalid_ks_raises(self) -> None:
        with pytest.raises(ValueError, match="k_s"):
            TyphoonWindSimulator(k_s=0.0)

    def test_repr(self) -> None:
        sim = TyphoonWindSimulator()
        r = repr(sim)
        assert "TyphoonWindSimulator" in r
        assert "1.15" in r


# ---------------------------------------------------------------------------
# Empirical formulas
# ---------------------------------------------------------------------------


class TestHollandB:
    """Validation of Holland B parameter."""

    def test_typical_typhoon(self) -> None:
        B = TyphoonWindSimulator.holland_b(950.0)
        # B = 1.5 + (980 - 950) / 120 = 1.5 + 0.25 = 1.75
        assert np.isclose(B, 1.75, rtol=1e-6)

    def test_weak_storm(self) -> None:
        B = TyphoonWindSimulator.holland_b(1000.0)
        # B = 1.5 + (980 - 1000) / 120 = 1.5 - 0.1667 = 1.333
        assert np.isclose(B, 1.333333, rtol=1e-4)

    def test_vectorized(self) -> None:
        B = TyphoonWindSimulator.holland_b(np.array([950.0, 980.0, 920.0]))
        assert B.shape == (3,)
        assert B[0] == pytest.approx(1.75, rel=1e-6)
        assert B[1] == pytest.approx(1.5, rel=1e-6)
        assert B[2] == pytest.approx(2.0, rel=1e-6)


class TestRadiusOfMaximumWind:
    """Validation of Rmax empirical formula."""

    def test_typical_typhoon(self) -> None:
        R = TyphoonWindSimulator.radius_of_maximum_wind(950.0, 25.0, vt=15.0)
        # Should be in typical range 20–80 km
        assert 20.0 < float(R) < 80.0

    def test_low_latitude(self) -> None:
        R_low = TyphoonWindSimulator.radius_of_maximum_wind(950.0, 10.0, vt=15.0)
        R_mid = TyphoonWindSimulator.radius_of_maximum_wind(950.0, 30.0, vt=15.0)
        # Rmax increases with latitude up to a point
        assert float(R_low) < float(R_mid)

    def test_vectorized(self) -> None:
        R = TyphoonWindSimulator.radius_of_maximum_wind(
            np.array([950.0, 960.0]), np.array([25.0, 30.0]), np.array([10.0, 20.0])
        )
        assert R.shape == (2,)


class TestCoriolis:
    """Validation of Coriolis parameter."""

    def test_equator(self) -> None:
        f = TyphoonWindSimulator.coriolis(0.0)
        assert np.isclose(f, 0.0, atol=1e-10)

    def test_mid_latitude(self) -> None:
        f = TyphoonWindSimulator.coriolis(30.0)
        expected = 2.0 * 7.2921e-5 * np.sin(np.radians(30.0))
        assert np.isclose(f, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Gradient wind speed
# ---------------------------------------------------------------------------


class TestGradientWindSpeed:
    """Validation of Batts gradient wind equation."""

    def test_at_rmax(self) -> None:
        sim = TyphoonWindSimulator()
        pc = 950.0
        lat = 25.0
        R = float(TyphoonWindSimulator.radius_of_maximum_wind(pc, lat))
        v = sim.calculate_gradient_wind_speed(R, pc, lat=lat)
        # Wind speed at Rmax should be substantial
        assert 30.0 < float(v) < 80.0

    def test_decays_with_distance(self) -> None:
        sim = TyphoonWindSimulator()
        r = np.array([30.0, 60.0, 120.0, 240.0])
        v = sim.calculate_gradient_wind_speed(r, 950.0, lat=25.0)
        # Wind speed should decrease beyond Rmax
        assert v[0] > v[-1]

    def test_stronger_storm_higher_wind(self) -> None:
        sim = TyphoonWindSimulator()
        r = np.array([50.0])
        v_weak = sim.calculate_gradient_wind_speed(r, 980.0, lat=25.0)
        v_strong = sim.calculate_gradient_wind_speed(r, 920.0, lat=25.0)
        assert float(v_strong[0]) > float(v_weak[0])

    def test_vectorized(self) -> None:
        sim = TyphoonWindSimulator()
        r = np.linspace(10.0, 300.0, 100)
        v = sim.calculate_gradient_wind_speed(r, 950.0, lat=25.0)
        assert v.shape == (100,)
        assert np.all(v >= 0.0)

    def test_zero_distance_handled(self) -> None:
        sim = TyphoonWindSimulator()
        v = sim.calculate_gradient_wind_speed(np.array([0.0]), 950.0, lat=25.0)
        assert np.all(np.isfinite(v))


# ---------------------------------------------------------------------------
# Height conversion
# ---------------------------------------------------------------------------


class TestHeightConvert:
    """Validation of power-law height scaling."""

    def test_gradient_height_no_change(self) -> None:
        sim = TyphoonWindSimulator(k_s=1.0, h_g=10.0, alpha_shear=0.143)
        v = sim.height_convert(np.array([50.0]), h=10.0)
        assert np.isclose(v[0], 50.0, rtol=1e-6)

    def test_higher_altitude_reduces(self) -> None:
        sim = TyphoonWindSimulator(k_s=0.9, h_g=10.0, alpha_shear=0.143)
        v10 = sim.height_convert(np.array([50.0]), h=10.0)
        v50 = sim.height_convert(np.array([50.0]), h=50.0)
        # Higher altitude → higher wind speed
        assert float(v50[0]) > float(v10[0])

    def test_roughness_reduces(self) -> None:
        sim_smooth = TyphoonWindSimulator(k_s=1.0)
        sim_rough = TyphoonWindSimulator(k_s=0.7)
        v_smooth = sim_smooth.height_convert(np.array([50.0]), h=30.0)
        v_rough = sim_rough.height_convert(np.array([50.0]), h=30.0)
        assert float(v_smooth[0]) > float(v_rough[0])


# ---------------------------------------------------------------------------
# Asset exposure
# ---------------------------------------------------------------------------


class TestAssetExposure:
    """Validation of circular sub-region asset exposure method."""

    def test_asset_in_path(self) -> None:
        sim = TyphoonWindSimulator(buffer_radius_km=200.0)
        track = _make_track(n_points=24, start_lon=125.0, start_lat=15.0,
                            delta_lon=0.3, delta_lat=0.25, pc=950.0)
        # Asset near the track midpoint
        mid_lon = 125.0 + 12 * 0.3
        mid_lat = 15.0 + 12 * 0.25
        result = sim.evaluate_asset_exposure(mid_lon, mid_lat, track)
        assert len(result) > 0
        assert "wind_speed_ms" in result.columns
        assert "gradient_wind_ms" in result.columns
        assert "distance_km" in result.columns

    def test_asset_far_away(self) -> None:
        sim = TyphoonWindSimulator(buffer_radius_km=100.0)
        track = _make_track(n_points=24, start_lon=125.0, start_lat=15.0,
                            delta_lon=0.3, delta_lat=0.25, pc=950.0)
        # Asset far from track
        result = sim.evaluate_asset_exposure(100.0, 50.0, track)
        assert len(result) == 0

    def test_empty_track(self) -> None:
        sim = TyphoonWindSimulator()
        track = pd.DataFrame(columns=["lon", "lat", "pc"])
        result = sim.evaluate_asset_exposure(125.0, 15.0, track)
        assert len(result) == 0

    def test_missing_columns_raises(self) -> None:
        sim = TyphoonWindSimulator()
        track = pd.DataFrame({"x": [1.0], "y": [2.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            sim.evaluate_asset_exposure(125.0, 15.0, track)

    def test_wind_speed_positive(self) -> None:
        sim = TyphoonWindSimulator(buffer_radius_km=300.0)
        track = _make_track(n_points=48, start_lon=125.0, start_lat=15.0,
                            delta_lon=0.2, delta_lat=0.15, pc=930.0)
        result = sim.evaluate_asset_exposure(128.0, 18.0, track)
        if len(result) > 0:
            assert np.all(result["wind_speed_ms"] >= 0.0)
            assert np.all(result["gradient_wind_ms"] >= 0.0)


class TestEvaluateAssets:
    """Validation of batch asset exposure."""

    def test_multiple_assets(self) -> None:
        sim = TyphoonWindSimulator(buffer_radius_km=200.0)
        track = _make_track(n_points=24, start_lon=125.0, start_lat=15.0,
                            delta_lon=0.3, delta_lat=0.25, pc=950.0)
        assets = [
            {"id": "sub_A", "lon": 128.0, "lat": 18.0},
            {"id": "sub_B", "lon": 100.0, "lat": 50.0},  # far away
        ]
        result = sim.evaluate_assets(assets, track)
        assert "sub_A" in result
        assert "sub_B" not in result
        assert len(result["sub_A"]) > 0


# ---------------------------------------------------------------------------
# Wind profile
# ---------------------------------------------------------------------------


class TestWindProfile:
    """Validation of radial wind profile."""

    def test_profile_shape(self) -> None:
        sim = TyphoonWindSimulator()
        profile = sim.wind_profile(pc=950.0, lat=25.0, r_max_km=300.0, n_points=200)
        assert len(profile) == 200
        assert "r_km" in profile.columns
        assert "gradient_wind_ms" in profile.columns
        assert np.all(profile["gradient_wind_ms"] >= 0.0)

    def test_peak_near_rmax(self) -> None:
        sim = TyphoonWindSimulator()
        pc = 950.0
        lat = 25.0
        R = float(TyphoonWindSimulator.radius_of_maximum_wind(pc, lat))
        profile = sim.wind_profile(pc=pc, lat=lat, r_max_km=300.0, n_points=500)
        idx_peak = int(np.argmax(profile["gradient_wind_ms"].values))
        r_peak = profile["r_km"].values[idx_peak]
        # Peak should be near Rmax
        assert abs(r_peak - R) < 30.0


# ---------------------------------------------------------------------------
# Historical typhoon track verification
# ---------------------------------------------------------------------------


class TestHistoricalTyphoon:
    """Validation against historical typhoon track characteristics."""

    def test_haiyan_like_profile(self) -> None:
        """Typhoon Haiyan (2013) had pc ≈ 895 hPa, lat ≈ 11°N at landfall."""
        sim = TyphoonWindSimulator()
        pc = 895.0
        lat = 11.0
        vt = 30.0  # km/h

        R = float(TyphoonWindSimulator.radius_of_maximum_wind(pc, lat, vt))
        B = float(TyphoonWindSimulator.holland_b(pc))

        # Rmax should be in reasonable range for a super typhoon
        assert 15.0 < R < 60.0
        # B should be > 1.5 for intense storms
        assert B > 1.8

        # Wind at Rmax should be extreme
        v_rmax = sim.calculate_gradient_wind_speed(np.array([R]), pc, vt, lat)
        assert float(v_rmax[0]) > 60.0

    def test_mangkhut_like_profile(self) -> None:
        """Typhoon Mangkhut (2018) had pc ≈ 905 hPa, lat ≈ 18°N."""
        sim = TyphoonWindSimulator()
        pc = 905.0
        lat = 18.0
        vt = 25.0

        R = float(TyphoonWindSimulator.radius_of_maximum_wind(pc, lat, vt))
        v_rmax = sim.calculate_gradient_wind_speed(np.array([R]), pc, vt, lat)
        assert float(v_rmax[0]) > 55.0

    def test_wind_decay_profile(self) -> None:
        """Wind speed should follow realistic radial decay."""
        sim = TyphoonWindSimulator()
        pc = 950.0
        lat = 25.0
        R = float(TyphoonWindSimulator.radius_of_maximum_wind(pc, lat))

        r_near = np.array([R])
        r_far = np.array([R * 3.0])

        v_near = float(sim.calculate_gradient_wind_speed(r_near, pc, lat=lat)[0])
        v_far = float(sim.calculate_gradient_wind_speed(r_far, pc, lat=lat)[0])

        # Wind at 3×Rmax should be significantly lower
        assert v_far < v_near * 0.7
