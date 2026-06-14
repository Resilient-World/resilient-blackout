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

"""Unit tests for ``resilient_blackout.economics.sensitivity_analysis``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resilient_blackout.economics.sensitivity_analysis import GlobalSensitivityAnalyzer


# ---------------------------------------------------------------------------
# Test model: Ishikawa function (additive, known sensitivities)
# ---------------------------------------------------------------------------


def _ishikawa(params: np.ndarray) -> float:
    """Ishikawa function: f(x) = sin(x0) + 5*sin²(x1) + 0.1*x2⁴*sin(x0).

    x0 dominates, x1 secondary, x2 has interaction with x0.
    """
    x0, x1, x2 = params[0], params[1], params[2]
    return float(np.sin(x0) + 5.0 * np.sin(x1) ** 2 + 0.1 * x2 ** 4 * np.sin(x0))


_PARAM_NAMES = ["x0", "x1", "x2"]
_PARAM_BOUNDS = [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]]
_PARAM_GROUPS = ["trig", "trig", "poly"]


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestInit:
    """Validation of constructor and parameter handling."""

    def test_default_construction(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        assert analyzer.n_params == 3
        assert analyzer.param_names == _PARAM_NAMES
        assert analyzer.param_groups is None
        assert analyzer.model_func is None
        assert analyzer.problem["num_vars"] == 3

    def test_with_groups(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(
            _PARAM_NAMES, _PARAM_BOUNDS, param_groups=_PARAM_GROUPS
        )
        assert analyzer.param_groups == _PARAM_GROUPS
        assert "groups" in analyzer.problem

    def test_with_model_func(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(
            _PARAM_NAMES, _PARAM_BOUNDS, model_func=_ishikawa
        )
        assert analyzer.model_func is _ishikawa

    def test_empty_names_raises(self) -> None:
        with pytest.raises(ValueError, match="param_names"):
            GlobalSensitivityAnalyzer([], [])

    def test_mismatched_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match="param_bounds"):
            GlobalSensitivityAnalyzer(["a", "b"], [[0, 1]])

    def test_mismatched_groups_raises(self) -> None:
        with pytest.raises(ValueError, match="param_groups"):
            GlobalSensitivityAnalyzer(
                ["a", "b"], [[0, 1], [0, 1]], param_groups=["g1"]
            )

    def test_invalid_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match="lower"):
            GlobalSensitivityAnalyzer(["a"], [[5, 1]])

    def test_repr(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        r = repr(analyzer)
        assert "GlobalSensitivityAnalyzer" in r
        assert "3" in r


# ---------------------------------------------------------------------------
# Sobol sampling
# ---------------------------------------------------------------------------


class TestSobolSampling:
    """Validation of Sobol sequence sampling."""

    def test_sample_shape(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        samples = analyzer.generate_sobol_samples(N=64, calc_second_order=True)
        # Total samples = 2 * N * (K + 1) = 2 * 64 * 4 = 512
        assert samples.shape == (512, 3)

    def test_sample_shape_no_second_order(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        samples = analyzer.generate_sobol_samples(N=64, calc_second_order=False)
        # Total samples = N * (K + 2) = 64 * 5 = 320
        assert samples.shape == (320, 3)

    def test_samples_within_bounds(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        samples = analyzer.generate_sobol_samples(N=32)
        for i in range(3):
            assert np.all(samples[:, i] >= _PARAM_BOUNDS[i][0])
            assert np.all(samples[:, i] <= _PARAM_BOUNDS[i][1])


# ---------------------------------------------------------------------------
# Morris sampling
# ---------------------------------------------------------------------------


class TestMorrisScreening:
    """Validation of Morris method screening."""

    def test_sample_shape(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        result = analyzer.evaluate_morris_screening(N=10, num_levels=4)
        samples = result["samples"]
        # Morris: N * (K + 1) samples
        assert samples.shape == (10 * 4, 3)
        assert "morris_problem" in result
        assert result["morris_problem"]["num_levels"] == 4

    def test_samples_within_bounds(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        result = analyzer.evaluate_morris_screening(N=10, num_levels=6)
        samples = result["samples"]
        for i in range(3):
            assert np.all(samples[:, i] >= _PARAM_BOUNDS[i][0])
            assert np.all(samples[:, i] <= _PARAM_BOUNDS[i][1])


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


class TestModelEvaluation:
    """Validation of model evaluation."""

    def test_sequential_evaluation(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        samples = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        outputs = analyzer.evaluate_model(
            samples, model_func=_ishikawa, parallel=False
        )
        assert outputs.shape == (2,)
        assert np.all(np.isfinite(outputs))

    def test_parallel_evaluation(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        samples = analyzer.generate_sobol_samples(N=16, calc_second_order=False)
        outputs = analyzer.evaluate_model(
            samples, model_func=_ishikawa, parallel=True, n_jobs=2
        )
        assert len(outputs) == len(samples)
        assert np.all(np.isfinite(outputs))

    def test_no_model_func_raises(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        with pytest.raises(ValueError, match="model_func"):
            analyzer.evaluate_model(np.array([[0.0, 0.0, 0.0]]))


# ---------------------------------------------------------------------------
# Sobol analysis
# ---------------------------------------------------------------------------


class TestSobolAnalysis:
    """Validation of Sobol index computation."""

    def test_analyze_indices(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        samples = analyzer.generate_sobol_samples(N=128, calc_second_order=True)
        outputs = analyzer.evaluate_model(samples, model_func=_ishikawa, parallel=False)
        indices = analyzer.analyze_sobol_indices(samples, outputs)

        assert "S1" in indices
        assert "ST" in indices
        assert "S2" in indices
        assert "S1_conf" in indices
        assert "ST_conf" in indices
        assert "S2_conf" in indices
        assert "summary" in indices
        assert len(indices["S1"]) == 3
        assert len(indices["ST"]) == 3

    def test_summary_sorted_by_st(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        samples = analyzer.generate_sobol_samples(N=128, calc_second_order=True)
        outputs = analyzer.evaluate_model(samples, model_func=_ishikawa, parallel=False)
        indices = analyzer.analyze_sobol_indices(samples, outputs)

        summary = indices["summary"]
        st_vals = summary["ST"].values
        for i in range(len(st_vals) - 1):
            assert st_vals[i] >= st_vals[i + 1]

    def test_x0_dominates(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        samples = analyzer.generate_sobol_samples(N=256, calc_second_order=True)
        outputs = analyzer.evaluate_model(samples, model_func=_ishikawa, parallel=False)
        indices = analyzer.analyze_sobol_indices(samples, outputs)

        summary = indices["summary"]
        # x1 should have the highest ST (5*sin² dominates)
        top_param = summary["parameter"].values[0]
        assert top_param == "x1"


# ---------------------------------------------------------------------------
# Morris analysis
# ---------------------------------------------------------------------------


class TestMorrisAnalysis:
    """Validation of Morris index computation."""

    def test_analyze_morris(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        screening = analyzer.evaluate_morris_screening(N=20, num_levels=4)
        samples = screening["samples"]
        outputs = analyzer.evaluate_model(samples, model_func=_ishikawa, parallel=False)
        indices = analyzer.analyze_morris(samples, outputs, screening["morris_problem"])

        assert "mu_star" in indices
        assert "sigma" in indices
        assert "mu_star_conf" in indices
        assert "summary" in indices
        assert len(indices["mu_star"]) == 3

    def test_summary_sorted(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        screening = analyzer.evaluate_morris_screening(N=20, num_levels=4)
        samples = screening["samples"]
        outputs = analyzer.evaluate_model(samples, model_func=_ishikawa, parallel=False)
        indices = analyzer.analyze_morris(samples, outputs, screening["morris_problem"])

        summary = indices["summary"]
        mu_vals = summary["mu_star"].values
        for i in range(len(mu_vals) - 1):
            assert mu_vals[i] >= mu_vals[i + 1]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Validation of end-to-end analysis pipelines."""

    def test_run_sobol_analysis(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        result = analyzer.run_sobol_analysis(
            N=64, model_func=_ishikawa, calc_second_order=True, parallel=False
        )
        assert "samples" in result
        assert "outputs" in result
        assert "indices" in result
        assert len(result["outputs"]) == len(result["samples"])

    def test_run_morris_analysis(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        result = analyzer.run_morris_analysis(
            N=10, num_levels=4, model_func=_ishikawa, parallel=False
        )
        assert "samples" in result
        assert "outputs" in result
        assert "indices" in result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


class TestVisualisation:
    """Validation of visualisation helpers."""

    def test_plot_sensitivity_bars(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        samples = analyzer.generate_sobol_samples(N=64, calc_second_order=True)
        outputs = analyzer.evaluate_model(samples, model_func=_ishikawa, parallel=False)
        indices = analyzer.analyze_sobol_indices(samples, outputs)

        fig = GlobalSensitivityAnalyzer.plot_sensitivity_bars(indices)
        assert fig is not None

    def test_plot_sensitivity_bars_top_n(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        samples = analyzer.generate_sobol_samples(N=64, calc_second_order=True)
        outputs = analyzer.evaluate_model(samples, model_func=_ishikawa, parallel=False)
        indices = analyzer.analyze_sobol_indices(samples, outputs)

        fig = GlobalSensitivityAnalyzer.plot_sensitivity_bars(indices, top_n=2)
        assert fig is not None

    def test_plot_interaction_heatmap(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        samples = analyzer.generate_sobol_samples(N=64, calc_second_order=True)
        outputs = analyzer.evaluate_model(samples, model_func=_ishikawa, parallel=False)
        indices = analyzer.analyze_sobol_indices(samples, outputs)

        fig = GlobalSensitivityAnalyzer.plot_interaction_heatmap(indices)
        assert fig is not None

    def test_plot_morris_bars(self) -> None:
        analyzer = GlobalSensitivityAnalyzer(_PARAM_NAMES, _PARAM_BOUNDS)
        screening = analyzer.evaluate_morris_screening(N=10, num_levels=4)
        samples = screening["samples"]
        outputs = analyzer.evaluate_model(samples, model_func=_ishikawa, parallel=False)
        indices = analyzer.analyze_morris(samples, outputs, screening["morris_problem"])

        fig = GlobalSensitivityAnalyzer.plot_morris_bars(indices)
        assert fig is not None
