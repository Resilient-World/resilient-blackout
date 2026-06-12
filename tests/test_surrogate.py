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

"""Unit tests for ``resilient_blackout.ml.surrogate``."""

import numpy as np
import pytest
import torch

from resilient_blackout.ml.surrogate import (
    GridSurrogateNet,
    predict_opf_states,
)


class TestGridSurrogateNet:
    """Test suite for GridSurrogateNet."""

    def test_forward_shapes(self):
        """Forward pass should produce correct output shapes."""
        n_buses, n_lines, n_gens = 10, 15, 4
        model = GridSurrogateNet(n_buses, n_lines, n_gens)
        model.eval()

        batch = 8
        x = torch.randn(batch, model.input_dim)
        with torch.no_grad():
            line_out, volt_out = model(x)

        assert line_out.shape == (batch, n_lines)
        assert volt_out.shape == (batch, n_buses)
        assert torch.all(line_out >= 0)
        assert torch.all(volt_out >= 0.85)
        assert torch.all(volt_out <= 1.15)

    def test_predict_with_confidence(self):
        """MC Dropout should return confidence scores."""
        n_buses, n_lines, n_gens = 8, 12, 3
        model = GridSurrogateNet(n_buses, n_lines, n_gens)

        batch = 4
        x = torch.randn(batch, model.input_dim)
        line_mean, volt_mean, confidence = model.predict_with_confidence(x, n_samples=5)

        assert line_mean.shape == (batch, n_lines)
        assert volt_mean.shape == (batch, n_buses)
        assert confidence.shape == (batch,)
        assert np.all(confidence >= 0.0)
        assert np.all(confidence <= 1.0)

    def test_confidence_fallback(self):
        """Low confidence should trigger fallback when grid_model is provided."""
        n_buses, n_lines, n_gens = 5, 7, 2
        model = GridSurrogateNet(n_buses, n_lines, n_gens)

        state = np.ones(n_lines + n_gens, dtype=np.float32)
        loads = np.ones(n_buses, dtype=np.float32) * 10.0

        result = predict_opf_states(
            model, state, loads,
            confidence_threshold=0.999,
        )

        assert not result["used_surrogate"]
        assert result["confidence"] == 1.0

    def test_high_confidence_uses_surrogate(self):
        """High confidence should use surrogate."""
        n_buses, n_lines, n_gens = 5, 7, 2
        model = GridSurrogateNet(n_buses, n_lines, n_gens)
        model.eval()

        state = np.ones(n_lines + n_gens, dtype=np.float32)
        loads = np.ones(n_buses, dtype=np.float32) * 10.0

        result = predict_opf_states(
            model, state, loads,
            confidence_threshold=0.0,
        )

        assert result["used_surrogate"]
        assert result["line_loadings"].shape == (n_lines,)
        assert result["bus_voltages"].shape == (n_buses,)

    def test_no_grid_model_raises_on_fallback(self):
        """Missing grid_model should raise when fallback needed."""
        n_buses, n_lines, n_gens = 5, 7, 2
        model = GridSurrogateNet(n_buses, n_lines, n_gens)

        state = np.ones(n_lines + n_gens, dtype=np.float32)
        loads = np.ones(n_buses, dtype=np.float32) * 10.0

        with pytest.raises(RuntimeError, match="no grid_model"):
            predict_opf_states(
                model, state, loads,
                confidence_threshold=0.999,
            )
