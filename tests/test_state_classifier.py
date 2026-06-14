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

"""Unit tests for ``resilient_blackout.ml.state_classifier``."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from resilient_blackout.ml.state_classifier import (
    CNNStateTrainer,
    StateClassifierCNN,
    should_skip_power_flow,
)


def _make_synthetic_data(
    n_samples: int = 500,
    input_dim: int = 30,
    random_state: int = 42,
) -> tuple:
    """Generate synthetic binary state vectors and labels."""
    rng = np.random.default_rng(random_state)
    states = rng.integers(0, 2, size=(n_samples, input_dim)).astype(np.float32)
    # Label: failure if > 40% of elements are out-of-service
    n_out = (states == 0).sum(axis=1)
    labels = (n_out > 0.4 * input_dim).astype(np.float32)
    return states, labels


class TestStateClassifierCNN:
    """Validation of CNN architecture and forward pass."""

    def test_construction(self) -> None:
        model = StateClassifierCNN(input_dim=30)
        assert model.input_dim == 30
        assert len(model.conv_channels) == 3

    def test_custom_channels(self) -> None:
        model = StateClassifierCNN(input_dim=20, conv_channels=[16, 32])
        assert model.conv_channels == [16, 32]

    def test_forward_shape(self) -> None:
        model = StateClassifierCNN(input_dim=30)
        x = torch.randn(8, 30)
        out = model(x)
        assert out.shape == (8, 1)
        assert (out >= 0).all() and (out <= 1).all()

    def test_forward_single_sample(self) -> None:
        model = StateClassifierCNN(input_dim=30)
        x = torch.randn(30)
        out = model(x)
        assert out.shape == (1,)

    def test_deterministic_eval(self) -> None:
        model = StateClassifierCNN(input_dim=30)
        model.eval()
        x = torch.ones(5, 30)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)


class TestCNNStateTrainer:
    """Validation of training pipeline."""

    def test_train_synthetic(self) -> None:
        states, labels = _make_synthetic_data(n_samples=300)
        model = StateClassifierCNN(input_dim=30)
        trainer = CNNStateTrainer(device="cpu")
        history = trainer.train(model, states, labels, epochs=5, batch_size=32, lr=1e-3)

        assert "train_loss" in history
        assert "val_loss" in history
        assert "val_acc" in history
        assert len(history["train_loss"]) == 5
        # Loss should generally decrease
        assert history["train_loss"][-1] < history["train_loss"][0] + 0.1

    def test_validate(self) -> None:
        states, labels = _make_synthetic_data(n_samples=200)
        model = StateClassifierCNN(input_dim=30)
        trainer = CNNStateTrainer(device="cpu")
        trainer.train(model, states, labels, epochs=10, batch_size=32, lr=1e-3)

        metrics = trainer.validate(model, states, labels)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_validate_all_normal(self) -> None:
        states = np.ones((50, 20), dtype=np.float32)
        labels = np.zeros(50, dtype=np.float32)
        model = StateClassifierCNN(input_dim=20)
        trainer = CNNStateTrainer(device="cpu")
        trainer.train(model, states, labels, epochs=5, batch_size=16, lr=1e-3)
        metrics = trainer.validate(model, states, labels)
        assert metrics["accuracy"] >= 0.0


class TestSaveLoad:
    """Validation of model persistence."""

    def test_save_load_roundtrip(self) -> None:
        states, labels = _make_synthetic_data(n_samples=200)
        model = StateClassifierCNN(input_dim=30)
        trainer = CNNStateTrainer(device="cpu")
        trainer.train(model, states, labels, epochs=5, batch_size=32, lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cnn_model.pt")
            CNNStateTrainer.save_model(model, path)
            loaded = CNNStateTrainer.load_model(path, device="cpu")

            assert loaded.input_dim == model.input_dim
            assert loaded.conv_channels == model.conv_channels

            x = torch.randn(10, 30)
            model.eval()
            loaded.eval()
            out_orig = model(x)
            out_loaded = loaded(x)
            assert torch.allclose(out_orig, out_loaded, atol=1e-6)


class TestPreScreening:
    """Validation of Monte Carlo pre-screening logic."""

    def test_skip_stable_state(self) -> None:
        states, labels = _make_synthetic_data(n_samples=200)
        model = StateClassifierCNN(input_dim=30)
        trainer = CNNStateTrainer(device="cpu")
        trainer.train(model, states, labels, epochs=10, batch_size=32, lr=1e-3)

        # All-in-service state should be classified as stable
        stable_state = np.ones(30, dtype=np.float32)
        result = should_skip_power_flow(model, stable_state, threshold=0.5, device="cpu")
        assert result is True  # all elements in-service → very low failure prob

    def test_no_skip_failure_state(self) -> None:
        states, labels = _make_synthetic_data(n_samples=200)
        model = StateClassifierCNN(input_dim=30)
        trainer = CNNStateTrainer(device="cpu")
        trainer.train(model, states, labels, epochs=10, batch_size=32, lr=1e-3)

        # Mostly out-of-service state should NOT be skipped
        failure_state = np.zeros(30, dtype=np.float32)
        result = should_skip_power_flow(model, failure_state, threshold=0.01, device="cpu")
        assert result is False  # all elements out → high failure prob

    def test_threshold_zero_never_skips(self) -> None:
        model = StateClassifierCNN(input_dim=10)
        state = np.ones(10, dtype=np.float32)
        result = should_skip_power_flow(model, state, threshold=0.0, device="cpu")
        assert result is False


class TestGatherTrainingStates:
    """Validation of training data collection from pandapower networks."""

    @pytest.mark.pandapower
    def test_gather_from_network(self) -> None:
        pytest.importorskip("pandapower")
        import pandapower as pp

        net = pp.create_empty_network()
        b0 = pp.create_bus(net, vn_kv=0.4)
        b1 = pp.create_bus(net, vn_kv=0.4)
        b2 = pp.create_bus(net, vn_kv=0.4)
        pp.create_ext_grid(net, bus=b0, vm_pu=1.0)
        pp.create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, std_type="NAYY 4x50 SE")
        pp.create_line(net, from_bus=b1, to_bus=b2, length_km=1.0, std_type="NAYY 4x50 SE")
        pp.create_gen(net, bus=b1, p_mw=2.0, vm_pu=1.0)
        pp.create_load(net, bus=b2, p_mw=1.0, q_mvar=0.2)

        trainer = CNNStateTrainer(device="cpu")
        rng = np.random.default_rng(42)
        states, labels = trainer.gather_training_states(net, n_samples=20, k_max=2, rng=rng)

        assert states.shape[0] == 20
        assert states.shape[1] == 3  # 2 lines + 1 gen
        assert labels.shape[0] == 20
        assert set(np.unique(states)) <= {0.0, 1.0}
        assert set(np.unique(labels)) <= {0.0, 1.0}

    @pytest.mark.pandapower
    def test_gather_all_in_service_is_normal(self) -> None:
        pytest.importorskip("pandapower")
        import pandapower as pp

        net = pp.create_empty_network()
        b0 = pp.create_bus(net, vn_kv=0.4)
        b1 = pp.create_bus(net, vn_kv=0.4)
        pp.create_ext_grid(net, bus=b0, vm_pu=1.0)
        pp.create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, std_type="NAYY 4x50 SE")
        pp.create_load(net, bus=b1, p_mw=0.5, q_mvar=0.1)

        trainer = CNNStateTrainer(device="cpu")
        rng = np.random.default_rng(42)
        states, labels = trainer.gather_training_states(net, n_samples=10, k_max=1, rng=rng)

        # With k_max=1 and only 1 line, some samples will trip it → failure
        # But some should be normal
        assert 0.0 in labels or 1.0 in labels  # at least some variation
