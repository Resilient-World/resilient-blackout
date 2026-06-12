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
Deep learning surrogate for AC Optimal Power Flow (AC-OPF).

Provides ``GridSurrogateNet``, a PyTorch MLP that maps grid topology
state and bus injections to line loading percentages and bus voltage
magnitudes.  Includes an offline training pipeline using randomized
N-1/N-2 contingencies and a runtime prediction interface with MC
Dropout uncertainty estimation and confidence-based fallback to the
full AC-OPF solver.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from resilient_blackout.grid.network import GridModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GridSurrogateNet
# ---------------------------------------------------------------------------

class GridSurrogateNet(nn.Module):
    """MLP surrogate for AC-OPF line loading and bus voltage prediction.

    Maps a concatenated input of binary line/generator states and bus
    active/reactive injections to:

    - Line loading percentages (0–∞ %).
    - Bus voltage magnitudes (per-unit).

    Uses MC Dropout at inference time to estimate prediction confidence.

    Parameters
    ----------
    n_buses : int
        Number of buses in the grid.
    n_lines : int
        Number of lines.
    n_gens : int
        Number of generators.
    hidden_dims : list of int
        Sizes of hidden layers.  Default ``[512, 256, 128]``.
    dropout : float
        Dropout probability.  Default 0.1.

    Attributes
    ----------
    n_buses : int
    n_lines : int
    n_gens : int
    input_dim : int
    """

    def __init__(
        self,
        n_buses: int,
        n_lines: int,
        n_gens: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.n_buses = n_buses
        self.n_lines = n_lines
        self.n_gens = n_gens
        self.input_dim = n_lines + n_gens + 2 * n_buses

        layers: List[nn.Module] = []
        in_dim = self.input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.line_head = nn.Linear(in_dim, n_lines)
        self.voltage_head = nn.Linear(in_dim, n_buses)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            ``(line_loadings, bus_voltages)``.
        """
        features = self.backbone(x)
        line_out = torch.relu(self.line_head(features))
        volt_out = torch.sigmoid(self.voltage_head(features)) * 0.3 + 0.85
        return line_out, volt_out

    def predict_with_confidence(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with MC Dropout uncertainty estimation.

        Runs multiple stochastic forward passes and returns the mean
        prediction and per-sample standard deviation as a confidence
        proxy.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, input_dim)``.
        n_samples : int
            Number of MC dropout samples.  Default 10.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray, np.ndarray)
            ``(mean_line_loadings, mean_bus_voltages, confidence)``
            where confidence is ``1 - mean_normalized_std``.
        """
        self.train()
        line_samples: List[np.ndarray] = []
        volt_samples: List[np.ndarray] = []

        with torch.no_grad():
            for _ in range(n_samples):
                l, v = self.forward(x)
                line_samples.append(l.cpu().numpy())
                volt_samples.append(v.cpu().numpy())

        line_stack = np.stack(line_samples, axis=0)
        volt_stack = np.stack(volt_samples, axis=0)

        line_mean = np.mean(line_stack, axis=0)
        volt_mean = np.mean(volt_stack, axis=0)
        line_std = np.std(line_stack, axis=0)
        volt_std = np.std(volt_stack, axis=0)

        line_norm_std = np.divide(
            line_std, np.maximum(line_mean, 1e-6),
            out=np.ones_like(line_std),
            where=line_mean > 1e-6,
        )
        volt_norm_std = np.divide(
            volt_std, np.maximum(volt_mean, 1e-6),
            out=np.ones_like(volt_std),
            where=volt_mean > 1e-6,
        )

        confidence = 1.0 - 0.5 * (
            np.mean(line_norm_std, axis=1) + np.mean(volt_norm_std, axis=1)
        )
        confidence = np.clip(confidence, 0.0, 1.0)

        return line_mean, volt_mean, confidence


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def _generate_training_sample(
    net: Any,
    rng: np.random.Generator,
    n_lines: int,
    n_gens: int,
    n_buses: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Generate one training sample via randomized contingency.

    Parameters
    ----------
    net : pandapowerNet
    rng : np.random.Generator
    n_lines : int
    n_gens : int
    n_buses : int

    Returns
    -------
    dict or None
        Training sample, or ``None`` if OPF failed.
    """
    import pandapower as pp

    test_net = copy.deepcopy(net)

    n_trip_lines = rng.integers(0, min(3, n_lines + 1))
    if n_trip_lines > 0:
        trip_candidates = rng.choice(n_lines, size=n_trip_lines, replace=False)
        for lidx in trip_candidates:
            if lidx in test_net.line.index:
                test_net.line.at[lidx, "in_service"] = False

    n_trip_gens = rng.integers(0, min(2, n_gens + 1))
    if n_trip_gens > 0:
        gen_candidates = rng.choice(n_gens, size=n_trip_gens, replace=False)
        for gidx in gen_candidates:
            if gidx in test_net.gen.index:
                test_net.gen.at[gidx, "in_service"] = False

    for bidx in range(n_buses):
        if bidx in test_net.load.index:
            scale = 1.0 + rng.uniform(-0.1, 0.1)
            test_net.load.at[bidx, "p_mw"] *= scale
            test_net.load.at[bidx, "q_mvar"] *= scale

    try:
        pp.runopp(test_net)
    except pp.OPFNotConverged:
        return None

    state = np.zeros(n_lines + n_gens, dtype=np.float32)
    for lidx in range(n_lines):
        if lidx in test_net.line.index:
            state[lidx] = float(test_net.line.at[lidx, "in_service"])
    for gidx in range(n_gens):
        if gidx in test_net.gen.index:
            state[n_lines + gidx] = float(test_net.gen.at[gidx, "in_service"])

    injections = np.zeros(2 * n_buses, dtype=np.float32)
    for bidx in range(n_buses):
        if bidx in test_net.res_bus.index:
            injections[2 * bidx] = float(test_net.res_bus.at[bidx, "p_mw"])
            injections[2 * bidx + 1] = float(test_net.res_bus.at[bidx, "q_mvar"])

    line_loadings = np.zeros(n_lines, dtype=np.float32)
    if hasattr(test_net, "res_line"):
        for lidx in range(n_lines):
            if lidx in test_net.res_line.index:
                line_loadings[lidx] = float(
                    test_net.res_line.at[lidx, "loading_percent"]
                )

    bus_voltages = np.zeros(n_buses, dtype=np.float32)
    if hasattr(test_net, "res_bus"):
        for bidx in range(n_buses):
            if bidx in test_net.res_bus.index:
                bus_voltages[bidx] = float(test_net.res_bus.at[bidx, "vm_pu"])

    return {
        "x": np.concatenate([state, injections]).astype(np.float32),
        "line_loadings": line_loadings,
        "bus_voltages": bus_voltages,
    }


def train_surrogate(
    grid_model: GridModel,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    n_samples: int = 5000,
    val_split: float = 0.2,
    device: Optional[str] = None,
) -> Tuple[GridSurrogateNet, Dict[str, List[float]]]:
    """Train a ``GridSurrogateNet`` on randomized contingency data.

    Generates training samples by running N-1/N-2 contingencies with
    perturbed loads through ``pp.runopp``, then fits the MLP surrogate.

    Parameters
    ----------
    grid_model : GridModel
        The grid model to learn from.
    epochs : int
        Number of training epochs.  Default 100.
    batch_size : int
        Mini-batch size.  Default 64.
    lr : float
        Initial learning rate.  Default 1e-3.
    n_samples : int
        Number of training samples to generate.  Default 5000.
    val_split : float
        Fraction of samples reserved for validation.  Default 0.2.
    device : str or None
        Torch device.  Defaults to ``"cuda"`` if available, else
        ``"cpu"``.

    Returns
    -------
    tuple of (GridSurrogateNet, dict)
        Trained model and training history
        ``{"train_loss": [...], "val_loss": [...]}``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    net = grid_model.net
    n_lines = len(net.line)
    n_gens = len(net.gen)
    n_buses = len(net.bus)

    rng = np.random.default_rng(42)
    samples: List[Dict[str, np.ndarray]] = []

    for _ in range(n_samples * 2):
        sample = _generate_training_sample(net, rng, n_lines, n_gens, n_buses)
        if sample is not None:
            samples.append(sample)
        if len(samples) >= n_samples:
            break

    if len(samples) < 100:
        raise RuntimeError(
            f"Only {len(samples)} valid training samples generated. "
            "Check that the grid model supports OPF."
        )

    logger.info("Generated %d training samples.", len(samples))

    X = np.stack([s["x"] for s in samples])
    Y_line = np.stack([s["line_loadings"] for s in samples])
    Y_volt = np.stack([s["bus_voltages"] for s in samples])

    n_val = int(len(X) * val_split)
    indices = rng.permutation(len(X))
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32, device=device)
    Y_line_train = torch.tensor(Y_line[train_idx], dtype=torch.float32, device=device)
    Y_volt_train = torch.tensor(Y_volt[train_idx], dtype=torch.float32, device=device)

    X_val = torch.tensor(X[val_idx], dtype=torch.float32, device=device)
    Y_line_val = torch.tensor(Y_line[val_idx], dtype=torch.float32, device=device)
    Y_volt_val = torch.tensor(Y_volt[val_idx], dtype=torch.float32, device=device)

    train_ds = TensorDataset(X_train, Y_line_train, Y_volt_train)
    val_ds = TensorDataset(X_val, Y_line_val, Y_volt_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = GridSurrogateNet(
        n_buses=n_buses,
        n_lines=n_lines,
        n_gens=n_gens,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    line_loss_fn = nn.MSELoss()
    volt_loss_fn = nn.MSELoss()

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yl, yv in train_loader:
            optimizer.zero_grad()
            pred_l, pred_v = model(xb)
            loss = line_loss_fn(pred_l, yl) + volt_loss_fn(pred_v, yv)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yl, yv in val_loader:
                pred_l, pred_v = model(xb)
                loss = line_loss_fn(pred_l, yl) + volt_loss_fn(pred_v, yv)
                val_loss += loss.item() * len(xb)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 20 == 0:
            logger.info(
                "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f",
                epoch + 1, epochs, train_loss, val_loss,
            )

    model.eval()
    return model, history


# ---------------------------------------------------------------------------
# Runtime prediction
# ---------------------------------------------------------------------------

def predict_opf_states(
    model: GridSurrogateNet,
    state_vector: np.ndarray,
    active_loads: np.ndarray,
    reactive_loads: Optional[np.ndarray] = None,
    grid_model: Optional[GridModel] = None,
    confidence_threshold: float = 0.85,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Predict OPF states using the surrogate, with fallback to AC-OPF.

    Parameters
    ----------
    model : GridSurrogateNet
        Trained surrogate model.
    state_vector : np.ndarray
        Binary state of shape ``(n_lines + n_gens,)``.
    active_loads : np.ndarray
        Active power injections per bus of shape ``(n_buses,)``.
    reactive_loads : np.ndarray or None
        Reactive power injections per bus.  Defaults to zeros.
    grid_model : GridModel or None
        Required for fallback AC-OPF.  If ``None`` and fallback is
        needed, raises ``RuntimeError``.
    confidence_threshold : float
        Minimum confidence to trust surrogate.  Default 0.85.
    device : str or None

    Returns
    -------
    dict
        ``{"line_loadings": np.ndarray, "bus_voltages": np.ndarray,
        "confidence": float, "used_surrogate": bool}``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if reactive_loads is None:
        reactive_loads = np.zeros(model.n_buses, dtype=np.float32)

    x = np.concatenate([
        np.asarray(state_vector, dtype=np.float32).ravel(),
        np.asarray(active_loads, dtype=np.float32).ravel(),
        np.asarray(reactive_loads, dtype=np.float32).ravel(),
    ])

    x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

    line_mean, volt_mean, confidence = model.predict_with_confidence(x_t)
    conf_val = float(confidence[0])

    if conf_val >= confidence_threshold:
        return {
            "line_loadings": line_mean[0],
            "bus_voltages": volt_mean[0],
            "confidence": conf_val,
            "used_surrogate": True,
        }

    logger.warning(
        "Surrogate confidence %.3f < threshold %.3f — falling back to AC-OPF.",
        conf_val, confidence_threshold,
    )

    if grid_model is None:
        raise RuntimeError(
            "Surrogate confidence too low and no grid_model provided for fallback."
        )

    return _fallback_opf(grid_model, state_vector, active_loads, reactive_loads)


def _fallback_opf(
    grid_model: GridModel,
    state_vector: np.ndarray,
    active_loads: np.ndarray,
    reactive_loads: np.ndarray,
) -> Dict[str, Any]:
    """Run full AC-OPF as fallback.

    Parameters
    ----------
    grid_model : GridModel
    state_vector : np.ndarray
    active_loads : np.ndarray
    reactive_loads : np.ndarray

    Returns
    -------
    dict
    """
    import pandapower as pp

    net = copy.deepcopy(grid_model.net)
    n_lines = len(net.line)
    n_gens = len(net.gen)

    for lidx in range(min(n_lines, len(state_vector))):
        if lidx in net.line.index:
            net.line.at[lidx, "in_service"] = bool(state_vector[lidx] > 0.5)
    for gidx in range(n_gens):
        sidx = n_lines + gidx
        if sidx < len(state_vector) and gidx in net.gen.index:
            net.gen.at[gidx, "in_service"] = bool(state_vector[sidx] > 0.5)

    for bidx in range(min(len(active_loads), len(net.load))):
        if bidx in net.load.index:
            net.load.at[bidx, "p_mw"] = float(active_loads[bidx])
    for bidx in range(min(len(reactive_loads), len(net.load))):
        if bidx in net.load.index:
            net.load.at[bidx, "q_mvar"] = float(reactive_loads[bidx])

    try:
        pp.runopp(net)
    except pp.OPFNotConverged:
        pp.runpp(net)

    line_loadings = np.zeros(n_lines, dtype=np.float32)
    if hasattr(net, "res_line"):
        for lidx in range(n_lines):
            if lidx in net.res_line.index:
                line_loadings[lidx] = float(net.res_line.at[lidx, "loading_percent"])

    bus_voltages = np.zeros(len(net.bus), dtype=np.float32)
    if hasattr(net, "res_bus"):
        for bidx in range(len(net.bus)):
            if bidx in net.res_bus.index:
                bus_voltages[bidx] = float(net.res_bus.at[bidx, "vm_pu"])

    return {
        "line_loadings": line_loadings,
        "bus_voltages": bus_voltages,
        "confidence": 1.0,
        "used_surrogate": False,
    }
