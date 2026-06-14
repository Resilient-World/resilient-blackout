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
Deep 1D CNN state classifier for grid topology screening.

Provides ``StateClassifierCNN``, a PyTorch 1D convolutional network that
maps a binary grid-state vector (in-service / out-of-service flags for
generators, lines, and transformers) to a failure probability.  Includes
``CNNStateTrainer`` for offline training on N-k contingency data and a
pre-screening helper for Monte Carlo loops.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# StateClassifierCNN
# ---------------------------------------------------------------------------


class StateClassifierCNN(nn.Module):
    """1D CNN for binary grid-state → failure-probability classification.

    .. math::

        h_1 &= \\sigma(w_1 * S + b_1) \\\\
        h_q &= \\sigma(w_q * h_{q-1} + b_q) \\\\
        z   &= \\sigma(w_f \\cdot \\text{flatten}(h_Q) + b_f)

    Parameters
    ----------
    input_dim : int
        Length of the binary state vector (n_gens + n_lines + n_trafos).
    conv_channels : list of int
        Output channel counts for each convolutional layer.
        Default ``[32, 64, 128]``.
    kernel_size : int
        Convolution kernel size.  Default 3.
    fc_hidden : int
        Hidden dimension of the fully-connected layer.  Default 64.
    dropout : float
        Dropout probability after the FC hidden layer.  Default 0.3.
    pool_output_size : int
        Target size for adaptive average pooling before flattening.
        Default 4.
    """

    def __init__(
        self,
        input_dim: int,
        conv_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        fc_hidden: int = 64,
        dropout: float = 0.3,
        pool_output_size: int = 4,
    ) -> None:
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128]

        self.input_dim = input_dim
        self.conv_channels = list(conv_channels)
        self.kernel_size = kernel_size
        self.pool_output_size = pool_output_size

        # Build convolutional stack
        layers: List[nn.Module] = []
        in_ch = 1
        for out_ch in conv_channels:
            layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            )
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.conv_stack = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(pool_output_size)

        conv_out_dim = conv_channels[-1] * pool_output_size
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)`` — failure probability ∈ [0, 1].
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        h = self.conv_stack(x)
        h = self.pool(h)
        h = h.flatten(1)
        return self.fc(h)


# ---------------------------------------------------------------------------
# CNNStateTrainer
# ---------------------------------------------------------------------------


class CNNStateTrainer:
    """Training pipeline for ``StateClassifierCNN``.

    Gathers labelled training states from randomised N-k contingency
    simulations, trains the CNN with weighted cross-entropy loss, and
    provides validation metrics.

    Parameters
    ----------
    device : str or torch.device
        Compute device.  Default ``"cpu"``.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    # Data gathering
    # ------------------------------------------------------------------

    def gather_training_states(
        self,
        net: Any,
        n_samples: int = 1000,
        k_max: int = 5,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate labelled training data via randomised N-k contingencies.

        For each sample, randomly trips *k* elements (lines, generators,
        or transformers), runs a DC power flow, and labels the state as
        failure (1) if any load is shed or the power flow does not
        converge.

        Parameters
        ----------
        net : pandapowerNet
            Base network (not mutated).
        n_samples : int
            Number of training samples.  Default 1000.
        k_max : int
            Maximum number of simultaneous outages.  Default 5.
        rng : np.random.Generator or None

        Returns
        -------
        tuple
            ``(states, labels)`` — states shape ``(n_samples, D)``,
            labels shape ``(n_samples,)``.
        """
        import copy

        import pandapower as pp

        if rng is None:
            rng = np.random.default_rng()

        n_lines = len(net.line)
        n_gens = len(net.gen)
        n_trafos = len(net.trafo) if hasattr(net, "trafo") else 0
        D = n_lines + n_gens + n_trafos

        states = np.zeros((n_samples, D), dtype=np.float32)
        labels = np.zeros(n_samples, dtype=np.float32)

        line_ids = list(range(n_lines))
        gen_ids = list(range(n_gens))
        trafo_ids = list(range(n_trafos))

        for i in range(n_samples):
            k = rng.integers(1, k_max + 1)
            n_trip = min(k, D)

            # Build binary state: 1 = in-service, 0 = out-of-service
            state = np.ones(D, dtype=np.float32)
            trip_indices = rng.choice(D, size=n_trip, replace=False)
            state[trip_indices] = 0.0
            states[i] = state

            # Apply contingency to a copy of the network
            net_copy = copy.deepcopy(net)
            for idx in trip_indices:
                if idx < n_lines:
                    net_copy.line.at[idx, "in_service"] = False
                elif idx < n_lines + n_gens:
                    gen_idx = idx - n_lines
                    net_copy.gen.at[gen_idx, "in_service"] = False
                else:
                    trafo_idx = idx - n_lines - n_gens
                    if trafo_idx < n_trafos:
                        net_copy.trafo.at[trafo_idx, "in_service"] = False

            # Run DC power flow
            try:
                pp.rundcpp(net_copy)
                total_shed = 0.0
                if hasattr(net_copy, "res_load") and "p_mw" in net_copy.res_load.columns:
                    served = net_copy.res_load["p_mw"].sum()
                    demand = net_copy.load["p_mw"].sum()
                    total_shed = max(0.0, demand - served)
                labels[i] = 1.0 if total_shed > 1e-6 else 0.0
            except (pp.LoadflowNotConverged, Exception):
                labels[i] = 1.0  # non-convergence → failure

        n_pos = int(labels.sum())
        logger.info(
            "Gathered %d training samples: %d failure, %d normal (%.1f%% positive)",
            n_samples, n_pos, n_samples - n_pos, 100.0 * n_pos / max(1, n_samples),
        )
        return states, labels

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        model: StateClassifierCNN,
        states: np.ndarray,
        labels: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        val_frac: float = 0.2,
    ) -> Dict[str, List[float]]:
        """Train the CNN on labelled state data.

        Uses weighted binary cross-entropy to handle class imbalance.

        Parameters
        ----------
        model : StateClassifierCNN
        states : np.ndarray
            Shape ``(N, D)``.
        labels : np.ndarray
            Shape ``(N,)``.
        epochs : int
        batch_size : int
        lr : float
        val_frac : float

        Returns
        -------
        dict
            ``{"train_loss": [...], "val_loss": [...], "val_acc": [...]}``.
        """
        N = len(labels)
        n_val = max(1, int(N * val_frac))
        indices = np.random.permutation(N)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        X_train = torch.tensor(states[train_idx], dtype=torch.float32)
        y_train = torch.tensor(labels[train_idx], dtype=torch.float32)
        X_val = torch.tensor(states[val_idx], dtype=torch.float32)
        y_val = torch.tensor(labels[val_idx], dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

        model = model.to(self.device)

        # Weighted loss: inverse class frequency
        n_pos = float(labels.sum())
        n_neg = float(N - n_pos)
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=self.device)
        criterion = nn.BCELoss(weight=pos_weight)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = model(xb).squeeze(1)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
            avg_train = total_loss / len(train_idx)
            history["train_loss"].append(avg_train)

            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = model(xb).squeeze(1)
                    val_loss += criterion(pred, yb).item() * len(xb)
                    correct += ((pred >= 0.5) == yb).sum().item()
            avg_val = val_loss / len(val_idx)
            acc = correct / len(val_idx)
            history["val_loss"].append(avg_val)
            history["val_acc"].append(acc)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f, val_acc=%.2f%%",
                    epoch + 1, epochs, avg_train, avg_val, 100.0 * acc,
                )

        return history

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        model: StateClassifierCNN,
        states: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Compute classification metrics.

        Parameters
        ----------
        model : StateClassifierCNN
        states : np.ndarray
        labels : np.ndarray

        Returns
        -------
        dict
            ``accuracy``, ``precision``, ``recall``, ``f1``.
        """
        model.eval()
        X = torch.tensor(states, dtype=torch.float32).to(self.device)
        y_true = torch.tensor(labels, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_pred = model(X).squeeze(1)
            y_pred_bin = (y_pred >= 0.5).float()

        tp = ((y_pred_bin == 1) & (y_true == 1)).sum().item()
        fp = ((y_pred_bin == 1) & (y_true == 0)).sum().item()
        fn = ((y_pred_bin == 0) & (y_true == 1)).sum().item()
        tn = ((y_pred_bin == 0) & (y_true == 0)).sum().item()

        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_model(model: StateClassifierCNN, path: str) -> None:
        """Save model state dict and metadata to disk.

        Parameters
        ----------
        model : StateClassifierCNN
        path : str
        """
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "state_dict": model.state_dict(),
            "input_dim": model.input_dim,
            "conv_channels": model.conv_channels,
            "kernel_size": model.kernel_size,
            "pool_output_size": model.pool_output_size,
        }
        torch.save(payload, path)
        logger.info("Model saved to %s", path)

    @staticmethod
    def load_model(path: str, device: str = "cpu") -> StateClassifierCNN:
        """Load a previously saved model.

        Parameters
        ----------
        path : str
        device : str

        Returns
        -------
        StateClassifierCNN
        """
        payload = torch.load(path, map_location=device, weights_only=False)
        model = StateClassifierCNN(
            input_dim=payload["input_dim"],
            conv_channels=payload.get("conv_channels"),
            kernel_size=payload.get("kernel_size", 3),
            pool_output_size=payload.get("pool_output_size", 4),
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.eval()
        logger.info("Model loaded from %s", path)
        return model


# ---------------------------------------------------------------------------
# Monte Carlo pre-screening
# ---------------------------------------------------------------------------


def should_skip_power_flow(
    model: StateClassifierCNN,
    state_vector: np.ndarray,
    threshold: float = 0.01,
    device: str = "cpu",
) -> bool:
    """Pre-screen a grid state to decide whether to skip power flow.

    If the predicted failure probability is below *threshold*, the state
    is classified as stable and the costly power flow solve can be
    bypassed.

    Parameters
    ----------
    model : StateClassifierCNN
    state_vector : np.ndarray
        Binary state vector of shape ``(D,)``.
    threshold : float
        Probability below which the state is considered stable.
        Default 0.01.
    device : str

    Returns
    -------
    bool
        ``True`` if power flow should be skipped.
    """
    model.eval()
    x = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = model(x).item()
    return prob < threshold
