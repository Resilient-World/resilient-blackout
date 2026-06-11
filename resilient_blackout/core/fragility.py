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
Log-normal fragility curve evaluation for physical vulnerability modelling.

Implements the ``ImpactFunction`` class — a mathematically rigorous
log-normal cumulative distribution evaluator — and the
``ImpactFunctionSet`` container for managing collections of vulnerability
curves with JSON and DataFrame import/export.
"""

from __future__ import annotations

import json
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

_EPSILON: float = 1e-12


@dataclass(frozen=True)
class ImpactFunction:
    """A log-normal fragility curve for a specific asset class and hazard type.

    Models the probability of failure as a function of hazard intensity
    using the log-normal cumulative distribution:

    .. math::

        P_f(I) = \\Phi\\!\\left(\\frac{\\ln(I) - \\mu}{\\sigma}\\right)

    where :math:`\\Phi` is the standard normal CDF, :math:`\\mu` is the
    log-median failure threshold, and :math:`\\sigma` is the log-standard
    deviation.

    The class is **immutable** — modifier methods return new instances,
    supporting the "Minimize" dimension of the M-A-R-C (Mitigation,
    Adaptation, Resilience, Coping) framework by allowing :math:`\\mu`
    and :math:`\\sigma` to be dynamically shifted to represent structural
    hardening upgrades.

    Attributes
    ----------
    function_id : str
        Unique identifier for this fragility curve.
    name : str
        Human-readable label (e.g., "Wood pole — TC winds").
    hazard_type : str
        Peril code (``"TC"``, ``"WF"``, ``"EQ"``, ``"FL"``, etc.).
    intensity_unit : str
        Physical unit of the intensity measure (``"m/s"``, ``"m"``, ``"g"``).
    mu : float
        Log-median failure threshold.  :math:`\\exp(\\mu)` is the
        intensity at which the failure probability reaches 50 %.
    sigma : float
        Log-standard deviation controlling the steepness of the curve.
        Must be strictly positive.
    """

    function_id: str
    name: str
    hazard_type: str
    intensity_unit: str
    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(
                f"ImpactFunction.sigma must be positive, got {self.sigma}"
            )

    def evaluate_failure_probability(self, intensity: float) -> float:
        """Evaluate the failure probability at a single intensity value.

        Parameters
        ----------
        intensity : float
            Hazard intensity.  Values ≤ 0 are clamped to a small
            epsilon to avoid domain errors in the logarithm.

        Returns
        -------
        float
            Failure probability in [0, 1].
        """
        safe = max(intensity, _EPSILON)
        z = (np.log(safe) - self.mu) / self.sigma
        return float(norm.cdf(z))

    def evaluate_batch(self, intensities: np.ndarray) -> np.ndarray:
        """Vectorized evaluation over an array of intensity values.

        Parameters
        ----------
        intensities : np.ndarray
            1-D array of hazard intensity values.

        Returns
        -------
        np.ndarray
            Failure probabilities with the same shape as *intensities*.
        """
        intensities = np.asarray(intensities, dtype=np.float64)
        safe = np.maximum(intensities, _EPSILON)
        z = (np.log(safe) - self.mu) / self.sigma
        return norm.cdf(z)

    def shift_mu(self, delta: float) -> ImpactFunction:
        """Return a new ``ImpactFunction`` with :math:`\\mu` shifted by *delta*.

        Positive *delta* increases the log-median threshold (higher
        resistance / hardening).  Negative *delta* reduces it (weakening).

        Parameters
        ----------
        delta : float
            Additive offset applied to ``mu``.

        Returns
        -------
        ImpactFunction
            A new instance with ``mu + delta``.
        """
        return ImpactFunction(
            function_id=self.function_id,
            name=self.name,
            hazard_type=self.hazard_type,
            intensity_unit=self.intensity_unit,
            mu=self.mu + delta,
            sigma=self.sigma,
        )

    def scale_sigma(self, factor: float) -> ImpactFunction:
        """Return a new ``ImpactFunction`` with :math:`\\sigma` scaled by *factor*.

        A factor < 1 reduces uncertainty (steeper curve), representing
        more uniform construction quality.  A factor > 1 increases
        uncertainty (shallower curve).

        Parameters
        ----------
        factor : float
            Multiplicative factor applied to ``sigma``.  Must be
            positive.

        Returns
        -------
        ImpactFunction
            A new instance with ``sigma * factor``.

        Raises
        ------
        ValueError
            If *factor* is not positive.
        """
        if factor <= 0:
            raise ValueError(f"scale_sigma factor must be positive, got {factor}")
        return ImpactFunction(
            function_id=self.function_id,
            name=self.name,
            hazard_type=self.hazard_type,
            intensity_unit=self.intensity_unit,
            mu=self.mu,
            sigma=self.sigma * factor,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns
        -------
        dict
            Dictionary with keys matching the dataclass fields.
        """
        return {
            "function_id": self.function_id,
            "name": self.name,
            "hazard_type": self.hazard_type,
            "intensity_unit": self.intensity_unit,
            "mu": self.mu,
            "sigma": self.sigma,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ImpactFunction:
        """Construct from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with keys matching the dataclass fields.

        Returns
        -------
        ImpactFunction
        """
        return cls(**data)


class ImpactFunctionSet(MutableMapping[str, ImpactFunction]):
    """A dict-like container mapping ``function_id`` to ``ImpactFunction``.

    Supports standard dictionary operations (get, set, delete, iteration,
    length) plus bulk import/export to JSON files and pandas DataFrames.

    Parameters
    ----------
    functions : iterable of ImpactFunction, optional
        Initial set of functions to populate the container.

    Examples
    --------
    >>> func = ImpactFunction("tc_wood", "Wood pole TC", "TC", "m/s", mu=3.5, sigma=0.4)
    >>> ifs = ImpactFunctionSet([func])
    >>> ifs["tc_wood"] is func
    True
    >>> df = ifs.to_dataframe()
    """

    def __init__(self, functions: Optional[list[ImpactFunction]] = None) -> None:
        self._store: Dict[str, ImpactFunction] = {}
        if functions:
            for func in functions:
                self._store[func.function_id] = func

    # ------------------------------------------------------------------
    # MutableMapping abstract methods
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> ImpactFunction:
        return self._store[key]

    def __setitem__(self, key: str, value: ImpactFunction) -> None:
        if key != value.function_id:
            raise ValueError(
                f"Key '{key}' does not match ImpactFunction.function_id "
                f"'{value.function_id}'"
            )
        self._store[key] = value

    def __delitem__(self, key: str) -> None:
        del self._store[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"ImpactFunctionSet({list(self._store.keys())})"

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def add(self, func: ImpactFunction) -> None:
        """Add an ``ImpactFunction`` to the set.

        Parameters
        ----------
        func : ImpactFunction
            The function to add.  Its ``function_id`` is used as the key.
        """
        self[func.function_id] = func

    def get_by_hazard_type(self, hazard_type: str) -> list[ImpactFunction]:
        """Retrieve all functions for a given hazard type.

        Parameters
        ----------
        hazard_type : str
            Peril code to filter by.

        Returns
        -------
        list of ImpactFunction
        """
        return [f for f in self._store.values() if f.hazard_type == hazard_type]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Export all functions as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``function_id``, ``name``, ``hazard_type``,
            ``intensity_unit``, ``mu``, ``sigma``.
        """
        records = [f.to_dict() for f in self._store.values()]
        return pd.DataFrame(records)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> ImpactFunctionSet:
        """Construct an ``ImpactFunctionSet`` from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns ``function_id``, ``name``,
            ``hazard_type``, ``intensity_unit``, ``mu``, ``sigma``.

        Returns
        -------
        ImpactFunctionSet
        """
        functions = [ImpactFunction.from_dict(row) for row in df.to_dict("records")]
        return cls(functions)

    def to_json(self, path: str | Path) -> None:
        """Write all functions to a JSON file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        records = [f.to_dict() for f in self._store.values()]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, path: str | Path) -> ImpactFunctionSet:
        """Construct an ``ImpactFunctionSet`` from a JSON file.

        Parameters
        ----------
        path : str or Path
            Path to a JSON file containing a list of function dicts.

        Returns
        -------
        ImpactFunctionSet
        """
        with open(path, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        functions = [ImpactFunction.from_dict(r) for r in records]
        return cls(functions)
