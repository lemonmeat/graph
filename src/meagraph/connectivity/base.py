"""The shared interface of every connectivity estimator.

An estimator turns ``SpikeTrains`` into a ``ConnectivityResult``. Adding a method means adding
one class with a ``name``, a pydantic ``Config`` and an ``estimate`` method, decorated with
``@register``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Protocol

import numpy as np
from pydantic import BaseModel

from meagraph.spiketrains import SpikeTrains


@dataclass(frozen=True, eq=False)
class ConnectivityResult:
    """Pairwise connectivity among ``node_ids``. Matrices are (source row, target column).

    Entries for pairs that were not tested (too few spikes, or the diagonal) are NaN in
    ``weights``, ``delays_ms`` and ``p_values`` and False in ``significant``.

    ``signed`` results (methods that also detect inhibition) mark both kinds of edge in
    ``significant``; the sign of the weight says which (negative = inhibitory).

    Methods without a significance test (``cch_gauss2``) leave ``p_values`` NaN and give the
    tested pairs in ``tested_mask`` instead.
    """

    method: str
    node_ids: tuple[str, ...]
    weights: np.ndarray
    delays_ms: np.ndarray
    p_values: np.ndarray
    significant: np.ndarray
    directed: bool
    params: dict
    n_spikes: np.ndarray
    duration_s: float
    positions_um: np.ndarray | None = None
    extra: dict = field(default_factory=dict)  # method-specific arrays, e.g. correlograms
    signed: bool = False
    tested_mask: np.ndarray | None = None

    @property
    def tested(self) -> np.ndarray:
        return self.tested_mask if self.tested_mask is not None else ~np.isnan(self.p_values)

    def edges(self) -> list[dict]:
        """Significant edges, strongest first."""
        rows = []
        for i, j in zip(*np.nonzero(self.significant)):
            if not self.directed and j < i:
                continue
            rows.append(dict(source=self.node_ids[i], target=self.node_ids[j], weight=float(self.weights[i, j]),
                             delay_ms=float(self.delays_ms[i, j]), p_value=float(self.p_values[i, j])))  # fmt: skip
        return sorted(rows, key=lambda r: -abs(r["weight"]))


class Estimator(Protocol):
    name: ClassVar[str]
    Config: ClassVar[type[BaseModel]]

    def estimate(self, trains: SpikeTrains, config: BaseModel | None = None) -> ConnectivityResult: ...


ESTIMATORS: dict[str, type] = {}


def register(cls):
    ESTIMATORS[cls.name] = cls
    return cls


def estimate(trains: SpikeTrains, method: str, config: BaseModel | dict | None = None) -> ConnectivityResult:
    """Run a registered estimator by name; ``config`` may be the method's Config or a dict of overrides."""
    if method not in ESTIMATORS:
        raise KeyError(f"unknown method {method!r}; available: {sorted(ESTIMATORS)}")
    cls = ESTIMATORS[method]
    if isinstance(config, dict):
        config = cls.Config(**config)
    return cls().estimate(trains, config or cls.Config())


def empty_matrices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nan = np.full((n, n), np.nan)
    return nan.copy(), nan.copy(), nan.copy(), np.zeros((n, n), dtype=bool)


def testable(trains: SpikeTrains, min_spikes: int) -> np.ndarray:
    """(n, n) mask of ordered pairs whose trains both have enough spikes; diagonal excluded."""
    ok = trains.n_spikes() >= min_spikes
    mask = ok[:, None] & ok[None, :]
    np.fill_diagonal(mask, False)
    return mask
