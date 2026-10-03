"""Spike trains on the recording clock, the common currency between detection and analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, eq=False)
class SpikeTrains:
    """Spike times per unit, in seconds on the MCS recording clock (DECISIONS.md D5).

    For threshold detection each unit is one electrode, so ``unit_ids == channel_ids``.
    For sorted data several units can share an electrode; ``channel_ids`` says which.
    """

    unit_ids: tuple[str, ...]
    times_s: tuple[np.ndarray, ...]
    t_start_s: float
    t_stop_s: float
    channel_ids: tuple[str, ...] = field(default=())
    positions_um: np.ndarray | None = None

    def __post_init__(self) -> None:
        unit_ids = tuple(str(u) for u in self.unit_ids)
        times = tuple(np.asarray(t, dtype=np.float64).ravel() for t in self.times_s)
        channel_ids = tuple(str(c) for c in self.channel_ids) if self.channel_ids else unit_ids
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("unit_ids must be unique")
        if len(times) != len(unit_ids) or len(channel_ids) != len(unit_ids):
            raise ValueError("unit_ids, times_s and channel_ids must have the same length")
        if not self.t_stop_s > self.t_start_s:
            raise ValueError(f"t_stop_s ({self.t_stop_s}) must be after t_start_s ({self.t_start_s})")
        for u, t in zip(unit_ids, times):
            if t.size and np.any(np.diff(t) < 0):
                raise ValueError(f"spike times of unit {u!r} are not sorted")
            if t.size and (t[0] < self.t_start_s or t[-1] > self.t_stop_s):
                raise ValueError(f"spike times of unit {u!r} fall outside [t_start_s, t_stop_s]")
        positions = None
        if self.positions_um is not None:
            positions = np.asarray(self.positions_um, dtype=np.float64)
            if positions.ndim != 2 or positions.shape[0] != len(unit_ids):
                raise ValueError("positions_um must have shape (n_units, n_dims)")
        object.__setattr__(self, "unit_ids", unit_ids)
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "channel_ids", channel_ids)
        object.__setattr__(self, "positions_um", positions)

    @classmethod
    def from_dict(
        cls,
        times_s: Mapping[str, ArrayLike],
        t_start_s: float,
        t_stop_s: float,
        channel_ids: Mapping[str, str] | None = None,
    ) -> SpikeTrains:
        """Build from ``{unit_id: times}``; times are sorted, unit order is kept."""
        unit_ids = tuple(times_s)
        times = tuple(np.sort(np.asarray(times_s[u], dtype=np.float64).ravel()) for u in unit_ids)
        chans = tuple(channel_ids[u] for u in unit_ids) if channel_ids else ()
        return cls(unit_ids, times, float(t_start_s), float(t_stop_s), chans)

    @property
    def duration_s(self) -> float:
        return self.t_stop_s - self.t_start_s

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)

    def n_spikes(self) -> np.ndarray:
        return np.array([t.size for t in self.times_s], dtype=np.int64)

    def rates_hz(self) -> np.ndarray:
        return self.n_spikes() / self.duration_s

    def as_dict(self) -> dict[str, np.ndarray]:
        return dict(zip(self.unit_ids, self.times_s))

    def select(self, unit_ids: Sequence[str]) -> SpikeTrains:
        """Subset (and reorder) units."""
        index = {u: i for i, u in enumerate(self.unit_ids)}
        missing = [u for u in unit_ids if u not in index]
        if missing:
            raise KeyError(f"unknown unit ids: {missing}")
        idx = [index[u] for u in unit_ids]
        positions = None if self.positions_um is None else self.positions_um[idx]
        return SpikeTrains(
            tuple(self.unit_ids[i] for i in idx),
            tuple(self.times_s[i] for i in idx),
            self.t_start_s,
            self.t_stop_s,
            tuple(self.channel_ids[i] for i in idx),
            positions,
        )

    def with_positions(self, positions_um: ArrayLike) -> SpikeTrains:
        return SpikeTrains(
            self.unit_ids, self.times_s, self.t_start_s, self.t_stop_s, self.channel_ids, np.asarray(positions_um)
        )

    def to_sorting(self, sampling_frequency_hz: float):
        """SpikeInterface ``NumpySorting``; sample 0 is ``t_start_s`` (the recording's first sample)."""
        from spikeinterface.core import NumpySorting

        units = {
            u: np.round((t - self.t_start_s) * sampling_frequency_hz).astype(np.int64)
            for u, t in zip(self.unit_ids, self.times_s)
        }
        sorting = NumpySorting.from_unit_dict([units], sampling_frequency=sampling_frequency_hz)
        sorting.set_property("channel_id", np.array(self.channel_ids))
        return sorting

    @classmethod
    def from_sorting(cls, sorting, t_start_s: float, t_stop_s: float) -> SpikeTrains:
        """Inverse of :meth:`to_sorting` for a single-segment SpikeInterface sorting."""
        fs = sorting.get_sampling_frequency()
        unit_ids = tuple(str(u) for u in sorting.unit_ids)
        times = tuple(t_start_s + sorting.get_unit_spike_train(u, segment_index=0) / fs for u in sorting.unit_ids)
        chans = ()
        if "channel_id" in sorting.get_property_keys():
            chans = tuple(str(c) for c in sorting.get_property("channel_id"))
        return cls(unit_ids, times, t_start_s, t_stop_s, chans)
