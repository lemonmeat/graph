"""ProbeSpec -> ProbeInterface probe, attached to a SpikeInterface recording.

SpikeInterface returns 2D channel locations unless asked otherwise (DECISIONS.md D3), so
distance and position helpers here always request all axes.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from probeinterface import Probe
from spikeinterface.core import BaseRecording

from meagraph.probe.spec import ProbeSpec, load_probe_spec


class UnverifiedGeometryWarning(UserWarning):
    """The probe's physical dimensions are placeholders; distances are not physical."""


def build_probe(spec: ProbeSpec, channel_ids: Sequence[str] | None = None) -> Probe:
    """Probe with one contact per spec contact present in ``channel_ids``.

    ``device_channel_indices`` index into ``channel_ids`` (default: spec order).
    """
    labels = spec.labels
    if channel_ids is None:
        keep = list(range(len(labels)))
        device = np.arange(len(labels))
    else:
        where = {str(c): i for i, c in enumerate(channel_ids)}
        keep = [i for i, lbl in enumerate(labels) if lbl in where]
        device = np.array([where[labels[i]] for i in keep], dtype=np.int64)
    positions = spec.positions_um()[keep]
    grid = spec.grid_indices()[keep]

    probe = Probe(ndim=spec.ndim, si_units="um", name=spec.name)
    kwargs = dict(
        positions=positions,
        shapes="circle",
        shape_params={"radius": spec.contact_radius_um},
        contact_ids=[labels[i] for i in keep],
    )
    if spec.ndim == 3:
        # Contacts assumed to face +z, so each contact lies in the x-y plane (unverified, see spec notes).
        kwargs["plane_axes"] = np.tile(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (len(keep), 1, 1))
    probe.set_contacts(**kwargs)
    probe.set_device_channel_indices(device)
    probe.annotate(spec_name=spec.name, geometry_verified=spec.geometry_verified)
    probe.annotate_contacts(grid_row=grid[:, 0], grid_col=grid[:, 1], grid_layer=grid[:, 2])
    return probe


def attach_probe(recording: BaseRecording, spec: ProbeSpec | str) -> tuple[BaseRecording, tuple[str, ...]]:
    """Return a view of ``recording`` restricted to the probe's contacts, with the probe attached.

    The second value lists channels that were dropped because the spec has no contact for them
    (for example the reference electrode). Grid indices are also set as channel properties
    ``grid_row``, ``grid_col`` and ``grid_layer``.
    """
    if isinstance(spec, str):
        spec = load_probe_spec(spec)
    channel_ids = [str(c) for c in recording.channel_ids]
    probe = build_probe(spec, channel_ids)
    if probe.get_contact_count() == 0:
        raise ValueError(f"no channel of the recording matches probe {spec.name!r}")
    dropped = tuple(c for c in channel_ids if c not in set(spec.labels))
    unknown = [c for c in dropped if c not in spec.reference_labels]
    if unknown:
        warnings.warn(f"channels {unknown} are not on probe {spec.name!r} and were dropped", stacklevel=2)
    if not spec.geometry_verified:
        warnings.warn(
            f"probe {spec.name!r} uses placeholder dimensions; distances in um are not physical",
            UnverifiedGeometryWarning,
            stacklevel=2,
        )

    sub = recording.select_channels_with_probe(probe)
    index = {lbl: i for i, lbl in enumerate(spec.labels)}
    grid = spec.grid_indices()[[index[str(c)] for c in sub.channel_ids]]
    sub.set_property("grid_row", grid[:, 0])
    sub.set_property("grid_col", grid[:, 1])
    sub.set_property("grid_layer", grid[:, 2])
    sub.annotate(probe_spec=spec.name, geometry_verified=spec.geometry_verified)
    return sub, dropped


def channel_positions_um(recording: BaseRecording) -> np.ndarray:
    """(n_channels, 3) positions; planar probes get z = 0. Never use the 2D default (D3)."""
    if recording.get_probe().ndim == 3:
        return recording.get_channel_locations(axes="xyz")
    xy = recording.get_channel_locations(axes="xy")
    return np.column_stack([xy, np.zeros(len(xy))])


def channel_distances_um(recording: BaseRecording) -> np.ndarray:
    """(n_channels, n_channels) Euclidean distances in 3D."""
    pos = channel_positions_um(recording)
    return np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
