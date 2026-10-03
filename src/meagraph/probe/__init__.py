"""Electrode geometry as data: probe specs (YAML + CSV map) and their ProbeInterface form."""

from meagraph.probe.build import (
    UnverifiedGeometryWarning,
    attach_probe,
    build_probe,
    channel_distances_um,
    channel_positions_um,
)
from meagraph.probe.spec import GridContact, ProbeSpec, available_probe_specs, load_probe_spec

__all__ = [
    "GridContact",
    "ProbeSpec",
    "UnverifiedGeometryWarning",
    "attach_probe",
    "available_probe_specs",
    "build_probe",
    "channel_distances_um",
    "channel_positions_um",
    "load_probe_spec",
]
