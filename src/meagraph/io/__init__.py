"""Reading MCS HDF5 recordings, stimulation events, MCS spike streams and legacy outputs."""

from meagraph.io.inventory import FileInventory, format_inventory, inspect_file
from meagraph.io.legacy import LegacySpikes, read_spikes_sidecar, sidecar_path
from meagraph.io.mcs_events import (
    EventEntity,
    StimEvents,
    read_event_entities,
    read_mcs_spikes,
    read_stim_events,
)
from meagraph.io.mcs_h5 import McsH5Recording, read_mcs_h5
from meagraph.io.session import Session, load_session

__all__ = [
    "EventEntity",
    "FileInventory",
    "LegacySpikes",
    "McsH5Recording",
    "Session",
    "StimEvents",
    "format_inventory",
    "inspect_file",
    "load_session",
    "read_event_entities",
    "read_mcs_h5",
    "read_mcs_spikes",
    "read_spikes_sidecar",
    "read_stim_events",
    "sidecar_path",
]
