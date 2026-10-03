"""A session bundles one recording with its probe, stimulation events and file inventory."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from spikeinterface.core import BaseRecording

from meagraph.config.models import SessionConfig
from meagraph.io.inventory import FileInventory, inspect_file
from meagraph.io.mcs_events import StimEvents, read_stim_events
from meagraph.io.mcs_h5 import McsH5Recording
from meagraph.probe.build import attach_probe
from meagraph.probe.spec import ProbeSpec, load_probe_spec


@dataclass(frozen=True, eq=False)
class Session:
    config: SessionConfig
    recording: BaseRecording  # probe attached; channels not on the probe removed
    probe_spec: ProbeSpec
    excluded_channel_ids: tuple[str, ...]  # e.g. the reference electrode
    stim: tuple[StimEvents, ...]
    inventory: FileInventory

    @property
    def path(self) -> Path:
        return self.config.path


def _assign_sites(stim: list[StimEvents], site: str | Mapping[str, str] | None) -> list[StimEvents]:
    if site is None:
        return stim
    if isinstance(site, str):
        sources = sorted({s.source for s in stim})
        if len(sources) > 1:
            raise ValueError(f"several STG outputs {sources}; give stim_site as a mapping per output")
        return [s.with_site(site) for s in stim]
    unknown = set(site) - {s.source for s in stim}
    if unknown:
        raise KeyError(f"stim_site names STG outputs not in the file: {sorted(unknown)}")
    return [s.with_site(site.get(s.source)) for s in stim]


def load_session(config: SessionConfig | str | Path, **overrides) -> Session:
    """Open a recording lazily with its probe and stimulation events.

    ``config`` is a :class:`SessionConfig` or a path to the ``.h5`` file. Keyword overrides
    (``stream``, ``probe``, ``stim_site``, ``recording_index``) update the config.
    """
    if not isinstance(config, SessionConfig):
        config = SessionConfig(path=Path(config))
    if overrides:
        config = config.model_copy(update=overrides)
    spec = load_probe_spec(config.probe)
    raw = McsH5Recording(config.path, stream=config.stream, recording_index=config.recording_index)
    recording, excluded = attach_probe(raw, spec)
    stim = _assign_sites(read_stim_events(config.path, config.recording_index), config.stim_site)
    for s in stim:
        if s.site is not None and s.site not in set(raw.channel_ids):
            raise ValueError(f"stimulation site {s.site!r} is not a channel of this recording")
    return Session(
        config=config,
        recording=recording,
        probe_spec=spec,
        excluded_channel_ids=excluded,
        stim=tuple(stim),
        inventory=inspect_file(config.path, config.recording_index),
    )
