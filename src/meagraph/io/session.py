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
from meagraph.io.stimlog import Alignment, StimLog, align_stimlog, find_stimlog, read_stimlog
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
    stimlog: StimLog | None = None
    alignment: tuple[Alignment, ...] = ()

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
    (``stream``, ``probe``, ``stim_site``, ``stimlog``, ``recording_index``) update the config.
    Stimulation sites come from ``stim_site`` (one per output) or from a stimulation log (one per
    pulse; found next to the recording unless ``stimlog`` is None), never both.
    """
    if not isinstance(config, SessionConfig):
        config = SessionConfig(path=Path(config))
    if overrides:
        config = config.model_copy(update=overrides)
    spec = load_probe_spec(config.probe)
    raw = McsH5Recording(config.path, stream=config.stream, recording_index=config.recording_index)
    recording, excluded = attach_probe(raw, spec)
    stim = _assign_sites(read_stim_events(config.path, config.recording_index), config.stim_site)
    log, alignment = None, ()
    log_path = None
    if config.stimlog == "auto":
        log_path = find_stimlog(config.path)
    elif config.stimlog is not None:
        log_path = Path(config.stimlog)
    if log_path is not None:
        if config.stim_site is not None:
            raise ValueError(f"both stim_site and a stimulation log ({log_path.name}) give sites; use one (--stimlog none ignores the log)")
        log = read_stimlog(log_path)
        stim, reports = align_stimlog(stim, log)
        alignment = tuple(reports)
    channels = set(raw.channel_ids)
    for s in stim:
        sites = s.sites
        unknown = sorted({x for x in sites if x and x not in channels}) if sites is not None else []
        if unknown:
            raise ValueError(f"stimulation site(s) {unknown} ({s.source}) are not channels of this recording")
    return Session(
        config=config,
        recording=recording,
        probe_spec=spec,
        excluded_channel_ids=excluded,
        stim=tuple(stim),
        inventory=inspect_file(config.path, config.recording_index),
        stimlog=log,
        alignment=alignment,
    )
