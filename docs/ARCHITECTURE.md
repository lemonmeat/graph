# Architecture

This is a working draft that grows with each phase; Phase 7 finalises it. The roadmap is `PLAN.md`, and the design rationale is `DECISIONS.md`.

## Principles

- The library (`src/meagraph`) has no GUI imports, no global state and no hardcoded paths. Functions take in-memory objects and return in-memory objects.
- SpikeInterface objects are the currency for continuous data (`BaseRecording`). `meagraph.SpikeTrains` is the currency for spike times.
- **Time:** seconds on the MCS recording clock (D5).
- **Electrode identity:** the electrode label string (`"47"`), with probe coordinates attached.
- **Geometry and parameters are data** (probe YAML + CSV, config YAML). Outputs record their config and provenance.

## Data flow (implemented through Phase 1)

```
recording.h5 ──► io.McsH5Recording ──► probe.attach_probe ──► Session.recording (lazy, 3D probe, t_start)
      │                                      ▲
      │                       probe.load_probe_spec (YAML + CSV map)
      ├────────► io.read_stim_events ──► Session.stim (onsets/offsets; site from config)
      ├────────► io.read_mcs_spikes  ──► SpikeTrains (MCS online detections, comparison only)
      └────────► io.inspect_file     ──► FileInventory (`meagraph info`)
legacy *.spikes.h5 ──► io.read_spikes_sidecar ──► LegacySpikes (regression baseline)
```

Phase 2 adds `preprocess` and `detect` between `Session.recording` and `SpikeTrains`. Later phases consume `SpikeTrains` and `Session.stim`.

## Modules

| Module | Status | Responsibility |
|---|---|---|
| `io._mcs_layout` | done | MCS layout rules: stream discovery and lineage, label decoding, time stamps |
| `io.mcs_h5` | done | `McsH5Recording(BaseRecording)` (D1) |
| `io.mcs_events` | done | `EventEntity`, `StimEvents`, `read_stim_events`, `read_mcs_spikes` (D4) |
| `io.inventory` | done | `inspect_file`, `format_inventory` |
| `io.legacy` | done | `read_spikes_sidecar` for `spikes.py` output |
| `io.session` | done | `Session`, `load_session(SessionConfig)` |
| `probe` | done | `ProbeSpec` (data), `build_probe`, `attach_probe`, 3D positions and distances (D3, D8) |
| `config` | done | `SessionConfig`, YAML I/O, `collect_provenance`, `write_run_folder` |
| `spiketrains` | done | `SpikeTrains` with SpikeInterface `Sorting` conversion |
| `cli` | partial | `meagraph info`, `meagraph probes` |
| `preprocess`, `detect`, `stimulation` | Phase 2 | |
| `viz`, `apps/viewer` | Phase 3 | |
| `connectivity`, `synth`, `benchmark` | Phases 4–5 | |
| `reservoir`, `realtime` | Phase 7 | |

## Extension points

**A new electrode layout** needs files, not code:

1. Add `probe/data/<name>.yaml` with `ndim`, pitches, layer spacing, contact radius, reference labels and `geometry_verified`.
2. Add a CSV map `label,row,col,layer` (layer = 1 for planar).
3. Select it with `load_session(path, probe="<name>")`.

A spec can also live outside the package; pass its YAML path instead of a name.

**Another MCS stream.** Analog streams are selected by label (`stream="Filter (3)"`), or by `"raw"` for the hardware stream. Segment streams are selected by label (`"Spike Detector"`, `"Spike Sorter"`). New entity types belong in `io/mcs_events.py` and must be decoded by their ID, not by table row.

**Connectivity estimators, encoders and decoders** get their interfaces in Phases 4 and 7.

## Testing

`pytest` runs:

- **Unit tests** on small generated files in the MCS layout (`tests/mcs_fixture.py`). The fixture includes the traps found in Phase 0: a permuted `RowIndex`, non-zero ADZero, a filter stream stored before raw, shuffled InfoEvent rows, and sorter SegmentIDs out of row order.
- **Real-file tests**, marked `data`, which run when the recordings are in the repo root or in `$MEAGRAPH_DATA_DIR`.
- **A baseline guard:** a checksum test on the committed regression baseline in `tests/data/legacy_baseline/`.
