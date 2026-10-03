# Architecture

This is a working draft that grows with each phase; Phase 7 finalises it. The roadmap is `PLAN.md`, and the design rationale is `DECISIONS.md`.

## Principles

- The library (`src/meagraph`) has no GUI imports, no global state and no hardcoded paths. Functions take in-memory objects and return in-memory objects.
- SpikeInterface objects are the currency for continuous data (`BaseRecording`). `meagraph.SpikeTrains` is the currency for spike times.
- **Time:** seconds on the MCS recording clock (D5).
- **Electrode identity:** the electrode label string (`"47"`), with probe coordinates attached.
- **Geometry and parameters are data** (probe YAML + CSV, config YAML). Outputs record their config and provenance.

## Data flow (implemented through Phase 3)

```
recording.h5 ──► io.load_session ──► Session
                   │  McsH5Recording (lazy, t_start) + probe.attach_probe (3D, grid properties)
                   │  read_stim_events (onsets/offsets; site from config, never from the file)
                   ▼
       stimulation.measure_recovery ──► per-channel blanking windows      (default profile)
       stimulation.fixed_windows    ──► shared legacy windows             (legacy profile)
                   ▼
       preprocess.detection_band: scale_to_uV → InterpolateWindowsRecording → bandpass (SI)
                   ▼
       detect.median_abs_noise_uv ─► SI detect_peaks(by_channel, both signs)
                   ▼                     ├─ negative peaks → guard, cap, rebound → SpikeTrains + waveforms
                   ▼                     └─ all peaks → polarity-control events → ChannelQC (active channels)
       detect.save_detection ──► results/<recording>/detect_<profile>/ (npz, csv, config, provenance)
                   ▼
       viewer (meagraph view) ◄── load_detection, or a legacy *.spikes.h5 sidecar
```

Later phases consume `SpikeTrains`, `DetectionResult.active_channels` and `Session.stim`.

## Modules

| Module | Status | Responsibility |
|---|---|---|
| `io` | done | MCS reader (D1), events (D4), MCS spike streams, inventory, legacy sidecars, `Session` |
| `probe` | done | `ProbeSpec` data files, ProbeInterface probes, 3D positions and distances (D3, D8) |
| `config` | done | `SessionConfig`, YAML I/O, provenance, run folders |
| `spiketrains` | done | `SpikeTrains` with SpikeInterface `Sorting` conversion |
| `preprocess` | done | `InterpolateWindowsRecording` (D9), `detection_band` |
| `stimulation` | done | fixed and per-pulse windows, `measure_recovery`, `infer_site`, `audit_stimulation` |
| `detect` | done | `DetectionConfig` profiles, `detect_spikes`, polarity QC (D10), noise, save/load |
| `viz` | done | `plot_cube_map`, `plot_raster`, `plot_trace`, `plot_waveforms` (matplotlib, no I/O) |
| `viewer` | done | interactive viewer (D12), `meagraph view`; never imported by the core |
| `cli` | done | `info`, `probes`, `audit`, `detect`, `view` |
| `connectivity`, `synth`, `benchmark` | Phases 4–5 | |
| `reservoir`, `realtime` | Phase 7 | |

## Extension points

**A new electrode layout** needs files, not code:

1. Add `probe/data/<name>.yaml` with `ndim`, pitches, layer spacing, contact radius, reference labels and `geometry_verified`.
2. Add a CSV map `label,row,col,layer` (layer = 1 for planar).
3. Select it with `load_session(path, probe="<name>")`.

A spec can also live outside the package; pass its YAML path instead of a name.

**Another MCS stream.** Analog streams are selected by label (`stream="Filter (3)"`), or by `"raw"` for the hardware stream. Segment streams are selected by label (`"Spike Detector"`, `"Spike Sorter"`). New entity types belong in `io/mcs_events.py` and must be decoded by their ID, not by table row.

**A detection setting.** Every parameter is a field of `DetectionConfig`. Pass a modified copy: `PROFILES["default"].model_copy(update={...})`. Results record the config they were made with.

**A figure.** Add a function to `meagraph.viz` that takes data and an optional `ax` and returns artists. The viewer and notebooks can then share it.

**Connectivity estimators, encoders and decoders** get their interfaces in Phases 4 and 7.

## Testing

`pytest` runs:

- **Unit tests** on small generated files in the MCS layout (`tests/mcs_fixture.py`). The fixture includes the traps found in Phase 0: a permuted `RowIndex`, non-zero ADZero, a filter stream stored before raw, shuffled InfoEvent rows, and sorter SegmentIDs out of row order.
- **Real-file tests**, marked `data`, which run when the recordings are in the repo root or in `$MEAGRAPH_DATA_DIR`.
- **A baseline guard:** a checksum test on the committed regression baseline in `tests/data/legacy_baseline/`.
