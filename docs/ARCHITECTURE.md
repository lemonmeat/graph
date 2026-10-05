# Architecture

How the `meagraph` package is organised, what each part is responsible for, and where new code goes. The roadmap is `PLAN.md`; the reasons behind individual choices are in `DECISIONS.md`. Phase 7 finalises this document.

## Principles

- **Library and apps are separate.** `src/meagraph` is a library: no GUI imports, no global state, no hardcoded paths. Functions take in-memory objects and return in-memory objects. Two thin apps sit on top: the command line (`cli.py`) and the viewer (`viewer.py`). A test enforces that the core never loads matplotlib.
- **Three currencies.** Continuous data is a SpikeInterface `BaseRecording`. Spikes are a `meagraph.SpikeTrains`. Connectivity is a `ConnectivityResult`. Each step consumes one of these and produces the next.
- **One clock.** Times are seconds on the MCS recording clock (D5), the same clock as the stimulation events.
- **Stable identity.** An electrode is its MCS label (`"47"`), and its 3D position travels with it: on the recording (probe), on the spike trains (`positions_um`), and on graph nodes.
- **Geometry and parameters are data.** Probes are YAML + CSV files. Every analysis step has a typed pydantic config. Every output folder stores its config and its provenance (code version, git commit, input files).
- **Saved results are complete.** A detection folder contains everything later steps need, including electrode positions and stimulation events. Graphs and later analyses never reopen the raw `.h5` file.

## Layers

Each layer imports only from layers below it. There are no cycles.

```
 apps          cli.py                 viewer.py ── viz (plotting functions, matplotlib)
                 │                       │
 validation    benchmark ── synth (Hawkes networks with known connections)
                 │
 graphs        connectivity.pipeline  (detection result → burst-controlled graphs, GraphConfig)
                 │            │
 estimators      │      connectivity  (cch_hollow, cch_jitter, sttc; ConnectivityResult; graph files)
                 │            │         depends only on SpikeTrains: any spike source can feed it
 detection     detect   (threshold detection, QC, network bursts, result folders)
                 │
 stimulation   stimulation  (blanking windows, artifact recovery, site inference, audit, stimulation periods)
                 │
 signal        preprocess   (window bridging, detection band)
                 │
 input         io  (MCS reader, events, sessions, inventory, legacy sidecars)
                 │
 foundation    probe   config   spiketrains   intervals
```

## Data flow

```
recording.h5 ─► io.load_session ─► Session
                   McsH5Recording (lazy, chunked) + probe.attach_probe (3D positions, grid indices)
                   StimEvents per STG output (site from --stim-site; never in the file)
                     │
                     ▼
               detect.detect_spikes(recording, stim, DetectionConfig)
                   stimulation.measure_recovery ─► per-channel blanking windows
                   preprocess.detection_band ─► SI detect_peaks ─► spikes + polarity QC
                     │
                     ▼
               DetectionResult ─► detect.save_detection ─► results/<recording>/detect_<profile>/
                     │                                      (meagraph view reads this too)
                     ▼
               connectivity.pipeline.graphs_from_detection(detection, methods, GraphConfig)
                   drop stimulation periods (D18) ─► detect.network_bursts (D15)
                   estimate(...) on all spikes and without bursts ─► BurstControlled
                     │
                     ▼
               save_burst_controlled ─► results/<recording>/graph_<method>/{all,no_bursts,robust}/

synth.simulate_network ─► SpikeTrains with known W ─► benchmark.run_benchmark ─► scores per method
```

## Core objects

| Object | Module | What it is |
|---|---|---|
| `Session` | `io.session` | One recording with its probe attached, stimulation events and file inventory. Built by `load_session`. |
| `StimEvents` | `io.mcs_events` | Pulse onsets and offsets of one STG output, plus the stimulated `site` if known. |
| `ProbeSpec` | `probe.spec` | An electrode layout loaded from YAML + CSV. `DEFAULT_PROBE` names the 4×4×4 cube. |
| `SpikeTrains` | `spiketrains` | Spike times per electrode, the recording span, optional positions. Converts to and from SpikeInterface sortings. |
| `DetectionConfig` / `DetectionResult` | `detect.threshold` | Every detection parameter; spikes, waveforms, noise, QC, excluded channels, recovery times and the stimulation events. |
| `BurstConfig` | `detect.bursts` | ISI_N network-burst parameters. |
| `ConnectivityResult` | `connectivity.base` | Weight, delay, p-value and significance matrices (source row, target column) for one method. NaN marks untested pairs. |
| `GraphConfig` / `BurstControlled` | `connectivity.pipeline` | How detection becomes graphs (channels, stimulation exclusion, bursts); the three graph versions. |
| `NetworkConfig` / `SyntheticNetwork` | `synth.hawkes` | A simulated network and its ground truth. |

Every config is a frozen pydantic model next to the step that uses it, saved as `config.yaml` in that step's output folder. `config.SessionConfig` is the one exception for historical reasons.

## Modules

| Module | Responsibility |
|---|---|
| `io` | `McsH5Recording` (D1), event and segment decoding by ID (D4), MCS online spike streams (comparison only, D2), `inspect_file` inventory, legacy `*.spikes.h5` reader, `load_session`. `_mcs_layout.py` holds the HDF5 layout rules. |
| `probe` | Probe specs (`probe/data/*.yaml` + map CSV), ProbeInterface probes, `attach_probe`, 3D positions and distances (D3, D8). |
| `config` | YAML ↔ pydantic, provenance, `write_run_folder`. |
| `spiketrains`, `intervals` | `SpikeTrains` (select, positions, `without_periods`, SI conversion); merging and membership of `[start, stop]` intervals. |
| `preprocess` | `InterpolateWindowsRecording` (D9) and the `detection_band` chain. |
| `stimulation` | Fixed and per-pulse blanking windows, `measure_recovery`, `infer_site`, `audit_stimulation`, `stimulation_periods`. |
| `detect` | `DetectionConfig` profiles (`default`, `legacy`), `detect_spikes`, polarity QC (D10), noise, network bursts (D15), result folders. |
| `connectivity` | Estimator registry and `estimate`, `cch_hollow`, `cch_jitter`, `sttc`, surrogates and fitted p-values (D17), FDR, graph files. `pipeline.py` connects it to detection results. |
| `synth` | Linear Hawkes network with known connections and optional confounds (D16). |
| `benchmark` | Scenarios, scoring (including indirect false positives), parallel runner. |
| `viz` | Plotting functions: data and an optional `ax` in, artists out, no file I/O. |
| `viewer` | Interactive viewer (D12), `meagraph view`. Nothing else imports it. |
| `cli` | `info`, `probes`, `audit`, `detect`, `graph`, `benchmark`, `view`. Each command is a thin wrapper over library calls. |
| `realtime`, `reservoir` | Empty placeholders for Phase 7. |

## Output folders

Results go to `results/<recording name>/` in the recording's folder, so with the recordings in `data/` they are in `data/results/`. Benchmark output goes to `results/benchmark/` in the working directory.

```
results/<recording>/
  detect_<profile>/              meagraph detect
    spikes.npz                   spike times, amplitudes, waveforms (flat arrays)
    channels.csv                 per electrode: position, rate, noise, QC, exclusion, artifact recovery
    stimulation.csv              per pulse: STG output, onset, offset, site (header only if spontaneous)
    config.yaml, provenance.json
  graph_<method>/                meagraph graph
    network_bursts.csv           burst periods removed for no_bursts/
    all/, no_bursts/, robust/    each: graph.graphml, graph.json, edges.csv, matrices.npz, result.json,
                                 config.yaml (estimator params + GraphConfig), provenance.json (input: spikes.npz)
```

## Extension points

**A new electrode layout** needs files, not code:

1. Add `probe/data/<name>.yaml` with `ndim`, pitches, layer spacing, contact radius, reference labels and `geometry_verified`.
2. Add a CSV map `label,row,col,layer` (layer = 1 for planar).
3. Select it with `load_session(path, probe="<name>")` or `meagraph detect --probe <name>`. Positions then flow through detection into the graphs.

A spec can also live outside the package; pass its YAML path instead of a name.

**Another MCS stream.** Analog streams are selected by label (`stream="Filter (3)"`), or by `"raw"` for the hardware stream. Segment streams are selected by label (`"Spike Detector"`, `"Spike Sorter"`). New entity types belong in `io/mcs_events.py` and must be decoded by their ID, not by table row.

**A detection setting.** Every parameter is a field of `DetectionConfig`. Pass a modified copy: `PROFILES["default"].model_copy(update={...})`. Results record the config they were made with.

**A connectivity estimator** (Phase 5). Add one class to `meagraph/connectivity/` with:

- `name`;
- a pydantic `Config`;
- `estimate(trains, config) -> ConnectivityResult`, with matrices as (source row, target column) and NaN for untested pairs.

Decorate it with `@register` and import it in `connectivity/__init__.py`. It then works with `estimate(...)`, `meagraph graph --method` and the benchmark without further changes. An estimator that needs more than spike times (for example stimulus-evoked mapping) reads them from the `DetectionResult`, which carries the stimulation events.

**A figure.** Add a function to `meagraph.viz` that takes data and an optional `ax` and returns artists. The viewer, notebooks and report figures can then share it.

**A command.** Add a `_name(args)` function and a subparser in `cli.py`. Import library modules inside the function, so `meagraph --help` stays fast.

**Encoders and decoders** get their interfaces in Phase 7.

## Testing

`pytest` runs:

- **Unit tests** on small generated files in the MCS layout (`tests/mcs_fixture.py`). The fixture includes the traps found in Phase 0: a permuted `RowIndex`, non-zero ADZero, a filter stream stored before raw, shuffled InfoEvent rows, and sorter SegmentIDs out of row order.
- **Ground-truth tests** of the connectivity methods on simulated networks (`tests/test_connectivity.py`).
- **Structure guards** (`tests/test_architecture.py`): the core never imports matplotlib, and the estimators never import file reading or detection.
- **Real-file tests**, marked `data`, which run when the recordings are in `data/` or in `$MEAGRAPH_DATA_DIR`. The `slow` regression test compares the `legacy` profile with `spikes.py`.
- **A baseline guard:** a checksum test on the committed regression baseline in `tests/data/legacy_baseline/`.
