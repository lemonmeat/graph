# CLAUDE.md

Context for Claude Code sessions in this repo. Fields marked TODO are filled in by the owner. Fields marked (verified) were confirmed from the files during Phase 0 on 2026-10-03.

## Project

Analysis of extracellular recordings from in vitro neuronal cultures on a custom 3D microelectrode array, aimed at inferring functional connectivity graphs and understanding how stimulation reshapes them for reservoir computing and closed-loop biocomputation. This is Princeton ECE independent work (ECE 398), so methods must be defensible and well documented. The device design follows Kumar et al. 2026, Nat. Electron. 9:532 ("3D-MIND", `papers/Kumar2026.pdf`).

## Status

- Phase 0 (explore and verify) is done. The owner approved the plan "with defaults" on 2026-10-03.
- **Phase 1 (foundation) is done:** package `meagraph` with `io`, `probe`, `config`, `spiketrains`. See `docs/ARCHITECTURE.md`.
- **Phases 2 and 3 are done** (2026-10-03):
  - **Phase 2:** detection with `legacy` and `default` profiles, polarity QC, stimulation audit, CLI `audit`/`detect`, and a regression test against `spikes.py` that passes.
  - **Phase 3:** the viewer (`meagraph view`, custom matplotlib per D12).
- **Phase 4 is done** (2026-10-04):
  - connectivity core (`cch_hollow`, `cch_jitter`, `sttc`), with fitted-tail p-values (D17) and the burst control;
  - ISI_N network bursts;
  - Hawkes simulator and benchmark;
  - `meagraph graph` and `meagraph benchmark`.
  - First real edges in DIV142: 78→87 and 32→14 (the latter replicated in the associative file).
- **Full benchmark done** (2026-10-05): results in `docs/METHODS.md` § Validation and `docs/benchmark/benchmark.csv`. The D19 proposals await the owner.
  - It takes about 1.5 CPU-hours: run `caffeinate -i meagraph benchmark --n-jobs 4`.
  - macOS throttles long background jobs about 100× when idle. That, not the code, was why a run stalled overnight.
- **Architecture cleanup done** (2026-10-05, D20): layer map and output folder formats in `docs/ARCHITECTURE.md`.
  - Detection folders are self-contained (positions + `stimulation.csv`); `meagraph graph` reads only them and records a `GraphConfig`.
  - Shared helpers: `meagraph.intervals`, `SpikeTrains.without_periods`, `StimEvents.ends_s`, `stimulation.stimulation_periods`, `probe.DEFAULT_PROBE`, `ProbeSpec.positions_xyz_um`.
  - `tests/test_architecture.py` guards the layering (no matplotlib in the core; estimators independent of io/detect).
  - Pending owner OK (file deletions were blocked): turn `benchmark/` and `viz/` packages into single modules, delete the empty `realtime/` and `reservoir/` stubs, move `SessionConfig` from `config/models.py` into `io/session.py`.
- **Next is Phase 5:** more methods, namely GLM, stimulus-evoked, CFP, and possibly Elephant's TSPE (`elephant.functional_connectivity.total_spiking_probability_edges`).
- Decisions D1–D17 are Accepted. D18 (excluding 200 ms after each pulse), D19 (benchmark-based defaults) and D20 (self-contained result folders) are Proposed.
- **Legacy scripts retired** 2026-10-05 (see Existing code below).

## Hardware and data

- **Recording hardware:** MCS MEA2100-Mini with STG stimulator (verified from stream labels). MEA layout string: `ME21Combi60`.
- **Array:** custom 4x4x4 grid, but **only 60 channels are recorded**. 59 sit on the cube, and label `15` is the reference electrode (owner confirmed). 5 grid cells are empty.
  - Electrode pitch within a layer: TODO µm.
  - Layer spacing: TODO µm (the paper reports 25–250 µm spacers).
  - Electrode diameter: TODO (the paper says 30 µm).
- **Channel map:** label → (row, col, layer) is `src/meagraph/probe/data/cube4x4x4_E-00303_map.csv` (formerly `MEA_CUBE` in the legacy `mcs.py`). It encodes the headstage-to-electrode wiring (owner confirmed).
- **File format:** MCS HDF5, Multi Channel Experimenter 2.21 / DataManager 1.14, protocol RawData v3. Full layout in `docs/DATA_FORMAT.md`.
- **Sampling rate:** 10 kHz in the current files (`Tick` = 100 µs). Always read it from `InfoChannel.Tick`.
- **Analog stream index is not processing order.** `Stream_0` = raw, `Stream_3` = Filter 1, `Stream_2` = Filter 2, `Stream_1` = Filter 3. Select by label or lineage.
- **Time base:** events and MCS segments are in µs on the recording clock. Analog data starts at `ChannelDataTimeStamps[0, 0]`, which is 0.5 s in the beforestim file.
- **Stimulation events:** `EventEntity_<EventID>`. EventID 1 = STG 1 Single Pulse Start, 2 = Stop (+3 ms).
  - The **stimulated electrode is not recorded in the file.**
  - Current protocol: 3 biphasic pulses (about 500 µs per phase) every 4.003 s, 26 trains.
  - TODO: amplitude units and mode.
- **Older recordings** come from a standard 60-channel planar MEA. The pipeline should handle both through swappable probe definitions (TODO: MEA type and sample files).
- **Recording types:** spontaneous activity and electrical stimulation sessions, including before/after stimulation blocks for plasticity comparison.
- **Sample data location:** `data/` (moved from the repo root 2026-10-05), with each recording's results in `data/results/<recording>/`. `data/` and `results/` (benchmark output) are git-ignored.
  - 3 files from exp3 DIV140 (2026-07-27), plus `*.spikes.h5` sidecars from `spikes.py`.
  - 2 files from DIV142 (2026-07-29): a 10 min spontaneous recording, and a 32 min (8 GB) "Associative Stimulation 1" recording using STG 1 + STG 2. Their names contain spaces.
- **Associative protocol:** single pulses of about 2 ms; trains of 1/2/3 pulses every 5 s; alternating 50-train blocks per STG output. The whole array saturates during pulses, so the **stimulated sites cannot be inferred (Q14)**.

## Key Phase 0 findings (do not rediscover)

- **Signal quality.**
  - exp3 (DIV140) is noise-dominated: only electrode 12 passes the polarity QC.
  - DIV142 is much better: 8 active channels in the spontaneous file (14, 23, 22, 32, 72, 87, 78, 57).
  - MCS detector output is mostly noise in baseline, and 55 % of it is artifact in stim files.
  - Details: `docs/DATA_FORMAT.md` and `docs/DECISIONS.md` D10.
- **`mcs.py` labels events off by one** (Start is shown as Stop). Do not reuse `mcs.triggers()` labels. See `docs/EXISTING_CODE.md`.
- **SpikeInterface** `read_mcsh5` gives correct values but drops `t_start`, ignores `RowIndex`, and has a wrong ADZero formula. Hence D1, a custom `BaseRecording`.
- **SpikeInterface** `get_channel_locations()` defaults to 2D `xy`, and SI's sparsity and neighbour code uses that default. Always pass `axes="xyz"` (D3).
- `stim_connectivity.py` cannot run on real files: it expects `[src N]` event labels that MCS does not write.
- MCS segment streams can contain events from before the saved analog data starts. `read_mcs_spikes` drops them with a warning.
- **Elephant 1.2.1 bugs (D13).** `jitter_spikes` shifts spikes by 3 × `t_start` when `t_start` ≠ 0, and it uses global RNG. STTC's `np.isclose` widens dt by 10⁻⁵·t. meagraph has its own versions. Use Elephant's STTC only as a reference for t < 0.5 s.
- **SpikeInterface `compute_correlograms(sorting)`:** `ccg[a, b]` counts t_a − t_b. `meagraph.connectivity.cross_correlograms` transposes it to (source, target).
- **Monte-Carlo p-values** cannot go below 1/(N+1). Benjamini–Hochberg over many pairs then finds nothing, so use the fitted-tail p (D17).
- **Pairwise false positives in the simulations are indirect** (chains and common input). Benchmark rows report this as `fp_indirect`.
- **SpikeInterface 0.105 probe handling:** `set_probe` is in-place only and requires one contact per channel. To drop unmapped channels such as the reference, use `select_channels_with_probe`; `attach_probe` already does this.

## Existing code

- **Legacy scripts removed 2026-10-05** (owner approved): `mcs.py`, `spikes.py`, `visualize.py`, `spontaneous_ccg.py`, `stim_audit.py`. Recover any with `git show f3045f7:<name>`. What they did, and their bugs, is in `docs/EXISTING_CODE.md`.
- `stim_connectivity.py` remains only as the reference for the Phase 5 stimulus-evoked estimator. It cannot run (it imports the removed `mcs.py`). Delete it once that port is done.
- The regression baseline is the committed copy of the `spikes.py` sidecars in `tests/data/legacy_baseline/`, protected by a checksum test. `spikes.py` timestamps were relative to the first sample, not the recording clock.
- The three DIV140 `*.spikes.h5` sidecars in `data/` are kept; the viewer only falls back to them when a recording has no `meagraph detect` results.

## Environment

- Python 3.13.13 in a uv-managed `.venv`. The venv has no pip; use `uv pip install --python .venv/bin/python ...`.
- **Installed in `.venv`:** spikeinterface 0.105.0, probeinterface 0.4.0, neo 0.14.5, elephant 1.2.1, numpy 2.5.3, pydantic 2.13, pytest 9.1, plus `meagraph` itself in editable mode.
- To reinstall the package: `uv pip install --python .venv/bin/python -e ".[dev]"`.
- **Tests:** `.venv/bin/python -m pytest`. Real-file tests are marked `data` and look for recordings in `data/` or in `$MEAGRAPH_DATA_DIR`.
- **Primary machine:** MacBook Pro, Apple M3 Pro, 18 GB RAM. Avoid tools that require CUDA for core functionality; GPU spike sorters are optional extras only.
- **Poppler is not installed,** so the Read tool cannot render PDFs. Use `pypdf` to extract text.

## Conventions

- Package code lives in `src/meagraph`, installed in editable mode. The GUI goes in `apps/`.
- Units are SI-based with explicit suffixes in names (`_s`, `_ms`, `_uv`, `_um`).
- Time is in seconds on the MCS recording clock unless named otherwise (D5). The first analog sample is at `t_start`.
- Probe dimensions are placeholders until Q1 is answered (D8). Grid indices are exact; µm distances are not physical while `geometry_verified` is false.
- Electrode identity is always carried as a stable channel ID (the MCS electrode label, e.g. `"47"`) with probe coordinates attached, never as a bare array index.
- Configs are YAML, validated by typed models; outputs store the config and git hash used.
- Run `pytest` before every commit.

## Do not

- Do not delete or overwrite raw `.h5` files.
- Do not hardcode geometry, sampling rate, or file paths in library code.
- Do not choose scientific parameters silently; flag them and ask.
- Do not select analog streams by index or decode MCS entities by row.

## Open questions

The full list with context is in `docs/PLAN.md` § Open questions.

- **Q1.** Pitch, layer spacing, electrode diameter, layer orientation.
- **Q2 (remaining part).** Are the 5 empty cells absent, or present but unwired?
- **Q4.** Planar 60 MEA type and sample files.
- **Q5.** MCS Filter 1/2/3 and Spike Detector settings.
- **Q6.** Stimulation amplitude units and mode, polarity, and the site for the 11-15 file. The artifact says 12, and Phase 2 results used 12.
- **Q7.** Afterstim baseline, and longer or more active recordings. (The `data/` folder part is resolved: recordings live there.)
- **Q8.** Detection threshold for analysis.
- **Q9.** Stimulation blanking strategy.
- **Q10.** Connectivity significance defaults.
- **Q14.** Which electrodes did STG 1 and STG 2 drive in the DIV142 associative file? They cannot be inferred from the data.
- **Q15.** In stim47, positive QC events are elevated on several channels. Are slow artifact components outlasting the 50 ms QC exclusion? Check in Phase 6.
- **Q16.** The synaptic window is [1, 4) ms (D14). In DIV142, the 32→14 correlogram peaks at about 6 ms. Should a wider window be used, or a sensitivity analysis run?
- **Q17.** Accept D18, excluding spikes up to 200 ms after each pulse in spontaneous connectivity?
- **Q18.** Accept D19? It would make the burst-removed analysis primary, set `drop_symmetric=False`, and make `cch_jitter` the primary method.

Resolved 2026-10-03:

- **Q3.** 15 is the reference.
- **Q2 (provenance).** The map is the headstage wiring.
- **Q8 / Q9.** The recommended defaults are approved (D10).

Resolved earlier on 2026-10-03 ("approved with defaults"):

- **Q11.** Package name is `meagraph`.
- **Q12.** Use the existing `.venv`.
- **Q13.** Freeze the legacy scripts (D7).
- Q1–Q3 are deferred, with placeholder geometry flagged (D8).
