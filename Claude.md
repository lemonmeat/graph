# CLAUDE.md

Context for Claude Code sessions in this repo. Fields marked TODO are filled in by the owner. Fields marked (verified) were confirmed from the files during Phase 0 on 2026-10-03.

## Project

Analysis of extracellular recordings from in vitro neuronal cultures on a custom 3D microelectrode array, aimed at inferring functional connectivity graphs and understanding how stimulation reshapes them for reservoir computing and closed-loop biocomputation. This is Princeton ECE independent work (ECE 398), so methods must be defensible and well documented. The device design follows Kumar et al. 2026, Nat. Electron. 9:532 ("3D-MIND", `papers/Kumar2026.pdf`).

## Status

- Phase 0 (explore and verify) is done. The owner approved the plan "with defaults" on 2026-10-03.
- **Phase 1 (foundation) is done:** package `meagraph` with `io`, `probe`, `config`, `spiketrains` and a CLI (`meagraph info`, `meagraph probes`). See `docs/ARCHITECTURE.md`.
- **Next is Phase 2:** preprocessing and detection reproducing `spikes.py`, plus a regression test. Ask Q8 (threshold) and Q9 (blanking) before choosing defaults beyond the legacy profile.
- Decisions D1–D8 in `docs/DECISIONS.md` are Accepted.

## Hardware and data

- **Recording hardware:** MCS MEA2100-Mini with STG stimulator (verified from stream labels). MEA layout string: `ME21Combi60`.
- **Array:** custom 4x4x4 grid, but **only 60 channels are recorded**. 59 sit on the cube; label `15` behaves like the reference electrode (TODO: confirm). 5 grid cells are empty.
  - Electrode pitch within a layer: TODO µm.
  - Layer spacing: TODO µm (the paper reports 25–250 µm spacers).
  - Electrode diameter: TODO (the paper says 30 µm).
- **Channel map:** label → (row, col, layer) is `MEA_CUBE` in `mcs.py`, reproduced in `docs/DATA_FORMAT.md`. Its provenance is TODO, and the data cannot validate it.
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
- **Sample data location:** repo root, 3 files from exp3 DIV140 (2026-07-27), plus `*.spikes.h5` sidecars from `spikes.py`. Raw data is git-ignored.

## Key Phase 0 findings (do not rediscover)

- **Signal quality is the binding constraint.** On most electrodes, negative and positive threshold crossings are about equally frequent, which looks like noise. Only electrode 12 has robust negative-going spiking. MCS detector output is mostly noise in baseline, and 55 % of it is stimulation artifact in stim files. Details: `docs/DATA_FORMAT.md` § Signal quality.
- **`mcs.py` labels events off by one** (Start is shown as Stop). Do not reuse `mcs.triggers()` labels. See `docs/EXISTING_CODE.md`.
- **SpikeInterface** `read_mcsh5` gives correct values but drops `t_start`, ignores `RowIndex`, and has a wrong ADZero formula. Hence D1, a custom `BaseRecording`.
- **SpikeInterface** `get_channel_locations()` defaults to 2D `xy`, and SI's sparsity and neighbour code uses that default. Always pass `axes="xyz"` (D3).
- `stim_connectivity.py` cannot run on real files: it expects `[src N]` event labels that MCS does not write.
- MCS segment streams can contain events from before the saved analog data starts. `read_mcs_spikes` drops them with a warning.
- **SpikeInterface 0.105 probe handling:** `set_probe` is in-place only and requires one contact per channel. To drop unmapped channels such as the reference, use `select_channels_with_probe`; `attach_probe` already does this.

## Existing code

- `spikes.py` is the current spike analysis script and the reference for regression tests. Its spike timestamps are relative to the first sample, not the recording clock.
- `visualize.py` is the current viewer, including the 4x4x4 electrode selector that must be preserved. The UX contract is listed in `docs/EXISTING_CODE.md`.
- `mcs.py`, `spontaneous_ccg.py`, `stim_audit.py` and `stim_connectivity.py` are also documented in `docs/EXISTING_CODE.md`.
- **The legacy scripts are frozen (D7), bugs included.** Never run `spikes.py` without arguments in the repo root: it overwrites the sidecars. The regression baseline is the committed copy in `tests/data/legacy_baseline/`, protected by a checksum test.

## Environment

- Python 3.13.13 in a uv-managed `.venv`. The venv has no pip; use `uv pip install --python .venv/bin/python ...`.
- **Installed in `.venv`:** spikeinterface 0.105.0, probeinterface 0.4.0, neo 0.14.5, elephant 1.2.1, numpy 2.5.3, pydantic 2.13, pytest 9.1, plus `meagraph` itself in editable mode.
- To reinstall the package: `uv pip install --python .venv/bin/python -e ".[dev]"`.
- **Tests:** `.venv/bin/python -m pytest`. Real-file tests are marked `data` and look for recordings in the repo root or in `$MEAGRAPH_DATA_DIR`.
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
- Do not remove `spikes.py` or `visualize.py` until the refactored pipeline passes the regression test and the owner approves.
- Do not hardcode geometry, sampling rate, or file paths in library code.
- Do not choose scientific parameters silently; flag them and ask.
- Do not select analog streams by index or decode MCS entities by row.

## Open questions

The full list with context is in `docs/PLAN.md` § Open questions.

- **Q1.** Pitch, layer spacing, electrode diameter, layer orientation.
- **Q2.** Provenance of `MEA_CUBE`. Are the 5 empty cells absent or unwired?
- **Q3.** Is `15` the reference electrode?
- **Q4.** Planar 60 MEA type and sample files.
- **Q5.** MCS Filter 1/2/3 and Spike Detector settings.
- **Q6.** Stimulation amplitude units and mode, polarity, and the site for the 11-15 file (the artifact says 12).
- **Q7.** Afterstim baseline, longer or more active recordings, and the `data/` folder.
- **Q8.** Detection threshold for analysis.
- **Q9.** Stimulation blanking strategy.
- **Q10.** Connectivity significance defaults.

Resolved 2026-10-03 ("approved with defaults"):

- **Q11.** Package name is `meagraph`.
- **Q12.** Use the existing `.venv`.
- **Q13.** Freeze the legacy scripts (D7).
- Q1–Q3 are deferred, with placeholder geometry flagged (D8).
