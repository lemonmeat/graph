# Existing code (Phase 0 snapshot)

Snapshot of the pre-refactor scripts as of 2026-10-03. These are the reference implementation. `spikes.py` is the regression baseline, and the viewer UX in `visualize.py` must survive the refactor. Line numbers refer to this snapshot.

The scripts form a flat layout around a shared reader:

```
mcs.py  <──  spikes.py  <──  visualize.py
        <──  spontaneous_ccg.py
        <──  stim_audit.py
        <──  stim_connectivity.py
```

Generated artefacts in the repo root:

- `*.spikes.h5` (3 files, written 2026-09-30 by `spikes.py` at default parameters). These are the regression baseline.
- `ccg_connectivity.npz`: output of `spontaneous_ccg.py`. Its provenance (which input file, which segment stream) is not recorded in the file. Its shape (55 channels at ≥ 50 spikes) matches the beforestim file with MCS Spike Sorter events.

---

## `mcs.py`: MCS HDF5 reader and electrode geometry

| Name | What it does |
|---|---|
| `_s(x)` | Decodes HDF5 bytes or numpy scalars to `str`. |
| `electrode_number(label)` | `'E-00303 47'` → `'47'`. Returns the last whitespace token, with `;` treated as a space. |
| `mea_geometry(labels, pitch=200.0)` | Planar 8×8 MCS layout. Label digits are (column, row), giving x/y in µm = ((c−1)·pitch, (r−1)·pitch). Used only by `spontaneous_ccg.py`. |
| `MEA_CUBE` | Dict mapping label → (row, col, layer) for 59 labels on the 4×4×4 grid. Has a uniqueness assert. Label `15` is absent. Five cells are empty (see `DATA_FORMAT.md`). |
| `cube_geometry(labels)` | Returns an (n, 3) array of (row, col, layer), with NaN for labels not in the cube. |
| `Channel` | Holds label, data row, µV scale, ADZero. |
| `AnalogStream` | Wraps `AnalogStream/Stream_i`. Scale µV = `ConversionFactor · 10^Exponent · 1e6`. Honours `RowIndex`. `fs = 1e6 / Tick` from the first channel. `t_start` comes from `ChannelDataTimeStamps[0, 0]`. `index_of()` accepts a full label, an electrode number, or an int. `read_uv(t0, t1, channels)` returns float32 µV, with times **relative to the first sample**, reading one HDF5 row per channel. |
| `McsFile` | Opens `Data/Recording_0` (hardcoded). `analog_stream(i)` sorts streams by numeric suffix. `triggers()` returns {event label: times (s)}. `segment_triggers()` and `segment_events(stream)` read MCS spike timestamps; the latter merges sorter units per electrode. Segment times are on the absolute recording clock. |

**Hardcoded:** `Data/Recording_0`, the 200 µm planar pitch, the whole `MEA_CUBE` map, and µV as the output unit.

**Defects found in Phase 0:**

1. **`triggers()` labels event entities off by one** (`mcs.py:168-169`). Datasets are named `EventEntity_<EventID>`, and EventIDs start at 1. This is the McsPy convention (`McsPy/McsData.py:804`) and is visible in the files. `triggers()` instead labels `EventEntity_K` with InfoEvent *row* K. In the stim files this produces two wrong labels:
   - `EventEntity_1` is actually "STG 1 Single Pulse Start" but is labelled "STG 1 Single Pulse Stop".
   - `EventEntity_2` is actually "STG 1 Single Pulse Stop" (pulse start + 3 ms) but is labelled "STG 1 Marker Start".

   The raw data confirms the true mapping: the artifact begins 1–2 samples after `EventEntity_1` and ends at `EventEntity_2` (see `DATA_FORMAT.md`). Consequences:
   - `visualize.py` keeps only keys containing "Start", so it draws stimulation lines at **pulse offset**, 3 ms late.
   - `stim_audit.py` prefers a "start" entity, so it also uses offsets. Its printed conclusion that "the trigger marks pulse offset" is produced by this bug.
   - `spikes.py` is **not** affected, because it blanks around every event regardless of label.
2. **Mixed time bases.** Analog reads and `spikes.py` timestamps are relative to the first sample. Event and segment timestamps are on the MCS recording clock. The two coincide when `FirstTimeStamp = 0` (both stim files) but differ by 0.5 s in the beforestim file.
3. `segment_events()` / `segment_triggers()` index `InfoSegment` by row, while McsPy keys by `SegmentID`. These are identical in the current files, so this is a latent issue only.

---

## `spikes.py`: threshold spike detection to a sidecar file (regression reference)

`detect_file()` runs the following per channel, on raw analog `Stream_0` (`detect_channel`, lines 54-72):

1. Subtract the channel median.
2. **Bridge stimulation windows by linear interpolation** before filtering. `stim_windows()` (line 37) builds sample windows `[t − blank_pre, t + blank_post)` around *every* event of *every* entity (Start and Stop). It clips them to the recording, honours `t_start`, and merges overlaps. Because Stop = Start + 3 ms, the effective window is −1 ms to +9 ms around pulse onset.
3. Apply a 3rd-order Butterworth band-pass, 300 to min(3000, fs/2 − 1) Hz, with `sosfiltfilt` over the whole channel (zero-phase).
4. Compute σ = median(|y|) / 0.6745 over the whole channel, including the interpolated windows.
5. Find negative peaks with `find_peaks(−y, height=k·σ, distance=refractory)`.
6. Keep a peak only if all of these hold: it has a full cutout inside the recording; |amplitude| ≤ `max_uv`; it is not inside `[lo, hi + guard)` of any window.
7. **Rebound rule:** drop the peak if max(y) within the cutout ≥ |negative peak|. This targets ringing artifacts.
8. Cut waveforms of `pre + post` samples (−1 ms to +2 ms, 30 samples at 10 kHz) from the filtered trace.

**Defaults:** k = 5, band = (300, 3000) Hz, refractory = 1 ms, max_uv = 1000 µV, blank_pre = 1 ms, blank_post = 6 ms, guard = 1 ms, cutout = (1, 2) ms, stream = 0. The CLI exposes only `-k`, `--blank-post` and `--max-uv`. With no arguments it processes every `*.h5` next to the script.

**Output** is `<recording>.spikes.h5`, written by `write_spikes`:

```
attrs: source (basename), params (JSON: all of the above + fs, t_start, duration, n_stim_windows)
SpikeStream/Electrode_<label>/ts          float64, s, = sample_index / fs  (relative to FIRST SAMPLE, not recording clock)
SpikeStream/Electrode_<label>/amp         float32, µV, filtered negative-peak value (negative)
SpikeStream/Electrode_<label>/waveforms   float32 (30, n), µV, gzip
SpikeStream/Electrode_<label>.attrs       sigma_uv
```

`load_spikes(h5_path)` returns `(ts{label}, wf{label}, (−pre_ms, post_ms), params)`, or `None` if there is no sidecar. `visualize.py` consumes it.

**Hardcoded:** the filter order (3), the 0.6745 MAD constant, the sidecar suffix, the default parameters above, and the input-file glob in `main()`.

**Notes:**

- The module docstring says timestamps are "on the same clock as the analog stream". They are actually relative to the first sample (`t_start` is ignored). This is consistent with `read_uv` but not with event times.
- Each per-channel read decompresses HDF5 chunks shaped (60, 1092), which hold all 60 channels. Reading channel by channel therefore decompresses everything about 60 times. Runtime is fine at the current file sizes.
- No code version or git hash is stored in the output.

**Current outputs:**

| File | Spikes | Electrodes with 0 spikes |
|---|---|---|
| beforestim | 165 | 29 |
| stim | 220 | 37 |
| stim47 | 249 | 28 |

**Behaviour that must be preserved:** the algorithm and defaults above, which the regression test checks within tolerance, and the ability to produce the sidecar schema, or an equivalent the viewer can read, until the owner retires it.

---

## `visualize.py`: interactive viewer (UX must survive the refactor)

**`print_tree(path)`** (`--tree`) prints the HDF5 tree and root attributes. Repeated `SegmentData_*` / `EventEntity_*` entries beyond `_0`–`_2` are hidden.

**`Recording(path)`** collects everything the viewer needs:

- Opens all analog streams. Stream names are shortened to the text before `;` (for example `"Filter (3)"`).
- Takes electrode order from `Stream_0` labels and positions from `cube_geometry`.
- Keeps events = non-empty `triggers()`. `stim_times` = events whose key contains "Start", or all events if none do. This is affected by the label bug.
- Loads spikes and cutouts from the sidecar via `load_spikes`. If there is no sidecar it prints a hint and runs with no spikes.
- `cutout_uv(elec)` subsamples at most 300 waveforms with a fixed RNG (seed 0).
- `rates()` = spike count / duration.

**`Viewer`** is a 15 × 9.2 in matplotlib figure with four panels and controls:

| Element | Behaviour to preserve |
|---|---|
| 3D map (top left) | Faint 4×4×4 lattice. Electrodes coloured by spike rate (Blues, with colourbar). Axes: x = col, y = row (inverted, so row 1 is at the front/top as in the pad map), z = layer. Box aspect (1, 1, 2). Fixed view (elev 20°, azim −30°), mouse rotation disabled. **Left-click release selects the electrode nearest on screen within 25 px.** The selection is shown with a warm-coloured ring and a text label "electrode N (row r, col c, layer l)". |
| Raster (top right) | All electrodes in `Stream_0` order, with every 5th label shown. Stimulation times as red lines. The current window shown as a blue span. A horizontal line marks the selected electrode. **Click re-centres the window at that time and selects the electrode on that row.** |
| Trace (bottom left) | The selected electrode on the selected analog stream over `[t0, t0 + win]`. Optional display band-pass: 3rd-order Butterworth, 300–3000 Hz, `filtfilt` after mean removal. Y limits come from the 0.5/99.5 percentiles plus 25 % padding, so stimulation artifacts are clipped deliberately. Stimulation lines, and ▼ spike markers along the top. |
| Waveforms (bottom right) | Up to 300 cutouts (alpha 0.08) plus their mean in the warm colour. Title shows n. |
| Controls | Start slider [0, duration − 0.05]. Window slider [0.05, 20] s in 0.05 s steps. Radio buttons for the analog stream. "band-pass 300-3000 Hz" checkbox. ←/→ keys step by half a window. |
| CLI | `path` (or an interactive chooser over `*.h5`, excluding sidecars), `--tree`, `--save PNG` (Agg backend), `--t0`, `--win`, `--electrode`, `--stream`, `--bandpass`. |

**Hardcoded:** colour constants, the 300–3000 Hz display filter, the 25 px pick radius, the 300-waveform cap, the figure layout, and the 1..4 lattice extent.

**Known issues:**

- Stimulation lines sit at pulse offset because of the `mcs.py` bug.
- Spike ticks are on the relative clock and stimulation lines on the absolute clock. They misalign by `t_start` when a file has both a non-zero start and events (none of the current files do).
- Electrode `15` appears in the raster but not on the map ("not in cube").
- `Recording` reads files itself. After the refactor it must go through the package API.

---

## `spontaneous_ccg.py`: CCG connectivity from spontaneous activity

The method follows Stark & Abeles (2009) and English et al. (2017):

- **Input spikes:** the MCS `SegmentStream` (default `Stream_0` = MCS Spike Sorter, merged per electrode). It does **not** use `spikes.py` output.
- **Channel selection:** channels with at least 50 spikes (`--min-spikes`).
- **Binning:** 0.5 ms bins, lags ±30 ms, sparse binary binning (`np.unique` per bin, so multiple spikes in one bin count once).
- **Baseline:** each CCG convolved with a partially hollow Gaussian (sd 10 ms, centre weight scaled by 0.4, support ±4 sd).
- **Test:** in the 1–4 ms window, Poisson p-values for excess and for deficit in each bin. Take the minimum over the window, multiply by the number of bins (Bonferroni), and call an edge excitatory or inhibitory if the result is below `alpha` = 0.001.
- **Common-input filter:** drop pairs where both directions are excitatory with latencies within 1 ms of each other.
- **Strength:** (max observed − max λ) / max λ in the window.
- **Positions:** from `mea_geometry` (planar 8×8, 200 µm), which is the **wrong geometry** for the cube.
- **Output and validation:** an `.npz` file, plus a `--truth` hook that compares against a ground-truth file.

**Issues:**

- The docstring promises "FDR across pairs", but no correction across pairs is applied.
- Duration is taken from the analog stream on the relative clock while spike times are absolute. In the beforestim file this drops spikes in the final 0.5 s.
- Inhibitory latency reuses the excitatory argmin.

**Existing result:** `ccg_connectivity.npz` has 55 channels, 2 excitatory edges and 0 inhibitory edges.

---

## `stim_audit.py`: is a stimulation file analysable?

The script reports the following:

- **Event entities** and their inter-event-interval CV.
- **Stimulus times.** It prefers a regular "start" entity, which the label bug corrupts. Fallbacks are segment triggers, then raw-trace detection: a time counts as a stimulus when more than 50 % of channels exceed 25 sd within a 20 s chunk.
- **Stimulated site:** the channel with the largest artifact in the first 3 ms.
- **Artifact timing:** the artifact-peak offset from the trigger.
- **Recovery time:** the point after which |mean epoch| stays ≤ 5 × baseline sd for 2 ms.
- **Template repeatability:** correlation of each trial with the median template over the first 5 ms.
- **Response drift:** evoked event counts in the first third versus the last third of the block, at 4.5 σ.

**Hardcoded:** the thresholds above, the epoch window (±50 ms), the 4 ms recovery limit, and the 300–3000 Hz band.

**Issues:** epoch indexing uses `t·fs` and ignores `t_start`, and it is affected by the label bug. **Worth porting** to `stimulation/audit` as a QC report.

---

## `stim_connectivity.py`: evoked connectivity from site-by-site stimulation

This script was written for a protocol where every site is stimulated in turn and event labels contain `"[src <electrode>]"`. The `--truth mcs_stim_truth.npz` option suggests those labels came from a simulator. **No such labels exist in the real MCS files**, so on the current files the script exits with "could not match trigger entities to channels".

The method runs per stimulating site:

1. Cut epochs of ±30 ms around each trigger.
2. Subtract the across-trial median as an artifact template.
3. Blank 0.6 ms.
4. Band-pass at 300–3000 Hz.
5. Detect threshold-crossing onsets at 4.5 σ, with σ estimated from the pre-stimulus period.
6. Count onsets in [0.6, 12] ms.
7. Compute a permutation null from pre-stimulus windows (2000 permutations).
8. Apply BH-FDR at q = 0.01.
9. Estimate latency from the first PSTH bin above baseline + 3 sd, and jitter as the sd of first-onset times.
10. Network deconvolution, `A = I − (I + R)⁻¹` (Feizi et al. 2013).
11. Keep edges with weight > 0.03, latency ≤ 5 ms and jitter ≤ 1.5 ms.

The usage string references a `plot_connectivity.py` that does not exist. The logic is worth porting as the `evoked` connectivity estimator once the stimulated site can be supplied from metadata or inferred from the artifact.
