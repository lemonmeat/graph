# Design decisions

Each entry records what was decided, why, and what evidence supports it. Status values:

- **Proposed:** awaiting owner approval.
- **Accepted:** approved by the owner.
- **Superseded:** replaced by a later entry.

---

## D1. Custom MCS HDF5 `BaseRecording` instead of `spikeinterface.extractors.read_mcsh5`

**Status:** Accepted 2026-10-03 (implemented in Phase 1: `meagraph.io.mcs_h5`)

**Decision.** Write a thin `McsH5Recording(BaseRecording)` in `io/` that returns µV-scalable traces. It stays fully SpikeInterface-native: chunked `get_traces`, gains and offsets, properties, and `t_start`. Downstream preprocessing, peak detection, sorting and `SortingAnalyzer` all run on it unchanged.

**Why not the built-in reader.** It is tested and bit-identical in values (see `DATA_FORMAT.md`), but on these files it has five problems:

1. It sets segment `t_start = 0` and drops `FirstTimeStamp`. In beforestim, spikes and stimulus events would then be misaligned by 0.5 s.
2. It ignores `InfoChannel.RowIndex`. The mapping is the identity in current files, but the spec allows otherwise.
3. Its ADZero offset formula is wrong. It is harmless only while ADZero = 0.
4. Its channel IDs are `Ch0..Ch59`. CLAUDE.md requires stable electrode IDs (`"47"`), with the MCS label and ChannelID kept as properties.
5. It cannot select streams by label or lineage, and exposes no events or segment streams.

Neo has no MCS HDF5 reader (`RawMCSRawIO` handles `.raw` only). McsPy (`McsPyDataTools` 0.4.3) reads everything, including events, but is not lazy in a SpikeInterface-compatible way.

**Scope.** The subclass is under 150 lines and does traces only. Events and MCS segment streams are read by small h5py functions in `io/mcs_events.py`, using McsPy's EventID/SegmentID naming conventions. Their outputs are SpikeInterface `NumpySorting` objects (for MCS spikes) and plain typed arrays (for stimulation events).

**Upstream.** An upstream fix to `read_mcsh5` covering `t_start`, `RowIndex` and ADZero is worth proposing to SpikeInterface later. If it lands, D1 can be superseded.

---

## D2. Analyse the raw stream; ignore MCS filter streams and MCS spike streams for analysis

**Status:** Accepted 2026-10-03

**Decision.** All analysis starts from raw `Stream_0` and does its own filtering in documented SpikeInterface pipelines. The MCS filter streams remain selectable in the viewer. MCS Detector and Sorter events are readable, for comparison only.

**Why.**

- The filter parameters are not stored in the file. Filter 3 is unknown and was only measured empirically.
- The MCS detector's threshold and polarity are not stored.
- In the stim files, 55 % of MCS detector events sit within 15 ms of a stimulus pulse.
- `spikes.py` already uses `Stream_0`, so regression is direct.

---

## D3. 3D geometry: never rely on SpikeInterface's default 2D channel locations

**Status:** Accepted 2026-10-03

**Finding.** ProbeInterface stores a 3D probe correctly. However, `BaseRecording.get_channel_locations()` defaults to `axes="xy"`. SI code that builds channel neighbourhoods or sparsity calls it without `axes`: `core/sparsity.py` (radius sparsity), `core/recording_tools.py`, and the `matched_filtering` peak detector, observed in SI 0.105.0. For the cube, that silently projects onto x-y. Electrodes stacked in different layers then appear to be at distance 0.

**Decision.**

- Package code always asks for `axes="xyz"`.
- Peak detection uses `method="by_channel"`. This matches `spikes.py` and is geometry-free.
- Any SI step that needs sparsity receives an explicit sparsity mask computed from true 3D distances.
- A unit test asserts that the probe round-trips with z intact.

This entry must be re-checked whenever the SI version changes.

---

## D4. Decode MCS entities by ID, not by row

**Status:** Accepted 2026-10-03

**Decision.** Event datasets map as `EventEntity_<EventID>` and segment datasets as `SegmentData_<SegmentID>`, following the McsPy convention.

**Why.** Legacy `mcs.py` indexes by InfoEvent row, which mislabels the stimulation events (Start appears as Stop, and Stop as "Marker Start"). This was verified against the raw artifact timing; see `EXISTING_CODE.md`.

---

## D5. One time base: seconds on the MCS recording clock

**Status:** Accepted 2026-10-03

**Decision.**

- t = 0 is the MCS recording clock zero, the same clock as event and segment timestamps.
- The analog segment's `t_start` is `FirstTimeStamp`.
- All spike, event and analysis times are stored on this clock, with an `_s` suffix.
- Legacy `spikes.py` timestamps are relative to the first sample. They are converted (+`t_start`) when loaded for regression.

---

## D6. Environment and dependency policy

**Status:** Accepted 2026-10-03

**Environment.** Keep the existing uv-managed `.venv` (Python 3.13.13). Make the package installable with `uv pip install -e .`.

**Core dependencies:** `spikeinterface` (pinned to a minor version, currently 0.105.x), `probeinterface`, `h5py`, `numpy`, `scipy`, `pydantic`, `pyyaml`, `networkx`, `neo`, `elephant`.

**Optional extras:**

| Extra | Packages |
|---|---|
| `viewer` | `matplotlib` |
| `nwb` | `pynwb` |
| `glm` | `nemos` |
| `te` | `idtxl`, if practical (to be evaluated) |
| `synth` | `brian2` |
| `sorting` | CPU sorters only |

**Phase 0 test.** SI 0.105.0, ProbeInterface 0.4.0, Neo 0.14.5 and Elephant 1.2.1 install and import cleanly on Python 3.13 and numpy 2.5.3. This was tested in a scratch venv.

**Phase 1.** The core dependencies were installed into `.venv`. numpy, scipy, h5py and matplotlib kept their versions, and the legacy scripts still run. `te` and `sorting` extras are not declared yet; they are added once evaluated.

---

## D7. Package name, layout and the legacy scripts

**Status:** Accepted 2026-10-03

**Package.** The package is `meagraph` (src layout, hatchling, `meagraph` console command). The CLI lives inside the package so installing it provides the command.

**Legacy scripts.** `mcs.py`, `spikes.py`, `visualize.py` and the other scripts stay frozen as the reference implementation, including the event-label bug. `spikes.py` output does not depend on event labels, so the regression baseline is unaffected. Fixes live only in `meagraph`.

**Running the legacy scripts.** Do not run `spikes.py` with default arguments in the repo root: it overwrites the `*.spikes.h5` sidecars next to the recordings.

**Regression baseline.** The baseline is a committed copy in `tests/data/legacy_baseline/`. It holds the three sidecars written 2026-09-30, after the last edit to `spikes.py`, plus a `SHA256SUMS` file. A test fails if the copy changes.

---

## D8. Placeholder geometry is explicit, never silent

**Status:** Accepted 2026-10-03

**Decision.** Until the owner supplies pitch, layer spacing and diameter (Q1), probe specs carry placeholder dimensions with `geometry_verified: false`:

- cube: 100 µm everywhere;
- planar: 200 µm, the legacy default.

The flag is enforced in three places:

- `attach_probe` emits `UnverifiedGeometryWarning`;
- the recording and probe annotations carry `geometry_verified`;
- `meagraph probes` prints "[placeholder geometry]".

Grid indices (row, col, layer) are always exact. They are stored as channel properties, so the viewer and any grid-based analysis do not depend on the placeholders.

**Rule for later phases.** Any analysis that interprets distances in µm, such as latency-versus-distance or spatial priors, must check `geometry_verified` and refuse or warn.

---

## D9. Stimulation windows are bridged by a small custom preprocessor, not `remove_artifacts`

**Status:** Accepted 2026-10-03 (Phase 2: `meagraph.preprocess.InterpolateWindowsRecording`)

**Why not SpikeInterface's `remove_artifacts(mode="linear")`.** It does not fit, for three reasons found by reading its source in SI 0.105:

1. It applies one `ms_before`/`ms_after` to every trigger and every channel. Adaptive blanking (D10) needs a different window per channel, and windows that span pulse onset to pulse offset when pulse length varies.
2. It anchors the interpolation on 5-sample medians around the window edges rather than on the boundary samples, so it cannot reproduce `spikes.py`.
3. Overlapping triggers are bridged one after another, so a later window can anchor inside an earlier one.

**What the custom step does.** About 80 lines: it merges each channel's windows, then draws a straight line from the last sample before each window to the first sample after it. This is exactly the legacy rule.

**Behaviour.** The step is chunk-safe: reads that cross a window fetch the anchor samples. A unit test checks that piecewise reads equal a full read.

**Recording edges (the one deliberate difference).** At the start or end of the recording, where no anchor exists on one side, it holds the other anchor. Legacy used the channel median there.

---

## D10. Detection defaults (Q8, Q9) and the signal-quality control

**Status:** Accepted 2026-10-03 (owner approved the recommended defaults)

### Detection (Q8)

The detection parameters are the `spikes.py` settings:

| Setting | Value |
|---|---|
| Threshold | 5 × median(\|x\|)/0.6745 over the whole recording |
| Filter | 3rd-order Butterworth, 300–3000 Hz, zero-phase |
| Refractory / isolation | 1 ms |
| Amplitude cap | 1000 µV |
| Cutout | −1 to +2 ms |
| Rebound rule | on |

Connectivity analyses (Phase 4 onward) use only channels flagged **active** by the polarity control below. Spikes are still reported for every channel.

### Blanking (Q9, profile `default`)

For each STG output, `measure_recovery` works as follows:

1. Bridge `[onset − 1 ms, offset + 1 ms)` and band-pass.
2. Take the trial **median** of the band-passed epochs after the pulse offsets. This is the deterministic artifact. Evoked spikes are jittered and absent on many trials, so they barely move the median.
3. Call a channel recovered once |median| stays below 1 single-trial noise σ for 1 ms.

Each channel is then blanked over `[onset − 1 ms, offset + recovery)`:

- **Bounds:** recovery is at least 1 ms. It is capped at 50 ms; channels still showing artifact at the cap are blanked for the full 50 ms.
- **Guard:** peaks less than 1 ms after a window are dropped, as in legacy.
- **Stimulation sites:** sites given in the config or `--stim-site` are excluded from detection entirely.

The `legacy` profile keeps the old fixed rule: −1/+6 ms around every Start and Stop event, with no site exclusion. It exists for the regression test.

**Measured recovery after pulse offset:**

| Recording | Median | Slowest channels |
|---|---|---|
| exp3 | 5 ms | the stimulating electrode, 9–11 ms |
| Associative (DIV142) | 6.6–6.8 ms | 17 ms |

For comparison, legacy blanking effectively ended 7 ms after the offset for every channel.

### Polarity control (QC)

**Principle.** Extracellular spikes are mostly negative-going, while noise crossings are symmetric.

**Counting.** For each channel, threshold crossings of either sign closer than one cutout span (3 ms) form one **event**. The event's polarity is that of its largest peak. Count negative and positive events, excluding pulses and the 50 ms after each one (`qc_exclude_post_ms`), where evoked responses and artifact residue are not spontaneous-like.

**Decision.** A channel is **active** when a one-sided binomial test of negative > positive gives p < 0.001 and it has at least 20 negative events.

**Why events, not peaks.** Two simpler versions failed synthetic tests:

- *Counting raw peaks.* After band-passing, a spike's positive overshoot often also crosses +5σ, so every spike counted once on each side and real units looked like noise.
- *Ignoring positive peaks near a negative peak.* Symmetric ringing bursts then produced a false negative excess; this is how noisy channel 41 was flagged in DIV142.

With events, a spike counts as negative because its trough outweighs its overshoot. A symmetric burst is equally likely to count either way, so under the null the counts are binomial with p = ½. `tests/test_detection.py` covers both cases.

**Limitations.** This is a heuristic. It can miss genuinely positive-going units, and a strongly bursting channel can still pass.

---

## D11. Regression tolerances against `spikes.py`

**Status:** Accepted 2026-10-03 (`tests/test_regression.py`)

**Tolerances.** The `legacy` profile must reproduce the committed baseline as follows:

- ≥ 99 % of spikes matched within ±1 sample, in both directions;
- σ within 0.1 %;
- spikes at the same sample have filtered waveforms within 0.01 µV.

Spikes in the first 50 ms are ignored, because of the D9 edge rule.

**Measured.** 99.2–100 % matched. σ is within 0.03 %; the 3rd-order SpikeInterface filter chain matches `sosfiltfilt` to 2×10⁻⁶ µV. Waveforms at the same sample are identical.

**Remaining differences (for completeness).** Besides the t = 0 edge, a few spikes inside bursts differ in peak selection:

- `scipy.find_peaks(distance=…)` suppresses neighbours greedily, so a small peak survives when its larger neighbour was itself suppressed.
- SpikeInterface's `by_channel` keeps only peaks that are the minimum within ±1 ms.

The SI rule is the stricter one. It is used for all profiles.

**Noise estimate.** `median_abs_noise_uv` computes the legacy whole-recording median with a logarithmic histogram, which gives < 0.03 % error in one chunked pass. SpikeInterface's `get_noise_levels` samples random chunks instead, which is not reproducible to that precision.

---

## D12. The viewer is custom matplotlib, inside the package

**Status:** Accepted 2026-10-03 (Phase 3: `meagraph.viewer`, `meagraph view`)

**Options evaluated** (SI 0.105):

| Option | Why not |
|---|---|
| `spikeinterface.widgets` | Mostly static figures built around sorted units and `SortingAnalyzer`. The interactive backends each add a heavy dependency: `ipywidgets` (Jupyter only), `ephyviewer` (PyQt), `sortingview`/`figpack` (web, cloud-backed). None has a clickable 3D electrode selector. |
| `spikeinterface-gui` | A curation GUI for sorted units. It needs a full `SortingAnalyzer` and Qt or panel, and its probe view is 2D. |
| `probeinterface.plotting` | Can draw a 3D probe but has no picking. |

**Choice.** Keep the legacy matplotlib UX (`docs/EXISTING_CODE.md`), rebuilt on the package API.

- `meagraph.viz` holds reusable drawing functions (cube map, raster, trace, waveforms) that also work in notebooks and report figures.
- `meagraph.viewer` is about 250 lines of state and interaction.
- **Dependency:** matplotlib only (the `viewer` extra). It runs anywhere with no Qt, Jupyter or account.
- **Launch:** `meagraph view file.h5`.

**Location.** It lives in the package rather than `apps/viewer/`, so one install provides it. The core library never imports it.

**Behaviour changes from `visualize.py`:**

- stimulation lines at pulse onset (the legacy label bug is gone), coloured per STG output;
- the reference electrode `15` is not listed;
- analog streams are listed in processing order;
- QC-active electrodes have a dark rim;
- spikes come from `meagraph detect` results, falling back to a legacy sidecar.
