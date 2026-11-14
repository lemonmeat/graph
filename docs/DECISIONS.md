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
  - *Update 2026-11-03:* the owner supplied the settings (2nd-order Butterworth HP 200 Hz, 2nd-order Butterworth LP 3500 Hz, 60 Hz notch), and they were verified in five recordings (`DATA_FORMAT.md` § Analog streams). The decision still stands. The online filters are causal and so delay and reshape spikes; they run before any artifact bridging can be applied; and the notch is not gain-normalized (+2 % above 1 kHz).
- The MCS detector's threshold and polarity are not stored.
- In the stim files, 55 % of MCS detector events sit within 15 ms of a stimulus pulse.
- `spikes.py` already uses `Stream_0`, so regression is direct.
- *2026-11-04:* recordings saved without a raw stream are read from the end of the filter chain instead (D27).

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

**Retired 2026-10-05.** With the regression test passing, the owner approved removing `mcs.py`, `spikes.py`, `visualize.py`, `spontaneous_ccg.py` and `stim_audit.py` (recoverable from commit `f3045f7`). `stim_connectivity.py` stays until the Phase 5 port. The regression test does not need the scripts: it compares against the committed baseline.

**Regression baseline** (removed 2026-10-05, D21; recoverable from git history). The baseline was a committed copy in `tests/data/legacy_baseline/`. It holds the three sidecars written 2026-09-30, after the last edit to `spikes.py`, plus a `SHA256SUMS` file. A test fails if the copy changes.

---

## D8. Placeholder geometry is explicit, never silent

**Status:** Accepted 2026-10-03

**Decision.** Until the owner supplies pitch, layer spacing and diameter (Q1), probe specs carry placeholder dimensions with `geometry_verified: false`:

- cube: 100 µm everywhere; **updated 2026-10-05:** layer spacing 250 µm and sensor diameter 30 µm from Kumar et al. 2026 (same device design; owner confirmed layer 1 is the bottom). The in-layer pitch remains a 100 µm placeholder, so `geometry_verified` stays false;
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

### Blanking (Q9)

For each STG output, `measure_recovery` works as follows:

1. Bridge `[onset − 1 ms, offset + 1 ms)` and band-pass.
2. Take the trial **median** of the band-passed epochs after the pulse offsets. This is the deterministic artifact. Evoked spikes are jittered and absent on many trials, so they barely move the median.
3. Call a channel recovered once |median| stays below 1 single-trial noise σ for 1 ms.

Each channel is then blanked over `[onset − 1 ms, offset + recovery)`:

- **Bounds:** recovery is at least 1 ms. It is capped at 50 ms; channels still showing artifact at the cap are blanked for the full 50 ms.
- **Guard:** peaks less than 1 ms after a window are dropped, as in legacy.
- **Stimulation sites:** sites given in the config or `--stim-site` are excluded from detection entirely.

The old fixed rule (−1/+6 ms around every Start and Stop event, no site exclusion) was kept as a `legacy` profile for the regression test until 2026-10-05 (D21).

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

**Status:** Superseded 2026-10-05 by D21. The test passed on all three baseline files until it was retired.

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
- spikes come from `meagraph detect` results (the fallback to legacy sidecars was removed 2026-10-05, D21).

---

## D13. Phase 4 library survey: what is reused, what is custom

**Status:** Accepted 2026-10-04

Surveyed in SpikeInterface 0.105.0 and Elephant 1.2.1 before writing any connectivity code.

**Reused:**

- **SpikeInterface `postprocessing.compute_correlograms`** for all cross-correlograms. It takes about 0.01 s for 59 units without numba.
  - *Sign convention:* `ccg[A, B]` peaks at −2 ms when B fires 2 ms after A. `meagraph.connectivity.cross_correlograms` therefore returns `ccg_si.transpose(1, 0, 2)`, so that `ccg[i, j]` at +lag counts spikes of j that occur `lag` after spikes of i.
  - *Bins:* half-open, `[lo, hi)`, which matches `window_counts`. A test pins both properties.
- **SciPy `stats.false_discovery_control`** for Benjamini–Hochberg correction.
- **NetworkX** for graphs.
- **Elephant `spike_time_tiling_coefficient`** as the reference implementation in tests.
- **Elephant `functional_connectivity.total_spiking_probability_edges`** (TSPE, an MEA-specific method) is a Phase 5 candidate.

**Not available in either library:**

- connectivity inference beyond pairwise correlograms;
- network-burst detection (SpikeInterface lists burst metrics as a TODO; Elephant's synchrony tools address a different question);
- networks with known directed connections for validation (SpikeInterface's generators produce independent Poisson units only).

These are written in `meagraph`.

**Custom replacements, and why:**

- **Interval jitter.** Elephant's `jitter_spikes` defines the right surrogate (spikes redrawn uniformly within fixed windows anchored at `t_start`), but in 1.2.1 it has two problems:
  1. It **shifts every spike by 3 × `t_start`** when `t_start ≠ 0`. Tested: with `t_start` = 0.5 s, surrogates moved by +1.5 s. Our beforestim recording starts at 0.5 s.
  2. It draws from NumPy's **global** random state.

  `meagraph.connectivity.interval_jitter` implements the same definition with a local, seeded generator. This is worth reporting upstream.
- **STTC.** Elephant's implementation has two problems:
  1. **Speed.** 3.6 ms per pair. Surrogate testing at 1000 surrogates would take 3 minutes for the 8 active electrodes and about 3.4 hours for all 59.
  2. **A tolerance bug.** It tests coincidence with `np.isclose(a, b, atol=dt)`, whose default *relative* tolerance makes the effective window dt + 10⁻⁵·t. That is 0.2 ms wider at t = 20 s and 19 ms wider at t = 1920 s. Tested: at t = 1000 s with dt = 5 ms, it scores two spikes 5.1 ms apart as perfectly coincident (STTC = 1.0).

  `meagraph.connectivity.stats.sttc` is vectorized and uses the exact definition (|a − b| ≤ dt). It matches a brute-force implementation exactly, and matches Elephant to 1e-16 where Elephant's relative term is negligible (t < 0.5 s). Both checks are tests.

---

## D14. Connectivity significance defaults (Q10)

**Status:** Accepted 2026-10-04 (owner approved the proposal)

| Setting | Value |
|---|---|
| Correlogram bins | 0.5 ms |
| Correlogram lags | ±30 ms |
| Synaptic window | [1, 4) ms after the source spike |
| Hollow-Gaussian baseline | σ = 10 ms, centre weight scaled by 0.4 |
| Jitter surrogates | 1000, interval jitter with **10 ms windows** |
| Multiple comparisons | Benjamini–Hochberg, q = 0.05, across all tested ordered pairs of one recording and method |
| Minimum spikes per electrode | 100 |
| STTC | Δt = 5 ms tested; Δt = 50 ms reported as a descriptive burst-scale value |
| Electrodes | QC-active only |

**About the jitter window.** The approved "±5 ms interval jitter" is implemented as interval jitter with 10 ms windows: each spike is redrawn uniformly within the fixed 10 ms window that contains it, i.e. within ±5 ms of the window centre. Co-firing slower than about 10 ms (bursts, rate co-modulation) is preserved, so the test asks only about fine timing.

**Network bursts.** Each graph is computed with all spikes and again with network-burst periods removed. Edges significant in both are marked **robust**.

---

## D15. Network-burst detection

**Status:** Accepted 2026-10-04 (`meagraph.detect.bursts`)

**Method.** ISI_N (Bakkum et al. 2013):

1. Pool all non-excluded channels.
2. Wherever 10 consecutive pooled spikes fall within 100 ms, those spikes belong to a burst.
3. Overlapping runs merge.
4. Keep bursts in which at least 3 channels take part.

**No merge step.** In the DIV142 spontaneous recording, gaps between bursts spread evenly from 0.1 s to 10 s with no natural cutoff, so no merge parameter is used.

**Observed:**

| Recording | Bursts | Median duration | Spikes inside bursts |
|---|---|---|---|
| DIV142 spontaneous (600 s) | 143 | 98 ms | 57 % |
| DIV142 associative | 607 | — | — |

In the associative file, 283 of the 606 gaps fall between 3 and 5 s. These bursts are stimulus-evoked and locked to the 5 s train period; excluding post-stimulus periods (D18) halves the count.

**Use.** Bursts are used as a control (the analysis is repeated without burst periods), not as an outcome measure. The parameters are therefore not tuned further.

---

## D16. Synthetic ground truth: a linear Hawkes network

**Status:** Accepted 2026-10-04 (owner chose the simple model; `meagraph.synth`)

**Model.** Plain NumPy, simulated exactly as a branching process. Each spike of unit i causes, on average, `W[i, j]` extra spikes in unit j after a delay of 1.5–3.5 ms with 0.25 ms jitter. `W` is the spike transmission probability, the quantity the correlogram estimators measure. Units sit on randomly chosen contacts of the probe, with baseline rates log-uniform between 0.2 and 2 Hz.

**Confounds, each optional:**

- **Network bursts:** shared rate surges. The defaults follow DIV142: 0.24 bursts/s, 100 ms long, ×20 gain, 80 % participation.
- **Periodic stimulation:** shared, time-locked responses.
- **Detection errors:** missed and false spikes, plus a 1 ms dead time.

**Not used.** Brian2 and neuron-model simulators are not used. They can be added if this model proves too idealised.

**Known simplifications:**

- linear interactions, so no inhibition or saturation;
- delays always fall inside the default synaptic window;
- no biophysical bursting.

---

## D17. Jitter p-values come from a distribution fitted to the surrogates

**Status:** Accepted 2026-10-04 (fixes a power failure found by `tests/test_connectivity.py`)

**The failure.** A Monte-Carlo p-value can never be below 1/(N+1). Benjamini–Hochberg over m pairs needs p ≤ q·k/m for k discoveries, so with N = 1000:

- an edge is discoverable only if about m/50 edges reach the floor;
- with 59 electrodes (3,422 pairs), roughly 70 would have to;
- with 200 surrogates and 132 pairs, nothing can ever be significant.

A sparse synthetic network gave 0 of 12 true edges.

**The fix.** The 1000 interval-jitter surrogates (D14) are kept, but each p-value comes from a distribution moment-matched to them:

- **Correlogram window counts:** negative binomial when overdispersed (as bursts make them), otherwise Poisson. The mean gets +0.5/N so that all-zero surrogates still give a finite p.
- **STTC:** normal.

The Monte-Carlo p-value is stored alongside as `extra["p_empirical"]`.

**Calibration on null networks** (16 units, 3 seeds, 10 min; fraction of p < 0.05 and p < 0.01):

| Method | Without bursts | With bursts |
|---|---|---|
| `cch_jitter` | 1.3 % / 0.1 % | 2.4 % / 0.3 % |
| `sttc` | 5.3 % / 1.4 % | 6.7 % / 1.7 % |

`cch_jitter` is conservative. STTC is near nominal and slightly liberal under bursts, which BH absorbs. In the range Monte Carlo resolves, fitted and empirical p agree to 0.01–0.06 (median absolute difference).

---

## D18. Stimulation periods are excluded from spontaneous connectivity

**Status:** Proposed 2026-10-04 (default `--exclude-stim-ms 200`; awaiting owner review)

**Why.** Shared stimulus drive makes unconnected units co-fire. In the "stim null" benchmark scenario, STTC called 32 % of unconnected pairs connected.

**Default.** `meagraph graph` drops spikes from 1 ms before each pulse to 200 ms after its offset whenever the recording has stimulation events. 200 ms covers the stimulus-evoked network bursts seen in the associative file, but the value is provisional. Evoked responses themselves are the subject of the Phase 5 stimulus-triggered estimator.

---

## D19. Defaults suggested by the benchmark

**Status:** Proposed 2026-10-05 (awaiting owner approval; current defaults unchanged)

From the full benchmark (`docs/METHODS.md` § Validation; `docs/benchmark/benchmark.csv`):

1. **Make the burst-removed analysis primary; report "robust" as a conservative subset.** Bursts lowered recall but produced no false edges in any null scenario. Removing burst periods restored recall, for example 0.66 vs 0.49 for `cch_jitter` at 2 min. Requiring both analyses cut recall to 0.24 at 2 min in exchange for about 2 points of precision. This would change D14.
2. **Set `cch_hollow`'s `drop_symmetric` default to False.** The legacy common-input rule removed 6 of 12 reciprocal connections, while all 119 one-way connections were found. Common input is better handled by model-based methods (Phase 5 GLM).
3. **Use `cch_jitter` as the primary method,** with `cch_hollow` and `sttc` as supporting evidence. `cch_jitter` has the highest precision (0.94–0.97), a delay for every edge, and 0 false edges in every null scenario. STTC must never be applied to stimulation periods (52 % false edges under shared drive).
4. **Recording length (for experiment design).** About 30 min of spontaneous activity recovers about 95 % of connections at transmission probability 0.03. 10 min recovers about 72 %.

---

## D20. Saved results are self-contained; the graph step reads only the detection folder

**Status:** Proposed 2026-10-05 (structural cleanup before Phase 5; no effect on spikes or edges)

**What changed.**

- `DetectionResult` carries the stimulation events it blanked (with their sites) and the electrode positions of the session's probe. The detection folder stores them in `stimulation.csv` and in `x_um`, `y_um`, `z_um` columns of `channels.csv`.
- `meagraph graph` takes stimulation times and node positions from the detection folder. Its `--probe` option is gone. Given a recording or its detection folder, it now does the same thing; before, a folder silently skipped the stimulation exclusion (D18).
- The graph settings (channels, stimulation exclusion, burst parameters) are a typed `GraphConfig`, recorded in each graph folder's `config.yaml`. The provenance lists the `spikes.npz` the graph came from.

**Why.** The probe could be given differently at detection and graph time, and graph folders did not record the stimulation exclusion or channel choice. The Phase 5 stimulus-evoked estimator and the Phase 6 plasticity comparisons need spikes and stimulation times together. Reading them from one folder avoids reopening multi-GB files.

**Evidence.** Re-running detection and graphs on all five recordings gave the same spikes, QC and edges as before the change.

**Cost.** Detection folders written before this change lack `stimulation.csv`. `load_detection` refuses them with a message to re-run `meagraph detect`.

---

## D21. Legacy detection retired; a snapshot test guards detection instead

**Status:** Accepted 2026-10-05 (owner chose "everything legacy")

**What was removed.** The `legacy` detection profile, with its fixed blanking and the option to keep the stimulated electrode. Also the `--profile` option, the `spikes.py` regression test and its committed baseline (`tests/data/legacy_baseline/`), the `*.spikes.h5` reader (`io/legacy.py`) and the viewer's fallback to it. Detection now has one configuration, `DetectionConfig()`. Results go to `results/<recording>/detect/`.

**What the defaults are.** Unchanged: the `spikes.py` detection settings (D10, Q8) with adaptive per-channel blanking and the stimulated electrode excluded (Q9). Re-running detection on all five recordings after the change reproduced the previous default results exactly.

**What replaces the regression test.** `tests/test_detection_snapshot.py` fingerprints detection on four real recordings (spike counts, a hash of every spike's sample index, noise, active and excluded channels). Any change in which spikes are found fails the test. Intended changes regenerate the snapshot (`MEAGRAPH_UPDATE_SNAPSHOT=1`) in the same commit. The parity with `spikes.py` that D11 established remains documented there and in `docs/METHODS.md`; recover the old test and baseline from git history (commit before this change) if it must be re-run.


---

## D22. Phase 4.5: TSPE, CFP and inhibition in the benchmark

**Status:** Proposed 2026-10-05 (implemented with the defaults below; each numbered choice awaits owner review)

**What was added.** Two estimators (`tspe`, `cfp`), inhibitory units in the simulator, signed scoring and inhibition scenarios in the benchmark. `ConnectivityResult.signed` marks results that contain inhibitory edges (negative weight). `meagraph graph` now runs all five methods by default; `meagraph plot` draws inhibitory edges in blue.

**Choices made, for review:**

1. **Inhibition model** (simulator). A fraction of units are inhibitory and all their connections are inhibitory (Dale's principle). An inhibitory spike deletes each target spike in [delay, delay + 10 ms] with probability S (benchmark: 20 % inhibitory units, S = 0.6, excitatory weight 0.06). This is a simple suppression model, chosen because its ground truth is unambiguous; it is not a conductance model. Excitatory-only networks are unchanged (verified spike for spike against the previous simulator).
2. **TSPE significance.** Elephant returns scores only. Scores are tested against 10 ms interval-jitter surrogates (as `cch_jitter`) with a normal fit (D17), two-sided, BH q = 0.05 across ordered pairs. The weight is the score's excess over the surrogate mean; its sign is the edge type. Elephant's defaults are kept (1 ms bins, its edge-filter windows, delays up to 25 ms). Elephant's matrices are (target, source) and are transposed; this was checked on simulations. Elephant's experimental `normalize` option is not used because it indexes delays by value instead of position (a bug).
3. **TSPE reverse artefact.** A strong excitatory i → j makes j → i score negative. The benchmark reports these as `inh_fp_reverse` and scores a variant, `tspe_noreverse`, that drops inhibitory edges opposite a significant excitatory edge. Whether to adopt that rule depends on the benchmark (see `docs/METHODS.md` § Validation). It would also hide true reciprocal excitatory–inhibitory pairs.
4. **CFP test.** "Clearly deviates from flat" is implemented as: peak of the 5 ms-smoothed CFP curve minus its mean, against circular-shift surrogates (each train shifted by an independent random offset), normal fit, BH. The original acceptance procedure (le Feber et al. 2007) could not be checked; the fitted curve M / (1 + ((τ − T)/w)²) + offset follows the le Feber group's form but should be verified against the paper.
5. **CFP validity rule.** The published minimum peak width (5 ms at 80 % of the peak; Martiniuc et al. 2015) rejects monosynaptic peaks, which are about 1 ms wide: on simulations, CFP's test found every connection and the width rule then rejected all of them. The default keeps the published rule, so `cfp` reports burst-scale functional relations only. The benchmark also scores `cfp_narrow` (no width rule) for comparison.
6. **CFP delays at the lower boundary.** CFP looks only at lags ≥ 0. A peak at or before zero lag (synchrony, or the reverse direction) is fitted at the 0.5 ms boundary. In DIV142, several CFP edges have exactly this delay. Proposal: also reject relations with delay < 1 ms.
7. **Benchmark cost.** TSPE and CFP use 200 surrogates in `meagraph benchmark` (their p-values come from a fitted normal, D17). Real-data defaults are 1000.

**Benchmark evidence** (`docs/METHODS.md` § Validation, points 7–11):

- TSPE is less sensitive than `cch_jitter` for excitatory connections; its value here is delays beyond 4 ms.
- Inhibition is undetectable at culture-like rates (0.2–2 Hz) by every method. DIV142's active electrodes fire at 0.1–1.7 Hz.
- 56 of TSPE's 64 false inhibitory edges are the reverse artefact; the reverse rule removes them all and loses 3 of 4 true ones.
- CFP with the published width rule finds no monosynaptic connections and marks 84 % of co-bursting unconnected pairs.

**Recommendations, for the owner:**

- (a) Make the reverse rule TSPE's default (point 3), and do not interpret inhibitory edges from recordings whose electrodes fire below a few Hz.
- (b) Keep CFP with its published rule as a burst-scale functional measure for Phase 6, and exclude it from connectivity maps; add the delay ≥ 1 ms rule (point 6).
- (c) Make the ISI_N burst thresholds scale with the pooled firing rate before using burst removal on higher-rate cultures.

**First real-data results (DIV142).** TSPE replicates 78 → 87 (3 ms) and 32 → 14 at its actual peak delay (5–6 ms, see Q16). It adds longer-delay edges, 87 → 14 (15–16 ms) and 87 → 32 (10–11 ms), that replicate in the associative recording; their correlogram peaks are broad, suggesting multi-step paths. No inhibitory edges were found.

---

## D23. Geometry from Kumar et al. 2026, and `cch_gauss2` for comparison with earlier lab analyses

**Status:** Proposed 2026-10-05 (geometry values applied at the owner's instruction; the estimator's open settings await the owner)

**Geometry.** The array is the 3D-MIND design of Kumar et al. 2026, and layer 1 is the bottom (owner). From the paper: sensors are 30 µm discs (radius 15 µm, unchanged), and layers are separated by a 250 µm SU-8 spacer (the fabrication recipe's thickness; the paper quotes a 25–250 µm range in general). The in-layer sensor pitch is **not** in the paper's main text, so the x–y pitch stays a 100 µm placeholder and `geometry_verified` stays false. Positions are stored as (row, col, layer), the project's convention; the paper's layer–row–column sensor names are not used.

**`cch_gauss2`.** Reproduces the paper's connection weight so new results can be checked against the lab's earlier analyses: a cross-correlogram (2 ms bins) fitted with two Gaussians, weight (a1 + a2)/(c1 + c2), net direction from the side of zero where the fitted curve peaks. Like the paper, it has no significance test; `p_values` are NaN and every tested pair with an off-zero peak is an edge (`min_weight` can set a threshold).

**Settings that must match the original NeuroExplorer/MATLAB analysis before weights can be compared numerically** (not stated in the paper):

1. correlogram normalization (here counts per bin; `rate_hz` also available);
2. lag range (here ±100 ms);
3. fit bounds and starting values (the MATLAB fit is unbounded; here widths are limited to 10 × the lag span, and three starting points are tried).

**What the implementation showed.** The paper's model has no constant term, so on correlograms with a flat baseline the second Gaussian absorbs the baseline and its width runs to hundreds or thousands of ms; the weight is then dominated by that width. In a simulated 30 min network, 7 of 22 true-connection fits reached the width bound. The weight still ranks true connections above others (median 0.17 vs 0.03 there), and the direction and delay are accurate, but without a test nearly every pair becomes an edge (115 of 120 pairs). Benchmark (96 networks): recall 0.81–0.89, precision 0.18, weight ROC area 0.84–0.86 (`cch_jitter`: 0.99); null networks get an edge on about half of all ordered pairs. Use it to compare a pair's weight across sessions with the lab's earlier numbers, not to decide whether a connection exists.

---

## D24. Spike sorting and a lower detection threshold: not adopted for now

**Status:** Proposed 2026-10-05 (owner asked for the debate; recommendation below)

**Context.** Kumar et al. 2026 thresholded at "4 × (median of the filtered signal)" in NeuroExplorer and spike-sorted. If that median is of |x|, the threshold is about 2.7σ (σ = median|x|/0.6745); meagraph uses 5σ and one train per electrode. The active DIV142 electrodes have σ ≈ 0.9–1.3 µV and spike troughs of 5–13 µV.

**Evidence: threshold** (DIV142 spontaneous, 10 min; "noise fraction" = positive / negative QC events on the active electrodes, the expected share of noise among negative events):

| Threshold | Spikes (all electrodes) | QC-active electrodes | Spikes on active electrodes | Noise fraction | Estimated real spikes | Real / √(all) |
|---|---|---|---|---|---|---|
| 5σ (current) | 4,489 | 8 | 3,179 | 0.38 | 1,971 | 35 |
| 4σ | 20,815 | 8 | 7,747 | 0.59 | 3,176 | 36 |
| 3.5σ | 88,274 | 7 | 17,408 | 0.80 | 3,482 | 26 |
| 2.7σ (paper, if median of \|x\|) | 925,576 | 1 | 17,800 | 0.96 | 712 | 5 |

For correlation methods, independent noise spikes add variance without adding signal, so detection power scales roughly with real spikes / √(all spikes). 4σ gains 60 % more real spikes but no power; below 4σ power falls, and at 2.7σ the polarity QC can no longer find active electrodes because noise dominates.

**Evidence: sorting** (5σ waveforms on the 8 active electrodes, 3 ms at 10 kHz = 30 samples): k-means with two clusters on the first 3 principal components gives separations of d' = 2.1–3.0, which is what splitting one Gaussian cloud gives (≈ 2.7); the first component explains only 17–24 % of the variance, so the waveforms are noise-dominated; 63–912 spikes per electrode in 10 min. There is no evidence of separable units.

**Arguments for sorting / a lower threshold:** continuity with the paper; 30 µm sensors in 3D tissue may record several neurons; thresholding and sorting go together (sorting removes the noise clusters a low threshold lets in); more spikes would help the slowest-firing electrodes.

**Arguments against, for these recordings:** the numbers above; 10 kHz sampling (the paper used 20 kHz) gives few samples per waveform; sorting adds operator-dependent choices; the main source of false edges in the benchmark is indirect paths, which sorting does not fix.

**Recommendation.**

1. Keep 5σ and per-electrode trains as the default.
2. Record at 20 kHz in future sessions (the MEA2100-Mini supports it), and revisit sorting with a SpikeInterface sorter when SNR and spike counts allow; the threshold should then be lowered together with sorting, not alone.
3. Optionally run 4σ as a sensitivity check on any key result (`DetectionConfig(threshold_sigma=4.0)`).
4. Check adjacent electrodes for the same neuron recorded twice (zero-lag synchrony, e.g. 78–87) instead of sorting.

---

## D25. Per-pulse stimulation sites from a stimulation log

**Status:** Proposed 2026-10-08 (implemented; the blanking choice in point 2 awaits the owner)

**What was added.** `meagraph.io.stimlog` reads `<tag>.stimlog.csv` written by the stimulation script, finds it next to the recording (or takes `--stimlog`), and matches it to the recorded STG events (`docs/DATA_FORMAT.md`). `StimEvents.pulse_sites` holds one site per pulse; detection folders save it. `meagraph audit` checks each logged site against its artifact.

**Choices made, for review:**

1. **Matching by order, checked by intervals.** Within each output, the k-th logged pulse is the k-th recorded one if the counts agree and the logged and recorded intervals agree within 0.1 s after fitting an offset and a linear clock drift. Otherwise each logged pulse is matched to the recorded pulse its interval predicts (within 0.1 s), and the rest are reported unmatched. The 0.1 s tolerance is far below the 2 s minimum interval, so pulses cannot be confused. On a simulated 2,900-pulse session with a 7 s offset, 50 ppm drift, 3 ms jitter, missing and extra pulses, every pulse was matched correctly.
2. **Stimulated electrodes stay in detection, with longer blanking on their own pulses.** With one site per output (`--stim-site`), that electrode is excluded entirely, as before. With per-pulse sites, excluding every stimulated electrode would remove up to 16 recording electrodes. Instead, each is blanked for `recovery_max_post_ms` (50 ms) after its own pulses, rather than for its measured recovery, because the stimulated electrode recovers slowest from its own pulse. Spontaneous connectivity is unaffected either way: it drops 200 ms after every pulse (D18).
3. **Both kinds of site are not combined.** A recording uses `--stim-site` or a log, not both; giving both is an error.
4. **Finding a log by time.** Besides a matching name or tag, a log with the stimulation script's default name (its own start time) is used if it started while the recording ran. Added 2026-10-08 for `2026-10-08T19-17-47.stimlog.csv` next to `2026-10-08T19-17-36Flex_Electronics_Andre_Random1_2.h5`.

**First real log (2026-10-08, device E-00278, 10 kHz). The culture was not connected and the headstage not closed during this recording (owner), so only the log matching and the stimulator behaviour below are meaningful; the neural and artifact observations are not.** All 186 pulses matched by order (timing within 26 ms). The audit could not confirm any logged site from the artifact: the artifact is not site-specific (electrode 57 shows a large artifact on every pulse; every electrode goes flat during pulses, consistent with hardware blanking), and the pulses were ±1 mV. **Cause, found afterwards:** the stimulation script switched on hardware blanking for each electrode only when it first stimulated it, so each electrode's artifact fell from about 2.3 mV to 0.02 mV after its first use (all 55 electrodes first used mid-session), and electrode 57 (API index 59, not in the script's table) was never set up and kept its full artifact. The script now sets up all 60 electrodes before the first pulse (mea-stim-protocols, random-sites-2). The recording also has almost no spikes (noise 9.4 µV; 0.3 negative crossings per electrode per minute away from pulses), so no electrode passes the QC and no graph can be built.

---

## D26. Discard crossings just before blanked windows too (proposal, not implemented)

**Status:** Proposed 2026-10-08 (detection unchanged until the owner decides)

**Problem.** Bridging replaces each stimulation window with a straight line. When the artifact leaves a large offset, the line has a sharp corner at its start, and the zero-phase filter spreads that corner's ringing about 2 ms in both directions. Detection already discards peaks up to `guard_ms` (1 ms) after a window, but not before one. In the 2026-10-08 recording, about 300 positive crossings per 4 electrodes fell 1–2 ms before the recorded pulse onset, where the raw signal had not yet changed; they count against the polarity QC. (That recording was later found to have no culture connected and an open headstage, so this example must be re-checked on a valid recording; the mechanism does not depend on the culture.) Separately, in the DIV142 associative recordings a third of electrode-pulses deviate from 2.9 ms before the recorded onset, earlier than `blank_pre_ms` (1 ms).

**Proposal.** A symmetric guard: also discard peaks (and exclude QC events) within 2 ms before each window; and measure, per recording, how long before the recorded onset the raw signal starts to change, to set `blank_pre_ms`. This changes detection output on stimulation recordings, so the detection snapshot would be regenerated with the change.


---

## D27. Recordings without a raw stream: analyse the end of the filter chain

**Status:** Proposed 2026-11-04 (the fallback is implemented; the phase correction in point 3 is not, and awaits the owner)

**Problem.** The owner plans to save only the `Filter (3)` stream. Every reader asked for the raw stream (D2), so such files could not be opened for analysis.

**What was implemented.**

1. **Stream choice.** `SessionConfig.stream` now defaults to `"auto"`: the raw stream if it was recorded (D2 unchanged), otherwise the end of the recorded filter chain (`meagraph.io._mcs_layout.analysis_stream`). If several filter streams were saved without raw and they are not one chain, the reader stops with an error. `"raw"` and stream labels still work as before. No CLI option was added.
2. **Reporting.** `meagraph info`, `audit` and `detect` print the stream used when there is no raw stream. `DetectionResult.analog_stream` holds it, and the detection folder's `provenance.json` records it (`analog_stream`). Detection itself is unchanged: the 300–3000 Hz zero-phase band-pass (D10) runs on top of the MCS filters.
3. **Also fixed.** A filter stream whose source was not saved counted as a root of the lineage, so files with only raw and `Filter (3)` (all 2026-11-03 recordings) listed `Filter (3)` first, and the viewer showed it as the default trace. Raw now always comes first. The real-file tests no longer assume every file is a 10 kHz, four-stream recording.

**Evidence: detection on Filter (3) is not interchangeable with raw.** Default `DetectionConfig`, run on the raw and the `Filter (3)` stream of the same file. Spikes on the same electrode within ±0.5 ms count as matched, over the union of QC-active electrodes:

| Recording | Spikes matched (% of raw, % of Filter 3) | QC-active electrodes |
|---|---|---|
| DIV142 spontaneous, 10 kHz | 57 %, 60 % | 3 shared; 5 raw-only, 3 Filter-3-only |
| 2026-11-03 Baseline1, 20 kHz | 67 %, 72 % | 11 shared; 4 raw-only, 2 Filter-3-only |
| 2026-11-03 Random1, 20 kHz (log ignored) | 67 %, 69 % | 8 shared; 3 raw-only, 9 Filter-3-only |

Amplitudes (median ratio 0.95–0.99) and noise (0.97–0.99) barely differ, and the median lag of matched spikes is 0–0.05 ms. The difference is the phase of the causal online chain (`DATA_FORMAT.md` § Analog streams). It reshapes spikes, which moves near-threshold events across the 5σ threshold, and changes which peak of an event is larger, which moves the polarity QC and the rebound rejection.

**Proposal (not implemented): undo the phase.** Run the documented chain (Butterworth HP 200 Hz and LP 3500 Hz, both 2nd order; notch at 60 Hz, r = 0.98) backward in time over `Filter (3)` before the detection band-pass. The online forward pass and this backward pass together are zero-phase with gain |H|², which is 0.85–1.03 across 300–3000 Hz. Measured on the same files:

| Recording | Spikes matched with raw | QC-active electrodes | Model check (raw run forward and backward through the chain, vs corrected Filter 3) |
|---|---|---|---|
| DIV142, 10 kHz | 96.0 %, 96.5 % | all 8 identical | 99.9 % of raw-chain spikes matched |
| 2026-11-03 Baseline1, 20 kHz | 96.7 %, 95.8 % | all 15 identical | 100.0 % |

The model check also confirms the filter settings at 20 kHz, where they had not been measured before. The risk is that the settings are not stored in the file (D2): if they are changed in Experimenter, the correction becomes silently wrong, and with no raw stream it cannot be re-checked. The recommendation to the owner is therefore to keep saving the raw stream (alone, or with `Filter (3)`), and to adopt the correction only if filter-only files are recorded anyway (Q23).

---

## D28. Stimulus-evoked responses per site, for choosing closed-loop inputs and readouts

**Status:** Proposed 2026-11-12 (implemented as `meagraph evoked`; the parameter choices below await the owner)

**Problem.** In the 2026-11-12 recordings no connectivity edge survives the burst control, yet bursts follow stimulation. Closed-loop training needs to know which sites drive the culture, which electrodes respond, and whether different sites give different responses (otherwise the agent has fewer actions than sites). Connectivity graphs do not answer this.

**What was added.** `meagraph.evoked` (`evoked_from_detection`, `evoked_responses`, `save_evoked`) and `meagraph evoked <results/recording>`. It reads only the detection folder and writes `results/<recording>/evoked/` (`sites.csv`, `responses.csv`, `channels.csv`, `trials.csv`, `decoding.json`, `psth.npz`, `network_bursts.csv`, config, provenance). For each site:

1. **Evoked bursts.** P(network-burst onset from pulse onset to 150 ms), with a Wilson interval, against sham times (a 0.25 s grid at least 1 s after any pulse and with no pulse in its window) in the same network state, by a one-sided binomial test. States: `in_burst` (pulse inside a burst), `active` (last burst ended < 5 s before), `silent`. The "all" comparison weights the sham rate by the site's own mix of states.
2. **Spike responses.** Spikes per readout electrode 12–150 ms after the pulse against shams, both outside bursts (Mann–Whitney, BH over site × electrode), first-spike latency and PSTHs.
3. **Decoding.** Multinomial logistic regression (ridge 1.0, standardized sqrt counts in bins 12/30/60/100/150 ms), stratified 5-fold CV repeated 5 times, balanced accuracy, label-permutation p (200; the whole CV is repeated per permutation). Four tasks: sites; sites + sham; sites from electrodes never stimulated in the session; and every pair of sites (100 permutations, BH), which is the basis for choosing a set of mutually distinguishable inputs.

**Choices made, for review:**

1. **Blanking is equalized across conditions.** Each electrode's response window starts at the latest blanking end over every output and site (pulse + recovery, or 50 ms for a stimulated electrode (D25) or an unrecovered one, + guard), for pulses and shams alike. Without this the decoder could recognize a site from where its electrode is blanked. Cost: every stimulated electrode is read only from ~52 ms.
2. **Spikes in the 2 ms before each pulse are dropped** (`drop_pre_ms`). In both 2026-11-12 stimulation recordings every active electrode has crossings 1.0–1.5 ms before onset (36 and 8 on active electrodes; 156 and 51 over all electrodes) and none in the 4.5 ms before that. This is D26's problem on a valid recording. Kept, they start a "burst" 1 ms before the pulse: 41/120 pulses were labelled `in_burst` and their evoked bursts were not counted.
3. **Burst onsets count from 0 ms, not 12 ms.** Evoked bursts start at 9–12 ms (the first spikes after blanking), so a 12 ms lower edge missed most of them.
4. **The state threshold (5 s) and the 12–150 ms window** are the defaults chosen from the Culture1 burst statistics; they are parameters.

**First results (2026-11-12, E-00278, 1000 mV biphasic 500 µs/phase, 8 sites).** `21-55-43`, 15 pulses per site; `21-28-58`, 5 per site (76 in place of 56):

| Site | Evoked bursts, 21-55-43 | 21-28-58 |
|---|---|---|
| 68 | 15/15 | 4/5 |
| 36 | 14/15 | 4/5 |
| 58 | 14/15 | 4/5 |
| 46 | 12/15 | 4/5 |
| 56 | 10/15 | not used |
| 76 | not used | 4/5 |
| 21 | 6/15 | 1/5 |
| 55, 61 | 0/15 | 0/5 |

Sham rates are 0.2–1.6 %. Bursts are evoked from silent periods too (e.g. 36 8/8, 68 5/5), unlike 500 mV on 2026-11-10 (0/23). Evoked spikes have spontaneous-like waveforms (all negative, same amplitude; correlation 0.97 for stimulated electrodes' own post-pulse spikes). Order: 76, 62 and 67 at 15–25 ms, then 51 and 43 at 25–40 ms, with 46 elevated for over 150 ms. 61, 55 and 21 do not evoke bursts but their own electrodes fire after their pulses (61: 3.1 spikes, first at 74 ms, on 15/15).

Decoding (21-55-43, chance 0.125): sites 0.48, sites from unstimulated readouts (44, 43, 52, 51, 62, 76, 67) 0.33, both p 0.005. The confusion matrix has two blocks: burst sites (36, 46, 58, 68; 56 partly) and non-burst sites (21, 55, 61). Pairs within the non-burst block are at 0.50. Within the burst block, 36 vs 58 is 0.81 (q < 0.05) and 1.00 in 21-28-58; 46 is not separable from 36, 58 or 68 in either session. 56 separates from every site (0.74–0.86), perhaps only through its 67 % burst probability. With 5 pulses per site no pair reaches q < 0.05 (the permutation p cannot fall much below 0.01–0.03), so site-pair claims need at least ~15 pulses per site.
