# Refactor plan

Read alongside `DATA_FORMAT.md` (what the files contain), `EXISTING_CODE.md` (what the scripts do) and `DECISIONS.md` (design choices D1–D8).

**Status (2026-10-03).** The owner approved the plan with defaults.

| Phase | Status |
|---|---|
| 0 | done |
| 1 | done |
| 2 | done |
| 3 | done |
| 4 | done |
| 5 | next |

Also resolved:

- **Q2, Q3:** 15 is the reference; the map is the headstage wiring.
- **Q8, Q9:** defaults approved; see D10.
- **Viewer:** custom matplotlib inside the package (D12).

New open questions:

- **Q14:** sites in the associative file.
- **Q15:** stim47 QC.
- **Q16:** synaptic window width.
- **Q17:** D18 and the primary burst control.

Q11–Q13 are resolved: the package is `meagraph`, the existing `.venv` is used, and the legacy scripts are frozen. Q1–Q3 are deferred, with placeholder geometry flagged (D8).

## What Phase 0 changed about the plan

1. **The data is signal-limited.** In the three current recordings, only electrode 12 (and perhaps 72, 31 and 73 in one file) shows clear negative-going spiking above the noise floor. Most threshold crossings, including most of the 6,283 MCS-detector events in the baseline file, look like noise. Every estimator therefore has to report how much data it was given and what it could have detected. The synthetic validation suite also has to model this SNR regime, not just clean spike trains.
2. **The array has 60 recorded channels**: 59 cube positions plus label `15`, which looks like the reference. The 4×4×4 grid has 5 empty cells. The probe definition must allow missing cells.
3. **Stimulation site identity is not in the file.** It has to come from config or metadata, or be inferred from the artifact (which agreed with "stim47"). Each train is three biphasic pulses spanning 3 ms, and the stimulated electrode stays railed or drifting for more than 6 ms.
4. **Two bugs in the legacy code** matter for the refactor: event labels are off by one, and time bases are mixed. Both are fixed in the new package; the old scripts are left frozen (question Q13).

## Architecture

This is the layout planned in Phase 0. The layout as built, with its layers and output folders, is in `ARCHITECTURE.md`; some modules below (for example `detect/qc.py`, `connectivity/significance.py`, `apps/viewer/`) ended up elsewhere.

```
src/meagraph/                    (package name: proposal, see Q11)
  io/
    mcs_h5.py        McsH5Recording(BaseRecording): lazy, chunked, t_start, RowIndex, labels as channel IDs (D1)
    mcs_events.py    StimEvents (onset/offset per STG, EventID-correct), MCS segment streams → NumpySorting
    inventory.py     file → typed summary (streams, lineage, durations, events) for `meagraph info`
    legacy.py        read *.spikes.h5 sidecars (regression, viewer compatibility)
    nwb.py           export recording metadata, spikes, events, graphs to NWB
  probe/
    data/            cube map CSV (label,row,col,layer) + geometry YAML (pitch, spacing, diameter); planar 60 layouts
    build.py         → probeinterface.Probe (ndim=3 or 2), attach to recording; distances always in 3D (D3)
  preprocess/        SI pipelines from config: artifact removal around StimEvents, band-pass, (optional) CMR
  detect/
    threshold.py     SI detect_peaks(by_channel) + legacy post-filters (max_uv, guard, rebound) → SpikeTrains
    qc.py            noise σ, polarity control (neg vs pos crossings), "active channel" flags
    bursts.py        single-channel bursts (MaxInterval), network bursts (population-rate threshold)
    sorting.py       optional wrappers (CPU sorters), off by default
  connectivity/
    base.py          SpikeTrains, ConnectivityResult, Estimator protocol, registry
    cch.py           CCH: hollow-Gaussian baseline (Stark & Abeles), interval-jitter surrogate variant
    sttc.py          spike time tiling coefficient (Elephant) + surrogate significance
    cfp.py           conditional firing probability (le Feber et al.)
    evoked.py        stimulus-triggered estimator (port of stim_connectivity.py)
    glm.py           Poisson GLM coupling filters (NeMoS, optional extra)
    te.py            transfer entropy (IDTxl if practical, else documented omission)
    significance.py  surrogates (Elephant), BH-FDR, bootstrap CIs
    graph.py         → networkx.DiGraph with 3D node positions; GraphML / JSON I/O
    compare.py       graph diffs across conditions (edge overlap, weight change with CIs, permutation tests)
  stimulation/       audit (port of stim_audit.py), PSTH, latency maps, response probability, pre/post diffs
  synth/             ground-truth networks on any probe geometry; confound knobs; optional voltage synthesis
  benchmark/         estimator × (duration, rate, SNR, confound) sweeps → precision/recall/ROC/AUPR/delay error
  reservoir/         state extraction, sklearn readouts, memory capacity, separation/generalisation rank
  realtime/          SpikeSource (replay | live stub), Encoder, Decoder, StimulationSink stub, LoopRunner, Pong1D
  viz/               figure functions: data in, Figure out, no file I/O
  config/            pydantic models, YAML load/save, provenance (config + package version + git hash + input file)
  cli.py             `meagraph info | detect | audit | graph | compare | view` (console entry point)
apps/viewer/         rebuilt visualize.py on the package API (4x4x4 selector preserved)
tests/               unit tests with a tiny generated MCS-format fixture; real-file tests marked and skippable
docs/
```

**Changes from your suggested layout:**

- **`cli/` moves into the package** as `cli.py`, so `pip install` provides the command.
- **New `synth/` and `benchmark/` modules** are added, because validation is a first-class deliverable.
- **QC lives in `detect/qc.py`.**

**Core in-memory objects:**

- `Session`: the SI recording with its probe attached, `StimEvents`, and metadata (device, condition, stimulated site).
- `SpikeTrains`: channel IDs, times in seconds on the recording clock, 3D positions, `t_start` and `t_stop`. It converts to and from SI `Sorting` and Neo/Elephant `SpikeTrain`.
- `ConnectivityResult`:

  | Field | Contents |
  |---|---|
  | `weights` | n×n array |
  | `delays_ms` | n×n array |
  | `pvalues` | n×n array |
  | `significant` | n×n array |
  | `node_ids`, `positions_um` | node identity and 3D position |
  | `method`, `params` | estimator name and parameters |
  | `n_spikes`, `duration_s`, `provenance` | data and provenance metadata |

  It serialises to GraphML or JSON.

## Phases

Each phase ends with a commit and a report covering what was done, how it was verified, and what is uncertain. Scientific parameters are raised for decision when a phase first needs them, not chosen silently.

| Phase | Deliverables | Verification |
|---|---|---|
| **1. Foundation** | `pyproject.toml` (uv, editable), package skeleton, `io/` (D1, D4, D5), `probe/` from data files (cube + planar 60), config and provenance, pytest setup | Tests compare the reader against raw h5py (values, `t_start`, `RowIndex` permutation in the fixture, event mapping); probe round-trips with z intact; works on all 3 real files |
| **2. Preprocess and detection** | SI pipeline reproducing `spikes.py`; QC (noise, polarity control); `stimulation/audit`; `meagraph info/detect/audit` | **Regression test against the 3 existing sidecars** (proposed tolerance: ≥ 99 % of spikes matched within ±1 sample in both directions, σ within 1 %); old scripts untouched |
| **3. Viewer** | `apps/viewer` on the package API, preserving the UX contract in `EXISTING_CODE.md`; stimulation lines at pulse onset | `--save` renders compared side by side with the old viewer; selector and click behaviours checked |
| **4. Validation harness + first estimators** | `synth/` (multivariate Hawkes with delays on the probe geometry, numpy-only; optional Brian2 Izhikevich; confound knobs for network bursts, shared stimulus drive and rate heterogeneity; optional voltage synthesis at the measured noise level), `benchmark/`, `connectivity/` core, CCH (hollow and jitter), STTC, graph I/O | Recovery curves against duration, rate and SNR; false-positive rate under confounds with no true edges; `spontaneous_ccg.py` reproduced on its own inputs |
| **4.5. TSPE, CFP and inhibition** (done 2026-10-05) | `tspe` (Elephant + jitter significance), `cfp`; inhibitory units in `synth`; signed scoring and inhibition scenarios in `benchmark`; D22 | Same benchmark grid plus inhibition scenarios; TSPE orientation and delays checked on simulations |
| **5. More estimators** | CFP, evoked (stimulus-triggered), Poisson GLM, TE (if practical), `METHODS.md` with references | Each estimator on the same benchmark grid; `METHODS.md` states what each method can and cannot claim |
| **6. Stimulation and comparison** | PSTH, latency maps, evoked vs spontaneous, pre/post diffs, cross-session graph comparison, NWB export | Synthetic "plasticity" (edges added or removed between conditions) detected at known rates; NWB file validates with `pynwb` |
| **7. Reservoir and real-time** | State extraction, readouts, memory capacity and separability; `SpikeSource`/`Encoder`/`Decoder`/`StimulationSink`/`LoopRunner` with Pong1D; `REALTIME.md`; README and ARCHITECTURE finalised | End-to-end loop on a replayed `.h5` with per-step latency logged; readout tests on synthetic reservoirs |

Old scripts stay runnable until the Phase 2 regression passes and you approve retiring them.

## Open questions

### Physical facts (cannot be read from the files)

- **Q1.** In-layer electrode pitch (row and column, µm), layer spacing (µm; spacer thickness for device E-00303), and electrode diameter (the paper says 30 µm). Is layer 1 the bottom, as in the paper? Do contacts face up?
- **Q2.** Where does the `MEA_CUBE` map come from (pad drawing, mask)? The data cannot confirm it. Are the 5 empty cells absent or present-but-unwired?
- **Q3.** Is label `15` the reference electrode? It shows 0.73 µV noise, no spikes and almost no artifact. Should it be excluded everywhere?
- **Q4.** For the planar 60-channel recordings: which MCS MEA type (pitch 100 or 200 µm, 8×8 naming)? Are sample files available?
- **Q5.** What are the Filter 1/2/3 settings and the MCS Spike Detector settings in Multi Channel Experimenter? This is for documentation only (D2).
- **Q6.** Stimulation protocol. Is `500amplitude` in mV (voltage mode) or µA? Is the pulse negative-first? Is it 3 biphasic pulses with 500 µs per phase, as measured? Was the 11-15 file stimulated on electrode 12, as the artifact suggests?
- **Q7.** Data:
  - Is there an "afterstim" baseline for exp3, for the pre/post comparison?
  - Are there longer (for example 15 min) or more active recordings?
  - Is it acceptable to move raw files into `data/`? `spikes.py` and `visualize.py` currently glob next to themselves, so I would leave the files where they are unless you say otherwise.

### Scientific choices (needed in Phases 2, 4 and 5; Phase 1 can proceed without them)

- **Q8. Detection threshold for analysis.** Proposal: keep the `spikes.py` profile as the regression default (k = 5 MAD, 300–3000 Hz, 3rd-order zero-phase, 1 ms refractory, rebound rule, 1000 µV cap). Add the polarity control as a per-channel "active" flag, and run connectivity only on active channels.
- **Q9. Stimulation blanking.** Today the effective window is −1 to +9 ms around onset, plus a 1 ms guard. The stimulated electrode is still drifting at +6 ms. Options:
  - (a) keep the fixed window;
  - (b) a per-channel window set by the measured recovery time (from the audit);
  - (c) (b) plus excluding the stimulated electrode during its own stimulation block.

  I recommend (c).
- **Q10. Significance defaults.** Proposal:
  - CCH: 0.5 ms bins, ±30 ms lags, a 1–4 ms test window, and a 10 ms hollow-Gaussian baseline, as in `spontaneous_ccg.py`.
  - Jitter surrogates: interval jitter of ±5 ms, 1000 surrogates.
  - Multiple comparisons: BH-FDR at q = 0.05 across ordered pairs. `spontaneous_ccg.py` claims this correction but does not apply it.
  - Minimum of 50 spikes per train.

### Engineering

- **Q11.** Package name: `meagraph`, or your preference.
- **Q12.** Environment: add dependencies to the existing uv `.venv` (Python 3.13)? Phase 0 left it untouched and tested everything in a scratch venv.
- **Q13.** Legacy bugs: leave `mcs.py` frozen as the reference (recommended; `spikes.py` output does not depend on event labels), or patch the event-label bug now so the old viewer and `stim_audit.py` show onsets correctly?
