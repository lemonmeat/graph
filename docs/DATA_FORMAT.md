# MCS HDF5 data format (as observed)

This document describes what is actually in this lab's files, not what the MCS specification allows. It was verified on 2026-10-03 with h5py 3.16 on five recordings from device `E-00303`: three from exp3 (DIV140, 2026-07-27) and two from DIV142 (2026-07-29). Items marked **(unverified)** need confirmation from the owner.

## Files examined

| Short name | File | Duration | First sample at | Stimulation |
|---|---|---|---|---|
| beforestim | `2026-07-27T10-52-01nic_exp3_plastic_DIV140_beforestim_E-00303.h5` (1.39 GB) | 317.1 s | 0.5 s | none |
| stim | `2026-07-27T11-15-56..._stim_500width_500amplitude_4s_interval_3pulse_E-00303.h5` (444 MB) | 100.2111 s | 0 s | 26 trains; site inferred as **12** |
| stim47 | `2026-07-27T12-04-26..._stim_..._3pulse_stim47_E-00303.h5` (447 MB) | 100.2872 s | 0 s | 26 trains; site inferred as **47** (matches filename) |
| DIV142 spontaneous | `2026-07-29T15-16-32Flex Electronics Nick Acrylic Day 142_E-00303.h5` (2.5 GB) | 600.2 s | 0 s | none |
| DIV142 associative | `2026-07-29T15-32-23Flex Electronics Nick Acrylic Day 142 Associative Stimulation 1_E-00303.h5` (8.0 GB) | 1920.2 s | 0 s | STG 1 and STG 2, 300 pulses each; **sites cannot be inferred** (see below) |

**Structure of the DIV142 files.** Both have the same structure as the exp3 files: 60 channels, 10 kHz, the same four analog streams and lineage, and MCS spike streams. Their names contain spaces, so quote them in shells.

**Provenance attributes.** Root attributes: Multi Channel DataManager 1.14.10, `McsHdf5ProtocolType = RawData`, protocol version 3, McsDataTools 1.7.1.15. `/Data` attributes: `ProgramName = Multi Channel Experimenter 2.21.1`, `MeaLayout = MeaName = "ME21Combi60"`, `Date`, `DateInTicks` (.NET ticks), `FileGUID`. `/Data/Recording_0` attributes: `Duration` (µs; equals the end of data on the recording clock) and `TimeStamp` (µs).

The hardware, identified from the stream labels, is an **MCS MEA2100-Mini** with an STG stimulator.

## Tree

```
/Data/Recording_0/
  AnalogStream/Stream_{0,1,2,3}/
      ChannelData            (60, n_samples) int32, gzip, chunks (60, 1092)
      ChannelDataTimeStamps  (1, 3) int64: [FirstTimeStamp (µs), FirstIndex, LastIndex]
      InfoChannel            (60,) compound (see below)
  EventStream/Stream_0/
      InfoEvent              (8,) compound
      EventEntity_<EventID>  (5, n_events) int64   (absent when no events occurred)
  SegmentStream/Stream_{0,1}/
      InfoSegment            (n_entities,) compound
      SourceInfoChannel      (60,) compound
      SegmentData_<SegmentID>     (31, n) int32 cutouts
      SegmentData_ts_<SegmentID>  (1, n) int64 timestamps (µs)
```

The files contain no `TimeStampStream` or `FrameStream`. Only `Recording_0` exists.

## Analog streams

**Stream index is not processing order.** The lineage below is taken from `StreamGUID` → `SourceStreamGUID`. Readers must select streams by label or lineage, never by index.

| HDF5 stream | Label | Source | Measured response vs raw |
|---|---|---|---|
| `Stream_0` | `Data Acquisition (1);MEA2100-Mini; Electrode Raw Data1` | hardware | raw, DC-coupled |
| `Stream_3` | `Filter (1);Filter; Filter Data1` | Stream_0 | high-pass ≈ 200 Hz (−3 dB), slope consistent with 2nd-order Butterworth |
| `Stream_2` | `Filter (2);Filter; Filter Data2` | Stream_3 | adds a low-pass ≈ 3.5 kHz (|H| = 0.71 at 3.5 kHz, 0.10 at 4.5 kHz) |
| `Stream_1` | `Filter (3);Filter; Filter Data3` | Stream_2 | adds further low-frequency attenuation (|H| 0.018 vs 0.062 at 50 Hz). Purpose unknown **(unverified)** |

The responses were measured as the median over channels of √(PSD_stream / PSD_raw), Welch, beforestim 10–30 s:

| f (Hz) | 50 | 100 | 200 | 300 | 500 | 1000 | 3000 | 3500 | 4000 | 4500 |
|---|---|---|---|---|---|---|---|---|---|---|
| Filter 1 | 0.062 | 0.242 | 0.707 | 0.914 | 0.987 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |
| Filter 2 | 0.062 | 0.242 | 0.707 | 0.914 | 0.986 | 0.998 | 0.896 | 0.706 | 0.376 | 0.096 |
| Filter 3 | 0.018 | 0.189 | 0.697 | 0.920 | 1.002 | 1.016 | 0.914 | 0.720 | 0.383 | 0.098 |

Filters 1 and 2 together match the Kumar et al. 2026 methods: a "second-order Butterworth band-pass 200 Hz to 3,500 Hz". All filter streams show zero-sample lag against raw.

**The software filter settings are not stored in the file.** The `HighPass*` and `LowPass*` fields of `InfoChannel` are empty or −1 for every stream. This is the main reason the pipeline should filter raw `Stream_0` itself (see `DECISIONS.md`).

**Length mismatch.** In beforestim, the filter streams have 3,170,000 samples against 3,171,000 for raw. They share the same first timestamp, so they end 0.1 s earlier. In the stim files all four streams have equal length.

### `InfoChannel` and unit conversion

These values are identical on every channel and in every stream and file examined:

| Field | Value | Meaning |
|---|---|---|
| `Unit` | `V` | |
| `Exponent` | −12 | |
| `ConversionFactor` | 8670 | |
| `ADZero` | 0 | |
| `Tick` | 100 | µs per sample, so **fs = 1e6 / Tick = 10,000 Hz** (the paper used 20 kHz; these files are 10 kHz) |
| `ADCBits` | 24 | |
| `RawDataType` | `Int` | |
| `GroupID`, `ElectrodeGroup` | 0 | |

`ChannelID`, `RowIndex` and `Label` vary per channel. `ChannelID == RowIndex == 0..59` in InfoChannel order.

```
volts = (raw − ADZero) × ConversionFactor × 10^Exponent      → 8.67 nV per LSB (0.00867 µV)
ADC rail = ±2^23 LSB = ±72.73 mV                            (the stimulated electrode saturates exactly here)
ChannelData row for a channel = InfoChannel.RowIndex         (identity here, but readers must honour it)
time of sample i (µs, recording clock) = FirstTimeStamp + i × Tick
```

**Units sanity check.** For raw data band-passed at 300–3000 Hz (3rd-order, zero-phase), noise σ (MAD/0.6745) has a median of **4.67 µV** across channels, with a range of 0.73–9.57 µV (beforestim, 10–30 s). These are typical MEA noise levels, so the conversion is correct.

### Time base

All event and segment timestamps are in µs on the **MCS recording clock**. Analog data starts at `FirstTimeStamp`, which is 0.5 s in beforestim and 0 in the stim files. `Recording_0.Duration` equals `FirstTimeStamp + n_samples × Tick`: 317.6 s for beforestim, which is 0.5 s + 317.1 s.

## Channel labels and the 4×4×4 map

Labels have the form `"E-00303 NN"`. `E-00303` is the same in every channel, file and event label; it appears to be the device ID **(unverified)**. `NN` is the standard MCS 60-electrode label.

Channel order (InfoChannel order, identical in all files and all four streams):

```
47 48 46 45 38 37 28 36 27 17 26 16 35 25 15 14 24 34 13 23 12 22 33 21 32 31 44 43 41 42
52 51 53 54 61 62 71 63 72 82 73 83 64 74 84 85 75 65 86 76 87 77 66 78 67 68 55 56 58 57
```

**Only 60 channels are recorded, not 64.** `MEA_CUBE` in `mcs.py` places 59 labels on the 4×4×4 grid. Label `15` is not placed. Five cells are empty, as (row, col, layer): (1,2,4), (1,3,4), (1,4,4), (4,3,4), (4,4,4).

**Label `15` is the reference electrode** (confirmed by the owner, 2026-10-03). Consistent with that, its noise is 0.73 µV, it shows no spikes, and its stimulation artifact is 137–237 µV, against ≥ 7 mV on every other channel.

The current map (`MEA_CUBE`, as (row, col) per layer):

**Layer 1**

| row \ col | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | 48 | 46 | 45 | 38 |
| 2 | 85 | 75 | 65 | 86 |
| 3 | 84 | 74 | 64 | 83 |
| 4 | 41 | 43 | 44 | 31 |

**Layer 2**

| row \ col | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | 37 | 28 | 36 | 27 |
| 2 | 76 | 87 | 77 | 66 |
| 3 | 73 | 82 | 72 | 63 |
| 4 | 32 | 21 | 33 | 22 |

**Layer 3**

| row \ col | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | 17 | 26 | 16 | 35 |
| 2 | 78 | 67 | 68 | 55 |
| 3 | 71 | 62 | 61 | 54 |
| 4 | 12 | 23 | 13 | 34 |

**Layer 4**

| row \ col | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | 25 | · | · | · |
| 2 | 56 | 58 | 57 | 47 |
| 3 | 53 | 51 | 52 | 42 |
| 4 | 24 | 14 | · | · |

**The map cannot be validated from the data.** It is not stored in the file. A spatial check gave no confirmation: stimulation-artifact amplitude does not fall off with grid distance from the stimulated site (Spearman ρ = −0.06, p = 0.63 for stim at 12; ρ = −0.18, p = 0.19 for stim at 47). The artifact is larger on electrodes that share the stimulated electrode's *row* (median 19.1 vs 14.8 mV for stim 12, and 23.5 vs 16.9 mV for stim 47). That pattern points to crosstalk along shared interconnects rather than conduction through tissue. **Provenance (owner, 2026-10-03):** the map encodes how the headstage channels are wired to the electrodes of the folded array.

**Physical units are unknown (unverified).** Electrode pitch within a layer, layer spacing, and contact orientation are needed. Kumar et al. 2026 give a 30 µm sensor diameter, SU-8 spacers of 25–250 µm per device, and layer 1 = bottom. The in-layer pitch is not stated in the main text.

## Stimulation: `EventStream/Stream_0`

The stream is labelled `Stimulator (1);Stimulator; STG Events1` (`DataSubType = StgSideband`). `InfoEvent` has 8 rows:

| EventID | Label | Source |
|---|---|---|
| 1 | `STG 1 Single Pulse Start` | `Sideband Data 0` (ID 70) |
| 2 | `STG 1 Single Pulse Stop` | `Sideband Data 0` |
| 3 | `STG 1 Marker Start` | `Sideband Data 0` |
| 4 | `STG 1 Marker Stop` | `Sideband Data 0` |
| 5–8 | the same for `STG 2` | `Sideband Data 1` (ID 71) |

`SourceChannel*` refers to STG sideband channels, **not electrodes**. **The file does not record which electrode was stimulated.**

**Dataset naming.** Datasets are named `EventEntity_<EventID>`, the McsPy convention (`McsPy/McsData.py:804`). In the stim files only `EventEntity_1` (Single Pulse Start) and `EventEntity_2` (Single Pulse Stop) exist. In beforestim, none exist. `mcs.py` maps these off by one (see `EXISTING_CODE.md`).

**Rows of each entity:**

| Row | Content |
|---|---|
| 0 | timestamp (µs, recording clock) |
| 1 | duration (all 0) |
| 2 | info type |
| 3–4 | N/A |

### Observed protocol (both stim files)

- **Train count and spacing:** 26 trains. The first is at t = 0, the very first sample, so it has no pre-stimulus baseline. The rest follow at a fixed **4.003 s** interval.
- **Start/Stop spacing:** Single Pulse Stop = Single Pulse Start + **3.000 ms** exactly.
- **Waveform**, from the trial-averaged raw epoch on non-stimulated electrodes:
  - The artifact begins 1–2 samples (0.1–0.2 ms) after Start.
  - It contains **three biphasic cycles of about 1 ms each**: roughly 500 µs at one polarity, then 500 µs at the other, three times.
  - It ends at Stop, followed by about 0.5 ms of settling.
  - This matches the filename tokens `500width ... 3pulse`. The meaning of `500amplitude` (mV in voltage mode vs µA) is **(unverified)**.
- **Stimulated electrode:** saturates at −72.7 mV for about 3 ms.
  - Electrode 12 then sits at about −7 mV.
  - Electrode 47 rebounds to **+43 mV at +6 ms** and is still drifting, so it is unusable for tens of ms.
- **Other electrodes:** peak artifact of **7–37 mV**, larger on same-row electrodes. Electrode 15 sees about 0.2 mV.
- **Stimulated site:** taken as the largest-artifact channel. This is **12** for the 11-15 file, where the filename does not say, and **47** for the 12-04 file, which agrees with "stim47".

### Associative stimulation protocol (DIV142)

This file uses both STG outputs:

| Event | Entity |
|---|---|
| STG 1 Single Pulse Start / Stop | EventEntity 1 / 2 |
| STG 2 Single Pulse Start / Stop | EventEntity 5 / 6 |

**Events mark individual pulses.** In the exp3 files, one event spanned a whole 3-pulse burst. Here each event is a single pulse.

- **Pulses:** Start to Stop is 2.0–2.1 ms. The raw artifact shows about 1 ms per phase.
- **Trains:** every 5 s (0.2 Hz). Each train has 1, 2 or 3 pulses, 3 ms apart onset to onset, cycling 1-2-3. Each output delivers 150 trains and 300 pulses.
- **Blocks:** the outputs alternate in blocks of 50 trains (245 s), with a 375 s gap between one output's blocks:

  | STG output | Blocks (s) |
  |---|---|
  | STG 2 | 60–305, 680–925, 1300–1545 |
  | STG 1 | 370–615, 990–1235, 1610–1855 |

  The two outputs never fire together.

**The stimulated sites cannot be inferred.** During every pulse, 36 (STG 1) or 38 (STG 2) of the 59 channels reach the ADC rail (±72.7 mV). After each pulse, every channel carries a 4–10 mV offset that decays over tens of ms. **Which electrodes STG 1 and STG 2 drove must come from the owner (Q14).**

**Artifact recovery** (`meagraph audit`), in the detection band after a 1 ms post-pulse blank:

| Recording | Median | 90th percentile | Slowest |
|---|---|---|---|
| exp3 | 5.0–5.3 ms | 5.4–6.8 ms | the stimulating electrode, 9.3 ms (12) and 10.8 ms (47) |
| DIV142 associative | 6.6–6.8 ms | 7.1 ms | 17.1 ms (56, STG 2) |

## MCS online spike streams: `SegmentStream`

| Stream | Label | Source | Entities |
|---|---|---|---|
| `Stream_1` | `Spike Detector (1)` | analog `Stream_1` (Filter 3) | 60, one per electrode |
| `Stream_0` | `Spike Sorter (1)` | the Spike Detector | 121 / 311 / 244 units (`Sorter Unit` column) |

**Cutouts:** 31 samples. `PreInterval` is 1000 µs and `PostInterval` is 2000 µs, so the detection sample is at index 10. Values are int32 in the source stream's ADC units, at the same 8.67 nV scale. Timestamps are in µs on the recording clock.

**Events can predate the saved analog data.** In beforestim, 2 events (both streams) are timestamped at 0.4036 s, before the first analog sample at 0.5 s. MCS's online processing ran before saving started. `meagraph.io.read_mcs_spikes` drops such events with a warning.

**Detector settings (threshold, polarity) are not stored.** From the data:

- **Trough depth:** event troughs have a median of about −4.4 σ of the source stream.
- **Polarity:** some events have no meaningful negative trough, which suggests both-polarity detection.
- **Counts:**

  | File | MCS detector events | `spikes.py` spikes |
  |---|---|---|
  | beforestim | 6283 | 165 |
  | stim | 8243 | 220 |
  | stim47 | 9330 | 249 |

- **Stimulation contamination:** in the stim files, **55 %** of MCS detector events fall within −1 to +15 ms of a pulse onset.

## Signal quality (Phase 0 measurement, and important)

**Test.** Extracellular action potentials are predominantly negative-going. Noise is roughly symmetric. So for each channel I counted negative and positive peaks at the same threshold, on raw data band-passed at 300–3000 Hz, with peaks ≥ 1 ms apart. In the stim files I masked −2 to +50 ms around every pulse.

| File | k | Negative peaks | Positive peaks | Channels with negative ≫ positive (binomial p < 0.001, ≥ 20 negative) |
|---|---|---|---|---|
| beforestim (317 s) | 4.5 | 925 | 1043 | 1: `12` |
| beforestim | 5.0 | 189 | 189 | 1: `12` |
| stim (99 s unmasked) | 4.5 | 1337 | 1948 | 0 |
| stim47 (99 s unmasked) | 4.5 | 1720 | 1790 | 2: `12`, `72` |
| stim47 | 5.0 | 1369 | 1335 | 3: `12`, `31`, `73` |

**Interpretation.** In these three recordings, threshold crossings on most electrodes are about as frequent on the positive side as on the negative side, which is what noise produces. Electrode 12 is the only electrode with a robust negative excess in every file. Electrode 63, for example, has 162 MCS-detector events but a band-passed signal with std = MAD-σ = 3.21 µV, a maximum excursion of 16 µV (5 σ), and 110 peaks at 4 σ over 317 s, consistent with Gaussian noise.

This test is a heuristic: positive or biphasic spikes exist, and a 3D device could record unusual polarities. But it implies that both the MCS detector output and low-threshold detections in these particular files are dominated by noise. Any connectivity estimate from them is signal-limited, whatever the method.

## SpikeInterface and Neo compatibility

**`spikeinterface.extractors.read_mcsh5`** (SI 0.105.0, tested in a scratch venv) loads `AnalogStream/Stream_<i>`. With `return_in_uV=True` its values are **identical to `mcs.py`** (max |Δ| = 0.0 µV over 10 s × 60 channels). It has these limitations:

- It **drops `FirstTimeStamp`**: segment `t_start` is 0, not 0.5 s.
- It ignores `RowIndex` and assumes InfoChannel order equals data row order.
- Its ADZero offset formula is dimensionally wrong (`−ADZero·10^Exp·gain`). This is harmless only because ADZero = 0.
- It derives fs by treating `ChannelDataTimeStamps` columns inconsistently (µs vs index). The result is correct for these files.
- It reads no events or segment streams, no stream labels or lineage, and `Recording_0` only.
- Its channel IDs are `"Ch0".."Ch59"`, with labels stored only as a property.

**Neo 0.14.5** has only `RawMCSRawIO`, which handles MCS binary `.raw` files, not HDF5.

**ProbeInterface 0.4.0** supports 3D probes. It requires `plane_axes` per contact (the contact orientation). After attaching one, SI's `get_channel_locations()` **defaults to `axes="xy"`, which silently drops z**. SI's neighbourhood and sparsity code (`core/sparsity.py`, `core/recording_tools.py`, `matched_filtering` peak detection) calls it with that default. See `DECISIONS.md`.

## Minimal correct reading recipe

```python
g = f["Data/Recording_0/AnalogStream/Stream_0"]            # choose by Label, not index
info = g["InfoChannel"][:]
fs = 1e6 / info["Tick"][0]
t0 = g["ChannelDataTimeStamps"][0, 0] * 1e-6               # first-sample time on the recording clock
uv = (g["ChannelData"][info["RowIndex"][k], i0:i1] - info["ADZero"][k]) * info["ConversionFactor"][k] * 10.0**info["Exponent"][k] * 1e6
ev = f["Data/Recording_0/EventStream/Stream_0"]
labels = {r["EventID"]: r["Label"] for r in ev["InfoEvent"][:]}
onsets_s = ev["EventEntity_1"][0] * 1e-6                   # EventEntity_<EventID>; EventID 1 = "STG 1 Single Pulse Start"
```
