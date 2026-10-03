# Design decisions

Each entry records what was decided, why, and what evidence supports it. Status values:

- **Proposed:** awaiting owner approval.
- **Accepted:** approved by the owner.
- **Superseded:** replaced by a later entry.

---

## D1. Custom MCS HDF5 `BaseRecording` instead of `spikeinterface.extractors.read_mcsh5`

**Status:** Proposed (Phase 0, 2026-10-03)

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

**Status:** Proposed

**Decision.** All analysis starts from raw `Stream_0` and does its own filtering in documented SpikeInterface pipelines. The MCS filter streams remain selectable in the viewer. MCS Detector and Sorter events are readable, for comparison only.

**Why.**

- The filter parameters are not stored in the file. Filter 3 is unknown and was only measured empirically.
- The MCS detector's threshold and polarity are not stored.
- In the stim files, 55 % of MCS detector events sit within 15 ms of a stimulus pulse.
- `spikes.py` already uses `Stream_0`, so regression is direct.

---

## D3. 3D geometry: never rely on SpikeInterface's default 2D channel locations

**Status:** Proposed

**Finding.** ProbeInterface stores a 3D probe correctly. However, `BaseRecording.get_channel_locations()` defaults to `axes="xy"`. SI code that builds channel neighbourhoods or sparsity calls it without `axes`: `core/sparsity.py` (radius sparsity), `core/recording_tools.py`, and the `matched_filtering` peak detector, observed in SI 0.105.0. For the cube, that silently projects onto x-y. Electrodes stacked in different layers then appear to be at distance 0.

**Decision.**

- Package code always asks for `axes="xyz"`.
- Peak detection uses `method="by_channel"`. This matches `spikes.py` and is geometry-free.
- Any SI step that needs sparsity receives an explicit sparsity mask computed from true 3D distances.
- A unit test asserts that the probe round-trips with z intact.

This entry must be re-checked whenever the SI version changes.

---

## D4. Decode MCS entities by ID, not by row

**Status:** Proposed

**Decision.** Event datasets map as `EventEntity_<EventID>` and segment datasets as `SegmentData_<SegmentID>`, following the McsPy convention.

**Why.** Legacy `mcs.py` indexes by InfoEvent row, which mislabels the stimulation events (Start appears as Stop, and Stop as "Marker Start"). This was verified against the raw artifact timing; see `EXISTING_CODE.md`.

---

## D5. One time base: seconds on the MCS recording clock

**Status:** Proposed

**Decision.**

- t = 0 is the MCS recording clock zero, the same clock as event and segment timestamps.
- The analog segment's `t_start` is `FirstTimeStamp`.
- All spike, event and analysis times are stored on this clock, with an `_s` suffix.
- Legacy `spikes.py` timestamps are relative to the first sample. They are converted (+`t_start`) when loaded for regression.

---

## D6. Environment and dependency policy

**Status:** Proposed

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

**Phase 0 test.** SI 0.105.0, ProbeInterface 0.4.0, Neo 0.14.5 and Elephant 1.2.1 install and import cleanly on Python 3.13 and numpy 2.5.3. This was tested in a scratch venv; the project venv is untouched.
