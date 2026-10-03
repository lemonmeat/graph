# Methods

This file is written so it can be adapted for the independent work report. Parameter values are the `default` profile of `meagraph` unless stated otherwise. Each method names the module that implements it. Design rationale is in `DECISIONS.md`.

## Recording

- **Hardware.** Extracellular signals were recorded from 3D neuronal cultures on a folded 4×4×4 microelectrode array (3D-MIND design; Kumar et al., 2026) connected to an MCS MEA2100-Mini system.
- **Channels.** 59 electrodes plus one reference (electrode 15). Of the 64 grid positions, 5 have no recorded electrode.
- **Sampling.** 10 kHz, 24-bit (8.67 nV per bit, ±72.7 mV range).
- **Data used.** Raw, unfiltered data was saved by Multi Channel Experimenter in MCS HDF5 format. All analysis starts from the raw stream; filtering in the acquisition software was not used (DECISIONS.md D2).
- **Software.** Data were read and processed with SpikeInterface (Buccino et al., 2020). The electrode geometry was represented as a ProbeInterface probe (Garcia et al., 2022). See `meagraph.io` and `meagraph.probe`.

## Spike detection (`meagraph.detect`)

1. **Artifact bridging.** Stimulation windows (next section) were replaced by a straight line between the samples bordering each window, before any filtering. This keeps the stimulus artifact from ringing through the filter.
2. **Filtering.** Traces were band-pass filtered at 300–3000 Hz with a 3rd-order Butterworth filter applied forwards and backwards (zero phase).
3. **Noise level.** Each channel's noise was estimated as σ = median(|x|)/0.6745 over the whole recording (Quiroga et al., 2004). This estimate is robust to the spikes themselves.
4. **Threshold.** A spike was a negative peak below −5σ that was the minimum within ±1 ms.
5. **Rejection rules.** Peaks were rejected if any of these held:
   - |amplitude| > 1000 µV;
   - the peak fell inside a stimulation window or within 1 ms after one;
   - the largest positive value in the −1 to +2 ms cutout exceeded the peak's magnitude. Stimulation artifacts ring symmetrically, while a spike's trough dominates.
6. **Waveforms.** Cutouts of −1 to +2 ms of the filtered trace were stored with each spike.

**Validation.** This pipeline reproduces the lab's earlier detection script: ≥ 99 % of spikes matched within ±0.1 ms, noise levels agree within 0.03 %, and waveforms are identical (`tests/test_regression.py`).

## Stimulation artifacts (`meagraph.stimulation`)

**Pulse times.** Pulse onset and offset times were read from the stimulator's event stream; this ties each pulse's Start and Stop event to the correct timestamps. The stimulating electrode is not recorded in the file. It was taken from the experiment notes, or inferred when exactly one channel saturated the amplifier during the pulse.

**Recovery measurement.** Each channel's blanking window was set from its measured artifact recovery:

1. Bridge each pulse from 1 ms before onset to 1 ms after offset, then band-pass as above.
2. Take the median across trials of the post-pulse epochs. This estimates the deterministic artifact residue. Evoked spikes are jittered in time and absent on many trials, so they barely affect the median.
3. The recovery time is the first time after which |median| stays below one noise σ (measured before the pulses) for 1 ms.

**Blanking and exclusion.** Each channel was blanked from 1 ms before onset to its recovery time after offset, with a minimum of 1 ms and a maximum of 50 ms. The stimulating electrode was excluded from detection in recordings where it was stimulated.

**Measured recovery** after pulse offset:

| Recordings | Median | Slowest channel |
|---|---|---|
| DIV140 (exp3) | 5 ms | 11 ms (the stimulating electrode) |
| DIV142 (associative) | 6.6–6.8 ms | 17 ms |

These recovery times set the shortest response latency that can be measured on each channel.

## Signal-quality control (`meagraph.detect`, DECISIONS.md D10)

**Principle.** Extracellular action potentials are predominantly negative-going, while threshold crossings caused by noise are symmetric in sign.

**Events.** For each channel, all peaks crossing ±5σ were grouped into events: peaks less than 3 ms apart belong to one event. Each event takes the polarity of its largest peak. Events during stimulation and in the 50 ms after each pulse were not counted.

**Test.** A channel was classified as **active**, meaning it records spiking activity above the noise floor, if a one-sided binomial test showed more negative than positive events (p < 0.001) with at least 20 negative events. Only active channels are used for connectivity inference.

**Why events.** A spike's positive overshoot can itself cross threshold; grouping counts each spike once, as negative. A symmetric noise burst is equally likely to count as positive or negative, so the binomial null still holds. Synthetic tests confirm both properties.

## Connectivity inference

To be added in Phases 4–5: methods, assumptions and validation on synthetic ground truth.

## References

- Buccino, A. P., Hurwitz, C. L., Garcia, S., Magland, J., Siegle, J. H., Hurwitz, R., & Hennig, M. H. (2020). SpikeInterface, a unified framework for spike sorting. *eLife*, 9, e61834.
- Garcia, S., Sprenger, J., Holtzman, T., & Buccino, A. P. (2022). ProbeInterface: a unified framework for probe handling in extracellular electrophysiology. *Frontiers in Neuroinformatics*, 16, 823056.
- Kumar et al. (2026). A three-dimensional micro-instrumented neural network device. *Nature Electronics*, 9, 532–543.
- Quiroga, R. Q., Nadasdy, Z., & Ben-Shaul, Y. (2004). Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering. *Neural Computation*, 16(8), 1661–1687.
