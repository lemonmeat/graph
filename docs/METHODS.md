# Methods

This file is written so it can be adapted for the independent work report. Parameter values are the `meagraph` defaults unless stated otherwise. Each method names the module that implements it. Design rationale is in `DECISIONS.md`.

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

## Network bursts (`meagraph.detect.bursts`)

**Detection.** Network bursts were detected with the ISI_N method (Bakkum et al., 2013). Spikes from all channels were pooled, and any 10 consecutive pooled spikes falling within 100 ms were assigned to a burst. Overlapping runs were merged, and bursts with fewer than 3 participating channels were discarded.

**Use.** Bursts served as a control: every connectivity analysis was repeated with burst periods removed.

## Connectivity inference (`meagraph.connectivity`)

### Scope

**Recordings.** Connectivity was estimated from spontaneous activity among QC-active electrodes with at least 100 spikes. In stimulation recordings, spikes from 1 ms before each pulse to 200 ms after its offset were excluded first, because shared, stimulus-locked drive makes unconnected units co-fire.

**What an edge means.** Each electrode's spikes are treated as one train. An edge means that spikes on one electrode change the probability of spikes on another at a short, consistent delay. That is **functional** connectivity: evidence for, not proof of, a synapse.

**Three things it cannot separate.** Simulations (Validation) show that pairwise methods also report:

- two-step chains (i → k → j);
- common input (k → i and k → j);
- for very short delays, possibly the same neuron recorded on two nearby electrodes.

### Cross-correlograms

For each ordered pair (source i, target j), the cross-correlogram counts target spikes at each lag after source spikes, in 0.5 ms bins from −30 to +30 ms. It was computed with SpikeInterface (Buccino et al., 2020).

**Synaptic window.** A putative excitatory connection appears as excess target spikes 1–4 ms after source spikes.

**Weight and delay.** The weight is the spike transmission probability: excess target spikes in the window per source spike (English et al., 2017). The delay is the lag of the largest excess.

**Two tests of the excess:**

- **Smoothed-baseline test** (`cch_hollow`; Stark & Abeles, 2009; English et al., 2017).
  1. Estimate the expected correlogram by convolving it with a partially hollow Gaussian (σ = 10 ms, centre weight × 0.4), which follows slow co-modulation without absorbing a sharp synaptic peak.
  2. Test each window bin against a Poisson distribution with that expectation.
  3. Bonferroni-correct the smallest p within the window.
- **Jitter test** (`cch_jitter`; Amarasingham et al., 2012).
  1. Redraw every spike of both trains uniformly within the fixed 10 ms window that contains it, 1000 times. This interval jitter preserves each train's spike count in every 10 ms window, and therefore all co-firing slower than about 10 ms, such as bursts and rate co-modulation.
  2. Destroy the fine timing that a synapse would create.
  3. Fit a negative binomial (or Poisson) distribution to the 1000 surrogate window counts by moments, and take the p-value from its tail. This avoids the 1/1001 resolution limit of a pure Monte-Carlo test, which would otherwise prevent discoveries when many pairs are tested.

### Spike time tiling coefficient

The STTC (Cutts & Eglen, 2014) is an undirected measure of co-firing within ±Δt that does not depend on firing rate. It was computed with Δt = 5 ms. Significance came from the same interval-jitter surrogates, with a normal distribution fitted to the surrogate values. STTC with Δt = 50 ms was reported as a descriptive measure of burst-scale co-firing.

### Multiple comparisons and burst control

**Multiple comparisons.** p-values were corrected with the Benjamini–Hochberg procedure (Benjamini & Hochberg, 1995) at a false discovery rate of 5 %:

- correlogram methods: across all tested ordered pairs;
- STTC: across unordered pairs.

**Burst control.** Each method was run on all spikes and again with network-burst periods removed. Edges significant in both analyses are reported as robust.

### Validation on synthetic networks (`meagraph.synth`, `meagraph.benchmark`)

**Model.** The methods were validated on simulated networks with known directed connections: a linear Hawkes process simulated as a branching process.

- Each spike of unit i adds on average W_ij spikes to unit j after a 1.5–3.5 ms delay.
- Units had baseline rates of 0.2–2 Hz and sat on the electrode positions of the array.
- Optional confounds were network bursts matched to the DIV142 recording, periodic stimulation, and detection errors.

**Scoring.** For each method, recording length and connection strength, the scores were:

- precision, recall and false-positive rate against the true connections;
- ROC area, computed from the p-values;
- delay error.

**Setup.**

- 16 units, connection probability 0.1, transmission probabilities 0.03, 0.06 or 0.12.
- Recordings of 2, 5, 10 or 30 min, with and without DIV142-like network bursts.
- Null networks (no connections) with and without bursts, and with periodic stimulation.
- 3 random seeds per scenario, 1000 jitter surrogates, FDR q = 0.05.
- Command: `meagraph benchmark`. Per-network rows are in `docs/benchmark/benchmark.csv`.

**Recall vs recording length** (mean over connection strengths and seeds):

| Method | Spikes analysed | 2 min | 5 min | 10 min | 30 min |
|---|---|---|---|---|---|
| `cch_jitter` | no bursts present | 0.60 | 0.78 | 0.88 | 0.98 |
| `cch_jitter` | bursts present, all spikes | 0.49 | 0.67 | 0.84 | 0.98 |
| `cch_jitter` | bursts present, burst periods removed | 0.66 | 0.76 | 0.89 | 0.99 |
| `cch_jitter` | bursts present, robust (both) | 0.24 | 0.55 | 0.82 | 0.98 |
| `cch_hollow` | no bursts present | 0.85 | 0.82 | 0.90 | 0.92 |
| `sttc` | no bursts present | 0.35 | 0.70 | 0.82 | 0.96 |

**Recall vs connection strength** (`cch_jitter`, no bursts):

| Transmission probability | 10 min | 30 min |
|---|---|---|
| 0.03 | 0.72 | 0.95 |
| 0.06 | 0.95 | 0.99 |
| 0.12 | 0.99 | 1.00 |

**Precision and false positives** (all connected scenarios pooled):

| Method | Precision (all spikes) | False positives that are indirect* | Delay error | ROC area |
|---|---|---|---|---|
| `cch_jitter` | 0.94–0.96 | 29 of 36 | 0.11 ms | 0.98–0.99 |
| `cch_hollow` | 0.88–0.91 | 69 of 85 | 0.11 ms | 0.99 |
| `sttc` | 0.82–0.85 | 78 of 93 | — | 0.95–0.97 |

\*Indirect means a two-step chain (i → k → j) or common input (k → i and k → j). These pairs truly correlate, and pairwise methods cannot distinguish them from direct connections.

**Null networks** (no connections; mean false edges per network; 240 ordered pairs tested by the correlogram methods, 120 unordered pairs by STTC):

| Scenario | `cch_jitter` | `cch_hollow` | `sttc` |
|---|---|---|---|
| no bursts | 0 | 0 | 0 |
| network bursts | 0 | 0 | 0 |
| periodic stimulation (shared drive) | 0 | 0 | **62 (52 %)** |

### What the validation shows

1. **No false edges from rate effects.** In these simulations, none of the methods produced false edges from network bursts or rate differences.
2. **STTC is not usable during stimulation.** Shared, time-locked stimulation made STTC report half of all pairs as connected, while the directional 1–4 ms correlogram tests were unaffected. Stimulation periods must be excluded for STTC (D18).
3. **Bursts cost power, not specificity.** Burst co-firing inflates the null distribution and lowers recall in short recordings. Removing burst periods restores it. Requiring significance in both analyses ("robust") costs much more recall in short recordings (0.24 vs 0.66 at 2 min) than it gains in precision (0.97 vs 0.95).
4. **The legacy symmetric-pair rule removes reciprocal connections.** `cch_hollow` levels off at 0.92 recall because this rule, inherited from `spontaneous_ccg.py`, drops pairs significant in both directions at similar delays. In a check with 30 min recordings, it found 119 of 119 one-way connections but only 6 of 12 reciprocal ones.
5. **Recording length.** About 30 min of spontaneous activity recovers nearly all connections down to a transmission probability of 0.03. 10 min misses about a quarter of the weakest ones.
6. **`cch_jitter` is the best primary method.** It has the highest precision, a delay for every edge, and no false edges in any null scenario. It is also the most conservative under the null (D17), which costs some recall in short recordings.

**Limitations of the validation.** The simulated network is linear, has excitatory connections only, and keeps all delays inside the test window. Its bursts are rate surges rather than propagating network events. Real data can violate each of these assumptions.

## References

- Amarasingham, A., Harrison, M. T., Hatsopoulos, N. G., & Geman, S. (2012). Conditional modeling and the jitter method of spike resampling. *Journal of Neurophysiology*, 107(2), 517–531.
- Bakkum, D. J., Radivojevic, M., Frey, U., Franke, F., Hierlemann, A., & Takahashi, H. (2013). Parameters for burst detection. *Frontiers in Computational Neuroscience*, 7, 193.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society B*, 57(1), 289–300.
- Cutts, C. S., & Eglen, S. J. (2014). Detecting pairwise correlations in spike trains: an objective comparison of methods and application to the study of retinal waves. *Journal of Neuroscience*, 34(43), 14288–14303.
- English, D. F., McKenzie, S., Evans, T., Kim, K., Yoon, E., & Buzsáki, G. (2017). Pyramidal cell-interneuron circuit architecture and dynamics in hippocampal networks. *Neuron*, 96(2), 505–520.
- Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika*, 58(1), 83–90.
- Stark, E., & Abeles, M. (2009). Unbiased estimation of precise temporal correlations between spike trains. *Journal of Neuroscience Methods*, 179(1), 90–100.

- Buccino, A. P., Hurwitz, C. L., Garcia, S., Magland, J., Siegle, J. H., Hurwitz, R., & Hennig, M. H. (2020). SpikeInterface, a unified framework for spike sorting. *eLife*, 9, e61834.
- Garcia, S., Sprenger, J., Holtzman, T., & Buccino, A. P. (2022). ProbeInterface: a unified framework for probe handling in extracellular electrophysiology. *Frontiers in Neuroinformatics*, 16, 823056.
- Kumar et al. (2026). A three-dimensional micro-instrumented neural network device. *Nature Electronics*, 9, 532–543.
- Quiroga, R. Q., Nadasdy, Z., & Ben-Shaul, Y. (2004). Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering. *Neural Computation*, 16(8), 1661–1687.
