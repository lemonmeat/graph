# Methods

This file is written so it can be adapted for the independent work report. Parameter values are the `meagraph` defaults unless stated otherwise. Each method names the module that implements it. Design rationale is in `DECISIONS.md`.

## Recording

- **Hardware.** Extracellular signals were recorded from 3D neuronal cultures on a folded 4×4×4 microelectrode array (3D-MIND design; Kumar et al., 2026) connected to an MCS MEA2100-Mini system.
- **Channels.** 59 electrodes plus one reference (electrode 15). Of the 64 grid positions, 5 have no recorded electrode.
- **Geometry.** Electrodes are 30 µm gold discs on four stacked layers separated by 250 µm (Kumar et al., 2026); layer 1 is the bottom. Positions are given as (row, column, layer). The in-layer electrode pitch was not available, so distances within a layer are placeholders and no analysis interprets them physically.
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

### Total spiking probability edges

TSPE (De Blasi et al., 2019) was designed for in vitro MEA recordings and detects both excitatory and inhibitory relations. It was computed with Elephant (Denker et al., 2018) on spike trains binned at 1 ms, with Elephant's default edge-filter windows and delays up to 25 ms. For each pair, the normalized cross-correlation is filtered with edge filters: a local peak after the source spike gives a positive score, a local dip a negative one.

**Significance.** Elephant returns scores without a test. Each score was compared with the same 10 ms interval-jitter surrogates as the jitter test (1000 surrogates), using a normal distribution fitted to the surrogate scores. The two-sided p-value was corrected with Benjamini–Hochberg across ordered pairs. **Weight:** score minus surrogate mean; positive = excitatory, negative = inhibitory. **Delay:** the lag of the largest absolute score.

**Known artefact.** A strong excitatory i → j makes the reverse direction j → i score negative, because the filter's leading window sees the i → j peak at negative lag. The benchmark counts these reverse "inhibitory" edges separately.

### Conditional firing probability

CFP (le Feber et al., 2007) is the probability that electrode j fires in the 1 ms bin at lag τ after a spike of electrode i, for τ from 0 to 500 ms. Related pairs show a peak; its height above the curve's offset is the relation's strength and its latency the delay. Strength and delay come from a least-squares fit of M / (1 + ((τ − T)/w)²) + offset; the exact form of the original fit has not been checked against the paper.

**Significance (our implementation).** The original criterion is a curve that "clearly deviates from flat". Here the statistic was the peak of the curve after 5 ms smoothing minus its mean, tested against 1000 surrogates in which every train was shifted circularly by an independent random offset. Shifting keeps each train's own structure, including its bursts, but breaks the timing between trains. A fitted normal gave the p-value, corrected with Benjamini–Hochberg across ordered pairs.

**Validity.** As in Martiniuc et al. (2015), a relation was accepted only if the peak was at least 5 ms wide at 80 % of its height and its delay at most 250 ms.

**What CFP measures.** Shared network bursts are part of the CFP, so co-bursting pairs are related even without a synapse. CFP describes functional coupling at the timescale of network bursts, which is how it is used in studies of stimulation-induced change (le Feber et al., 2010), not monosynaptic connectivity.

### Two-Gaussian correlogram weight (comparison with earlier analyses)

To compare with the lab's earlier analyses (Kumar et al., 2026), the cross-correlogram (2 ms bins, ±100 ms) of each pair was also fitted with a sum of two Gaussians, a1·exp(−(x − b1)²/2c1²) + a2·exp(−(x − b2)²/2c2²), and the connection weight was (a1 + a2)/(c1 + c2). The net direction was taken from the side of zero on which the fitted curve peaks. As in the original analysis, no significance test was applied. The correlogram normalization and lag range of the original analysis were not reported; the values used here are assumptions (DECISIONS.md D23).

### Multiple comparisons and burst control

**Multiple comparisons.** p-values were corrected with the Benjamini–Hochberg procedure (Benjamini & Hochberg, 1995) at a false discovery rate of 5 %:

- correlogram methods: across all tested ordered pairs;
- STTC: across unordered pairs.

**Burst control.** Each method was run on all spikes and again with network-burst periods removed. Edges significant in both analyses (and, for TSPE, of the same sign) are reported as robust.

### Validation on synthetic networks (`meagraph.synth`, `meagraph.benchmark`)

**Model.** The methods were validated on simulated networks with known directed connections: a linear Hawkes process simulated as a branching process.

- Each spike of unit i adds on average W_ij spikes to unit j after a 1.5–3.5 ms delay.
- Units had baseline rates of 0.2–2 Hz and sat on the electrode positions of the array.
- Optional confounds were network bursts matched to the DIV142 recording, periodic stimulation, and detection errors.
- **Inhibition** (optional). A fraction of units were inhibitory, and all their connections were inhibitory (Dale's principle). After an inhibitory spike, each target spike within the connection's delay plus 10 ms was deleted with the connection's suppression probability; deleted spikes caused no further spikes. With inhibition the cascade was simulated in time order; without it, the network is identical to the excitatory-only model.

**Scoring.** For each method, recording length and connection strength, the scores were:

- precision, recall and false-positive rate against the true connections;
- ROC area, computed from the p-values;
- delay error.

**Setup.**

- 16 units, connection probability 0.1, transmission probabilities 0.03, 0.06 or 0.12.
- Recordings of 2, 5, 10 or 30 min, with and without DIV142-like network bursts.
- Null networks (no connections) with and without bursts, and with periodic stimulation.
- 3 random seeds per scenario, 1000 jitter surrogates, FDR q = 0.05.
- Inhibition scenarios: 20 % of units inhibitory (suppression probability 0.6 for 10 ms), excitatory weight 0.06; 10 and 30 min, with and without bursts, plus one 10 min network with 2–10 Hz rates. TSPE and CFP used 200 surrogates in the benchmark.
- Command: `meagraph benchmark`. Per-network rows for all five methods are in `docs/benchmark/benchmark.csv`.

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

**TSPE and CFP on the same grid** (excitatory scenarios; recall at 2 / 5 / 10 / 30 min):

| Method | Spikes analysed | 2 min | 5 min | 10 min | 30 min |
|---|---|---|---|---|---|
| `tspe` | no bursts present | 0.47 | 0.69 | 0.79 | 0.95 |
| `tspe` | bursts present, all spikes | 0.38 | 0.45 | 0.66 | 0.90 |
| `tspe` | bursts present, burst periods removed | 0.51 | 0.66 | 0.81 | 0.95 |
| `cfp` (published 5 ms width rule) | no bursts present | 0.00 | 0.00 | 0.00 | 0.00 |
| `cfp` (published 5 ms width rule) | bursts present, all spikes | 0.44 | 0.39 | 0.31 | 0.34 |
| `cfp_narrow` (no width rule) | no bursts present | 0.73 | 0.87 | 0.95 | 1.00 |
| `cfp_narrow` (no width rule) | bursts present, burst periods removed | 0.78 | 0.87 | 0.94 | 1.00 |

| Method | Precision, no bursts / bursts (all spikes) | False positives that are indirect | Delay error | ROC area |
|---|---|---|---|---|
| `tspe` | 0.89 / 0.97 | 135 of 148 | 0.23 ms | 0.95 |
| `cfp` | 0.50 / 0.05 | 1587 of 5667 | 2.2 ms | 0.95 |
| `cfp_narrow` | 0.79 / 0.11 (0.79 with bursts removed) | 1846 of 6129 | 0.14 ms (bursts removed) | 0.95–0.98 |

Null networks (mean false edges per network): `tspe` 1, 0 and 0 (no bursts, bursts, stimulation); `cfp` 0, **202 (84 %)** and 0.3; `cfp_narrow` 1, **214 (89 %)** and 19 (8 %).

**Two-Gaussian weight** (`cch_gauss2`, no significance test): recall 0.81–0.89 at every recording length; precision 0.18; ROC area of the weight 0.84–0.86 (vs 0.99 for `cch_jitter`); delay error 0.3–0.6 ms. In null networks it reported an edge for about half of all ordered pairs (one net direction per pair): 49 % without bursts, 48 % with bursts, 35 % with stimulation.

**Inhibition** (TSPE, all spikes; 18 inhibitory connections per scenario over 3 seeds):

| Scenario | Inhibitory found | False inhibitory (of which reverse artefact) | With the reverse rule: found / false |
|---|---|---|---|
| no bursts, 10 min | 1 | 10 (7) | 1 / 3 |
| no bursts, 30 min | 1 | 25 (22) | 0 / 3 |
| bursts, 10 min | 0 | 2 (2) | 0 / 0 |
| bursts, 30 min | 1 | 12 (10) | 0 / 2 |
| 2–10 Hz rates, 10 min | 1 | 15 (15) | 0 / 0 |

Ranking inhibitory pairs by p-value (ROC area, 0.5 = chance): TSPE 0.54–0.81, `cch_jitter`'s one-sided inhibition p 0.58–0.70. In single-network checks, TSPE's raw score separated inhibitory pairs well at 2–10 Hz (ROC area 0.94) but not at 0.2–2 Hz (0.49), whatever the jitter window (10, 25 or 50 ms).

### What the validation shows

1. **No false edges from rate effects.** In these simulations, none of the methods produced false edges from network bursts or rate differences.
2. **STTC is not usable during stimulation.** Shared, time-locked stimulation made STTC report half of all pairs as connected, while the directional 1–4 ms correlogram tests were unaffected. Stimulation periods must be excluded for STTC (D18).
3. **Bursts cost power, not specificity.** Burst co-firing inflates the null distribution and lowers recall in short recordings. Removing burst periods restores it. Requiring significance in both analyses ("robust") costs much more recall in short recordings (0.24 vs 0.66 at 2 min) than it gains in precision (0.97 vs 0.95).
4. **The legacy symmetric-pair rule removes reciprocal connections.** `cch_hollow` levels off at 0.92 recall because this rule, inherited from `spontaneous_ccg.py`, drops pairs significant in both directions at similar delays. In a check with 30 min recordings, it found 119 of 119 one-way connections but only 6 of 12 reciprocal ones.
5. **Recording length.** About 30 min of spontaneous activity recovers nearly all connections down to a transmission probability of 0.03. 10 min misses about a quarter of the weakest ones.
6. **`cch_jitter` is the best primary method.** It has the highest precision, a delay for every edge, and no false edges in any null scenario. It is also the most conservative under the null (D17), which costs some recall in short recordings.

7. **TSPE does not improve excitatory mapping.** It has lower recall than `cch_jitter` at every recording length and lower precision without bursts. Its advantages are a signed output and delays beyond the 1–4 ms window (Q16).
8. **Inhibition is not detectable at culture-like firing rates.** At 0.2–2 Hz the target neuron fires too rarely for a 10 ms suppression to show, even in 30 min, and no method found more than 1 of 18 inhibitory connections. The active DIV142 electrodes fire at 0.1–1.7 Hz, so the absence of inhibitory edges in these recordings says nothing about inhibition in the culture.
9. **Most of TSPE's inhibitory calls are its reverse artefact:** 56 of 64 false inhibitory edges were the opposite direction of an excitatory connection. Dropping inhibitory edges that oppose a significant excitatory edge removes all 56, at the cost of 3 of the 4 true inhibitory edges found.
10. **CFP measures co-bursting, not synapses.** With the published width rule it found no monosynaptic connection, yet marked 84 % of unconnected pairs in burst networks. Without the rule it finds connections (recall comparable to `cch_jitter`) but with precision 0.79, and only after burst periods are removed. CFP is suited to tracking burst-scale functional change (Phase 6), not to mapping connections.
11. **The burst detector over-detects at high rates.** In the 2–10 Hz network, ISI_N (10 pooled spikes within 100 ms) marked much of the recording as bursts, so the burst-removed analyses lost most of their spikes (`cch_jitter` recall 1.00 → 0.45). Its thresholds should scale with the network's firing rate (D15).
12. **The earlier lab weight ranks pairs but cannot decide which are connected.** Without a test, about half of all pairs, connected or not, receive a directed edge. Its weight orders true connections above the rest less reliably than the jitter test's p-value (ROC area 0.85 vs 0.99), partly because the model has no baseline term and the second Gaussian's width absorbs the flat background. It is useful for comparing weights of the same pair across days, as in the original study, not for deciding whether a connection exists.

**Limitations of the validation.** The simulated network is linear apart from the suppression windows, keeps excitatory delays inside the test window, and models inhibition as spike deletion rather than conductance. Its bursts are rate surges rather than propagating network events. Real data can violate each of these assumptions.

## References

- Amarasingham, A., Harrison, M. T., Hatsopoulos, N. G., & Geman, S. (2012). Conditional modeling and the jitter method of spike resampling. *Journal of Neurophysiology*, 107(2), 517–531.
- Bakkum, D. J., Radivojevic, M., Frey, U., Franke, F., Hierlemann, A., & Takahashi, H. (2013). Parameters for burst detection. *Frontiers in Computational Neuroscience*, 7, 193.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society B*, 57(1), 289–300.
- De Blasi, S., Ciba, M., Bahmer, A., & Thielemann, C. (2019). Total spiking probability edges: a cross-correlation based method for effective connectivity estimation of cortical spiking neurons. *Journal of Neuroscience Methods*, 312, 169–181.
- Denker, M., Yegenoglu, A., & Grün, S. (2018). Collaborative HPC-enabled workflows on the HBP Collaboratory using the Elephant framework. *Neuroinformatics 2018*, P19.
- Cutts, C. S., & Eglen, S. J. (2014). Detecting pairwise correlations in spike trains: an objective comparison of methods and application to the study of retinal waves. *Journal of Neuroscience*, 34(43), 14288–14303.
- English, D. F., McKenzie, S., Evans, T., Kim, K., Yoon, E., & Buzsáki, G. (2017). Pyramidal cell-interneuron circuit architecture and dynamics in hippocampal networks. *Neuron*, 96(2), 505–520.
- Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika*, 58(1), 83–90.
- le Feber, J., et al. (2007). Conditional firing probabilities in cultured neuronal networks: a stable underlying structure in widely varying spontaneous activity patterns. *Journal of Neural Engineering*, 4(2), 54–67.
- le Feber, J., Stegenga, J., & Rutten, W. L. C. (2010). The effect of slow electrical stimuli to achieve learning in cultured networks of rat cortical neurons. *PLoS ONE*, 5(1), e8871. *(Citation to be verified.)*
- Martiniuc, A. V., et al. (2015). Paired spiking robustly shapes spontaneous activity in neural networks in vitro. arXiv:1510.09138.
- Stark, E., & Abeles, M. (2009). Unbiased estimation of precise temporal correlations between spike trains. *Journal of Neuroscience Methods*, 179(1), 90–100.

- Buccino, A. P., Hurwitz, C. L., Garcia, S., Magland, J., Siegle, J. H., Hurwitz, R., & Hennig, M. H. (2020). SpikeInterface, a unified framework for spike sorting. *eLife*, 9, e61834.
- Garcia, S., Sprenger, J., Holtzman, T., & Buccino, A. P. (2022). ProbeInterface: a unified framework for probe handling in extracellular electrophysiology. *Frontiers in Neuroinformatics*, 16, 823056.
- Kumar et al. (2026). A three-dimensional micro-instrumented neural network device. *Nature Electronics*, 9, 532–543.
- Quiroga, R. Q., Nadasdy, Z., & Ben-Shaul, Y. (2004). Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering. *Neural Computation*, 16(8), 1661–1687.
