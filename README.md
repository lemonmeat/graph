# meagraph

Analysis of Multi Channel Systems (MCS) HDF5 recordings from neuronal cultures on a custom 4×4×4 3D microelectrode array; standard 60-electrode planar MEAs also work.

The current version reads recordings, audits stimulation, and detects spikes with a signal-quality check. It infers connectivity graphs, with methods validated on simulated networks of known connections, and shows everything in an interactive viewer. Stimulation analyses and closed-loop interfaces come next (`docs/PLAN.md`).

## Install

You need Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/). From this folder:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[viewer,dev]"
```

Without uv, `python -m venv .venv` followed by `pip install -e ".[viewer,dev]"` works too.

## Use

Quote file names that contain spaces.

```bash
meagraph info recording.h5          # what is in the file: streams, stimulation events, MCS spike streams
meagraph audit recording.h5         # stimulation: pulse structure, stimulated site, artifact recovery per channel
meagraph detect recording.h5 --stim-site 47     # spikes + signal-quality check, saved under results/
meagraph graph recording.h5         # connectivity among signal-quality-checked electrodes (needs `detect` first)
meagraph view recording.h5          # interactive viewer (click electrodes in the 3D map)
meagraph benchmark --n-jobs 4       # score the connectivity methods on simulated networks (see below; --quick: seconds)
```

The full benchmark runs 81 simulated networks: about 1.5 CPU-hours, or roughly 25 minutes with `--n-jobs 4`. macOS throttles long jobs heavily when the machine idles or sleeps, so on a Mac start it as `caffeinate -i meagraph benchmark --n-jobs 4`.

- `--stim-site` names the electrode that was stimulated, because MCS files do not record it. With two stimulator outputs, use `--stim-site "STG 1=47" --stim-site "STG 2=82"`.
- `meagraph detect` writes `results/<recording>/detect_default/` next to the recording. It contains:
  - `spikes.npz`: spike times, amplitudes and waveforms;
  - `channels.csv`: per-electrode position, spike rate, noise, quality check and artifact recovery;
  - `stimulation.csv`: every stimulation pulse and its site, if given;
  - `config.yaml`: every parameter;
  - `provenance.json`: code version and input file.

  Later steps read only this folder, never the raw file. Folders written before 2026-10-05 lack `stimulation.csv`; re-run `meagraph detect` for them.
- `meagraph detect --profile legacy` reproduces the old `spikes.py`.
- `meagraph graph` writes `results/<recording>/graph_<method>/` with three versions of each graph:
  - `all/`: all spikes;
  - `no_bursts/`: network-burst periods removed;
  - `robust/`: edges significant in both.

  Each holds `graph.graphml` and `graph.json` (nodes with 3D positions), `edges.csv` (every tested pair) and `matrices.npz`. Its `config.yaml` records the method's parameters and how the spikes were chosen (channels, stimulation exclusion, burst settings). The methods are:
  - `cch_jitter`: cross-correlogram tested against spike-time jitter;
  - `cch_hollow`: cross-correlogram against a smoothed baseline;
  - `sttc`: spike time tiling coefficient, undirected.

  In stimulation recordings, spikes from each pulse to 200 ms after it are left out (`--exclude-stim-ms`). See `docs/METHODS.md` for what an edge means, and what it does not.

**Viewer controls.** Click an electrode in the 3D map, or a row in the raster, to select it. Click the raster to jump in time. The ←/→ keys step through time. Use the sliders for start and window length, and the radio buttons to switch between raw and filtered streams. `meagraph view file.h5 --save view.png` renders an image instead of opening a window.

## Use from Python

```python
from meagraph.io import load_session
from meagraph.detect import detect_spikes

session = load_session("recording.h5", stim_site="47")
rec = session.recording                          # lazy SpikeInterface recording with the 3D probe attached
result = detect_spikes(rec, session.stim)        # see meagraph.detect.DetectionConfig for every parameter
result.trains.as_dict()                          # {electrode: spike times in s}
result.active_channels                           # electrodes that pass the signal-quality check

from meagraph.connectivity import estimate
graph = estimate(result.trains.select(list(result.active_channels)), "cch_jitter")
graph.edges()                                    # significant edges: source, target, weight, delay, p

# What `meagraph graph` does: stimulation periods out, then all spikes vs. bursts removed
from meagraph.connectivity.pipeline import GraphConfig, graphs_from_detection
graphs = graphs_from_detection(result, ["cch_jitter"], GraphConfig())
graphs["cch_jitter"].no_bursts.edges()
```

`docs/ARCHITECTURE.md` explains how the package is organised.

## Things to know

- **Times** are seconds on the MCS recording clock, the same clock as the stimulation events.
- **Electrodes** are named by their MCS label (`"47"`). Electrode 15 is the reference and is dropped.
- **Geometry.** The electrode spacing is a placeholder until the real dimensions are entered in `src/meagraph/probe/data/cube4x4x4_E-00303.yaml`. Grid positions (row, column, layer) are exact.
- **Docs:**
  - `docs/DATA_FORMAT.md`: the file format;
  - `docs/METHODS.md`: methods, written for citation;
  - `docs/DECISIONS.md`: design choices;
  - `docs/ARCHITECTURE.md`: code layout and how to extend it.

## Tests

```bash
pytest                  # everything; tests marked `data` use the recordings in this folder
pytest -m "not slow"    # skip the full-pipeline regression against the old spikes.py (about 30 s)
```

The legacy scripts (`spikes.py`, `visualize.py`, …) are kept unchanged for reference until they are retired.
