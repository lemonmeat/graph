# meagraph

Analysis of MCS HDF5 recordings from in vitro neuronal cultures on a custom 4x4x4 3D microelectrode array (and standard 60-channel planar MEAs). The goal is to infer functional connectivity graphs and study how stimulation edits them, for reservoir computing and closed-loop experiments.

This README is a stub during the refactor. See `docs/PLAN.md` for the roadmap, `docs/DATA_FORMAT.md` for the file format, and `docs/DECISIONS.md` for design decisions.

## Install

```bash
uv pip install --python .venv/bin/python -e ".[dev,viewer]"
```

## Quickstart

```bash
meagraph info recording.h5     # streams (processing order), stimulation events, MCS spike streams
meagraph probes                # packaged electrode layouts
pytest                         # unit tests; tests marked `data` run when the sample recordings are present
```

```python
from meagraph.io import load_session

session = load_session("recording.h5", stim_site="47")    # stimulated electrode is not stored in MCS files
rec = session.recording                                   # lazy SpikeInterface recording, probe attached
traces_uv = rec.get_traces(start_frame=0, end_frame=10_000, return_in_uV=True)
times_s = rec.get_times()                                 # seconds on the MCS recording clock
onsets_s = session.stim[0].onsets_s
```

The legacy scripts (`spikes.py`, `visualize.py` and the others) stay runnable until the refactored pipeline passes the regression test.
