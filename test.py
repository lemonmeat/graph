import matplotlib.pyplot as plt
from meagraph import viz
from meagraph.synth import NetworkConfig, simulate_network
from meagraph.connectivity import estimate
from meagraph.benchmark import score
from meagraph.detect.store import find_detection, load_detection

det = load_detection(find_detection("2026-07-29T15-16-32Flex Electronics Nick Acrylic Day 142_E-00303.h5"))
r = estimate(det.trains.select(list(det.active_channels)), "cch_hollow")
i, j = r.node_ids.index("78"), r.node_ids.index("87")
viz.plot_ccg(r.extra["ccg"][i, j], r.extra["lag_edges_ms"], window_ms=(1, 4), baseline=r.extra["baseline"][i, j])
plt.xlim(-20, 20); plt.show()