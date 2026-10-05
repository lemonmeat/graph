"""Spike detection, signal-quality control, and result storage."""

from meagraph.detect.noise import median_abs_noise_uv
from meagraph.detect.store import load_detection, save_detection
from meagraph.detect.threshold import ChannelQC, DetectionConfig, DetectionResult, detect_spikes

__all__ = [
    "ChannelQC",
    "DetectionConfig",
    "DetectionResult",
    "detect_spikes",
    "load_detection",
    "median_abs_noise_uv",
    "save_detection",
]
