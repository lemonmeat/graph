"""MCS HDF5 analog stream as a lazy SpikeInterface recording (DECISIONS.md D1).

Differences from ``spikeinterface.extractors.read_mcsh5``:
- segment ``t_start`` is the stream's FirstTimeStamp, so times share the event clock (D5)
- data rows follow ``InfoChannel.RowIndex``
- offsets are ``-ADZero * gain``
- channel ids are electrode labels ('47'); the full MCS label and ChannelID are properties
- streams are chosen by label or ``'raw'``, never by HDF5 index
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment

from meagraph.io._mcs_layout import (
    RAW,
    channel_tick_us,
    decode,
    electrode_id,
    first_timestamp_s,
    recording_group,
    resolve_analog_stream,
)


class McsH5Recording(BaseRecording):
    """One analog stream of an MCS HDF5 file.

    Parameters
    ----------
    file_path
        Path to the ``.h5`` file written by Multi Channel Experimenter / DataManager.
    stream
        ``'raw'`` (the hardware stream) or a stream label such as ``'Filter (3)'``.
    recording_index
        Index of ``Data/Recording_<n>``.
    """

    def __init__(self, file_path: str | Path, stream: str = RAW, recording_index: int = 0):
        file_path = Path(file_path).resolve()
        with h5py.File(file_path, "r") as f:
            rec = recording_group(f, recording_index)
            stream_info = resolve_analog_stream(rec, stream)
            grp = rec["AnalogStream"][stream_info.name]
            dataset_path = f"{grp.name}/ChannelData"
            table = grp["InfoChannel"][:]
            dataset = grp["ChannelData"]
            n_rows, n_samples = dataset.shape
            dtype = dataset.dtype
            tick_us = channel_tick_us(table)
            t_start = first_timestamp_s(grp["ChannelDataTimeStamps"][:], tick_us, n_samples)
            file_guid = decode(f["Data"].attrs.get("FileGUID", ""))
            mea_layout = decode(f["Data"].attrs.get("MeaLayout", ""))

        units = {decode(u) for u in table["Unit"]}
        if units != {"V"}:
            raise ValueError(f"expected analog units 'V', found {sorted(units)}")
        labels = [decode(lbl) for lbl in table["Label"]]
        channel_ids = [electrode_id(lbl) for lbl in labels]
        if len(set(channel_ids)) != len(channel_ids):
            raise ValueError("electrode labels are not unique within the stream")
        row_index = table["RowIndex"].astype(np.int64)
        if sorted(row_index.tolist()) != list(range(n_rows)):
            raise ValueError("InfoChannel.RowIndex is not a permutation of the ChannelData rows")

        sampling_frequency = 1e6 / tick_us
        BaseRecording.__init__(self, sampling_frequency=sampling_frequency, channel_ids=channel_ids, dtype=dtype)

        gains_uv = table["ConversionFactor"].astype(np.float64) * 10.0 ** table["Exponent"].astype(np.float64) * 1e6
        self.set_channel_gains(gains_uv)
        self.set_channel_offsets(-table["ADZero"].astype(np.float64) * gains_uv)
        self.set_property("mcs_label", np.array(labels))
        self.set_property("mcs_channel_id", table["ChannelID"].astype(np.int64))
        self.set_property("mcs_row_index", row_index)
        # Largest representable |signal| (µV): stimulation artifacts that reach it are saturated.
        adc_max = 2.0 ** (table["ADCBits"].astype(np.float64) - 1) - 1
        self.set_property("adc_rail_uv", np.minimum(adc_max - table["ADZero"], adc_max + table["ADZero"]) * gains_uv)
        self.annotate(
            mcs_stream_label=stream_info.label,
            mcs_stream_guid=stream_info.guid,
            mcs_source_stream_guid=stream_info.source_guid,
            mcs_file_guid=file_guid,
            mea_layout=mea_layout,
        )

        self.add_recording_segment(
            McsH5RecordingSegment(
                file_path=file_path,
                dataset_path=dataset_path,
                row_index=row_index,
                num_samples=n_samples,
                sampling_frequency=sampling_frequency,
                t_start=t_start,
            )
        )
        self.extra_requirements.append("h5py")
        self._kwargs = {"file_path": str(file_path), "stream": stream, "recording_index": recording_index}


class McsH5RecordingSegment(BaseRecordingSegment):
    # Chunks span all rows, so reading every row costs the same decompression as reading one.
    # Below this fraction of rows, read rows individually to save memory on long single-channel reads.
    _FULL_READ_FRACTION = 0.25

    def __init__(self, file_path, dataset_path, row_index, num_samples, sampling_frequency, t_start):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency, t_start=t_start)
        self._file_path = Path(file_path)
        self._dataset_path = dataset_path
        self._row_index = np.asarray(row_index, dtype=np.int64)
        self._num_samples = int(num_samples)
        self._file = None
        self._dataset = None

    def _data(self) -> h5py.Dataset:
        # Opened lazily so each worker process gets its own handle.
        if self._dataset is None:
            self._file = h5py.File(self._file_path, "r")
            self._dataset = self._file[self._dataset_path]
        return self._dataset

    def get_num_samples(self) -> int:
        return self._num_samples

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None) -> np.ndarray:
        start = 0 if start_frame is None else int(start_frame)
        end = self._num_samples if end_frame is None else int(end_frame)
        rows = self._row_index if channel_indices is None else self._row_index[channel_indices]
        rows = np.atleast_1d(rows)
        data = self._data()
        unique_rows, inverse = np.unique(rows, return_inverse=True)
        if unique_rows.size >= self._FULL_READ_FRACTION * data.shape[0]:
            block = data[:, start:end][unique_rows]
        else:
            block = data[unique_rows.tolist(), start:end]
        return np.ascontiguousarray(block[inverse].T)


def read_mcs_h5(file_path: str | Path, stream: str = RAW, recording_index: int = 0) -> McsH5Recording:
    """Function form of :class:`McsH5Recording`."""
    return McsH5Recording(file_path, stream=stream, recording_index=recording_index)
