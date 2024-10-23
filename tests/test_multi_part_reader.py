from dataclasses import dataclass
from functools import cached_property

import av
import numpy as np
import pytest

from pupil_labs.video.multi_part_reader import MultiPartReader

Slice = type("", (object,), {"__getitem__": lambda _, key: key})()


@dataclass
class PacketData:
    pts: list[int]
    times: list[float]
    keyframe_indices: list[int]

    @cached_property
    def gop_size(self) -> int:
        return int(max(np.diff(self.keyframe_indices)))

    def _summarize_list(self, lst: list) -> str:
        return f"""[{
            (
                ", ".join(
                    x if isinstance(x, str) else str(round(x, 4))
                    for x in lst[:3] + ["..."] + lst[-3:]
                )
            )
        }]"""

    def __len__(self) -> int:
        return len(self.pts)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            + ", ".join(
                f"{key}={value}"
                for key, value in [
                    ("len", len(self.pts)),
                    ("pts", self._summarize_list(self.pts)),
                    ("times", self._summarize_list(self.times)),
                    ("keyframe_indices", self._summarize_list(self.keyframe_indices)),
                ]
            )
            + ")"
        )


@pytest.fixture
def correct_data(multi_part_video_paths: list[str]) -> PacketData:
    pts_bias = 0
    times_bias = 0
    pts = []
    times = []
    index = 0
    keyframe_indices = []
    for video_path in multi_part_video_paths:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        assert stream.time_base

        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            pts.append(packet.pts + pts_bias)
            times.append(float(packet.pts * stream.time_base) + times_bias)
            if packet.is_keyframe:
                keyframe_indices.append(index)
            index += 1

        pts_bias += container.duration
        times_bias = pts_bias * stream.time_base
    return PacketData(pts=pts, times=times, keyframe_indices=keyframe_indices)


@pytest.fixture
def reader(multi_part_video_paths: list[str]) -> MultiPartReader:
    return MultiPartReader(multi_part_video_paths)


def test_indexing(reader: MultiPartReader, correct_data: PacketData) -> None:
    for i in range(len(reader)):
        assert reader[i].index == i


def test_reverse_iteration(reader: MultiPartReader, correct_data: PacketData) -> None:
    for i in reversed(range(len(reader))):
        assert reader[i].index == i


@pytest.mark.parametrize(
    "slice_arg",
    [
        Slice[:],
        Slice[:100],
        Slice[:-100],
        Slice[-100:],
        Slice[-100:],
        Slice[-100:-50],
        Slice[50:100],
    ],
)
def test_slices(
    reader: MultiPartReader, slice_arg: slice, correct_data: PacketData
) -> None:
    for frame, index in zip(reader[slice_arg], range(*slice_arg.indices(len(reader)))):
        assert frame.index == index
