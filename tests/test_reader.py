from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import av
import numpy as np
import pytest

from pupil_labs.video.reader import Reader

from .utils import measure_fps


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
def correct_data(video_path: Path) -> PacketData:
    pts = []
    times = []
    keyframe_indices = []
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    assert stream.time_base
    index = 0
    for packet in container.demux(stream):
        if packet.pts is None:
            continue
        pts.append(packet.pts)
        times.append(float(packet.pts * stream.time_base))
        if packet.is_keyframe:
            keyframe_indices.append(index)
        index += 1
    return PacketData(pts=pts, times=times, keyframe_indices=keyframe_indices)


@pytest.fixture
def reader(video_path: Path) -> Reader:
    return Reader(video_path)


def test_pts(reader: Reader, correct_data: PacketData) -> None:
    assert list(reader._pts) == correct_data.pts


def test_iteration(reader: Reader, correct_data: PacketData) -> None:
    frame_count = 0
    for frame, expected_pts in measure_fps(zip(reader, correct_data.pts)):
        assert frame.pts == expected_pts
        frame_count += 1

    assert reader.stats.seeks == 0
    assert frame_count == len(correct_data.pts)


def test_backward_iteration_from_end(reader: Reader, correct_data: PacketData) -> None:
    total_keyframes = len(correct_data.keyframe_indices)
    assert total_keyframes <= len(correct_data.pts)

    expected_seeks = total_keyframes

    for i in reversed(range(len(reader))):
        assert reader[i].pts == correct_data.pts[i]
        assert reader.stats.seeks <= expected_seeks

    # we expect keyframe seeks to occur while iterating backwards, one per keyframe
    assert reader.stats.seeks == expected_seeks


def test_backward_iteration_from_N(reader: Reader, correct_data: PacketData) -> None:
    total_keyframes = len(correct_data.keyframe_indices)
    assert total_keyframes <= len(correct_data.pts)

    N = 100
    for i in reversed(range(N)):
        assert reader[i].pts == correct_data.pts[i]

    assert reader.stats.seeks == round(N / correct_data.gop_size)


def test_by_idx(reader: Reader, correct_data: PacketData) -> None:
    frame_count = 0
    for i, expected_pts in measure_fps(enumerate(correct_data.pts)):
        frame = reader[i]
        assert frame.pts == expected_pts
        frame_count += 1

    assert reader.stats.seeks == 0
    assert frame_count == len(correct_data.pts)


def test_by_pts(reader: Reader, correct_data: PacketData) -> None:
    for expected_pts in measure_fps(correct_data.pts):
        frame = reader._by_pts[expected_pts]
        assert frame.pts == expected_pts

    assert reader.stats.seeks == 1  # one seek needed to reset after loading all pts


def test_accessing_times_while_decoding(
    reader: Reader, correct_data: PacketData
) -> None:
    for i, frame in enumerate(reader):
        if i < 100:
            assert Reader._pts.attrname not in reader.__dict__  # type: ignore
        else:
            reader.times  # noqa: B018
            assert Reader._pts.attrname in reader.__dict__  # type: ignore
        assert frame.pts == correct_data.pts[i]


def test_accessing_times_while_decoding_by_frame_step(
    reader: Reader, correct_data: PacketData
) -> None:
    for i in range(0, len(correct_data.pts), 2):
        if i < 10:
            assert Reader._pts.attrname not in reader.__dict__  # type: ignore
            frame = reader[i]
            assert frame.pts == correct_data.pts[i]
        else:
            reader.times  # noqa: B018
            assert Reader._pts.attrname in reader.__dict__  # type: ignore
            frame = reader[i]
            assert frame.pts == correct_data.pts[i]


def test_accessing_times_before_decoding(
    reader: Reader, correct_data: PacketData
) -> None:
    assert Reader._pts.attrname not in reader.__dict__  # type: ignore
    reader.times  # noqa: B018
    assert Reader._pts.attrname in reader.__dict__  # type: ignore
    for i, frame in enumerate(reader):
        assert frame.pts == correct_data.pts[i]


def test_gop_size(reader: Reader, correct_data: PacketData) -> None:
    assert reader.gop_size == correct_data.gop_size
    assert reader.stats.seeks == 0


def test_gop_size_on_seeked_container_within_gop_size(
    reader: Reader, correct_data: PacketData
) -> None:
    index = correct_data.gop_size * 2

    # access some frames to cause a seek
    assert reader[10].pts == correct_data.pts[10]
    assert reader[index].pts == correct_data.pts[index]
    assert reader.stats.seeks == 2

    # now check the gop_size
    assert reader.gop_size == correct_data.gop_size
    assert reader.stats.seeks == 2

    assert reader[10].pts == correct_data.pts[10]
    assert reader[index].pts == correct_data.pts[index]


def test_seek_avoidance_arbitrary_seek(
    reader: Reader, correct_data: PacketData
) -> None:
    reader[correct_data.gop_size * 2]
    assert reader.stats.decodes < correct_data.gop_size
    assert reader.stats.seeks == 1  # one seek to get pts


def test_seek_avoidance(reader: Reader, correct_data: PacketData) -> None:
    assert reader.stats.seeks == 0
    assert reader.stats.decodes == 0

    # we dont need to seek when loading the first frame
    reader[0]
    assert reader.stats.seeks == 0
    assert reader.stats.decodes == 1

    # a second access will load the frame from buffer and not seek/decode
    reader[0]
    assert reader.stats.seeks == 0
    assert reader.stats.decodes == 1

    # getting the second frame will also not require a seek
    reader[1]
    assert reader.stats.seeks == 0
    assert reader.stats.decodes == 2

    # moving forward in same keyframe's worth of frames won't seek
    frames = reader[10:20]
    assert len(frames) == 10
    assert [f.pts for f in frames] == correct_data.pts[10:20]
    assert reader.stats.seeks == 0
    assert reader.stats.decodes == 20

    gop_size = correct_data.gop_size
    # moving forward till next keyframe won't seek
    frame = reader[gop_size]
    assert frame.index == gop_size
    assert reader.stats.seeks == 0
    assert reader.stats.decodes == gop_size + 1

    # no seek even when getting last frame of that next keyframe
    previous_decodes = reader.stats.decodes
    frame = reader[gop_size * 2 - 1]
    assert frame.index == gop_size * 2 - 1
    assert frame.pts == correct_data.pts[gop_size * 2 - 1]
    assert reader.stats.seeks == 0
    assert reader.stats.decodes > previous_decodes


Slice = type("", (object,), {"__getitem__": lambda _, key: key})()
"""
Syntax sugar helper for frame tests to define a slice selection

>>> Slice[:300]
slice(None, 300, None)
"""


# @pytest.mark.parametrize(
#     "slice_arg",
#     [
#         Slice[:],
#         Slice[:100],
#         Slice[:-100],
#         Slice[-100:],
#         Slice[-100:],
#         Slice[-100:-50],
#         Slice[50:100],
#     ],
# )
# @pytest.mark.parametrize(
#     "subslice_arg",
#     [
#         Slice[:],
#         Slice[:50],
#         Slice[:-50],
#         Slice[-50:],
#         Slice[-50:],
#         Slice[-40:-20],
#         Slice[30:100],
#     ],
# )
# def ztest_lazy_slice(slice_arg: slice, subslice_arg: slice, reader: Reader) -> None:
#     expected_start_index, expected_stop_index, _ = slice_arg.indices(len(reader))
#     reader.lazy_frame_slice_limit = 0
#     frame_slice = reader[slice_arg][subslice_arg]
#     assert isinstance(frame_slice, FrameSlice)
#     num_expected_frames = expected_stop_index - expected_start_index
#     assert len(frame_slice) == num_expected_frames
#     assert reader.stats.seeks == 0

#     count = 0
#     for expected_frame_index, frame in zip(
#         range(expected_start_index, expected_stop_index), frame_slice
#     ):
#         assert frame.index == expected_frame_index
#         count += 1

#     assert count == num_expected_frames


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
        Slice[100:101],
        Slice[10:20],
        Slice[20:30],
        Slice[5:8],
    ],
)
def test_slices(reader: Reader, slice_arg: slice, correct_data: PacketData) -> None:
    assert [f.pts for f in reader[slice_arg]] == correct_data.pts[slice_arg]


def test_consuming_lazy_frame_slice(reader: Reader, correct_data: PacketData) -> None:
    assert reader.gop_size > 30
    start = reader.gop_size + 10
    stop = start + reader.gop_size + 10
    assert stop - start > 30
    reader.lazy_frame_slice_limit = 30

    num_wanted_frames = stop - start
    frames = reader[start:stop]
    assert len(frames) == stop - start
    assert reader.stats.seeks == 0

    # cosuming the slice will require a seek, but the rest of the slice will not
    count = 0
    for frame in frames:
        assert frame.pts == correct_data.pts[start + count]
        count += 1

    assert count == num_wanted_frames
    assert reader.stats.seeks == 1

    # the slice started 10 frames after a keyframe, so we expect to decode the frames
    # after the keyframe as well as the ones in the slice range
    assert reader.stats.decodes == num_wanted_frames + 10


def test_arbitrary_index(reader: Reader, correct_data: PacketData) -> None:
    for i in [0, 1, 2, 10, 20, 59, 70, 150]:
        assert reader[i].pts == correct_data.pts[i]
    for i in [-1, -10, -20, -150]:
        assert reader[i].pts == correct_data.pts[i]


def test_access_next_keyframe(reader: Reader, correct_data: PacketData) -> None:
    frame = reader[correct_data.gop_size]
    index = frame.index
    assert reader.stats.seeks == 1
    frame = reader[correct_data.gop_size - 1]
    assert frame.index == index - 1
    assert reader.stats.seeks == 2

    # expect to decode one of the second keyframe plus all of the previous one
    assert reader.stats.decodes == correct_data.gop_size + 1


def test_access_frame_before_next_keyframe(
    reader: Reader, correct_data: PacketData
) -> None:
    frame = reader[correct_data.gop_size - 1]
    assert reader.stats.seeks == 0

    index = frame.index
    frame = reader[correct_data.gop_size - 2]
    assert frame.index == index - 1
    assert reader.stats.seeks == 0


def test_by_time(reader: Reader, correct_data: PacketData) -> None:
    for time in correct_data.times:
        if time > 1:
            first_after_1s = time
            break
    assert reader.by_time[1.0:5.0][0].ts == first_after_1s  # type: ignore
