from dataclasses import dataclass
from functools import cached_property
from heapq import merge
from pathlib import Path

import av
import av.stream
import numpy as np
import pytest
from upath import UPath

from pupil_labs.video.frame import VideoFrame
from pupil_labs.video.frame_slice import FrameSlice
from pupil_labs.video.reader import Reader

from .utils import measure_fps


@dataclass
class PacketData:
    video_pts: list[int]
    audio_pts: list[int]
    video_times: list[float]
    audio_times: list[float]
    video_keyframe_indices: list[int]
    video_audio_times: list[list[float]]

    @cached_property
    def gop_size(self) -> int:
        return int(max(np.diff(self.video_keyframe_indices)))

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
                    ("len", len(self.video_pts)),
                    ("pts", self._summarize_list(self.video_pts)),
                    ("times", self._summarize_list(self.video_times)),
                    (
                        "keyframe_indices",
                        self._summarize_list(self.video_keyframe_indices),
                    ),
                ]
            )
            + ")"
        )


@pytest.fixture
def correct_data(video_path: Path) -> PacketData:  # noqa: C901
    audio_pts, video_pts = [], []
    audio_times, video_times = [], []
    audio_index, video_index = 0, 0
    video_keyframe_indices = []
    container = av.open(str(video_path))
    for packet in container.demux():
        if packet.pts is None:
            continue
        if packet.stream.type == "audio":
            audio_pts.append(packet.pts)
            audio_index += 1
        else:
            video_pts.append(packet.pts)
            if packet.is_keyframe:
                video_keyframe_indices.append(video_index)
            video_index += 1

    if video_pts:
        video_time_base = container.streams.video[0].time_base
        assert video_time_base
        video_times = [float(pts * video_time_base) for pts in video_pts]

    if audio_pts:
        audio_time_base = container.streams.audio[0].time_base
        assert audio_time_base
        audio_times = [float(pts * audio_time_base) for pts in audio_pts]

    video_audio_times = list[list[float]]()
    audio_times_buffer = list[float]()

    for frame_time, kind in merge(
        ((time, "audio") for time in audio_times),
        ((time, "video") for time in video_times),
        key=lambda x: [1],
    ):
        if kind == "audio":
            audio_times_buffer.append(frame_time)
            continue

        video_frame_audio_times = []
        i = 0
        while i < len(audio_times_buffer):
            audio_time = audio_times_buffer[i]
            if audio_time < frame_time:
                video_frame_audio_times.append(audio_time)
            else:
                break
            i += 1
        audio_times_buffer = audio_times_buffer[i:]
        video_audio_times.append(video_frame_audio_times)
    video_audio_times[-1] = audio_times_buffer

    return PacketData(
        video_pts=video_pts,
        video_times=video_times,
        audio_pts=audio_pts,
        audio_times=audio_times,
        video_audio_times=video_audio_times,
        video_keyframe_indices=video_keyframe_indices,
    )


@pytest.fixture
def reader(video_path: Path) -> Reader:
    reader = Reader(video_path)
    reader._log = True
    return reader


def test_pts(reader: Reader[VideoFrame], correct_data: PacketData) -> None:
    assert list(reader.pts) == correct_data.video_pts


def test_iteration(reader: Reader[VideoFrame], correct_data: PacketData) -> None:
    frame_count = 0
    for frame, expected_pts in measure_fps(zip(reader, correct_data.video_pts)):
        assert frame.pts == expected_pts
        frame_count += 1

    assert reader.stats.seeks == 1
    assert frame_count == len(correct_data.video_pts)


def test_backward_iteration_from_end(
    reader: Reader[VideoFrame], correct_data: PacketData
) -> None:
    total_keyframes = len(correct_data.video_keyframe_indices)
    assert total_keyframes <= len(correct_data.video_pts)

    expected_seeks = 1  # for gop
    expected_seeks += total_keyframes

    for i in reversed(range(len(reader))):
        assert reader[i].pts == correct_data.video_pts[i]
        assert reader.stats.seeks <= expected_seeks

    # we expect keyframe seeks to occur while iterating backwards, one per keyframe
    assert reader.stats.seeks == expected_seeks


def test_backward_iteration_from_N(
    reader: Reader[VideoFrame], correct_data: PacketData
) -> None:
    total_keyframes = len(correct_data.video_keyframe_indices)
    assert total_keyframes <= len(correct_data.video_pts)

    N = 100
    for i in reversed(range(N)):
        assert reader[i].pts == correct_data.video_pts[i]

    expected_seeks = 1  # for gop
    expected_seeks += round(N / correct_data.gop_size)
    assert reader.stats.seeks == expected_seeks


def test_by_idx(reader: Reader[VideoFrame], correct_data: PacketData) -> None:
    frame_count = 0
    for i, expected_pts in measure_fps(enumerate(correct_data.video_pts)):
        frame = reader[i]
        assert frame.pts == expected_pts
        frame_count += 1

    assert reader.stats.seeks == 1  # for gop
    assert frame_count == len(correct_data.video_pts)


def test_by_pts(reader: Reader[VideoFrame], correct_data: PacketData) -> None:
    for expected_pts in measure_fps(correct_data.video_pts):
        frame = reader._by_pts[expected_pts]
        assert frame.pts == expected_pts

    # TODO(dan): we can get this to 0
    expected_seeks = 2  # for gop_size and packet loading from gop to end
    assert reader.stats.seeks == expected_seeks


def test_accessing_pts_while_decoding(
    reader: Reader[VideoFrame], correct_data: PacketData
) -> None:
    for i, frame in enumerate(reader):
        if i < 100:
            assert Reader.pts.attrname not in reader.__dict__  # type: ignore
        else:
            reader.pts  # noqa: B018
            assert Reader.pts.attrname in reader.__dict__  # type: ignore
        assert frame.pts == correct_data.video_pts[i]


def test_accessing_times_while_decoding_by_frame_step(
    reader: Reader[VideoFrame], correct_data: PacketData
) -> None:
    for i in range(0, len(correct_data.video_pts), 2):
        if i < 10:
            assert Reader.pts.attrname not in reader.__dict__  # type: ignore
            frame = reader[i]
            assert frame.pts == correct_data.video_pts[i]
        else:
            reader.pts  # noqa: B018
            assert Reader.pts.attrname in reader.__dict__  # type: ignore
            frame = reader[i]
            assert frame.pts == correct_data.video_pts[i]


# 100+74 = 174, max frames for smallest neon scene vid and 60 is keyframe
@pytest.mark.parametrize("start", [55, 59, 60, 61, 74])
@pytest.mark.parametrize("delta", [1, 5, 30, 59, 60, 61, 100])
def test_seeking_to_various_frames(
    reader: Reader[VideoFrame], correct_data: PacketData, start: int, delta: int
) -> None:
    assert reader[start].pts == correct_data.video_pts[start]
    assert reader[start + delta].pts == correct_data.video_pts[start + delta]


def test_accessing_times_before_decoding(
    reader: Reader[VideoFrame], correct_data: PacketData
) -> None:
    assert Reader.pts.attrname not in reader.__dict__  # type: ignore
    reader.pts  # noqa: B018
    assert Reader.pts.attrname in reader.__dict__  # type: ignore
    for i, frame in enumerate(reader):
        assert frame.pts == correct_data.video_pts[i]


def test_gop_size(reader: Reader[VideoFrame], correct_data: PacketData) -> None:
    assert reader.gop_size == correct_data.gop_size
    assert reader.stats.seeks == 0


def test_gop_size_on_seeked_container_within_gop_size(
    reader: Reader[VideoFrame], correct_data: PacketData
) -> None:
    index = correct_data.gop_size * 2

    # access some frames to cause a seek
    assert reader[10].pts == correct_data.video_pts[10]
    assert reader[index].pts == correct_data.video_pts[index]
    assert reader.stats.seeks == 3  # one for gop

    # now check the gop_size
    assert reader.gop_size == correct_data.gop_size
    assert reader.stats.seeks == 3

    assert reader[10].pts == correct_data.video_pts[10]
    assert reader[index].pts == correct_data.video_pts[index]


def test_seek_avoidance_arbitrary_seek(
    reader: Reader[VideoFrame], correct_data: PacketData
) -> None:
    reader[correct_data.gop_size * 2]
    assert reader.stats.decodes <= correct_data.gop_size

    expected_seeks = 0
    expected_seeks += 1  # seek to get pts
    expected_seeks += 1  # seek to start

    assert reader.stats.seeks == expected_seeks


def test_seek_avoidance(reader: Reader[VideoFrame], correct_data: PacketData) -> None:
    assert reader.stats.seeks == 0

    assert reader.stats.decodes == 0

    # we seek once for getting gop size to set buffers when loading the first frame
    expected_seeks = 1

    reader[0]
    assert reader.stats.seeks == expected_seeks
    assert reader.stats.decodes == 1

    # a second access will load the frame from buffer and not seek/decode
    reader[0]
    assert reader.stats.seeks == expected_seeks
    assert reader.stats.decodes == 1

    # getting the second frame will also not require a seek
    reader[1]
    assert reader.stats.seeks == expected_seeks
    assert reader.stats.decodes == 2

    # moving forward in same keyframe's worth of frames won't seek
    frames = reader[10:20]
    assert len(frames) == 10
    assert [f.pts for f in frames] == correct_data.video_pts[10:20]

    assert reader.stats.decodes == 20

    assert reader.stats.seeks == expected_seeks

    gop_size = correct_data.gop_size
    # moving forward till next keyframe won't seek
    frame = reader[gop_size]
    assert frame.index == gop_size

    if reader._stream.name == "mjpeg":
        assert reader.stats.decodes == 20
        # expected_seeks += 1
    else:
        assert reader.stats.decodes == gop_size + 1

    assert reader.stats.seeks == expected_seeks

    # no seek even when getting last frame of that next keyframe
    previous_decodes = reader.stats.decodes
    frame = reader[gop_size * 2 - 1]
    assert frame.index == gop_size * 2 - 1
    assert frame.pts == correct_data.video_pts[gop_size * 2 - 1]
    assert reader.stats.seeks == expected_seeks
    if reader._stream.name == "mjpeg":
        assert reader.stats.decodes == previous_decodes
    else:
        assert reader.stats.decodes > previous_decodes


Slice = type("", (object,), {"__getitem__": lambda _, key: key})()
"""
Syntax sugar helper for frame tests to define a slice selection

>>> Slice[:300]
slice(None, 300, None)
"""


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
@pytest.mark.parametrize(
    "subslice_arg",
    [
        Slice[:],
        Slice[:50],
        Slice[:-50],
        Slice[-50:],
        Slice[-50:],
        Slice[-40:-20],
        Slice[30:100],
    ],
)
def test_lazy_slice(
    slice_arg: slice,
    subslice_arg: slice,
    reader: Reader[VideoFrame],
    correct_data: PacketData,
) -> None:
    expected_indexes = range(len(correct_data.video_pts))[slice_arg][subslice_arg]
    reader.lazy_frame_slice_limit = 0
    upper_slice = reader[slice_arg]

    # the first slice will require a seek on containers that don't provide cheap length
    expected_seeks = 0
    if reader._stream.name == "mjpeg" and (
        (slice_arg.start is not None and slice_arg.start < 0)
        or (slice_arg.stop is None or slice_arg.stop < 0)
    ):
        expected_seeks += 1
    assert reader.stats.seeks == expected_seeks

    # a subslice will not need to do this since the index bounds are already known
    sub_slice = upper_slice[subslice_arg]
    assert reader.stats.seeks == expected_seeks

    num_expected_frames = expected_indexes.stop - expected_indexes.start
    assert isinstance(sub_slice, FrameSlice)
    assert len(sub_slice) == num_expected_frames

    assert reader.stats.seeks == expected_seeks

    count = 0
    for expected_frame_index, frame in zip(expected_indexes, sub_slice):
        assert frame.index == expected_frame_index
        count += 1

    assert count == num_expected_frames


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
def test_slices(
    reader: Reader[VideoFrame], slice_arg: slice, correct_data: PacketData
) -> None:
    assert [f.pts for f in reader[slice_arg]] == correct_data.video_pts[slice_arg]


def test_consuming_lazy_frame_slice(
    reader: Reader[VideoFrame], correct_data: PacketData
) -> None:
    if reader._stream.name == "mjpeg":
        start = 10
        stop = 30
        num_wanted_frames = stop - start
        frames = reader[start:stop]
        assert len(frames) == stop - start
        assert reader.stats.seeks == 2  # load the pts

        # cosuming the slice will require a seek, but the rest of the slice will not
        count = 0
        for frame in frames:
            assert frame.pts == correct_data.video_pts[start + count]
            count += 1

        assert count == num_wanted_frames
        assert reader.stats.seeks == 2  # one to load pts, one to seek back
        assert reader.stats.decodes == num_wanted_frames

    else:
        assert correct_data.gop_size > 30
        start = correct_data.gop_size + 10
        stop = start + correct_data.gop_size + 10
        assert stop - start > 30

        num_wanted_frames = stop - start
        frames = reader[start:stop]
        assert len(frames) == stop - start
        assert reader.stats.seeks == 0

        # cosuming the slice will require a seek, but the rest of the slice will not
        count = 0
        for frame in frames:
            assert frame.pts == correct_data.video_pts[start + count]
            count += 1

        assert count == num_wanted_frames
        assert reader.stats.seeks == 2  # one to load pts, one to seek back

        # the slice started 10 frames after a keyframe so we expect to decode frames
        # after the keyframe as well as the ones in the slice range
        assert reader.stats.decodes == num_wanted_frames + 10


def test_arbitrary_index(reader: Reader[VideoFrame], correct_data: PacketData) -> None:
    for i in [0, 1, 2, 10, 20, 59, 70, 150]:
        assert reader[i].pts == correct_data.video_pts[i]
    for i in [-1, -10, -20, -150]:
        assert reader[i].pts == correct_data.video_pts[i]


def test_access_previous_keyframe(
    reader: Reader[VideoFrame], correct_data: PacketData
) -> None:
    frame = reader[correct_data.gop_size]
    index = frame.index
    assert reader.stats.seeks == 1  # one to get pts for frame
    frame = reader[correct_data.gop_size - 1]
    assert frame.index == index - 1
    assert reader.stats.seeks == 2  # seek to previous keyframe

    # expect to decode one of the second keyframe plus all of the previous one
    assert reader.stats.decodes == correct_data.gop_size + 1


def test_access_frame_before_next_keyframe(
    reader: Reader[VideoFrame], correct_data: PacketData
) -> None:
    frame = reader[correct_data.gop_size * 2 - 1]

    expected_seeks = 1 if correct_data.gop_size == 1 else 2

    assert reader.stats.seeks == expected_seeks

    index = frame.index
    frame = reader[correct_data.gop_size * 2 - 2]
    assert frame.index == index - 1

    if correct_data.gop_size == 1:
        expected_seeks += 1
    assert reader.stats.seeks == expected_seeks


def test_times_return_container_times(
    reader: Reader[VideoFrame], correct_data: PacketData
) -> None:
    result_times = [
        frame.av_frame.time for frame in reader.by_container_timestamps[1.0:5.0]
    ]
    expected_times = [time for time in correct_data.video_times if 1.0 <= time < 5.0]
    assert expected_times == result_times


def test_external_times(video_path: str, correct_data: PacketData) -> None:
    external_timestamps = [i * 0.1 for i in range(len(correct_data.video_pts))]
    reader = Reader(source=video_path, container_timestamps=external_timestamps)
    assert np.all(reader.container_timestamps == external_timestamps)

    result_frames = reader.by_container_timestamps[1.0:5.0]
    result_timestamps = [frame.time for frame in result_frames]
    expected_times = [time for time in external_timestamps if 1.0 <= time < 5.0]
    assert np.all(expected_times == result_timestamps)


def test_external_times_being_set(
    reader: Reader[VideoFrame], correct_data: PacketData
) -> None:
    external_timestamps = [i * 0.1 for i in range(len(correct_data.video_pts))]
    reader.container_timestamps = external_timestamps
    assert np.all(reader.container_timestamps == external_timestamps)

    result_frames = reader.by_container_timestamps[1.0:5.0]
    result_times = [frame.time for frame in result_frames]
    expected_times = [time for time in external_timestamps if 1.0 <= time < 5.0]
    assert np.all(expected_times == result_times)


def test_by_container_times(
    reader: Reader[VideoFrame], correct_data: PacketData
) -> None:
    expected_times = [time for time in correct_data.video_times if 1.0 <= time < 5.0]
    result_times = [
        frame.av_frame.time for frame in reader.by_container_timestamps[1.0:5.0]
    ]
    assert expected_times == result_times


def test_audio_reader(reader: Reader[VideoFrame], correct_data: PacketData) -> None:
    if not correct_data.audio_pts:
        assert not reader.audio
    else:
        assert reader.audio
        audio_reader = reader.audio
        for i in range(len(correct_data.audio_pts)):
            assert audio_reader[i].pts == correct_data.audio_pts[i]

        assert len(audio_reader) == len(correct_data.audio_pts)


def test_upath_support(video_path: Path, correct_data: PacketData) -> None:
    path = UPath(video_path)
    reader = Reader(path)
    assert len(reader) == len(correct_data.video_pts)
