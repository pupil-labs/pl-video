from dataclasses import dataclass
from pathlib import Path

import av
import pytest

from pupil_labs.video.reader import (
    AVStreamPacketsInfo,
    PTSArray,
    Reader,
    TimesArray,
)

from .utils import measure_fps


@dataclass
class PacketData:
    pts: list[int]
    times: list[float]
    keyframe_indices: list[int]


@pytest.fixture
def correct_packet_info(video_path: Path) -> PacketData:
    pts = []
    times = []
    keyframe_indices = []
    container = av.open(str(video_path))  # type:ignore
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
def correct_pts(correct_packet_info: AVStreamPacketsInfo) -> PTSArray:
    return correct_packet_info.pts


@pytest.fixture
def correct_av_times(correct_packet_info: AVStreamPacketsInfo) -> TimesArray:
    return correct_packet_info.times


@pytest.fixture
def reader(video_path: Path) -> Reader:
    return Reader(video_path)


@pytest.fixture
def reader_with_ts(video_path: Path, correct_pts: PTSArray) -> Reader:
    timestamps = [i / 10.0 for i in range(len(correct_pts))]
    return Reader(video_path, timestamps)


def test_pts(reader: Reader, correct_pts: PTSArray) -> Reader:
    assert list(reader.pts) == correct_pts


def test_iteration(reader: Reader, correct_pts: PTSArray) -> None:
    frame_count = 0
    for frame, expected_pts in measure_fps(zip(reader, correct_pts)):
        assert frame.pts == expected_pts
        frame_count += 1

    assert reader.stats.seeks == 0
    assert frame_count == len(correct_pts)


def test_by_idx(reader: Reader, correct_pts: PTSArray) -> None:
    frame_count = 0
    for i, expected_pts in measure_fps(enumerate(correct_pts)):
        frame = reader.by_idx[i]
        assert frame.pts == expected_pts
        frame_count += 1

    assert reader.stats.seeks == 0
    assert frame_count == len(correct_pts)


def test_by_pts(reader: Reader, correct_pts: PTSArray) -> None:
    for expected_pts in measure_fps(correct_pts):
        frame = reader.by_pts[expected_pts]
        assert frame.pts == expected_pts

    assert reader.stats.seeks == 0


def test_seek_avoidance(reader: Reader) -> None:
    assert reader.stats.seeks == 0
    assert reader.stats.decodes == 0

    # we dont need to seek when loading the first frame
    reader.by_idx[0]
    assert reader.stats.seeks == 0
    assert reader.stats.decodes == 1

    # a second access will load the frame from buffer and not seek/decode
    reader.by_idx[0]
    assert reader.stats.seeks == 0
    assert reader.stats.decodes == 1

    # getting the second frame will also not require a seek
    reader.by_idx[1]
    assert reader.stats.seeks == 0
    assert reader.stats.decodes == 2

    # Short slices are loaded into the buffer, so they do require one seek
    frames = reader.by_idx[10:20]
    assert len(frames) == 10
    assert reader.stats.seeks == 1
    # since the keyframe is at 0, we will need to decode all frames from 0 to 20
    assert reader.stats.decodes == 22

    # Long slices are lazy, so they don't actually seek or decode
    frames = reader.by_idx[30:]
    assert len(frames) == len(reader) - 30
    assert reader.stats.seeks == 1
    assert reader.stats.decodes == 22

    # cosuming the slice will require a seek, but the rest of the slice will not
    for _ in frames:
        pass
    assert len(frames) == len(reader) - 30
    assert reader.stats.seeks == 2
    # Since we don't know where exactly the keyframe is, we can only compare against a minimum number of decodes
    assert reader.stats.decodes >= 22 + len(reader) - 30


def test_arbitrary_index(reader: Reader, correct_pts: PTSArray) -> None:
    for i in [0, 1, 2, 10, 20]:
        assert reader.by_idx[i].pts == correct_pts[i]
    for i in [-1, -10, -20]:
        assert reader.by_idx[i].pts == correct_pts[i]


def test_arbitrary_slices(reader: Reader, correct_pts: PTSArray) -> None:
    assert [f.pts for f in reader.by_idx[100:101]] == correct_pts[100:101]
    assert [f.pts for f in reader.by_idx[10:20]] == correct_pts[10:20]
    assert [f.pts for f in reader.by_idx[20:30]] == correct_pts[20:30]
    assert [f.pts for f in reader.by_idx[5:8]] == correct_pts[5:8]


def test_by_ts_without_passed_in_timestamps(reader: Reader, correct_packet_info: AVStreamPacketsInfo) -> None:
    for time in correct_packet_info.times:
        if time > 1:
            first_after_1s = time
            break
    assert reader.by_ts[1.0:5.0][0].ts == first_after_1s


def test_by_ts_with_passed_in_timestamps(reader_with_ts: Reader) -> None:
    assert reader_with_ts.by_ts[0.3].ts == 0.3


def test_backward_iteration(reader: Reader, correct_packet_info: PacketData) -> None:
    total_keyframes = len(correct_packet_info.keyframe_indices)
    assert total_keyframes <= len(correct_packet_info.pts)

    for i in reversed(range(len(reader))):
        reader[i]

    # we expect keyframe seeks to occur while iterating backwards, one per keyframe
    assert reader.stats.seeks == total_keyframes
