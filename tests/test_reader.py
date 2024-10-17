from dataclasses import dataclass
from pathlib import Path

import av
import pytest

from pupil_labs.video_simple.reader import AVStreamPacketsInfo, Reader

from .utils import measure_fps


@dataclass
class PacketData:
    pts: list[int]
    times: list[float]


@pytest.fixture
def correct_packet_info(video_path: Path):
    pts = []
    times = []
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    assert stream.time_base
    for packet in container.demux(stream):
        if packet.pts is None:
            continue
        pts.append(packet.pts)
        times.append(float(packet.pts * stream.time_base))
    return PacketData(pts=pts, times=times)


@pytest.fixture
def correct_pts(correct_packet_info: AVStreamPacketsInfo):
    return correct_packet_info.pts


@pytest.fixture
def correct_av_times(correct_packet_info: AVStreamPacketsInfo):
    return correct_packet_info.times


@pytest.fixture
def reader(video_path: Path):
    return Reader(video_path)


@pytest.fixture
def reader_with_ts(video_path: Path, correct_pts):
    timestamps = [i / 10.0 for i in range(len(correct_pts))]
    return Reader(video_path, timestamps)


def test_pts(reader: Reader, correct_pts):
    assert list(reader.pts) == correct_pts


def test_iteration(reader: Reader, correct_pts):
    frame_count = 0
    for frame, expected_pts in measure_fps(zip(reader, correct_pts)):
        assert frame.pts == expected_pts
        frame_count += 1

    assert reader.stats.seeks == 0
    assert frame_count == len(correct_pts)


def test_by_idx(reader: Reader, correct_pts):
    frame_count = 0
    for i, expected_pts in measure_fps(enumerate(correct_pts)):
        frame = reader.by_idx[i]
        assert frame.pts == expected_pts
        frame_count += 1

    assert reader.stats.seeks == 0
    assert frame_count == len(correct_pts)


def test_by_pts(reader: Reader, correct_pts):
    for expected_pts in measure_fps(correct_pts):
        frame = reader.by_pts[expected_pts]
        assert frame.pts == expected_pts

    assert reader.stats.seeks == 0


def test_seek_avoidance(reader: Reader):
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

    # slices are lazy, so they don't actually seek or decode
    frames = reader.by_idx[10:20]
    assert len(frames) == 10
    assert reader.stats.seeks == 0
    assert reader.stats.decodes == 2

    # cosuming the slice will require a seek, but the rest of the slice will not
    for frame in frames:
        pass
    assert len(frames) == 10
    assert reader.stats.seeks == 1
    # since the keyframe is at 0, we will need to decode all frames from 0 to 20
    assert reader.stats.decodes == 22


def test_arbitrary_index(reader: Reader, correct_pts):
    for i in [0, 1, 2, 10, 20]:
        assert reader.by_idx[i].pts == correct_pts[i]
    for i in [-1, -10, -20]:
        assert reader.by_idx[i].pts == correct_pts[i]


def test_arbitrary_slices(reader: Reader, correct_pts):
    assert [f.pts for f in reader.by_idx[100:101]] == correct_pts[100:101]
    assert [f.pts for f in reader.by_idx[10:20]] == correct_pts[10:20]
    assert [f.pts for f in reader.by_idx[20:30]] == correct_pts[20:30]
    assert [f.pts for f in reader.by_idx[5:8]] == correct_pts[5:8]


def test_by_ts_without_passed_in_timestamps(
    reader: Reader, correct_packet_info: AVStreamPacketsInfo
):
    for time in correct_packet_info.times:
        if time > 1:
            first_after_1s = time
            break
    assert reader.by_ts[1.0:5.0][0].time == first_after_1s


def test_by_ts_with_passed_in_timestamps(reader_with_ts: Reader):
    for time in reader_with_ts.timestamps:
        if time > 1:
            first_after_1s = time
            break

    assert reader_with_ts.by_ts[0.3].ts == 0.3
