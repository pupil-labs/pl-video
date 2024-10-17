from pathlib import Path
import av

from pupil_labs.video_simple.reader import Reader
import pytest

from .utils import measure_fps


@pytest.fixture
def correct_pts(video_path: Path):
    correct_pts = []
    for packet in av.open(str(video_path)).demux(video=0):
        if packet.pts is None:
            continue
        correct_pts.append(packet.pts)
    return correct_pts


@pytest.fixture
def reader(video_path: Path):
    return Reader(video_path)


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

    # getting the 10th frame will require a seek, but the rest of the slice will not
    reader.by_idx[10:20]
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
