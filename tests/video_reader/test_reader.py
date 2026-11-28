import re
from fractions import Fraction
from pathlib import Path
from typing import Any, assert_type

import av
import numpy as np
import pytest

from pupil_labs.video_reader import Frame, Reader, Selection
from pupil_labs.video_reader._decoder import Decoder
from pupil_labs.video_reader._mp4 import shown_times

DATA = Path(__file__).parent.parent / "data"
VIDEO = DATA / "Neon Scene Camera - audio off.mp4"
LENGTH = 175
UNUSUAL = [
    "duplicate_pts.mp4",
    "multi_duplicate_pts.mp4",
    "out_of_order_pts.mp4",
    "way_out_of_order_pts.mp4",
    "silently_skipped_corrupt_frame.mp4",
]


@pytest.fixture
def reader() -> Reader:
    return Reader(VIDEO)


KEYS: list[Any] = [
    slice(None),
    slice(10, 100, 7),
    slice(None, None, -1),
    slice(150, 20, -13),
    slice(-5, None),
    slice(200, 300),
    slice(None, None, 2**64 - 1),
    np.arange(LENGTH) % 3 == 0,
    [i % 2 == 0 for i in range(LENGTH)],
    [5, 3, 5, -1, 0],
    np.array([170, 2, 2, 90], dtype=np.int16),
    np.array([], dtype=np.int64),
]


@pytest.mark.parametrize("key", KEYS)
def test_a_key_selects_what_it_selects_from_arange(
    reader: Reader,
    key: Any,  # noqa: ANN401
) -> None:
    expected = np.arange(LENGTH)[key]

    selection = reader[key]

    assert len(selection) == len(expected)
    assert [frame.index for frame in selection] == expected.tolist()


@pytest.mark.parametrize(
    "key",
    [LENGTH, -LENGTH - 1, [0, LENGTH], np.array([True, False]), 1.5],
)
def test_a_key_fails_where_it_fails_on_arange(reader: Reader, key: Any) -> None:  # noqa: ANN401
    with pytest.raises(IndexError) as numpy_error:
        np.arange(LENGTH)[key]

    with pytest.raises(
        type(numpy_error.value), match=re.escape(str(numpy_error.value))
    ):
        reader[key]


@pytest.mark.parametrize(
    "key",
    [True, np.False_, np.array(3), np.array([[1, 2]]), None, (1, 2), ([1, 2],), ...],
)
def test_a_key_numpy_reads_differently_is_refused(reader: Reader, key: Any) -> None:  # noqa: ANN401
    with pytest.raises((TypeError, IndexError)):
        reader[key]


def test_an_integer_returns_the_frame_at_that_position(reader: Reader) -> None:
    assert reader[3].index == 3
    assert reader[np.int64(-1)].index == LENGTH - 1
    assert_type(reader[np.int64(3)], Frame)
    assert_type(reader[np.arange(3)], Selection)


def test_a_selection_of_a_selection_counts_from_the_selection(reader: Reader) -> None:
    assert reader[10:100][::2][3].index == 16
    assert [f.index for f in reader[::-1][[0, 2]]] == [LENGTH - 1, LENGTH - 3]


def test_a_selection_is_fixed_and_checked_when_made(reader: Reader) -> None:
    key = np.array([1, 2, 3])
    selection = reader[key]
    key[:] = 0

    assert [frame.index for frame in selection] == [1, 2, 3]
    with pytest.raises(IndexError):
        reader[[0, LENGTH]]


def test_a_selection_decodes_nothing_until_a_frame_is_asked_for() -> None:
    reader = Reader(VIDEO)
    reader.close()

    selection = reader[::2]

    assert len(selection) == (LENGTH + 1) // 2
    with pytest.raises(AssertionError, match="Container is not open"):
        selection[0]


def test_a_frame_stays_unchanged_after_the_reader_moves_on(reader: Reader) -> None:
    frames = [reader[70], *reader[[100, 30]]]
    images = [frame.bgr.copy() for frame in frames]

    for i in [71, 150, 0, 69, 70]:
        reader[i].bgr  # noqa: B018
    for frame in reader[::-3]:
        frame.bgr  # noqa: B018

    assert [frame.index for frame in frames] == [70, 100, 30]
    for frame, image in zip(frames, images, strict=True):
        assert np.array_equal(frame.bgr, image)


@pytest.mark.parametrize("name", UNUSUAL)
def test_random_access_returns_what_reading_in_order_returns(name: str) -> None:
    in_order = [(f.index, f.pts, f.gray.tobytes()) for f in Reader(DATA / name)]
    reader = Reader(DATA / name)
    order = np.random.default_rng(0).permutation(len(reader))

    for i in order[:300]:
        frame = reader[int(i)]
        assert (frame.index, frame.pts, frame.gray.tobytes()) == in_order[i]
    for i, frame in zip(order[:300], reader[order[:300]], strict=True):
        assert (frame.index, frame.pts, frame.gray.tobytes()) == in_order[i]
    for i in order[:5]:
        [frame] = reader[[i]]
        assert (frame.index, frame.pts, frame.gray.tobytes()) == in_order[i]


def test_a_dropped_frame_repeats_the_next_one_however_it_is_asked_for() -> None:
    reader = Reader(DATA / "silently_skipped_corrupt_frame.mp4")
    following = reader[221].gray

    [alone] = reader[[220]]

    assert (alone.index, alone.pts) == (220, 79596181)
    assert np.array_equal(alone.gray, following)
    assert np.array_equal(reader[220].gray, following)


@pytest.mark.parametrize(
    "name", sorted(str(p.relative_to(DATA)) for p in DATA.glob("**/*.mp4"))
)
def test_timestamps_are_the_frames_times(name: str) -> None:
    reader = Reader(DATA / name)
    rate = reader.rate

    assert reader.timestamps.tolist() == [frame.time for frame in reader]
    assert reader.rate == rate


def test_a_fragmented_mp4_keeps_no_times_in_its_tables(tmp_path: Path) -> None:
    path = tmp_path / "fragmented.mp4"
    with av.open(path, "w", options={"movflags": "frag_keyframe+empty_moov"}) as out:
        stream = out.add_stream("libx264", rate=1)
        stream.width = stream.height = 64
        stream.time_base = Fraction(1, 1)
        frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), "bgr24")
        out.mux([*stream.encode(frame), *stream.encode()])

    times = shown_times(path)

    assert times is not None
    assert len(times) == 0


def test_passed_timestamps_become_the_frames_times() -> None:
    timestamps = np.arange(LENGTH) * 0.5 + 100
    reader = Reader(VIDEO, timestamps=timestamps)

    assert reader[7].time == 103.5
    assert [frame.time for frame in reader[-2:]] == timestamps[-2:].tolist()
    assert reader.duration == pytest.approx(LENGTH * 0.5)
    with pytest.raises(ValueError, match="174 timestamps for 175 frames"):
        Reader(VIDEO, timestamps=timestamps[1:])


def test_a_decoder_seeks_again_for_a_frame_it_already_handed_out() -> None:
    decoder = Decoder(DATA / "out_of_order_pts.mp4")
    decoder.collect(2000, [2330])

    frames = decoder.collect(2000, [2329, 2330])

    assert sorted(frames) == [2329, 2330]
