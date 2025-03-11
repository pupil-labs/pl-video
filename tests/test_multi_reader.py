from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import av
import numpy as np
import pytest

from pupil_labs.video.multi_reader import MultiReader

from .utils import measure_fps

Slice = type("", (object,), {"__getitem__": lambda _, key: key})()


@dataclass
class PacketData:
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
        return len(self.times)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            + ", ".join(
                f"{key}={value}"
                for key, value in [
                    ("len", len(self.times)),
                    ("times", self._summarize_list(self.times)),
                    ("keyframe_indices", self._summarize_list(self.keyframe_indices)),
                ]
            )
            + ")"
        )


@pytest.fixture
def correct_data(multi_part_video_paths: list[str]) -> PacketData:
    times_bias = 0
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
            times.append(float(packet.pts * stream.time_base) + times_bias)
            if packet.is_keyframe:
                keyframe_indices.append(index)
            index += 1

        times_bias += float(stream.duration * stream.time_base)
    return PacketData(times=times, keyframe_indices=keyframe_indices)


@pytest.fixture
def reader(multi_part_video_paths: list[str]) -> MultiReader:
    return MultiReader(multi_part_video_paths)


def test_context_manager(multi_part_video_paths: list[str]) -> None:
    with MultiReader(multi_part_video_paths) as reader:
        assert reader is not None


def test_init_args(multi_part_video_paths: list[str]) -> None:
    assert MultiReader(multi_part_video_paths[0])

    with pytest.raises(ValueError):
        MultiReader([])


def test_iteration(reader: MultiReader, correct_data: PacketData) -> None:
    frame_count = 0
    for frame, expected_times in measure_fps(zip(reader, correct_data.times)):
        assert frame.ts == expected_times
        frame_count += 1

    assert frame_count == len(correct_data.times)


def test_backward_iteration_from_end(
    reader: MultiReader, correct_data: PacketData
) -> None:
    total_keyframes = len(correct_data.keyframe_indices)
    assert total_keyframes <= len(correct_data.times)

    for i in reversed(range(len(reader))):
        assert reader[i].ts == correct_data.times[i]


def test_backward_iteration_from_N(
    reader: MultiReader, correct_data: PacketData
) -> None:
    total_keyframes = len(correct_data.keyframe_indices)
    assert total_keyframes <= len(correct_data.times)

    N = 100
    for i in reversed(range(N)):
        assert reader[i].ts == correct_data.times[i]


def test_by_idx(reader: MultiReader, correct_data: PacketData) -> None:
    frame_count = 0
    for i, expected_time in measure_fps(enumerate(correct_data.times)):
        frame = reader[i]
        assert frame.ts == expected_time
        frame_count += 1

    assert frame_count == len(correct_data.times)


def test_arbitrary_index(reader: MultiReader, correct_data: PacketData) -> None:
    for i in [0, 1, 2, 10, 20, 59, 70, 150]:
        assert reader[i].ts == correct_data.times[i]
    for i in [-1, -10, -20, -150]:
        assert reader[i].ts == correct_data.times[i]


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
    reader: MultiReader, slice_arg: slice, correct_data: PacketData
) -> None:
    for frame, index in zip(reader[slice_arg], range(*slice_arg.indices(len(reader)))):
        assert frame.index == index
        assert frame.ts == correct_data.times[index]


def test_times(reader: MultiReader, correct_data: PacketData) -> None:
    assert np.allclose(reader.container_timestamps, correct_data.times)


def test_by_time(reader: MultiReader, correct_data: PacketData) -> None:
    for time in correct_data.times:
        if time > 1:
            first_after_1s = time
            break
    assert reader.by_container_timestamps[1.0:5.0][0].ts == first_after_1s  # type: ignore

    for time in correct_data.times:
        if time > 15:
            first_after_15s = time
            break
    assert reader.by_container_timestamps[15.0:20.0][0].ts == first_after_15s  # type: ignore


@pytest.fixture
def expected_size(multi_part_video_paths: list[str]) -> tuple[int, int]:
    path = Path(multi_part_video_paths[0])
    if path.name.startswith("PI"):
        return (1088, 1080)
    elif path.name.startswith("Neon"):
        return (1600, 1200)
    else:
        raise ValueError(f"Unknown video: {path}")


def test_width_and_height(reader: MultiReader, expected_size: tuple[int, int]) -> None:
    assert reader.width == expected_size[0]
    assert reader.height == expected_size[1]
