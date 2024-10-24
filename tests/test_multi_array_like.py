import numpy as np
import numpy.typing as npt
import pytest

from pupil_labs.video.multi_sequence import MultiArrayLike

Slice = type("", (object,), {"__getitem__": lambda _, key: key})()


@pytest.fixture
def base_array() -> npt.NDArray[np.int32]:
    return np.arange(100)


@pytest.fixture
def multi_array(base_array: npt.NDArray[np.int32]) -> MultiArrayLike[np.int32]:
    sequences = [base_array[i : i + 10] for i in range(0, 100, 10)]
    return MultiArrayLike[np.int32](sequences)


# def test_init_args(multi_part_video_paths: list[str]) -> None:
#     with pytest.raises(TypeError):
#         MultiPartReader(multi_part_video_paths[0])  # type: ignore

#     with pytest.raises(ValueError):
#         MultiPartReader([])


def test_iteration(
    multi_array: MultiArrayLike[np.int32], base_array: npt.NDArray[np.int32]
) -> None:
    for i, j in zip(multi_array, base_array):
        assert i == j


def test_backward_iteration_from_end(
    multi_array: MultiArrayLike[np.int32], base_array: npt.NDArray[np.int32]
) -> None:
    for i in reversed(range(len(multi_array))):
        assert multi_array[i] == base_array[i]


def test_backward_iteration_from_N(
    multi_array: MultiArrayLike[np.int32], base_array: npt.NDArray[np.int32]
) -> None:
    N = 100
    for i in reversed(range(N)):
        assert multi_array[i] == base_array[i]


def test_by_idx(
    multi_array: MultiArrayLike[np.int32], base_array: npt.NDArray[np.int32]
) -> None:
    frame_count = 0
    for i in range(len(multi_array)):
        assert multi_array[i] == base_array[i]
        frame_count += 1

    assert frame_count == len(multi_array)


def test_arbitrary_index(
    multi_array: MultiArrayLike[np.int32], base_array: npt.NDArray[np.int32]
) -> None:
    for i in [0, 1, 2, 10, 20, 59, 70]:
        assert multi_array[i] == base_array[i]
    for i in [-1, -10, -20]:
        assert multi_array[i] == base_array[i]


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
#         Slice[100:101],
#         Slice[10:20],
#         Slice[20:30],
#         Slice[5:8],
#     ],
# )
# def test_slices(
#     multi_array: MultiArrayLike[np.int32],
#     base_array: npt.NDArray[np.int32],
#     slice_arg: slice,
# ) -> None:
#     for i, j in zip(multi_array[slice_arg], base_array[slice_arg]):
#         assert i == j
