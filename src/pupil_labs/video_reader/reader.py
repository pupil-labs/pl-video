from collections.abc import Callable, Iterator, Sequence
from functools import cached_property
from itertools import chain
from os import PathLike
from types import EllipsisType, TracebackType
from typing import Any, Self, overload

import av
import numpy as np
import numpy.typing as npt

from pupil_labs.video_reader._decoder import Decoder
from pupil_labs.video_reader._parallel import Job, frames_in_order
from pupil_labs.video_reader.frame import Frame

type FrameKey = int | np.integer[Any]
type SelectionKey = (
    slice
    | Sequence[int | np.integer[Any]]
    | Sequence[bool | np.bool_]
    | npt.NDArray[np.integer[Any]]
    | npt.NDArray[np.bool_]
)


class Reader:
    """Frames of the first video stream of an MP4 file, indexed like a numpy array.

    `reader[i]` decodes one frame. A slice, a Boolean mask or an array of indices
    returns a `Selection`, which decodes nothing until a frame is asked for.
    `timestamps`, one per frame, become the frames' `time` instead of their pts.
    """

    def __init__(
        self,
        source: str | PathLike[str],
        timestamps: npt.ArrayLike | None = None,
    ) -> None:
        self._source = source
        self._timestamps = (
            None if timestamps is None else np.array(timestamps, dtype=np.float64)
        )
        self._decoder = Decoder(source, self._timestamps)
        self._length = len(self._decoder)
        if self._timestamps is not None and self._timestamps.shape != (self._length,):
            raise ValueError(
                f"{len(self._timestamps)} timestamps for {self._length} frames"
            )
        self._cache: dict[int, Frame] = {}
        self._cached_group = (0, 0)
        self._previous = -1

    def __len__(self) -> int:
        return self._length

    @overload
    def __getitem__(self, key: FrameKey) -> Frame: ...
    @overload
    def __getitem__(self, key: SelectionKey) -> "Selection": ...
    def __getitem__(self, key: FrameKey | SelectionKey) -> "Frame | Selection":
        selected = _select(self._length, key)
        if isinstance(selected, int):
            return self._frame(selected)
        return Selection(self._frame, self._frames, selected)

    def __iter__(self) -> Iterator[Frame]:
        return iter(self[:])

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._cache.clear()
        self._decoder.close()

    @property
    def width(self) -> int:
        return self._decoder.width

    @property
    def height(self) -> int:
        return self._decoder.height

    @cached_property
    def timestamps(self) -> npt.NDArray[np.float64]:
        """Frame times in seconds: those passed in, or each frame's pts in seconds."""
        if self._timestamps is None:
            return _read_timestamps(self._source)
        return self._timestamps

    @property
    def rate(self) -> float:
        """Average frame rate in Hz."""
        if self._timestamps is not None:
            return float(1 / np.mean(np.diff(self._timestamps)))
        return self._decoder.average_rate

    @property
    def duration(self) -> float:
        """Duration in seconds, including the last frame's."""
        if self._timestamps is not None:
            span = float(self._timestamps[-1] - self._timestamps[0])
            return span + 1 / self.rate
        return self._decoder.duration

    def _frame(self, index: int) -> Frame:
        previous, self._previous = self._previous, index
        frame = self._cache.get(index)
        if frame is not None:
            return frame
        decoder = self._decoder
        start, stop = self._cached_group
        if not start <= index < stop:
            start = decoder.keyframe_before(index)
            stop = decoder.group_end(start)
            self._cache = {i: f for i, f in self._cache.items() if start <= i < stop}
            self._cached_group = start, stop
        # Earlier frames of the group are kept only for a caller stepping backwards.
        keep_from = start if 0 < previous - index < stop - start else index
        if not decoder.produces(index):
            decoder.seek(start)
        while frame is None:
            for produced in decoder.step():
                if produced.index >= keep_from:
                    self._cache[produced.index] = produced
            frame = self._cache.get(index)
        return frame

    def _frames(self, indices: list[int]) -> Iterator[Frame]:
        jobs = self._jobs(indices)
        first = next(jobs, None)
        if first is None:
            return
        keyframe = first[0]
        yield from frames_in_order(
            self._decoder,
            lambda threads: Decoder(self._source, self._timestamps, threads),
            chain([first], jobs),
            self._decoder.group_end(keyframe) - keyframe,
        )

    def _jobs(self, indices: list[int]) -> Iterator[Job]:
        start = stop = 0
        targets: list[int] = []
        for index in indices:
            if not start <= index < stop:
                if targets:
                    yield start, targets
                    targets = []
                start = self._decoder.keyframe_before(index)
                stop = self._decoder.group_end(start)
            targets.append(index)
        if targets:
            yield start, targets


class Selection(Sequence[Frame]):
    """Frames picked from a reader, in the order the key picked them.

    The indices are fixed and checked when the selection is made; frames are
    decoded as they are asked for.
    """

    def __init__(
        self,
        frame_at: Callable[[int], Frame],
        frames_at: Callable[[list[int]], Iterator[Frame]],
        indices: npt.NDArray[np.intp],
    ) -> None:
        self._frame_at = frame_at
        self._frames_at = frames_at
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    @overload
    def __getitem__(self, key: FrameKey) -> Frame: ...
    @overload
    def __getitem__(self, key: SelectionKey) -> "Selection": ...
    def __getitem__(self, key: FrameKey | SelectionKey) -> "Frame | Selection":
        selected = _select(len(self._indices), key)
        if isinstance(selected, int):
            return self._frame_at(int(self._indices[selected]))
        return Selection(self._frame_at, self._frames_at, self._indices[selected])

    def __iter__(self) -> Iterator[Frame]:
        return self._frames_at(self._indices.tolist())

    def __repr__(self) -> str:
        return f"Selection({len(self)} frames)"


def _select(length: int, key: FrameKey | SelectionKey) -> int | npt.NDArray[np.intp]:
    if isinstance(key, bool | np.bool_ | tuple | EllipsisType) or (
        isinstance(key, np.ndarray) and key.ndim == 0
    ):
        raise TypeError(f"{key!r} does not select frames")
    if isinstance(key, int | np.integer):
        index = int(key)
        if not -length <= index < length:
            raise IndexError(
                f"index {index} is out of bounds for axis 0 with size {length}"
            )
        return index % length
    # numpy checks the key and raises its own errors on an array that takes no
    # memory; numpy's stubs take no sequence that mixes Python and numpy integers.
    numpy_key: Any = key
    if np.broadcast_to(np.intp(0), length)[numpy_key].ndim != 1:
        raise IndexError(f"{key!r} is not a one-dimensional key")
    if isinstance(key, slice):
        picked = range(*key.indices(length))
        step = picked.step if len(picked) > 1 else 0
        return picked.start + step * np.arange(len(picked))
    array = np.asarray(key)
    if array.dtype == np.bool_:
        return np.flatnonzero(array)
    return array.astype(np.intp) % length


def _read_timestamps(source: str | PathLike[str]) -> npt.NDArray[np.float64]:
    with av.open(source, "r") as container:
        stream = container.streams.video[0]
        assert stream.time_base is not None
        pts = [p.pts for p in container.demux(stream) if p.pts is not None]
        time_base = stream.time_base
    seconds = np.sort(np.array(pts, dtype=np.float64)) * time_base.numerator
    return seconds / time_base.denominator
