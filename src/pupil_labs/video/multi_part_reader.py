from functools import cached_property
from pathlib import Path
from types import TracebackType
from typing import Sequence, overload

import numpy as np

from .array_like import ArrayLike
from .frame import VideoFrame
from .frameslice import FrameSlice
from .indexer import Indexer
from .multi_array_like import MultiArrayLike
from .reader import Reader, TimesArray, index_key_to_indices


class MultiPartReader(MultiArrayLike[VideoFrame]):
    def __init__(self, paths: Sequence[str] | list[Path]):
        if isinstance(paths, (str, Path)):
            raise TypeError("paths must be a list")

        if len(paths) < 1:
            raise ValueError("paths must not be empty")

        # Declar that the ArrayLikes we are using in our MultiArrayLike are Readers
        self.arrays: Sequence[Reader] = []

        video_readers = [Reader(path) for path in paths]
        self._start_times = np.cumsum(
            [0] + [reader.duration for reader in video_readers]
        )
        super().__init__(video_readers)

    @cached_property
    def times(self) -> TimesArray:
        all_times = []
        for i in range(len(self.arrays)):
            times = self.arrays[i].times + self._start_times[i]
            all_times.append(times)
        return np.concatenate(all_times)

    @overload
    def __getitem__(self, key: int) -> VideoFrame: ...
    @overload
    def __getitem__(self, key: slice) -> ArrayLike[VideoFrame]: ...

    def __getitem__(self, key: int | slice) -> VideoFrame | ArrayLike[VideoFrame]:
        if isinstance(key, int):
            frame = super().__getitem__(key)
            frame.index = index_key_to_indices(key, self)[0]

            part_index = (
                np.searchsorted(self._start_indices, frame.index, side="right").item()
                - 1
            )
            frame.ts = frame.ts + self._start_times[part_index]
            return frame
        else:
            assert isinstance(key, slice)
            return FrameSlice[VideoFrame](self, key)

    @cached_property
    def by_time(self) -> Indexer[VideoFrame]:
        return Indexer(self.times, self)

    def __enter__(self) -> "MultiPartReader":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        for reader in self.arrays:
            reader.close()

    @property
    def width(self) -> int:
        return self.arrays[0].width

    @property
    def height(self) -> int:
        return self.arrays[0].height
