from functools import cached_property
from pathlib import Path
from types import TracebackType
from typing import SupportsIndex, cast, overload

import numpy as np

from .frameslice import FrameSlice
from .indexer import Indexer
from .multi_sequence import MultiArrayLike
from .reader import Reader, TimesArray, index_key_to_indices
from .sequence import ArrayLike
from .video_frame import VideoFrame


class MultiPartReader(MultiArrayLike[VideoFrame]):
    def __init__(self, paths: list[str] | list[Path]):
        if isinstance(paths, (str, Path)):
            raise TypeError("paths must be a list")

        if len(paths) < 1:
            raise ValueError("paths must not be empty")

        video_readers = [Reader(path) for path in paths]
        self._start_times = np.cumsum(
            [0] + [reader.duration for reader in video_readers]
        )
        sequences: list[ArrayLike] = list(video_readers)
        super().__init__(sequences)

    @cached_property
    def times(self) -> TimesArray:
        all_times = []
        for i in range(len(self.sequences)):
            times = cast(Reader, self.sequences[i]).times
            times = times.copy() + self._start_times[i]
            all_times.append(times)
        return np.concatenate(all_times)

    @overload
    def __getitem__(self, key: SupportsIndex) -> VideoFrame: ...
    @overload
    def __getitem__(self, key: slice) -> ArrayLike[VideoFrame]: ...

    def __getitem__(
        self, key: SupportsIndex | slice
    ) -> VideoFrame | ArrayLike[VideoFrame]:
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
        for reader in self.sequences:
            cast(Reader, reader).close()

    @property
    def width(self) -> int:
        reader = cast(Reader, self.sequences[0])
        return reader.width

    @property
    def height(self) -> int:
        reader = cast(Reader, self.sequences[0])
        return reader.height
