from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import TracebackType
from typing import Generic, TypeVar, cast, overload

import numpy as np

from .frameslice import FrameSlice
from .indexer import Indexer
from .reader import Reader, TimesArray, index_key_to_indices
from .video_frame import VideoFrame

T = TypeVar("T")


@dataclass
class MultiSequence(Generic[T], Sequence[T]):
    sequences: Sequence[Sequence[T]]

    def __post_init__(self) -> None:
        self._start_indices = np.cumsum([0] + [len(part) for part in self.sequences])

    @overload
    def __getitem__(self, key: int) -> T: ...
    @overload
    def __getitem__(self, key: slice) -> Sequence[T]: ...

    def __getitem__(self, key: int | slice) -> T | Sequence[T]:
        if isinstance(key, int):
            index = index_key_to_indices(key, len(self))[0]
            if index >= len(self):
                raise IndexError("Index out of range.")

            part_index = (
                np.searchsorted(self._start_indices, index, side="right").item() - 1
            )
            part_key = int(index - self._start_indices[part_index])
            return self.sequences[part_index][part_key]
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return sum(len(part) for part in self.sequences)


class MultiPartReader(MultiSequence[VideoFrame]):
    def __init__(self, paths: list[str] | list[Path]):
        if isinstance(paths, (str, Path)):
            raise TypeError("paths must be a list")

        if len(paths) < 1:
            raise ValueError("paths must not be empty")

        video_readers = [Reader(path) for path in paths]
        self._start_times = np.cumsum(
            [0] + [reader.duration for reader in video_readers]
        )
        super().__init__(video_readers)

    @cached_property
    def times(self) -> TimesArray:
        all_times = []
        for i in range(len(self.sequences)):
            times = cast(Reader, self.sequences[i]).times
            times = times.copy() + self._start_times[i]
            all_times.append(times)
        return np.concatenate(all_times)

    @overload
    def __getitem__(self, key: int) -> VideoFrame: ...
    @overload
    def __getitem__(self, key: slice) -> Sequence[VideoFrame]: ...

    def __getitem__(self, key: int | slice) -> VideoFrame | Sequence[VideoFrame]:
        if isinstance(key, int):
            frame = super().__getitem__(key)
            frame.index = index_key_to_indices(key, len(self))[0]

            part_index = (
                np.searchsorted(self._start_indices, frame.index, side="right").item()
                - 1
            )
            frame.ts = frame.ts + self._start_times[part_index]
            return frame
        else:
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
