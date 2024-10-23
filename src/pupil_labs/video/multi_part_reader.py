from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Generic, TypeVar, cast, overload

import numpy as np

from .frameslice import FrameSlice
from .reader import Reader
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
            if key >= len(self):
                raise IndexError("Index out of range.")

            part_index = (
                np.searchsorted(self._start_indices, key, side="right").item() - 1
            )
            part_key = int(key - self._start_indices[part_index])
            return self.sequences[part_index][part_key]
        else:
            raise NotImplementedError

    def _parse_key(self, key: int | slice) -> tuple[int, int]:
        if isinstance(key, slice):
            start_index, stop_index = key.start, key.stop
        elif isinstance(key, int):
            start_index, stop_index = key, key + 1
            if key < 0:
                start_index = len(self) + key
                stop_index = start_index + 1
        else:
            raise TypeError(f"key must be int or slice, not {type(key)}")

        if start_index is None:
            start_index = 0
        if start_index < 0:
            start_index = len(self) + start_index
        if stop_index is None:
            stop_index = len(self)
        if stop_index < 0:
            stop_index = len(self) + stop_index

        return start_index, stop_index

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

    @overload
    def __getitem__(self, key: int) -> VideoFrame: ...
    @overload
    def __getitem__(self, key: slice) -> Sequence[VideoFrame]: ...

    def __getitem__(self, key: int | slice) -> VideoFrame | Sequence[VideoFrame]:
        if isinstance(key, int):
            frame = super().__getitem__(key)
            frame.index = key

            part_index = (
                np.searchsorted(self._start_indices, key, side="right").item() - 1
            )
            frame.ts = frame.ts + self._start_times[part_index]
            return frame
        else:
            return FrameSlice[VideoFrame](self, key)

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
        raise NotImplementedError

    @property
    def width(self) -> int:
        reader = cast(Reader, self.sequences[0])
        return reader.width

    @property
    def height(self) -> int:
        reader = cast(Reader, self.sequences[0])
        return reader.height
