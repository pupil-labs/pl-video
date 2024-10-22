from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import overload

import numpy as np

from .reader import Reader
from .video_frame import VideoFrame


class MultiPartReader(Sequence[VideoFrame]):
    def __init__(self, paths: list[str] | list[Path]):
        if isinstance(paths, str):
            raise TypeError("paths must be a list")

        self.parts = [Reader(path) for path in paths]
        self._start_indices = np.cumsum([0] + [len(part) for part in self.parts])

    def __len__(self) -> int:
        return sum(len(part) for part in self.parts)

    @overload
    def __getitem__(self, key: int) -> VideoFrame: ...
    @overload
    def __getitem__(self, key: slice) -> Sequence[VideoFrame]: ...

    def __getitem__(self, key: int | slice) -> VideoFrame | Sequence[VideoFrame]:
        if isinstance(key, int):
            if key >= len(self):
                raise IndexError("Index out of range.")

            part_index = (
                np.searchsorted(self._start_indices, key, side="right").item() - 1
            )
            part_key = int(key - self._start_indices[part_index])
            frame = self.parts[part_index][part_key]
            frame.index = key
            # TODO(marc): How do we want to set frame.ts and frame.pts?
            return frame
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
        # TODO(marc): Add an appropriate attribute to the Reader class.
        return self.parts[0]._container.streams.video[0].width

    @property
    def height(self) -> int:
        # TODO(marc): Add an appropriate attribute to the Reader class.
        return self.parts[0]._container.streams.video[0].height
