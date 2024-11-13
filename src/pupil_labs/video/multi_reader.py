from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import TracebackType
from typing import Generic, Iterator, Sequence, overload

import numpy as np

from pupil_labs.video.constants import LAZY_FRAME_SLICE_LIMIT
from pupil_labs.video.frame import AudioFrame, ReaderFrameType, VideoFrame
from pupil_labs.video.frame_slice import FrameSlice
from pupil_labs.video.indexing import Indexer, index_key_to_absolute_indices
from pupil_labs.video.reader import Reader, Timestamps

ReaderLike = str | Path | Reader[ReaderFrameType]


@dataclass
class MultiReaderSlice:
    index: int
    offset: int
    slice: slice


class MultiReader(Generic[ReaderFrameType]):
    def __init__(self, *readers: ReaderLike | Sequence[ReaderLike]) -> None:
        parsed_readers: list[Reader] = []
        for arg in readers:
            args_to_check: list[ReaderLike] = []
            if isinstance(arg, Sequence):
                args_to_check.extend(arg)
            else:
                args_to_check.append(arg)

            for reader in args_to_check:
                if isinstance(reader, str | Path):
                    reader = Reader(reader)

                if isinstance(reader, Reader):
                    parsed_readers.append(reader)
                else:
                    raise TypeError(f"invalid reader argument: {arg}")

        if not parsed_readers:
            raise ValueError("at least one reader required")

        self.readers: Sequence[Reader[ReaderFrameType]] = parsed_readers
        self.lazy_frame_slice_limit = LAZY_FRAME_SLICE_LIMIT

    @cached_property
    def timestamps(self) -> Timestamps:
        all_times = []
        for reader_start_time, reader in zip(self.reader_start_times, self.readers):
            timestamps = np.array(reader.timestamps) + reader_start_time
            all_times.append(timestamps)
        return np.concatenate(all_times, dtype=np.float64)

    def __len__(self) -> int:
        return sum(len(reader) for reader in self.readers)

    @overload
    def __getitem__(self, key: int) -> ReaderFrameType: ...
    @overload
    def __getitem__(
        self, key: slice
    ) -> FrameSlice[ReaderFrameType] | list[ReaderFrameType]: ...

    def __getitem__(
        self, key: int | slice
    ) -> ReaderFrameType | FrameSlice[ReaderFrameType] | list[ReaderFrameType]:
        if isinstance(key, slice):
            frameslice = FrameSlice[ReaderFrameType](
                self, key, lazy_frame_slice_limit=self.lazy_frame_slice_limit
            )
            if len(frameslice) < self.lazy_frame_slice_limit:
                return list(frameslice)
            return frameslice

        try:
            key = int(key)
        except Exception as e:
            raise TypeError(f"key must be int or slice, not {type(key)}") from e

        try:
            reader_slice = next(self.reader_slices_for_key(key))
        except StopIteration as e:
            raise IndexError(f"{key} not found") from e

        reader_index = reader_slice.index
        reader = self.readers[reader_index]
        frame: ReaderFrameType = reader[reader_slice.slice.start]
        frame_index = frame.index + reader_slice.offset

        # if not reader._times_were_provided and reader_index > 0:
        #     frame_time = frame.time + self.reader_start_times[reader_index]
        frame_time = frame.time + self.reader_start_times[reader_index]

        # frame.av_frame.pts = int(frame_time / frame.av_frame.time_base)
        output_frame: ReaderFrameType = {
            VideoFrame: VideoFrame,
            AudioFrame: AudioFrame,
        }[type(frame)](
            av_frame=frame.av_frame,
            time=frame_time,
            index=frame_index,
            source=frame,
        )
        return output_frame

    @cached_property
    def reader_start_times(self) -> Timestamps:
        return np.cumsum([[0] + [reader.duration for reader in self.readers]])

    def reader_slices_for_key(self, key: int | slice) -> Iterator[MultiReaderSlice]:
        start, stop, _ = index_key_to_absolute_indices(key, self)  # TODO: handle step
        offset = 0
        for reader_index, reader in enumerate(self.readers):
            array_size = len(reader)
            start_index_in_array = max(0, start - offset)
            stop_index_in_array = min(stop - offset, array_size)
            if start_index_in_array < array_size and stop_index_in_array > 0:
                yield MultiReaderSlice(
                    reader_index,
                    offset,
                    slice(start_index_in_array, stop_index_in_array),
                )
            offset += array_size
            if offset > stop:
                break

    @cached_property
    def by_time(self) -> Indexer[ReaderFrameType]:
        return Indexer(self.timestamps, self)

    def __enter__(self) -> "MultiReader":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        for reader in self.readers:
            reader.close()

    def __iter__(self) -> Iterator[ReaderFrameType]:
        i = 0
        while True:
            try:
                yield self[i]
            except IndexError:
                break
            i += 1

    @cached_property
    def audio(self) -> "MultiReader[AudioFrame]":
        return MultiReader(*[reader.audio for reader in self.readers])

    @property
    def width(self) -> int | None:
        return self.readers[0].width

    @property
    def height(self) -> int | None:
        return self.readers[0].height
