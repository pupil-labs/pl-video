from dataclasses import dataclass
from functools import cached_property
from logging import Logger, getLogger
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Generator,
    Generic,
    Sequence,
    TypeVar,
    cast,
    overload,
)

import av
import av.audio.stream
import av.container.input
import av.frame
import av.stream
import av.video.stream
import numpy as np
import numpy.typing as npt

AVFrameType = TypeVar("AVFrameType", bound=av.VideoFrame | av.AudioFrame)
FrameType = TypeVar("FrameType")


@dataclass
class ContainerActionCounters:
    seeks: int = 0
    decodes: int = 0
    demuxes: int = 0


class AVStreamPacketsInfo:
    def __init__(
        self, av_stream: av.video.stream.VideoStream | av.audio.stream.AudioStream
    ):
        self.av_stream = av_stream
        av_container = cast(av.container.InputContainer, self.av_stream.container)
        av_container.seek(0)

        pts = []
        for packet in av_container.demux(self.av_stream):
            if packet.pts is None:
                continue

            pts.append(packet.pts)

        self.pts = np.array(pts, np.int64)
        assert self.av_stream.time_base
        self.times = self.pts * float(self.av_stream.time_base)
        av_container.seek(0)


@dataclass
class IndexedFrame(Generic[AVFrameType]):
    av_frame: AVFrameType
    index: int

    def __getattr__(self, key: str):
        return getattr(self.av_frame, key)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            + ", ".join(
                f"{key}={getattr(self, key, '?')}"
                for key in "av_frame index time timestamp".split()
            )
            + ")"
        )


class FrameSlice(Generic[FrameType], Sequence[FrameType]):
    def __init__(self, target: Sequence, slice: slice):
        self.target = target
        self.slice = slice
        self.start, self.stop, self.step = slice.indices(len(self.target))

    def __getitem__(self, key):
        if isinstance(key, int):
            if key > len(self) - 1:
                raise IndexError()
            return self.target[key + self.start]
        elif isinstance(key, slice):
            raise NotImplementedError()  # TODO(dan): implement
            return FrameSlice(self, new_slice)
        else:
            raise TypeError

    def __len__(self):
        return self.stop - self.start

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'{self.target})'
            '['
            f"{'' if self.slice.start is None else self.slice.start}"
            ':'
            f"{'' if self.slice.stop is None else self.slice.stop}"
            "]"
        )


class IndexableAVStream(Generic[AVFrameType], Sequence[IndexedFrame[AVFrameType]]):
    def __init__(
        self,
        stream: av.video.stream.VideoStream,
        frame_wrapper: Callable[[IndexedFrame], Any] = lambda x: x,
        logger: Logger | None = None,
    ):
        self.stream = stream
        self._av_decoder_frame_index: int = -1
        self._av_decoder: Generator[AVFrameType, None, None] | None = None
        self._av_frame_buffer = Deque[IndexedFrame[AVFrameType]](maxlen=100)
        self.frame_wrapper = frame_wrapper
        self.stats = ContainerActionCounters()
        self.logger = logger
        self.pts  # TODO(dan): this is accessed to fill pts, we can avoid this

    def __repr__(self):
        return f"{self.__class__.__name__}({self.av_container})"

    @cached_property
    def av_container(self):
        return cast(av.container.InputContainer, self.stream.container)

    @property
    def timestamps(self):
        if self._timestamps is None:
            return self.times
        return self._timestamps

    @cached_property
    def pts(self):
        return self.packets_info.pts

    @cached_property
    def times(self):
        return self.packets_info.times

    @cached_property
    def packets_info(self):
        return AVStreamPacketsInfo(self.stream)

    def get_index_for_pts(self, pts: int):
        return int(np.searchsorted(self.pts, pts))

    def seek_container_to_index(self, index: int):
        wanted_frame_pts = int(self.pts[index])
        if self.logger:
            self.logger.info(f"seeking to {wanted_frame_pts}")
        self.stats.seeks += 1
        self.av_container.seek(wanted_frame_pts, stream=self.stream)
        self._av_decoder = None
        self._av_decoder_frame_index = -1
        self._av_frame_buffer.clear()

    def _parse_key(self, key: int | slice) -> tuple[int, int]:
        if isinstance(key, slice):
            start_index, stop_index = key.start, key.stop
        elif isinstance(key, int):
            start_index, stop_index = key, key + 1
            if key < 0:
                start_index = len(self.pts) + key
                stop_index = start_index + 1
        else:
            raise TypeError(f"key must be int or slice, not {type(key)}")

        if start_index is None:
            start_index = 0
        if start_index < 0:
            start_index = len(self.pts) + start_index
        if stop_index is None:
            stop_index = len(self.pts)
        if stop_index < 0:
            stop_index = len(self.pts) + stop_index

        return start_index, stop_index

    @overload
    def __getitem__(self, key: int) -> IndexedFrame[AVFrameType]: ...

    @overload
    def __getitem__(
        self, key: slice
    ) -> FrameSlice[IndexedFrame[AVFrameType]] | list[IndexedFrame[AVFrameType]]: ...

    def __getitem__(
        self, key: int | slice
    ) -> (
        IndexedFrame[AVFrameType]
        | FrameSlice[IndexedFrame[AVFrameType]]
        | list[IndexedFrame[AVFrameType]]
    ):
        result = list[IndexedFrame[AVFrameType]]()
        start_index, stop_index = self._parse_key(key)
        if self.logger:
            self.logger.debug(f"getting frames [{start_index}:{stop_index}]")

        if start_index >= stop_index:
            return []

        # return from buffer if available
        buffer_contains_wanted_frames = (
            self._av_frame_buffer
            and self._av_frame_buffer[0].index <= start_index
            and self._av_frame_buffer[-1].index >= stop_index - 1
        )
        if buffer_contains_wanted_frames:
            for frame in self._av_frame_buffer:
                if start_index <= frame.index < stop_index:
                    result.append(frame)
                if frame.index >= stop_index:
                    break

            if isinstance(key, int):
                if self.logger:
                    self.logger.debug(f"returning buffered frame: {result[0]}")
                return result[0]
            if self.logger:
                self.logger.debug(
                    f"returning {len(result)} buffered frames: {result[0]}...{result[-1]}"
                )
            return result

        # if range requested, return a lazy list of the video frames
        if isinstance(key, slice):
            return FrameSlice(self, key)

        # otherwise return the frame, buffering up seen frames for further access
        if start_index != self._av_decoder_frame_index + 1:
            self.seek_container_to_index(start_index)

        if self._av_decoder is None:
            self._av_decoder = self.av_container.decode(self.stream)

        for av_frame in self._av_decoder:
            self.stats.decodes += 1

            if self._av_decoder_frame_index < 0:
                assert av_frame.pts is not None
                self._av_decoder_frame_index = self.get_index_for_pts(av_frame.pts)
            else:
                self._av_decoder_frame_index += 1

            frame = IndexedFrame(
                av_frame=av_frame,
                index=self._av_decoder_frame_index,
            )
            self._av_frame_buffer.append(frame)
            if self.logger:
                self.logger.info(f"decoded frame: {frame}")

            if self._av_decoder_frame_index == start_index:
                break

        decoded_frame = self._av_frame_buffer[-1]
        if self.logger:
            self.logger.debug(f"returning decoded frame: {decoded_frame}")
        return self.frame_wrapper(decoded_frame)

    def __len__(self):
        return len(self.pts)

    def __iter__(self) -> Generator[IndexedFrame[AVFrameType], None, None]:
        for i in range(len(self)):
            yield self[i]


@dataclass
class ReaderFrame:
    frame: IndexedFrame
    timestamp: int | float

    @property
    def ts(self):
        return self.timestamp

    def __getattr__(self, key: str):
        return getattr(self.frame, key)


class Reader:
    def __init__(
        self,
        path: str | Path,
        timestamps: npt.NDArray[np.int64 | np.float32] | None = None,
        logger: Logger = getLogger(__name__),
    ):
        self.path = path
        self.logger = logger
        self._timestamps = timestamps

    def wrap_frame(self, frame: IndexedFrame):
        return ReaderFrame(frame, self.timestamps[frame.index])

    @cached_property
    def video_stream(self):
        container = av.open(self.path)
        for stream in container.streams.video:
            stream.thread_type = "FRAME"
        return IndexableAVStream(
            container.streams.video[0],
            logger=self.logger,
            frame_wrapper=self.wrap_frame,
        )

    @cached_property
    def pts(self):
        return self.video_stream.pts

    @property
    def timestamps(self):
        return (
            self._timestamps
            if self._timestamps is not None
            else self.video_stream.times
        )

    @cached_property
    def by_idx(self):
        return self.video_stream

    @cached_property
    def by_pts(self):
        return Indexer(self.pts, self.video_stream)

    @cached_property
    def by_ts(self):
        return Indexer(self.timestamps, self.video_stream)

    def __getitem__(self, key: int | slice):
        return self.video_stream[key]

    def __len__(self):
        return len(self.video_stream)

    @property
    def stats(self):
        return self.video_stream.stats


IndexerValueType = TypeVar("IndexerValueType")
IndexerKeyType = int | float


class Indexer(Generic[IndexerValueType]):
    def __init__(
        self,
        keys: list[IndexerKeyType],
        values: Sequence[IndexerValueType],
    ):
        self.values = values
        self.keys = np.array(keys)

    @overload
    def __getitem__(self, key: IndexerKeyType) -> IndexerValueType: ...

    @overload
    def __getitem__(self, key: slice) -> list[IndexerValueType]: ...

    def __getitem__(
        self, key: IndexerKeyType | slice
    ) -> IndexerValueType | Sequence[IndexerValueType]:
        if isinstance(key, int | float):
            index = np.searchsorted(self.keys, [key])
            if self.keys[index] != key:
                raise IndexError()
            return self.values[int(index)]
        elif isinstance(key, slice):
            start_index, stop_index = np.searchsorted(self.keys, [key.start, key.stop])
            return self.values[start_index:stop_index]
        else:
            raise TypeError(f"key must be int or slice, not {type(key)}")
