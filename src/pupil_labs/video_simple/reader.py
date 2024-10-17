from dataclasses import dataclass
from functools import cached_property
from logging import Logger, getLogger
from pathlib import Path
from typing import (
    Deque,
    Generator,
    Generic,
    Iterator,
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
import av.subtitles
import av.subtitles.subtitle
import av.video.stream
import numpy as np
import numpy.typing as npt

from pupil_labs.video_simple.video_frame import VideoFrame

AVFrameTypes = (
    av.video.frame.VideoFrame
    | av.audio.frame.AudioFrame
    | av.subtitles.subtitle.SubtitleSet
)

FrameType = TypeVar("FrameType")
PTSArray = npt.NDArray[np.int64]
TimesArray = npt.NDArray[np.float64]
TimestampsArray = npt.NDArray[np.float64 | np.int64] | list[int | float]


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
        keyframe_indices = []
        frame_index = 0
        for packet in av_container.demux(self.av_stream):
            if packet.pts is None:
                continue

            pts.append(packet.pts)
            if packet.is_keyframe:
                keyframe_indices.append(frame_index)
            frame_index += 1
        self.pts = np.array(pts, np.int64)
        self.times = self.pts
        self.keyframe_indices = np.array(keyframe_indices)
        if self.av_stream.time_base is not None:
            self.times = self.pts * float(self.av_stream.time_base)
        av_container.seek(0)

    @property
    def largest_frame_group_size(self) -> int:
        return int(max(np.diff(self.keyframe_indices)))


IndexerValueType = TypeVar("IndexerValueType")
IndexerKeyType = int | float


class Indexer(Generic[IndexerValueType]):
    def __init__(
        self,
        keys: TimestampsArray,
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


class FrameSlice(Generic[FrameType], Sequence[FrameType]):
    def __init__(self, target: Sequence[FrameType], slice: slice):
        self.target = target
        self.slice = slice
        self.start, self.stop, self.step = slice.indices(len(self.target))

    @overload
    def __getitem__(self, key: int) -> FrameType: ...

    @overload
    def __getitem__(self, key: slice) -> Sequence[FrameType]: ...

    def __getitem__(self, key: int | slice) -> FrameType | Sequence[FrameType]:
        if isinstance(key, int):
            if key > len(self) - 1:
                raise IndexError()
            return self.target[key + self.start]
        elif isinstance(key, slice):
            # TODO(dan): implement FrameSlice(self, new_slice)
            raise NotImplementedError()
        else:
            raise TypeError

    def __len__(self) -> int:
        return self.stop - self.start

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f'{self.target})'
            '['
            f"{'' if self.slice.start is None else self.slice.start}"
            ':'
            f"{'' if self.slice.stop is None else self.slice.stop}"
            "]"
        )


class Reader(Sequence[VideoFrame]):
    def __init__(
        self,
        path: str | Path,
        timestamps: TimestampsArray | None = None,
        logger: Logger = getLogger(__name__),
    ):
        self.path = path
        self.logger = logger
        self._timestamps = timestamps
        self._av_decoder_frame_index: int = -1
        self._av_decoder: Iterator[av.video.VideoFrame] | None = None
        self._av_frame_buffer = Deque[VideoFrame](maxlen=100)
        self.stats = ContainerActionCounters()
        self.logger = logger
        self.pts  # TODO(dan): this is accessed to fill pts, we can avoid this

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.av_container})"

    @cached_property
    def av_container(self) -> av.container.InputContainer:
        container = av.open(str(self.path))  # type: ignore
        for stream in container.streams.video:
            stream.thread_type = "FRAME"
        return container

    @cached_property
    def av_video_stream(self) -> av.video.stream.VideoStream:
        return self.av_container.streams.video[0]

    @cached_property
    def video_packets_info(self) -> AVStreamPacketsInfo:
        info = AVStreamPacketsInfo(self.av_video_stream)
        # ensure that the buffer can fit an entire keyframe + subframes worth of frames
        buffer_size = max(
            info.largest_frame_group_size * 2, self._av_frame_buffer.maxlen
        )
        # sanity check
        assert buffer_size < 5000
        self._av_frame_buffer = Deque[VideoFrame](maxlen=buffer_size)
        return info

    @cached_property
    def pts(self) -> PTSArray:
        return self.video_packets_info.pts

    @cached_property
    def times(self) -> TimesArray:
        return self.video_packets_info.times

    def get_index_for_pts(self, pts: int) -> int:
        return int(np.searchsorted(self.pts, pts))

    def seek_container_to_index(self, index: int) -> None:
        wanted_frame_pts = int(self.pts[index])
        if self.logger:
            self.logger.info(f"seeking to {wanted_frame_pts}")
        self.stats.seeks += 1
        self.av_container.seek(wanted_frame_pts, stream=self.av_video_stream)
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
    def __getitem__(self, key: int) -> VideoFrame: ...

    @overload
    def __getitem__(self, key: slice) -> FrameSlice[VideoFrame] | list[VideoFrame]: ...

    def __getitem__(  # noqa: C901
        self, key: int | slice
    ) -> VideoFrame | FrameSlice[VideoFrame] | list[VideoFrame]:
        result = list[VideoFrame]()
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
            result = FrameSlice(self, key)
            if stop_index - start_index < self._av_frame_buffer.maxlen:
                return list(result)
            return result

        # otherwise return the frame, buffering up seen frames for further access
        if start_index != self._av_decoder_frame_index + 1:
            self.seek_container_to_index(start_index)

        if self._av_decoder is None:
            self._av_decoder = self.av_container.decode(self.av_video_stream)

        for av_frame in self._av_decoder:
            self.stats.decodes += 1

            if self._av_decoder_frame_index < 0:
                assert av_frame.pts is not None
                self._av_decoder_frame_index = self.get_index_for_pts(av_frame.pts)
            else:
                self._av_decoder_frame_index += 1

            frame_index = self._av_decoder_frame_index
            frame = VideoFrame(
                av_frame=av_frame,
                index=frame_index,
                ts=self.timestamps[frame_index],
            )
            self._av_frame_buffer.append(frame)
            if self.logger:
                self.logger.info(f"decoded frame: {frame}")

            if self._av_decoder_frame_index == start_index:
                break

        decoded_frame = self._av_frame_buffer[-1]
        if self.logger:
            self.logger.debug(f"returning decoded frame: {decoded_frame}")
        return decoded_frame

    def __len__(self) -> int:
        return len(self.pts)

    def __iter__(
        self,
    ) -> Generator[VideoFrame, None, None]:
        for i in range(len(self)):
            yield self[i]

    @property
    def timestamps(self) -> TimestampsArray:
        return (
            self._timestamps
            if self._timestamps is not None
            else self.video_packets_info.times
        )

    @cached_property
    def by_idx(
        self,
    ) -> "Reader":
        return self

    @cached_property
    def by_pts(self) -> Indexer[VideoFrame]:
        return Indexer(self.pts, self)

    @cached_property
    def by_ts(self) -> Indexer[VideoFrame]:
        return Indexer(self.timestamps, self)
