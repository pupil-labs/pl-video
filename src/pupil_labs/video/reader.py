import sys
from bisect import bisect_right
from collections import deque
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from logging import Logger, LoggerAdapter, getLogger
from pathlib import Path
from sys import maxsize
from types import TracebackType
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    cast,
    overload,
)

import av.audio
import av.container
import av.error
import av.packet
import av.stream
import av.video
import numpy as np
import numpy.typing as npt
from upath import UPath

from pupil_labs.video.constants import LAZY_FRAME_SLICE_LIMIT
from pupil_labs.video.frame import AudioFrame, ReaderFrameType, VideoFrame
from pupil_labs.video.frame_slice import FrameSlice
from pupil_labs.video.indexing import Indexer, index_key_to_absolute_indices

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


DEFAULT_LOGGER = getLogger(__name__)
AVFrame = av.video.frame.VideoFrame | av.audio.frame.AudioFrame

ContainerTimestamps = npt.NDArray[np.float64]
"Numpy array of PTS for frames in video"


class ReaderException(Exception): ...


class StreamNotFound(ReaderException): ...


class StreamNotSupported(ReaderException): ...


@dataclass
class Stats:
    """Tracks statistics on containers"""

    seeks: int = 0
    decodes: int = 0


class PrefixingLoggerAdapter(LoggerAdapter):
    def __init__(self, prefix: str, logger: Logger):
        super().__init__(logger)
        self.prefix = prefix

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        return f"[{self.prefix}] {msg}", kwargs


class Reader(Generic[ReaderFrameType]):
    @overload
    def __init__(
        self: "Reader[VideoFrame]",
        source: Path | str,
        stream: Literal["video"] = "video",
        container_timestamps: Optional[ContainerTimestamps | list[float]] | None = None,
        logger: Logger | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "Reader[AudioFrame]",
        source: Path | str,
        stream: Literal["audio"] = "audio",
        container_timestamps: Optional[ContainerTimestamps | list[float]] | None = None,
        logger: Logger | None = None,
    ) -> None: ...

    def __init__(
        self,
        source: Path | str,
        stream: Literal["audio", "video"]
        | tuple[Literal["audio", "video"], int] = "video",
        container_timestamps: Optional[ContainerTimestamps | list[float]] | None = None,
        logger: Optional[Logger] = None,
    ):
        """Create a reader for a video file.

        Args:
            source: Path to a video file. Can be a local path or an http-address.
            stream: The stream to read from, either "audio", "video". If the video file
                contains multiple streams of the deisred kind, a tuple can be provided
                to specify which stream to use, e.g. `("audio", 2)` to use the audio
                stream at index `2`.
            container_timestamps: Array containing the timestamps of the video frames in
                container time (equal to PTS * time_base). If not provided, timestamps
                will be inferred from the container. Providing pre-loaded values can
                speed up initialization for long videos by avoiding demuxing of the
                entire video to obtain PTS.
            logger: Python logger to use. Decreases performance.

        """
        self._container_timestamps: ContainerTimestamps | None = None
        if container_timestamps is not None:
            if isinstance(container_timestamps, list):
                container_timestamps = np.array(container_timestamps)
            self.container_timestamps = container_timestamps

        self.lazy_frame_slice_limit = LAZY_FRAME_SLICE_LIMIT
        self._times_were_provided = container_timestamps is not None
        self._source = source
        self._logger = logger or DEFAULT_LOGGER
        self.stats = Stats()

        if not isinstance(stream, tuple):
            stream = (stream, 0)
        self._stream_kind, self._stream_index = stream

        self._log = bool(logger)
        self._is_at_start = True
        self._last_processed_dts = -maxsize
        self._partial_pts = list[int]()
        self._partial_dts = list[int]()
        self._partial_pts_to_index = dict[int, int]()
        self._all_pts_are_loaded = False
        self._decoder_frame_buffer = deque[AVFrame]()
        self._current_decoder_index: int | None = -1
        self._indexed_frames_buffer: deque[ReaderFrameType] = deque(maxlen=1000)
        # TODO(dan): can we avoid it?
        # this forces loading the gopsize on initialization to set the buffer length
        assert self.gop_size

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def source(self) -> Any:
        """Return the source of the video"""
        return self._source

    @property
    def filename(self) -> str:
        """Return the filename of the video"""
        return str(self.source).split("/")[-1]

    def _get_logger(self, prefix: str) -> Any | Logger:
        if self._log:
            return PrefixingLoggerAdapter(
                f"{self.filename}",
                self.logger.getChild(f"{self.__class__.__name__}.{prefix}"),
            )
        return False

    @cached_property
    def _container(self) -> av.container.input.InputContainer:
        url = self.source
        if isinstance(self.source, UPath):
            try:
                url = self.source.fs.sign(self.source)
            except NotImplementedError:
                url = str(url)
        container = av.open(url)  # type: av.container.input.InputContainer
        for stream in container.streams.video:
            stream.thread_type = "FRAME"
        return container

    @property
    def container_timestamps(self) -> ContainerTimestamps:
        """Frame timestamps in container time.

        Container time is measured in seconds relative to begining of the video.
        Accordingly, the first frame typically has timestamp `0.0`.

        If these values were not provided when creating the Reader, they will be
        inferred from the video container.
        """
        if self._container_timestamps is None:
            return self._inferred_container_timestamps
        return self._container_timestamps

    @container_timestamps.setter
    def container_timestamps(
        self, video_time: ContainerTimestamps | list[float]
    ) -> None:
        self._times_were_provided = True
        if isinstance(video_time, list):
            video_time = np.array(video_time)
        self._container_timestamps = video_time

    @container_timestamps.deleter
    def container_timestamps(self) -> None:
        self._times_were_provided = False

    @cached_property
    def gop_size(self) -> int:
        """Return the amount of frames per keyframe in a video"""
        logger = self._get_logger(f"{Reader.gop_size.attrname}()")  # type: ignore
        logger and logger.info("loading gop_size")
        have_seen_keyframe_already = False
        self._seek_to_pts(0)
        count = 0
        for packet in self._demux():
            if packet.is_keyframe:
                if have_seen_keyframe_already:
                    break
                have_seen_keyframe_already = True
            count += 1
            if count > 1000:  # sanity check, eye videos usually have 400 frames per gop
                raise RuntimeError("too many packets demuxed trying to find a keyframe")

        gop_size = count or 1
        logger and logger.info(f"read {count} packets to get gop_size: {gop_size}")
        self._indexed_frames_buffer = deque(maxlen=max(60, gop_size))
        self._reset_decoder()
        return gop_size

    @cached_property
    def _stream(
        self,
    ) -> av.video.stream.VideoStream | av.audio.stream.AudioStream:
        try:
            if self._stream_kind == "audio":
                return self._container.streams.audio[self._stream_index]
            elif self._stream_kind == "video":
                return self._container.streams.video[self._stream_index]
            else:
                raise StreamNotSupported(
                    "{self._stream_kind} streams are not supported"
                )
        except IndexError as e:
            raise StreamNotFound(
                f"stream: {self._stream_kind}:{self._stream_index} not found"
            ) from e

    def _reset_decoder(self) -> None:
        if self._av_frame_decoder:
            del self._av_frame_decoder
        self._current_decoder_index = None
        self._indexed_frames_buffer.clear()
        self._decoder_frame_buffer.clear()

    def _seek_to_pts(self, pts: int) -> bool:
        logger = self._get_logger(f"{Reader._seek_to_pts.__name__}({pts})")
        if self._is_at_start and pts == 0:
            logger and logger.info("skipping seek, already at start")
            return False

        self._container.seek(pts, stream=self._stream)
        self.stats.seeks += 1
        logger and logger.warning(
            "seeked to: "
            + ", ".join([
                f"index={self._partial_pts_to_index[pts]}",
                f"pts={pts}",
                f"{self.stats}",
            ])
        )

        self._reset_decoder()
        if pts == 0:
            self._is_at_start = True
            self._current_decoder_index = -1

        return True

    def _seek_to_time(self, time: float) -> bool:
        logger = self._get_logger(f"{Reader._seek_to_index.__name__}({time})")
        logger and logger.info(f"seeking to time: {time}")
        if self._times_were_provided:
            index = max(bisect_right(self.container_timestamps, time) - 1, 0)
            closest_timestamp = self.container_timestamps[index]
            closest_from_zero = closest_timestamp - self.container_timestamps[0]
            time = self._duration_scale_factor * closest_from_zero
        self._container.seek(int(time * av.time_base))
        self._reset_decoder()
        return True

    def _seek_to_index(self, index: int) -> bool:
        logger = self._get_logger(f"{Reader._seek_to_index.__name__}({index})")
        logger and logger.info(f"seeking to index: {index}")
        pts = 0
        # TODO(dan): we can skip a seek if current decoder packet pts matches
        if 0 < index >= len(self._partial_pts):
            logger and logger.debug(f"index {index} not in loaded packets, loading..")
            pts = self._load_packets_till_index(index)
        elif index != 0:
            try:
                pts = self._partial_pts[index]
            except Exception as e:
                raise RuntimeError(
                    f"index not found in packets loaded so far:{index}"
                ) from e
        return self._seek_to_pts(pts)

    @cached_property
    def pts(self) -> list[int]:
        """Return all presentation timestamps in `video.time_base`"""
        self._load_packets_till_index(-1)
        assert self._all_pts_are_loaded
        return self._partial_pts

    def _load_packets_till_index(self, index: int) -> int:
        """Load pts up to a specific index"""
        logger = self._get_logger(
            f"{Reader._load_packets_till_index.__name__}({index})"
        )
        logger and logger.warning(
            f"getting packets till index:{index}"
            f", current max index: {len(self._partial_pts) - 1}"
        )
        assert index >= -1
        if index != -1 and index < len(self._partial_pts):
            pts = self._partial_pts[index]
            logger and logger.warning(f"found:{pts}")
            return pts

        if index == -1 or index >= len(self._partial_pts):
            last_pts = self._partial_pts[-1] if self._partial_pts else 0
            self._seek_to_pts(last_pts)
            for packet in self._demux():
                if packet.pts is None:
                    continue
                packet_index = self._partial_pts_to_index[packet.pts]
                if index != -1 and packet_index == index:
                    break
            if logger:
                logger.info(f"current max packet index: {len(self._partial_pts)}")
        return self._partial_pts[index]

    @overload
    def __getitem__(self, key: int) -> ReaderFrameType: ...
    @overload
    def __getitem__(
        self, key: slice
    ) -> FrameSlice[ReaderFrameType] | list[ReaderFrameType]: ...

    def __getitem__(
        self, key: int | slice
    ) -> ReaderFrameType | FrameSlice[ReaderFrameType] | list[ReaderFrameType]:
        """Index-based access to video frames.

        `reader[5]` returns the fifth frame in the video.
        `reader[5:10]` returns an `ArrayLike` of frames 5 to 10.

        Large slices are returned as a lazy view, which avoids immediately loading all
        frames into RAM.
        """
        frames = self._get_frames_by_indices(key)
        if isinstance(key, slice):
            return frames
        if not frames:
            raise IndexError(f"index: {key} not found")
        return frames[0]

    def _get_frames_by_indices(  # noqa: C901
        self, key: int | slice
    ) -> FrameSlice[ReaderFrameType] | list[ReaderFrameType]:
        """Return frames for an index or slice

        This returns a sequence of frames at a particular index or slice in the video

        - returns a view/lazy slice for results longer than self.lazy_frame_slice_limit
        - avoids seeking/demuxing entire video if possible, eg. iterating from start.
        - buffers decoded frames to avoid seeking / decoding when getting repeat frames
        - buffers frames after a keyframe to avoid seeking/decoding iterating backwards

        """
        # NOTE(dan): this is a function that will be called many times during iteration
        # a lot of the choices made here are in the interest of performance
        # eg.
        #   - avoid method calls unless necessary
        #   - minimize long.nested.attribute.accesses
        #   - avoid formatting log messages unless logging needed
        logger = self._get_logger(f"{Reader._get_frames_by_indices.__name__}({key})")
        log_buffer = logger and logger.debug
        log_frames = logger and logger.debug
        log_other = logger and logger.debug

        start, stop, step = index_key_to_absolute_indices(key, self)
        log_other and log_other(f"getting frames: [{start}:{stop}:{step}]")

        result = list[ReaderFrameType]()

        # BUFFERED FRAMES LOGIC
        # works out which frames in the current buffer we can use to fulfill the range
        if self._indexed_frames_buffer:
            log_buffer and log_buffer(
                f"buffer: {self._frame_summary(self._indexed_frames_buffer)}"
            )
            if len(self._indexed_frames_buffer) > 1:
                assert self._indexed_frames_buffer[-1].index is not None
                assert self._indexed_frames_buffer[0].index is not None

                assert (
                    self._indexed_frames_buffer[-1].index
                    - self._indexed_frames_buffer[0].index
                    == len(self._indexed_frames_buffer) - 1
                )

            offset = start - self._indexed_frames_buffer[0].index
            buffer_contains_wanted_frames = offset >= 0 and offset <= len(
                self._indexed_frames_buffer
            )
            if buffer_contains_wanted_frames:
                # TODO(dan): we can be faster here if we just use indices
                for buffered_frame in self._indexed_frames_buffer:
                    if start <= buffered_frame.index < stop:
                        result.append(buffered_frame)

            if result:
                if len(result) == stop - start:
                    log_buffer and log_buffer(
                        f"returning buffered frames: {self._frame_summary(result)}"
                    )
                    return result
                else:
                    log_buffer and log_buffer(
                        f"using buffered frames: {self._frame_summary(result)}"
                    )
            else:
                log_buffer and log_buffer("no buffered frames found")

            start = start + len(result)

        if isinstance(key, slice):
            resultview = FrameSlice[ReaderFrameType](
                self, key, lazy_frame_slice_limit=self.lazy_frame_slice_limit
            )
            if len(resultview) < self.lazy_frame_slice_limit:
                # small enough result set, return as is
                return list(resultview)
            return resultview

        try:
            key = int(key)
        except Exception as e:
            raise TypeError(f"key must be int or slice, not {type(key)}") from e

        # SEEKING LOGIC
        # Walk to the next frame if it's close enough, otherwise trigger a seek
        need_seek = True
        if self._current_decoder_index is not None:
            distance = start - self._current_decoder_index - 1
            assert self._indexed_frames_buffer.maxlen
            if -1 <= distance < self._indexed_frames_buffer.maxlen:
                need_seek = False
                log_other and log_other(f"distance to frame: {distance}, skipping seek")
            else:
                log_other and log_other(f"distance to frame: {distance}, need seek")

        if need_seek:
            self._seek_to_index(start)

        # DECODING LOGIC
        # Iterates over the av frame decoder, buffering the frames that come out
        # and checking them if they match the currently requested range
        logger and logger.debug("decoding")
        for frame in self._frame_generator():
            log_frames and log_frames(f"    received {frame}")
            self._indexed_frames_buffer.append(frame)
            assert self._current_decoder_index is not None

            if self._current_decoder_index >= start:
                result.append(frame)

            if self._current_decoder_index >= stop - 1:
                break

        log_frames and log_frames(f"returning frames: {self._frame_summary(result)}")
        return result

    def _frame_generator(self) -> Iterator[ReaderFrameType]:
        for av_frame in self._av_frame_decoder:
            assert av_frame.pts is not None
            assert av_frame.time is not None

            if av_frame.pts in self._partial_pts_to_index:
                self._current_decoder_index = self._partial_pts_to_index[av_frame.pts]

            frame_timestamp = av_frame.time
            if self._times_were_provided:
                if self._current_decoder_index is not None:
                    frame_timestamp = self.container_timestamps[
                        self._current_decoder_index
                    ]
                else:
                    # we're here from a seek
                    scaled_time = (
                        av_frame.time / self._duration_scale_factor
                        + self.container_timestamps[0]
                    )
                    frame_index = np.searchsorted(
                        self.container_timestamps, scaled_time, side="right"
                    )
                    frame_timestamp = self.container_timestamps[frame_index]
                    self._current_decoder_index = int(frame_index)

            if isinstance(av_frame, av.video.frame.VideoFrame):
                assert self._current_decoder_index is not None
                yield cast(
                    ReaderFrameType,
                    VideoFrame(
                        av_frame=av_frame,
                        time=frame_timestamp,
                        index=self._current_decoder_index,
                        source=self,
                    ),
                )
            elif isinstance(av_frame, av.audio.frame.AudioFrame):
                assert self._current_decoder_index is not None
                yield cast(
                    ReaderFrameType,
                    AudioFrame(
                        av_frame=av_frame,
                        time=frame_timestamp,
                        index=self._current_decoder_index,
                        source=self,
                    ),
                )
            else:
                self.logger.warning(f"unknown frame type found:{av_frame}")

    def __len__(self) -> int:
        """Return the number of frames in the video"""
        if self._stream.frames:
            return self._stream.frames
        return len(self.pts)

    def _demux(self) -> Iterator[av.packet.Packet]:
        """Demuxed packets from the stream"""
        logger = self._get_logger(f"{Reader._demux.__name__}()")
        logpackets = logger and logger.debug
        stream_time_base = (
            float(self._stream.time_base) if self._stream.time_base is not None else 1
        )
        prev_packet_dts = None
        for packet in self._container.demux(self._stream):
            is_new_dts = False
            if packet.dts is not None and packet.pts is not None:
                is_new_dts = (self._is_at_start and not self._partial_dts) or (
                    len(self._partial_dts) > 0
                    and self._partial_dts[-1] == prev_packet_dts
                    and packet.dts > self._partial_dts[-1]
                )
                if is_new_dts:
                    self._partial_pts.append(packet.pts)
                    self._partial_dts.append(packet.dts)
                    self._partial_pts_to_index[packet.pts] = len(self._partial_dts) - 1

            prev_packet_dts = packet.dts
            self._is_at_start = False

            if logpackets:
                index_str = " "
                if packet.pts is not None:
                    index_str = f"{self._partial_pts_to_index.get(packet.pts, '?')}"
                packet_time_str = "      "
                if packet.pts is not None:
                    packet_time_str = f"{packet.pts * stream_time_base:.3f}s"

                logpackets(
                    f"demuxed"
                    f" {packet.stream.type[0]}{(packet.is_keyframe and 'k') or ' '}"
                    f" {packet_time_str}"
                    f" index={index_str}"
                    f" pts={packet.pts}"
                    f" dts={packet.dts}"
                )
            yield packet
        self._all_pts_are_loaded = True

    @cached_property
    def _av_frame_decoder(self) -> Iterator[AVFrame]:
        """Yields decoded av frames from the stream

        This wraps the multithreaded av decoder in order to workaround the way pyav
        returns packets/frames in that case; it delays returning the first frame
        and yields multiple frames for the last decoded packet, which means:

        - the decoded frame does not match the demuxed packet per iteration
        - we would run into EOFError on the last few frames as demuxer has reached end

        This is how the packets/frames look like coming out of av demux/decode:

            packet.pts  packet   decoded           note
            0           0        []                no frames
            450         1
                        ...
                        14       []                no frames
            6761        15       [0]               first frame received
            7211        16       [1]               second frame received
                        ...
            None        30       [14, 15 ... 29]   rest of the frames

        So in this generator we buffer every frame that was decoded and then on the
        next iteration yield those buffered frames first. This ends up in a stream that
        avoids the second issue.
        """
        logger = self._get_logger(f"{Reader._av_frame_decoder.attrname}()")  # type: ignore
        log_decoded = logger and logger.debug

        while self._decoder_frame_buffer:
            # here we yield unconsumed frames from the previously packet decode
            frame = self._decoder_frame_buffer.popleft()
            log_decoded and log_decoded(f"  yielding previous packet frame: {frame}")
            yield frame

        for packet in self._demux():
            try:
                frames = cast(list[AVFrame], packet.decode())
            except av.error.EOFError as e:
                # this shouldn't happen but if it does, handle it
                if self.logger:
                    self.logger.warning(f"reached end of file: {e}")
                break
            else:
                log_decoded and log_decoded(f"  decoded packet frames: {frames}")
                self.stats.decodes += len(frames)

            # add all the decoded frames to the buffer first
            self._decoder_frame_buffer.extend(frames)

            # if we don't consume it entirely, will happen on next iteration of .decoder
            while self._decoder_frame_buffer:
                frame = self._decoder_frame_buffer.popleft()
                log_decoded and log_decoded(f"  yielding current packet frame: {frame}")
                yield frame

    def __next__(self) -> ReaderFrameType:
        return next(self._frame_generator())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            + ", ".join(
                f"{key}={value}"
                for key, value in [
                    ("source", self.source),
                    ("stats", self.stats),
                ]
            )
            + ")"
        )

    @cached_property
    def _by_pts(self) -> Indexer[ReaderFrameType]:
        return Indexer(np.array(self.pts), self)

    @cached_property
    def by_container_timestamps(self) -> Indexer[ReaderFrameType]:
        """Time-based access to video frames using container timestamps.

        Container time is measured in seconds relative to begining of the video.
        Accordingly, the first frame typically has timestamp `0.0`.

        When accessing a specific key, e.g. `reader[t]`, a frame with this exact
        timestamp needs to exist, otherwise an `IndexError` is raised.
        When acessing a slice, e.g. `reader[a:b]` an `ArrayLike` is returned such
        that ` a <= frame.time < b` for every frame.

        Large slices are returned as a lazy view, which avoids immediately loading all
        frames into RAM.

        Note that numerical imprecisions of float numbers can lead to issues when
        accessing individual frames by their container timestamp. It is recommended to
        prefer indexing frames via slices.
        """
        return Indexer(self.container_timestamps, self)

    @cached_property
    def _inferred_container_timestamps(self) -> ContainerTimestamps:
        assert self._stream.time_base
        return np.array(self.pts) * float(self._stream.time_base)

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
        self._container.close()

    @property
    def average_rate(self) -> float:
        """Return the average framerate of the video in Hz."""
        if self._times_were_provided or not self._stream.average_rate:
            return float(1 / np.mean(np.diff(self.container_timestamps)))
        return float(self._stream.average_rate)

    @cached_property
    def _duration_scale_factor(self) -> float:
        if not self._times_were_provided:
            return 1
        return self._duration_from_stream / self.duration

    @cached_property
    def _duration_from_stream(self) -> float:
        """Duration in seconds using container timestamps"""
        if not self._stream.duration:
            return float(self._inferred_container_timestamps[-1])
        assert self._stream.time_base is not None
        return float(self._stream.duration * self._stream.time_base)

    @property
    def duration(self) -> float:
        """Return the duration of the video in seconds.

        If the duration is not available in the container, it will be calculated based
        on the frames timestamps.
        """
        if self._times_were_provided or not self._stream.duration:
            last_frame_duration = 1.0 / self.average_rate
            return (
                float(self.container_timestamps[-1] - self.container_timestamps[0])
                + last_frame_duration
            )
        return self._duration_from_stream

    @property
    def width(self) -> int | None:
        """Width of the video in pixels."""
        if self._stream.type == "video":
            return self._stream.width
        return None

    @property
    def height(self) -> int | None:
        """Height of the video in pixels."""
        if self._stream.type == "video":
            return self._stream.height
        return None

    def __iter__(self) -> Iterator[ReaderFrameType]:
        # we iter like this to avoid calling len
        i = 0
        while True:
            try:
                yield self[i]
            except IndexError:
                break
            i += 1

    @cached_property
    def audio(self) -> "Reader[AudioFrame] | None":
        """Returns an `Reader` providing access to the audio data of the video only."""
        if not self._container.streams.audio:
            return None
        if (self._stream_kind, self._stream_index) == ("audio", 0):
            return cast(Reader[AudioFrame], self)
        return Reader(
            self.source, logger=self.logger if self._log else None, stream="audio"
        )

    @cached_property
    def video(self) -> "Reader[VideoFrame] | None":
        """Returns an `Reader` providing access to the video data of the video only."""
        if not self._container.streams.video:
            return None
        if (self._stream_kind, self._stream_index) == ("video", 0):
            return cast(Reader[VideoFrame], self)
        return Reader(
            self.source, logger=self.logger if self._log else None, stream="video"
        )

    def _frame_summary(
        self, result: list[ReaderFrameType] | deque[ReaderFrameType]
    ) -> str:
        indices = [frame.index for frame in result]
        if len(indices) > 1:
            return f"{len(indices)} frames from [{indices[0]} to {indices[-1]}]"
        return str(indices)

    @property
    def rate(self) -> Fraction | int | None:
        """Return the framerate of the video in Hz."""
        try:
            return self._stream.rate
        except AttributeError:
            return None
