from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from logging import Logger, getLogger
from pathlib import Path
from types import TracebackType
from typing import Any, Self, Sized, TypeVar, cast, overload

import av.audio
import av.container
import av.error
import av.packet
import av.stream
import av.video
import numpy as np
import numpy.typing as npt

from pupil_labs.video.array_like import ArrayLike
from pupil_labs.video.frame import AudioFrame, BaseFrame, VideoFrame
from pupil_labs.video.frameslice import FrameSlice
from pupil_labs.video.indexer import Indexer

DEFAULT_LOGGER = getLogger(__name__)


TimesArray = npt.NDArray[np.float64] | list[float]
"Timestamps for frames in video time seconds, eg. [0.033, 0.066, ...]"

AVFrame = av.video.frame.VideoFrame | av.audio.frame.AudioFrame
ReaderFrameType = TypeVar("ReaderFrameType", BaseFrame, VideoFrame, AudioFrame)


@dataclass
class Stats:
    """Tracks statistics on containers"""

    seeks: int = 0
    decodes: int = 0


def index_key_to_indices(key: int | slice, obj: Sized) -> tuple[int, int, int]:
    step = 1
    if isinstance(key, slice):
        start, stop, step = (
            key.start,
            key.stop,
            1 if key.step is None else key.step,
        )
    elif isinstance(key, int):
        start, stop = key, key + 1
        if key < 0:
            start = len(obj) + key
            stop = start + 1
    else:
        raise TypeError(f"key must be int or slice, not {type(key)}")

    if start is None:
        start = 0
    if start < 0:
        start = len(obj) + start
    if stop is None:
        stop = len(obj)
    if stop < 0:
        stop = len(obj) + stop

    return start, stop, step


class StreamReader(ArrayLike[ReaderFrameType]):
    def __init__(
        self,
        source: Path | str,
        times: TimesArray | None = None,
        logger: Logger | None = None,
    ):
        """Allows indexing and iterating over a video file.

        Args:
        ----
            source: Path to a video file. Can be a local path or an http-address.
            times: Timestamps for frames in video time in seconds, eg. `[0.033, 0.066]`.
                If not provided, times will be inferred from the container.
            logger: Python logger to use, decreases performance.

        """
        self.source = source
        self.logger = logger or DEFAULT_LOGGER

        if times is not None:
            self.times = times
        self.lazy_frame_slice_limit = 50
        self.stats = Stats()

        self._log = bool(logger)
        self._is_at_start = True
        self._partial_pts = list[int]()
        self._partial_pts_to_index = dict[int, int]()
        self._all_pts_are_loaded = False
        self._decoder_frame_buffer = deque[AVFrame]()
        self._current_decoder_index: int | None = -1
        self._indexed_frames_buffer = deque[BaseFrame](maxlen=1000)
        # TODO(dan): can we avoid it?
        # this forces loading the gopsize on initialization to set the buffer length
        assert self._gop_size

    def _get_logger(self, prefix: str) -> Any | Logger:
        if self._log:
            return self.logger.getChild(f"{self.__class__.__name__}.{prefix}")
        return False

    @cached_property
    def _container(self) -> av.container.input.InputContainer:
        container = av.open(self.source)
        for stream in container.streams.video:
            stream.thread_type = "FRAME"
        return container

    @cached_property
    def times(self) -> TimesArray:
        """Array of relative timestamps of the stream in seconds of video time.

        If no `times` argument was provided, the values will be inferred from the video
        container.
        """
        return self._container_times

    @cached_property
    def _gop_size(self) -> int:
        """Return the amount of frames per keyframe in a video"""
        logger = self._get_logger(f"{Reader._gop_size.attrname}()")  # type: ignore
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
        self._indexed_frames_buffer = deque[BaseFrame](maxlen=max(60, gop_size))
        self._reset_decoder()
        return gop_size

    @property
    def _stream(self) -> av.video.stream.VideoStream | av.audio.stream.AudioStream:
        return self._container.streams.video[0]

    def _seek_to_pts(self, pts: int) -> bool:
        logger = self._get_logger(f"{Reader._seek_to_pts.__name__}({pts})")
        if self._is_at_start and pts == 0:
            if logger:
                logger.info("skipping seek, already at start")
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

    def _reset_decoder(self) -> None:
        if self._av_frame_decoder:
            del self._av_frame_decoder
        self._current_decoder_index = None
        self._indexed_frames_buffer.clear()
        self._decoder_frame_buffer.clear()

    def _seek_to_index(self, index: int) -> bool:
        logger = self._get_logger(f"{Reader._seek_to_index.__name__}({index})")
        logger and logger.info(f"seeking to index: {index}")
        pts = 0
        # TODO(dan): we can skip a seek if current decoder packet pts matches
        if 0 < index >= len(self._partial_pts):
            logger and logger.warning(f"index {index} not in loaded packets, loading..")
            pts = self._get_pts_till_index(index)
        elif index != 0:
            try:
                pts = self._partial_pts[index]
            except Exception as e:
                raise RuntimeError(
                    f"index not found in packets loaded so far:{index}"
                ) from e
        return self._seek_to_pts(pts)

    @cached_property
    def _pts(self) -> list[int]:
        """Return all presentation timestamps in video.time_base"""
        self._get_pts_till_index(-1)
        assert self._all_pts_are_loaded
        return self._partial_pts

    def _get_pts_till_index(self, index: int) -> int:
        """Load pts up to a specific index"""
        logger = self._get_logger(f"{Reader._get_pts_till_index.__name__}({index})")
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
    def __getitem__(self, key: slice) -> ArrayLike[ReaderFrameType]: ...

    def __getitem__(
        self, key: int | slice
    ) -> ReaderFrameType | ArrayLike[ReaderFrameType]:
        """Index-based access to video frames.

        `reader[5]` returns the fifth frame in the video.
        `reader[5:10]` returns an `ArrayLike` of frames 5 to 10.

        Large slices are returned as a lazy view, which avoids immediately loading all
        frames into RAM.
        """
        frames = self._get_frames(key)
        if isinstance(key, int):
            if not frames:
                raise IndexError(f"index: {key} not found")
            return frames[0]
        return frames

    def _get_frames(self, key: int | slice) -> ArrayLike[ReaderFrameType]:  # noqa: C901
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
        logger = self._get_logger(f"{Reader._get_frames.__name__}({key})")
        log_buffer = logger and logger.debug
        log_frames = logger and logger.debug
        log_other = logger and logger.debug

        start, stop, step = index_key_to_indices(key, self)
        log_other and log_other(f"getting frames: [{start}:{stop}:{step}]")

        result = list[ReaderFrameType]()

        # BUFFERED FRAMES LOGIC
        # works out which frames in the current buffer we can use to fulfill the range
        if self._indexed_frames_buffer:
            log_buffer and log_buffer(
                f"buffer: {_frame_summary(self._indexed_frames_buffer)}"
            )
            if len(self._indexed_frames_buffer) > 1:
                assert (
                    self._indexed_frames_buffer[-1]._stream_index
                    - self._indexed_frames_buffer[0]._stream_index
                    == len(self._indexed_frames_buffer) - 1
                )

            offset = start - self._indexed_frames_buffer[0]._stream_index
            buffer_contains_wanted_frames = offset >= 0 and offset <= len(
                self._indexed_frames_buffer
            )
            if buffer_contains_wanted_frames:
                # TODO(dan): we can be faster here if we just use indices
                for buffered_frame in self._indexed_frames_buffer:
                    if start <= buffered_frame._stream_index < stop:
                        result.append(buffered_frame)  # type: ignore

            if result:
                if len(result) == stop - start:
                    log_buffer and log_buffer(
                        f"returning buffered frames: {_frame_summary(result)}"
                    )
                    return result
                else:
                    log_buffer and log_buffer(
                        f"using buffered frames: {_frame_summary(result)}"
                    )
            else:
                log_buffer and log_buffer("no buffered frames found")

            start = start + len(result)

        if isinstance(key, slice):
            resultview = FrameSlice[ReaderFrameType](self, key)
            if stop - start < self.lazy_frame_slice_limit:
                # small enough result set, return as is
                return list(resultview)
            return resultview

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

        for av_frame in self._av_frame_decoder:
            assert av_frame.pts is not None
            assert av_frame.time is not None

            self._current_decoder_index = self._partial_pts_to_index[av_frame.pts]
            frame_time = av_frame.time
            if "times" in self.__dict__:  # times was provided by user
                frame_time = float(self.times[self._current_decoder_index])

            frame = {
                av.video.frame.VideoFrame: VideoFrame,
                av.audio.frame.AudioFrame: AudioFrame,
            }[type(av_frame)](
                av_frame=av_frame,
                time=frame_time,
                index=self._current_decoder_index,
                _stream_index=self._current_decoder_index,
            )

            log_frames and log_frames(f"    received {frame}")
            self._indexed_frames_buffer.append(frame)
            if self._current_decoder_index >= start:
                result.append(frame)

            if self._current_decoder_index >= stop - 1:
                break

        log_frames and log_frames(f"returning frames: {_frame_summary(result)}")
        return result

    def __len__(self) -> int:
        """Return the number of frames in the video"""
        if self._stream.frames:
            return self._stream.frames
        return len(self._pts)

    def _demux(self) -> Iterator[av.packet.Packet]:
        """Demuxed packets from the stream"""
        logger = self._get_logger(f"{Reader._demux.__name__}()")
        logpackets = logger and logger.debug
        stream_time_base = (
            float(self._stream.time_base) if self._stream.time_base is not None else 1
        )
        prev_packet_pts = None
        for packet in self._container.demux(self._stream):
            is_new_pts = False
            if packet.pts is not None:
                is_new_pts = (self._is_at_start and len(self._partial_pts) < 1) or (
                    len(self._partial_pts) > 0
                    and self._partial_pts[-1] == prev_packet_pts
                    and packet.pts > self._partial_pts[-1]
                )
                if is_new_pts:
                    self._partial_pts.append(packet.pts)
                    self._partial_pts_to_index[packet.pts] = len(self._partial_pts) - 1

            prev_packet_pts = packet.pts
            self._is_at_start = False

            if logpackets:
                index_str = " "
                if packet.pts is not None:
                    index_str = f"{self._partial_pts_to_index[packet.pts]}"
                packet_time_str = "      "
                if packet.pts is not None:
                    packet_time_str = f"{packet.pts * stream_time_base:.3f}s"

                logpackets(
                    f"demuxed"
                    f" {packet.stream.type[0]}{packet.is_keyframe and 'k' or ' '}"
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
        return Indexer(np.array(self._pts), self)

    @cached_property
    def by_time(self) -> Indexer[ReaderFrameType]:
        """Time-based access to video frames using relative video time seconds.

        When accessing a specific key, e.g. `reader[t]`, a frame with this exact time
        needs to exist, otherwise an `IndexError` is raised.
        When acessing a slice, e.g. `reader[a:b]` an `ArrayLike` is returned such
        that ` a <= frame.time < b` for every frame.

        Large slices are returned as a lazy view, which avoids immediately loading all
        frames into RAM.
        """
        return Indexer(self.times, self)

    @cached_property
    def _by_container_time(self) -> Indexer[ReaderFrameType]:
        return Indexer(self._container_times, self)

    @cached_property
    def _container_times(self) -> TimesArray:
        assert self._stream.time_base
        return np.array(self._pts) * float(self._stream.time_base)

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
    def duration(self) -> float:
        """Return the duration of the video in seconds.

        If the duration is not available in the container, it will be calculated based
        on the frames timestamps.
        """
        if self._container.duration is None:
            return float(self.times[-1])
        return self._container.duration / av.time_base

    def __iter__(self) -> Iterator[ReaderFrameType]:
        # we iter like this to avoid calling len
        i = 0
        while True:
            try:
                yield self[i]
            except IndexError:
                break
            i += 1


def _frame_summary(result: list[ReaderFrameType] | deque[ReaderFrameType]) -> str:
    indices = [frame._stream_index for frame in result]
    if len(indices) > 1:
        return f"{len(indices)} frames from [{indices[0]} to {indices[-1]}]"
    return str(indices)


class AudioReader(StreamReader[AudioFrame]):
    @property
    def _stream(self) -> av.audio.stream.AudioStream:
        return self._container.streams.audio[0]


class Reader(StreamReader[VideoFrame]):
    @property
    def _stream(self) -> av.video.stream.VideoStream:
        return self._container.streams.video[0]

    @property
    def audio(self) -> AudioReader | None:
        """Returns an `AudioReader` providing access to the audio data of the video."""
        if not self._container.streams.audio:
            return None
        return AudioReader(self.source, logger=self.logger)

    @property
    def width(self) -> int | None:
        """Width of the video in pixels."""
        assert self._stream.type == "video"
        return self._stream.width

    @property
    def height(self) -> int | None:
        """Height of the video in pixels."""
        assert self._stream.type == "video"
        return self._stream.height
