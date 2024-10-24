from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from logging import Logger, getLogger
from pathlib import Path
from types import TracebackType
from typing import Sized, SupportsIndex, cast, overload

import av.container
import av.error
import av.packet
import av.video
import numpy as np
import numpy.typing as npt

from pupil_labs.video.frameslice import FrameSlice
from pupil_labs.video.indexer import Indexer
from pupil_labs.video.sequence import ArrayLike
from pupil_labs.video.video_frame import VideoFrame

DEFAULT_LOGGER = getLogger(__name__)


PTSArray = npt.NDArray[np.int64]
TimesArray = npt.NDArray[np.float64]


@dataclass
class Stats:
    """Tracks statistics on containers"""

    seeks: int = 0
    decodes: int = 0


def index_key_to_indices(key: SupportsIndex | slice, obj: Sized) -> tuple[int, int]:
    if isinstance(key, slice):
        start_index, stop_index = key.start, key.stop
    elif isinstance(key, int):
        start_index, stop_index = key, key + 1
        if key < 0:
            start_index = len(obj) + key
            stop_index = start_index + 1
    else:
        raise TypeError(f"key must be int or slice, not {type(key)}")

    if start_index is None:
        start_index = 0
    if start_index < 0:
        start_index = len(obj) + start_index
    if stop_index is None:
        stop_index = len(obj)
    if stop_index < 0:
        stop_index = len(obj) + stop_index

    return start_index, stop_index


class Reader(ArrayLike[VideoFrame]):
    def __init__(
        self,
        source: Path | str,
        logger: Logger | None = DEFAULT_LOGGER,
    ):
        """Reader reads video files providing a frame access api

        Arguments:
        ---------
            source: Path to a video file, local or http://

            logger: a python logger to use, pass in None to increase performance

        """
        self.source = source
        self.logger = logger
        self.decoder_index: int | None = -1
        self.stats = Stats()
        self.is_at_start = True
        self.lazy_frame_slice_limit = 50

        # TODO(dan): this should be based on the stream precision
        # 1ms is ok for video but perhaps not for audio later
        self.time_match_tolerance = 0.001  # seconds

        # TODO(dan): can we avoid it?
        # this forces loading the gopsize on initialization
        assert self.gop_size

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
    def gop_size(self) -> int:
        """Return the amount of frames per keyframe in a video"""
        self._container.seek(0)
        have_seen_keyframe_already = False
        count = 0
        # we use the native av demuxer to avoid logging overheads
        for packet in self._container.demux(self._stream):
            if packet.pts is None:
                continue
            if packet.is_keyframe:
                if have_seen_keyframe_already:
                    break
                have_seen_keyframe_already = True
            count += 1
            if count > 1000:  # sanity check, eye videos usually have 400 frames per gop
                raise RuntimeError("too many packets demuxed trying to find a keyframe")

        self._container.seek(0)
        if self.logger:
            self.logger.info(f"demuxed {count} packets to get gop_size")
        return count

    @cached_property
    def _container(self) -> av.container.input.InputContainer:
        container = av.open(self.source)
        for stream in container.streams.video:
            stream.thread_type = "FRAME"
        return container

    def _seek(self, video_secs: float) -> None:
        """Seek to a time in video seconds in the container"""
        want_start = video_secs == 0
        if want_start and self.is_at_start:
            return

        self.is_at_start = want_start
        seek_time = int(video_secs * av.time_base)
        if self.logger:
            self.logger.warning(f"seeking to {video_secs:.5f}s")
        self._container.seek(seek_time)
        self.stats.seeks += 1
        self.decoder_index = -1 if want_start else None
        self._get_frames_buffer.clear()
        self._decoder_frame_buffer.clear()

    @property
    def _stream(self) -> av.video.stream.VideoStream:
        return self._container.streams.video[0]

    @cached_property
    def times(self) -> TimesArray:
        """Return the presentation timestamps in float seconds from offset"""
        assert self._stream.time_base
        times = np.array(self._pts * float(self._stream.time_base), dtype=np.float64)
        return times

    @cached_property
    def _pts(self) -> PTSArray:
        """Return the presentation timestamps in video.time_base"""
        assert self._stream.time_base
        if self.logger:
            self.logger.warning("demuxing all packets to get pts")

        # we demux straight from the av container here, to avoid our overheads
        self._seek(0)
        count = 0
        pts = []
        for packet in self._container.demux(self._stream):
            if packet.pts is None:
                continue
            count += 1
            pts.append(packet.pts)

        # TODO(dan): can we use .seek() instead?
        self.decoder_index = None
        self.is_at_start = False

        if self.logger:
            self.logger.warning(f"demuxed {count} packets to get pts")
        return np.array(pts)

    def __len__(self) -> int:
        if self._stream.frames is not None:
            return self._stream.frames
        return len(self.times)

    @property
    def _demuxer(self) -> Iterator[av.packet.Packet]:
        """Demuxed packets from the video

        We only use this demuxer for the decoder as it has overheads for logging
        When loading all the pts in a container we use the native av demuxer instead
        """
        logpackets = self.logger.debug if self.logger else None
        stream_time_bases = {
            stream: float(stream.time_base) if stream.time_base is not None else 1
            for stream in self._container.streams
        }
        for packet in self._container.demux(self._stream):
            self.is_at_start = False
            if logpackets:
                packet_time_str = "      "
                if packet.pts is not None:
                    packet_time_str = (
                        f"{packet.pts * stream_time_bases[packet.stream]:.3f}s"
                    )

                logpackets(
                    f"demuxed"
                    f" {packet.stream.type[0]}{packet.is_keyframe and 'k' or ' '}"
                    f" {packet_time_str}"
                    f" packet={packet}"
                )
            yield packet

    @cached_property
    def _decoder_frame_buffer(self) -> deque[av.video.frame.VideoFrame]:
        """Returns a buffer that holds frames from the decoder"""
        return deque()

    @property
    def _decoder(self) -> Iterator[av.video.frame.VideoFrame]:
        """Yields decoded av frames from the video stream

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
        logdecoded = self.logger.debug if self.logger else None
        while self._decoder_frame_buffer:
            # here we yield unconsumed frames from the previously packet decode
            frame = self._decoder_frame_buffer.popleft()
            logdecoded and logdecoded(f"yielding previous packet frame: {frame}")
            yield frame

        for packet in self._demuxer:
            try:
                frames = cast(Iterator[av.video.frame.VideoFrame], packet.decode())
            except av.error.EOFError as e:
                # this shouldn't happen but if it does, handle it
                if self.logger:
                    self.logger.warning(f"reached end of file: {e}")
                break

            # add all the decoded frames to the buffer first
            self._decoder_frame_buffer.extend(frames)

            # if we don't consume it entirely, will happen on next iteration of .decoder
            while self._decoder_frame_buffer:
                frame = self._decoder_frame_buffer.popleft()
                logdecoded and logdecoded(f"yielding current packet frame: {frame}")
                self.stats.decodes += 1
                yield frame

    @cached_property
    def _get_frames_buffer(self) -> deque[VideoFrame]:
        return deque(maxlen=self.gop_size)

    def _get_frames(self, key: SupportsIndex | slice) -> ArrayLike[VideoFrame]:  # noqa: C901
        start_index, stop_index = index_key_to_indices(key, self)
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

        if self.logger:
            self.logger.info(f"get_frames: [{start_index}:{stop_index}]")

        loginfo = self.logger.info if self.logger else None
        logdebug = self.logger.debug if self.logger else None
        logwarning = self.logger.warning if self.logger else None

        start_index, stop_index = index_key_to_indices(key, self)
        loginfo and loginfo(f"get_frames: [{start_index}:{stop_index}]")

        result = list[VideoFrame]()

        # BUFFERED FRAMES LOGIC
        # works out which frames in the current buffer we can use to fulfill the range

        if self._get_frames_buffer:
            loginfo and loginfo(f"buffer: {_summarize_frames(self._get_frames_buffer)}")

            distance = start_index - self._get_frames_buffer[0].index
            buffer_contains_wanted_frames = distance >= 0 and distance <= len(
                self._get_frames_buffer
            )
            if buffer_contains_wanted_frames:
                # TODO(dan): we can be faster here if we just use indices
                for buffered_frame in self._get_frames_buffer:
                    if start_index <= buffered_frame.index < stop_index:
                        result.append(buffered_frame)

            if result:
                if len(result) == stop_index - start_index:
                    logdebug and logdebug(
                        f"returning buffered frames: {_summarize_frames(result)}"
                    )
                    return result
                else:
                    logdebug and logdebug(
                        f"using buffered frames: {_summarize_frames(result)}"
                    )
            else:
                logdebug and logdebug("no buffered frames found")

            start_index = start_index + len(result)

        if isinstance(key, slice):
            resultview = FrameSlice[VideoFrame](self, key)
            if stop_index - start_index < self.lazy_frame_slice_limit:
                # small enough result set, return as is
                return list(resultview)
            return resultview

        # SEEKING LOGIC
        # Most of this complexity here is to avoid making a seek

        # We have two modes of matching frames, only one will be used to match later:
        start_index_distance: int | None = None  # frame within keyframe distance away
        "distance of start_index frame from current index"

        start_index_time: float | None = None  # unknown distance, match on frame.time
        "frame.time of start_index frame"

        if self.decoder_index is not None and self.decoder_index + 1 == start_index:
            start_index_time = (
                0 if not self._get_frames_buffer else self._get_frames_buffer[-1].time
            )
        else:
            if self.decoder_index is not None:
                distance = distance = start_index - self.decoder_index - 1
                assert self._get_frames_buffer.maxlen
                if 0 < distance < self._get_frames_buffer.maxlen:
                    start_index_distance = distance
            if start_index_distance is None:
                start_index_time = float(self.times[start_index])
                self._seek(start_index_time)

        if logdebug and start_index_distance is not None:
            logdebug(
                f"iterating {start_index_distance} frames as within keyframe distance"
            )

        # DECODING LOGIC
        # Iterates over the av frame decoder, buffering the frames that come out
        # and checking them if they match the currently requested range

        # these variables are used to minimize attribute lookups in the hotloop
        pts_attribute = Reader._pts.attrname  # type: ignore
        pts_were_loaded = pts_attribute in self.__dict__

        count = 0
        for av_frame in self._decoder:
            count += 1  # noqa: SIM113

            if not pts_were_loaded and pts_attribute in self.__dict__:
                # something accessed the pts while we were decoding, we have to restart
                logwarning and logwarning("pts were loaded mid decoding")
                return self._get_frames(key)

            # decoder_index can be None if we have arrived here from a seek
            if self.decoder_index is None:
                self.decoder_index = int(np.searchsorted(self.times, av_frame.time))
            else:
                self.decoder_index += 1

            frame = VideoFrame(av_frame, self.decoder_index, av_frame.time)
            logdebug and logdebug(f"  decoded {frame}")

            # we can be iterating frames that are before the requested range since
            # seeks will start at a keyframe, we buffer them as access might come later
            self._get_frames_buffer.append(frame)

            # then match based on if we are in iterate distance mode or match time mode
            use_frame = False
            if start_index_distance is not None:
                use_frame = count > start_index_distance
            elif start_index_time is not None:
                use_frame = (
                    av_frame.time >= start_index_time - self.time_match_tolerance
                )

            # and only return frames that are in the currently requested range
            if use_frame:
                result.append(frame)

            if self.decoder_index >= stop_index - 1:
                break

        logdebug and logdebug(f"returning frames: {_summarize_frames(result)}")
        return result

    @overload
    def __getitem__(self, key: SupportsIndex) -> VideoFrame: ...
    @overload
    def __getitem__(self, key: slice) -> ArrayLike[VideoFrame]: ...

    def __getitem__(
        self, key: SupportsIndex | slice
    ) -> VideoFrame | ArrayLike[VideoFrame]:
        frames = self._get_frames(key)
        if isinstance(key, int):
            if not frames:
                raise IndexError(f"index: {key} not found")
            return frames[0]
        return frames

    @cached_property
    def _by_pts(self) -> Indexer[VideoFrame]:
        return Indexer(self._pts, self)

    @cached_property
    def by_time(self) -> Indexer[VideoFrame]:
        return Indexer(self.times, self)

    def __enter__(self) -> "Reader":
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
        if self._container.duration is None:
            return float(self.times[-1])
        return self._container.duration / av.time_base

    @property
    def width(self) -> int:
        return self._stream.width

    @property
    def height(self) -> int:
        return self._stream.height

    def __iter__(self) -> Iterator[VideoFrame]:
        for i in range(len(self)):
            yield self[i]


def _summarize_frames(result: list[VideoFrame] | deque[VideoFrame]) -> str:
    indices = [frame.index for frame in result]
    if len(indices) > 1:
        return f"{len(indices)} frames from [{indices[0]} to {indices[-1]}]"
    return str(indices)
