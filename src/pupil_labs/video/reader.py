from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from logging import Logger, getLogger
from pathlib import Path
from types import TracebackType
from typing import cast, overload

import av.container
import av.error
import av.packet
import av.video
import numpy as np
import numpy.typing as npt

from pupil_labs.video.frameslice import FrameSlice
from pupil_labs.video.indexer import Indexer
from pupil_labs.video.video_frame import VideoFrame

DEFAULT_LOGGER = getLogger(__name__)


PTSArray = npt.NDArray[np.int64]
TimesArray = npt.NDArray[np.float64]


@dataclass
class Stats:
    seeks: int = 0
    decodes: int = 0


@dataclass
class PacketData:
    pts: PTSArray
    times: TimesArray


class Reader(Sequence[VideoFrame]):
    def __init__(
        self,
        source: Path | str,
        times: TimesArray | None = None,
        logger: Logger | None = DEFAULT_LOGGER,
    ):
        self.source = source
        self.logger = logger

        # TODO(dan): this should not live here
        if times is None:
            time_file = Path(self.source).with_suffix(".time")
            if time_file.exists():
                epoch_ns_ts = np.fromfile(time_file, "<u8")
                times = (epoch_ns_ts - epoch_ns_ts[0]) / 1e9

        self._times = times
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
                    ("times", self._times),
                    ("stats", self.stats),
                ]
            )
            + ")"
        )

    @cached_property
    def gop_size(self) -> int:
        self.container.seek(0)
        have_seen_keyframe_already = False
        count = 0
        for packet in self.container.demux(self.stream):
            if packet.pts is None:
                continue
            if packet.is_keyframe:
                if have_seen_keyframe_already:
                    break
                have_seen_keyframe_already = True
            count += 1
            if count > 1000:  # sanity check, eye videos usually have 400 frames per gop
                raise RuntimeError("too many packets demuxed trying to find a keyframe")

        self.container.seek(0)
        if self.logger:
            self.logger.info(f"demuxed {count} packets to get gop_size")
        return count

    @property
    def times(self) -> TimesArray:
        if self._times is None:
            self._times = self.times_from_container
        return self._times

    @cached_property
    def container(self) -> av.container.input.InputContainer:
        container = av.open(self.source)
        for stream in container.streams.video:
            stream.thread_type = "FRAME"
        return container

    @cached_property
    def buffer(self) -> deque[VideoFrame]:
        return deque(maxlen=self.gop_size)

    def seek(self, ts: float) -> None:
        want_start = ts == 0
        if want_start and self.is_at_start:
            return

        self.is_at_start = want_start
        seek_time = int(ts * av.time_base)
        if self.logger:
            self.logger.warning(f"seeking to {ts:.5f}s")
        self.container.seek(seek_time)
        self.stats.seeks += 1
        self.decoder_index = -1 if want_start else None
        self.buffer.clear()
        self._frame_buffer.clear()

    @property
    def stream(self) -> av.video.stream.VideoStream:
        return self.container.streams.video[0]

    @cached_property
    def times_from_container(self) -> TimesArray:
        assert self.stream.time_base
        times = np.array(self.pts * float(self.stream.time_base), dtype=np.float64)
        return times

    @cached_property
    def pts(self) -> PTSArray:
        pts = []
        assert self.stream.time_base
        if self.logger:
            self.logger.warning("demuxing all packets to get pts")
        self.seek(0)
        count = 0
        for packet in self.container.demux(self.stream):
            if packet.pts is None:
                continue
            count += 1
            pts.append(packet.pts)

        # TODO(dan): use .seek() instead?
        self.decoder_index = None
        self.is_at_start = False

        if self.logger:
            self.logger.warning(f"demuxed {count} packets to get pts")
        return np.array(pts)

    def __len__(self) -> int:
        if self.stream.frames is not None:
            return self.stream.frames
        return len(self.times)

    @property
    def demuxer(self) -> Iterator[av.packet.Packet]:
        for packet in self.container.demux(self.stream):
            self.is_at_start = False
            if self.logger:
                packet_time_str = "      "
                if packet.pts is not None:
                    packet_time_str = f"{float(packet.dts * packet.time_base):.3f}s"

                self.logger.debug(
                    f"demuxed"
                    f" {packet.stream.type[0]}{packet.is_keyframe and 'k' or ' '}"
                    f" {packet_time_str}"
                    f" packet={packet}"
                )
            yield packet

    @cached_property
    def _frame_buffer(self) -> deque[av.video.frame.VideoFrame]:
        return deque()

    @property
    def decoder(self) -> Iterator[av.video.frame.VideoFrame]:
        while self._frame_buffer:
            frame = self._frame_buffer.popleft()
            self.logger.debug(f"decoded overage {frame}")
            yield frame

        for packet in self.demuxer:
            try:
                frames = cast(Iterator[av.video.frame.VideoFrame], packet.decode())
            except av.error.EOFError as e:
                if self.logger:
                    self.logger.warning(f"reached end of file: {e}")
                break

            self._frame_buffer.extend(frames)
            while self._frame_buffer:
                frame = self._frame_buffer.popleft()
                if self.logger:
                    self.logger.debug(f"decoding current {frame}")
                self.stats.decodes += 1
                yield frame

        while self._frame_buffer:
            frame = self._frame_buffer.popleft()
            if self.logger:
                self.logger.debug(f"decoding after {frame}")
            self.stats.decodes += 1
            yield frame

    def get_frames(self, key: int | slice) -> list[VideoFrame] | list[VideoFrame]:  # noqa: C901
        # start / stop index logic
        if isinstance(key, slice):
            start_index, stop_index = key.start, key.stop
        elif isinstance(key, int):
            start_index, stop_index = key, key + 1
            if key < 0:
                start_index = self.stream.frames + key
                stop_index = start_index + 1
        else:
            raise TypeError(f"key must be int or slice, not {type(key)}")

        if self.logger:
            self.logger.info(f"get_frames: [{start_index}:{stop_index}]")

        result = list[VideoFrame]()

        # buffered frames logic
        if self.buffer:
            if self.logger:
                self.logger.info(f"buffer: {_summarize_frames(self.buffer)}")

            distance = start_index - self.buffer[0].index
            buffer_contains_wanted_frames = distance >= 0 and distance <= len(
                self.buffer
            )
            if buffer_contains_wanted_frames:
                # TODO(dan): we can be faster here if we just use indices
                for buffered_frame in self.buffer:
                    if start_index <= buffered_frame.index < stop_index:
                        result.append(buffered_frame)

            if result:
                if len(result) == stop_index - start_index:
                    if self.logger:
                        self.logger.debug(
                            f"returning buffered frames: {_summarize_frames(result)}"
                        )
                    return result
                else:
                    if self.logger:
                        self.logger.debug(
                            f"using buffered frames: {_summarize_frames(result)}"
                        )
            else:
                if self.logger:
                    self.logger.debug("no buffered frames found")

            start_index = start_index + len(result)

        if isinstance(key, slice):
            # return a lazy list of the video frames
            if stop_index - start_index < self.lazy_frame_slice_limit:
                return list(FrameSlice[VideoFrame](self, key))
            return FrameSlice[VideoFrame](self, key)

        # seeking logic
        wanted_distance = None
        if self.decoder_index is not None and self.decoder_index + 1 == start_index:
            wanted_start_time = 0 if not self.buffer else self.buffer[-1].time
        else:
            if self.decoder_index is not None:
                distance = distance = start_index - self.decoder_index - 1
                if 0 < distance < self.buffer.maxlen:
                    wanted_distance = distance
            if wanted_distance is None:
                wanted_start_time = self.times[start_index]
                self.seek(wanted_start_time)

        if self.logger and wanted_distance is not None:
            self.logger.debug(
                f"going to iterate {wanted_distance} frames as within keyframe distance"
            )

        # decoding logic
        count = 0
        for av_frame in self.decoder:
            count += 1  # noqa: SIM113
            assert isinstance(av_frame, av.video.frame.VideoFrame)
            assert av_frame.time is not None

            if self.decoder_index is None:
                self.decoder_index = int(np.searchsorted(self.times, av_frame.time))
            else:
                self.decoder_index += 1

            timestamp = (
                self._times[self.decoder_index]
                if self._times is not None
                else av_frame.time
            )

            frame = VideoFrame(av_frame, self.decoder_index, timestamp)
            if self.logger:
                self.logger.debug(f"  decoded {frame}")
            self.buffer.append(frame)
            add_frame = (
                count > wanted_distance
                if wanted_distance is not None
                else av_frame.time >= wanted_start_time - self.time_match_tolerance
            )
            if add_frame:
                result.append(frame)
            if self.decoder_index >= stop_index - 1:
                break

        if self.logger:
            self.logger.debug(f"returning frames: {_summarize_frames(result)}")
        return result

    @overload
    def __getitem__(self, key: int) -> VideoFrame: ...
    @overload
    def __getitem__(self, key: slice) -> list[VideoFrame]: ...

    def __getitem__(self, key: int | slice) -> VideoFrame | list[VideoFrame]:
        frames = self.get_frames(key)
        if isinstance(key, int):
            if not frames:
                raise IndexError(f"index: {key} not found")
            return frames[0]
        return frames

    @property
    def by_idx(self) -> "Reader":
        return self

    @cached_property
    def by_pts(self) -> Indexer[VideoFrame]:
        return Indexer(self.pts, self)

    @cached_property
    def by_times(self) -> Indexer[VideoFrame]:
        return Indexer(self.times, self)

    @cached_property
    def by_ts(self) -> Indexer[VideoFrame]:
        assert self._times is not None
        return Indexer(self._times, self)

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
        self.container.close()


def _summarize_frames(result: list[VideoFrame] | deque[VideoFrame]) -> str:
    indices = [frame.index for frame in result]
    if len(indices) > 1:
        return f"{len(indices)} frames from [{indices[0]} to {indices[-1]}]"
    return str(indices)
