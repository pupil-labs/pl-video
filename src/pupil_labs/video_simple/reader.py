from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, overload
from pupil_labs.video_simple.video_frame import VideoFrame
import numpy.typing as npt
import av


@dataclass
class Reader:
    path: str | Path
    timestamps: Optional[npt.NDArray] = None

    def __post_init__(self):
        self.container = av.open(str(self.path))
        assert self.container.streams.video, "No video stream found"
        assert len(self.container.streams.video) == 1, "Only one video stream supported"

        self._pts = []
        self._pts_to_idx = {}
        for packet in self.container.demux(video=0):
            if packet.pts is None:
                continue
            self._pts.append(packet.pts)
            self._pts_to_idx[packet.pts] = len(self._pts) - 1

        self._int_indexer = IntegerIndexer(self)

    @property
    def duration(self) -> float | None:
        return self.container.duration

    @property
    def frames(self) -> List[VideoFrame]:
        raise NotImplementedError

    @property
    def pts(self) -> List[int]:
        return self._pts

    def __len__(self) -> int:
        return len(self._pts)

    @property
    def by_pts(self):
        raise NotImplementedError

    @property
    def by_idx(self):
        return self._int_indexer

    @property
    def by_ts(self):
        raise NotImplementedError

    def _seek_to_index(self, index: int) -> int:
        pts = self._pts[index]
        self.container.seek(
            offset=pts,
            stream=self.container.streams.video[0],
        )
        return pts


@dataclass
class IntegerIndexer:
    reader: Reader
    frame_buffer_size: int = 50
    previous_decoded_pts: Optional[int] = None

    def __post_init__(self):
        self.stream_buffer = deque(maxlen=self.frame_buffer_size)

    @overload
    def __getitem__(self, key: int) -> VideoFrame: ...

    @overload
    def __getitem__(self, key: slice) -> List[VideoFrame]: ...

    def __getitem__(self, key: int | slice) -> VideoFrame | List[VideoFrame]:
        """Get a frame by index or a list of frames by slice.

        This method tries to avoid seeking if possible. Indexing consecutive frames does not require seeking.
        """
        start_idx, stop_idx = self._parse_key(key)
        if start_idx >= stop_idx:
            return []

        buffer_contains_wanted_frames = (
            self.stream_buffer
            and self.stream_buffer[0].index <= start_idx
            and self.stream_buffer[-1].index >= stop_idx - 1
        )
        if buffer_contains_wanted_frames:
            result = []
            for frame in self.stream_buffer:
                if start_idx <= frame.index < stop_idx:
                    result.append(frame)
                if frame.index >= stop_idx:
                    break
            return result

        need_seek = self.check_if_seek_needed(start_idx)

        if need_seek:
            self.previous_decoded_pts = self.reader._seek_to_index(start_idx)
            self.stream_buffer.clear()

        result = []
        try:
            # TODO: Multi-threaded decoding for speed?
            for frame in self.reader.container.decode(video=0):
                assert frame
                frame_idx = self.reader._pts_to_idx[frame.pts]
                frame = VideoFrame(frame, frame_idx)
                self.stream_buffer.append(frame)

                if start_idx <= frame_idx < stop_idx:
                    result.append(frame)

                if frame_idx >= stop_idx - 1:
                    break
                frame_idx += 1
        except EOFError:
            self.previous_decoded_pts = None
            raise IndexError("frame index out of range")

        if isinstance(key, int):
            result = result[0]

        return result

    def check_if_seek_needed(self, start_idx):
        need_seek = True
        if self.previous_decoded_pts is not None:
            previous_decoded_index = self.reader._pts_to_idx[self.previous_decoded_pts]
            if previous_decoded_index == start_idx - 1:
                need_seek = False
        return need_seek

    def _parse_key(self, key: int | slice) -> Tuple[int, int]:
        if isinstance(key, slice):
            start_index, stop_index = key.start, key.stop
        elif isinstance(key, int):
            start_index, stop_index = key, key + 1
            if key < 0:
                start_index = len(self.reader) + key
                stop_index = start_index + 1
        else:
            raise TypeError(f"key must be int or slice, not {type(key)}")

        if start_index is None:
            start_index = 0
        if start_index < 0:
            start_index = len(self.reader) + start_index
        if stop_index is None:
            stop_index = len(self.reader)
        if stop_index < 0:
            stop_index = len(self.reader) + stop_index

        return start_index, stop_index
