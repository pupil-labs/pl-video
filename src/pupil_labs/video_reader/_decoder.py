from collections import deque
from collections.abc import Iterator
from os import PathLike

import av
import av.error
import numpy as np
import numpy.typing as npt
from av.video.stream import VideoStream

from pupil_labs.video_reader.frame import Frame

# Frames this small decode too fast to keep more than four threads busy.
_SMALL = 1_000_000


class Decoder:
    """Decodes one keyframe group at a time from its own open file.

    A frame's index is the rank of its pts among the packets of its keyframe group,
    counted from the group's keyframe. A frame the decoder drops is filled in with
    the next frame of its group, or the previous one at the group's end.
    """

    def __init__(
        self,
        path: str | PathLike[str],
        timestamps: npt.NDArray[np.float64] | None = None,
        threads: int | None = None,
    ) -> None:
        self._timestamps = timestamps
        self._container = av.open(path, "r")
        self._stream = self._container.streams.video[0]
        self._codec = self._stream.codec_context
        if threads is None:
            threads = 4 if self._codec.width * self._codec.height < _SMALL else 0
        self.threads = threads
        self._stream.thread_type = "FRAME"
        self._codec.thread_count = threads
        self._entries = self._stream.index_entries
        tb = self._stream.time_base
        assert tb is not None
        self._tb_num, self._tb_den = tb.numerator, tb.denominator
        self._packets: Iterator[av.Packet[VideoStream]] = self._container.demux(
            self._stream
        )
        self._next_keyframe: av.Packet[VideoStream] | None = None
        self._next_start = 0
        self._at_end = False
        self._queue: deque[av.Packet[VideoStream]] = deque()
        self._slots: dict[int, deque[int]] = {}
        self._groups: deque[_Group] = deque()
        self._drain_at_group_end = True
        self._ahead: dict[int, Frame] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def close(self) -> None:
        self._container.close()

    @property
    def width(self) -> int:
        return self._codec.width

    @property
    def height(self) -> int:
        return self._codec.height

    @property
    def average_rate(self) -> float:
        rate = self._stream.average_rate
        assert rate is not None
        return float(rate)

    @property
    def duration(self) -> float:
        assert self._stream.duration is not None
        return float(self._stream.duration * self._tb_num / self._tb_den)

    def group_end(self, keyframe: int) -> int:
        following = self._entries.search_timestamp(
            self._entries[keyframe].timestamp + 1, backward=False
        )
        return len(self._entries) if following < 0 else following

    def keyframe_before(self, index: int) -> int:
        return self._entries.search_timestamp(
            self._entries[index].timestamp, backward=True
        )

    def produces(self, index: int) -> bool:
        """Whether decoding on reaches `index` without decoding frames it skips."""
        for group in self._groups:
            if group.start <= index < group.stop:
                return not group.produced(index)
        return (
            bool(self._groups)
            and not self._queue
            and not self._at_end
            and self.keyframe_before(index) == self._next_start
        )

    def collect(self, keyframe: int, targets: list[int]) -> dict[int, Frame]:
        """Decode the targets, all in the group that starts at `keyframe`.

        Frames that come out past the last target are kept for the next call.
        """
        wanted = set(targets)
        if not all(i in self._ahead or self.produces(i) for i in wanted):
            self.seek(keyframe)
        last = max(targets)
        unneeded = self._unneeded_after(last)
        found = {i: self._ahead[i] for i in wanted if i in self._ahead}
        ahead: dict[int, Frame] = {}
        while len(found) < len(wanted):
            if unneeded is not None and len(self._queue) <= unneeded:
                frames = self._stop()
                if not wanted <= {f.index for f in frames} | found.keys():
                    self.seek(keyframe)
                    unneeded = None
            else:
                frames = self.step()
            for frame in frames:
                if frame.index in wanted:
                    found[frame.index] = frame
                elif frame.index > last:
                    ahead[frame.index] = frame
        self._ahead = ahead
        return found

    def _unneeded_after(self, last: int) -> int | None:
        """How many queued packets `last` does not need, if its group goes on."""
        for group in self._groups:
            if group.start <= last < group.stop - 1:
                cutoff = group.pts[last - group.start]
                needed = [
                    i
                    for i, p in enumerate(self._queue)
                    if p.pts is not None and p.pts <= cutoff
                ]
                return len(self._queue) - needed[-1] - 1 if needed else len(self._queue)
        return None

    def _stop(self) -> list[Frame]:
        """Decode what was fed and drop the rest; the next frame needs a seek."""
        try:
            decoded = self._codec.decode(None)
        except av.error.InvalidDataError:
            decoded = []
        frames = self._place(decoded)
        self._codec.flush_buffers()
        self._queue.clear()
        self._slots.clear()
        self._groups.clear()
        self._at_end = True
        return frames

    def seek(self, keyframe: int) -> None:
        # MP4 seeks by presentation time, which a keyframe has at or after its dts.
        self._container.seek(
            self._entries[keyframe].timestamp, stream=self._stream, backward=False
        )
        self._packets = self._container.demux(self._stream)
        self._next_keyframe = None
        self._next_start = keyframe
        self._at_end = False
        self._queue.clear()
        self._slots.clear()
        self._groups.clear()
        self._drain_at_group_end = True
        self._ahead = {}
        self._load_group()

    def step(self) -> list[Frame]:
        """Feed the decoder once and return the frames that came out."""
        if not self._queue:
            if self._groups and not self._drain_at_group_end and not self._at_end:
                self._load_group()
                return []
            return self._drain()
        packet = self._queue.popleft()
        try:
            decoded = self._codec.decode(packet)
        except av.error.InvalidDataError:
            decoded = []
        return self._place(decoded)

    def _drain(self) -> list[Frame]:
        try:
            decoded = self._codec.decode(None)
        except av.error.InvalidDataError:
            decoded = []
        frames = self._place(decoded)
        while self._groups:
            frames += self._complete_group()
        self._codec.flush_buffers()
        if not self._at_end:
            self._load_group()
            self._drain_at_group_end = False
        return frames

    def _place(self, decoded: list[av.VideoFrame]) -> list[Frame]:
        frames: list[Frame] = []
        for image in decoded:
            pts = image.pts
            assert pts is not None
            slots = self._slots.get(pts)
            if not slots:
                continue
            index = slots.popleft()
            if not slots:
                del self._slots[pts]
            frame = self._frame(index, pts, image)
            frames.append(frame)
            while self._groups and index >= self._groups[0].stop:
                frames += self._complete_group()
            if self._groups and self._groups[0].start <= index:
                group = self._groups[0]
                group.add(index, image)
                if group.complete:
                    self._groups.popleft()
        return frames

    def _complete_group(self) -> list[Frame]:
        group = self._groups.popleft()
        filled = []
        for index, pts in group.missing():
            self._slots[pts].remove(index)
            if not self._slots[pts]:
                del self._slots[pts]
            source = group.fill_for(index)
            if source is None:
                raise RuntimeError(f"frame {index}: no frame of its group decodes")
            filled.append(self._frame(index, pts, source))
        return filled

    def _frame(self, index: int, pts: int, image: av.VideoFrame) -> Frame:
        if self._timestamps is None:
            time = float(pts) * self._tb_num / self._tb_den
        else:
            time = float(self._timestamps[index])
        return Frame(index, time, pts, image)

    def _load_group(self) -> None:
        packets = [self._next_keyframe] if self._next_keyframe else []
        self._next_keyframe = None
        for packet in self._packets:
            if packet.pts is None:
                continue
            if packet.is_keyframe and packets:
                self._next_keyframe = packet
                break
            packets.append(packet)
        else:
            self._at_end = True
        group = _Group(
            self._next_start, sorted(p.pts for p in packets if p.pts is not None)
        )
        for index, pts in enumerate(group.pts, group.start):
            self._slots.setdefault(pts, deque()).append(index)
        self._next_start = group.stop
        self._groups.append(group)
        self._queue.extend(packets)


class _Group:
    """A keyframe group's frames: which came out, and what fills the ones that did not.

    A frame is kept only while a lower index in the group is still missing, as
    the next frame of a dropped one, and so is the latest frame, for a dropped one
    at the group's end.
    """

    def __init__(self, start: int, pts: list[int]) -> None:
        self.start = start
        self.stop = start + len(pts)
        self.pts = pts
        self._produced: set[int] = set()
        self._lowest_missing = start
        self._held: dict[int, av.VideoFrame] = {}
        self._latest: tuple[int, av.VideoFrame] | None = None

    def produced(self, index: int) -> bool:
        return index in self._produced

    @property
    def complete(self) -> bool:
        return len(self._produced) == len(self.pts)

    def add(self, index: int, image: av.VideoFrame) -> None:
        self._produced.add(index)
        if index > self._lowest_missing:
            self._held[index] = image
        while self._lowest_missing in self._produced:
            self._lowest_missing += 1
        if self._latest is None or index > self._latest[0]:
            self._latest = index, image

    def missing(self) -> list[tuple[int, int]]:
        return [
            (index, pts)
            for index, pts in enumerate(self.pts, self.start)
            if index not in self._produced
        ]

    def fill_for(self, index: int) -> av.VideoFrame | None:
        following = [i for i in self._held if i > index]
        if following:
            return self._held[min(following)]
        return None if self._latest is None else self._latest[1]
