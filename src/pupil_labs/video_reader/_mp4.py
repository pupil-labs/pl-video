import os
import struct
from os import PathLike
from typing import BinaryIO

import numpy as np
import numpy.typing as npt

type _Span = tuple[int, int]


def shown_times(source: str | PathLike[str]) -> npt.NDArray[np.float64] | None:
    """Each frame's shown time in seconds, from the first video track's tables.

    None when the track has an edit list or composition offsets, which make its
    shown times differ from its decode times.
    """
    with open(source, "rb") as file:
        moov = dict(_boxes(file, (0, os.fstat(file.fileno()).st_size)))[b"moov"]
        for kind, trak in _boxes(file, moov):
            if kind != b"trak":
                continue
            track = dict(_boxes(file, trak))
            mdia = dict(_boxes(file, track[b"mdia"]))
            if _read(file, mdia[b"hdlr"], 8, 4) != b"vide":
                continue
            stbl = dict(_boxes(file, dict(_boxes(file, mdia[b"minf"]))[b"stbl"]))
            if b"edts" in track or b"ctts" in stbl:
                return None
            return _decode_times(file, stbl[b"stts"]) / _timescale(file, mdia[b"mdhd"])
    return None


def _boxes(file: BinaryIO, span: _Span) -> list[tuple[bytes, _Span]]:
    boxes = []
    position, end = span
    while position + 8 <= end:
        file.seek(position)
        size, kind = struct.unpack(">I4s", file.read(8))
        header = 8
        if size == 1:
            (size,) = struct.unpack(">Q", file.read(8))
            header = 16
        elif size == 0:
            size = end - position
        boxes.append((kind, (position + header, position + size)))
        position += size
    return boxes


def _read(file: BinaryIO, span: _Span, offset: int, length: int) -> bytes:
    file.seek(span[0] + offset)
    return file.read(length)


def _timescale(file: BinaryIO, mdhd: _Span) -> int:
    version = _read(file, mdhd, 0, 1)[0]
    (timescale,) = struct.unpack(">I", _read(file, mdhd, 20 if version else 12, 4))
    return int(timescale)


def _decode_times(file: BinaryIO, stts: _Span) -> npt.NDArray[np.float64]:
    (count,) = struct.unpack(">I", _read(file, stts, 4, 4))
    runs = np.frombuffer(_read(file, stts, 8, 8 * count), dtype=">u4").reshape(-1, 2)
    durations = np.repeat(runs[:, 1].astype(np.int64), runs[:, 0])
    return (np.cumsum(durations) - durations).astype(np.float64)
