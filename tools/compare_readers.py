"""Compare every frame of pl-video's reader and pupil_labs.video_reader.

Prints each mismatch by video, field and frames, and exits 1 if there is any.
"""

import argparse
import sys
import warnings
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np

from pupil_labs.video import Reader as OldReader
from pupil_labs.video_reader import Reader

TEST_DATA = Path(__file__).parent.parent / "tests/data"
RECORDINGS = Path("/mnt/data/recordings")
NEON2 = RECORDINGS / "Neon 2/Markers/2026-11-25_14-09-35-520b2245"
VIDEOS = [
    RECORDINGS / "supermarket-no-audio/Neon Scene Camera v1 ps1.mp4",
    RECORDINGS / "supermarket-no-audio/Neon Sensor Module v1 ps1.mp4",
    RECORDINGS / "long/long-test/Neon Scene Camera v1 ps1.mp4",
    RECORDINGS / "long/long-test/Neon Sensor Module v1 ps1.mp4",
    NEON2 / "Neon2 Scene Camera ps1.mp4",
    NEON2 / "Neon2 Front IR Camera ps1.mp4",
    *sorted(TEST_DATA.glob("*.mp4")),
    *sorted(TEST_DATA.glob("multi-part/*.mp4")),
]
FIELDS = ["length", "index", "time", "pts", "bgr", "rgb", "gray"]


def compare(path: Path) -> dict[str, list[int]]:
    warnings.simplefilter("ignore")
    av.logging.set_level(av.logging.PANIC)
    old, new = OldReader(path), Reader(path)
    mismatches: dict[str, list[int]] = {field: [] for field in FIELDS}
    if len(old) != len(new):
        mismatches["length"].append(len(new))
    for position, (a, b) in enumerate(zip(old, new, strict=False)):
        for field in ("index", "time", "pts"):
            if getattr(a, field) != getattr(b, field):
                mismatches[field].append(position)
        for field in ("bgr", "rgb", "gray"):
            if not np.array_equal(getattr(a, field), getattr(b, field)):
                mismatches[field].append(position)
    return {field: frames for field, frames in mismatches.items() if frames}


def spans(frames: list[int]) -> Iterator[str]:
    start = previous = frames[0]
    for frame in [*frames[1:], None]:
        if frame is not None and frame == previous + 1:
            previous = frame
            continue
        yield str(start) if start == previous else f"{start}-{previous}"
        if frame is not None:
            start = previous = frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="*", type=Path, default=VIDEOS)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    clean = True
    with ProcessPoolExecutor(args.jobs) as pool:
        for path, mismatches in zip(
            args.videos, pool.map(compare, args.videos), strict=True
        ):
            clean = clean and not mismatches
            print(f"{'MISMATCH' if mismatches else 'identical'}  {path}", flush=True)
            for field, frames in mismatches.items():
                print(f"    {field}: frames {', '.join(spans(frames))}")
    sys.exit(0 if clean else 1)


if __name__ == "__main__":
    main()
