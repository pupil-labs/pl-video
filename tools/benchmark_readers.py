"""Time pl-video's reader against pupil_labs.video_reader on the benchmark videos.

Each scenario x video x reader runs in a fresh process, five times; the median
time and the highest peak memory are reported, then the checks.
"""

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import numpy.typing as npt

RECORDINGS = Path("/mnt/data/recordings")
VIDEOS = {
    "scene short": RECORDINGS / "supermarket-no-audio/Neon Scene Camera v1 ps1.mp4",
    "scene long": RECORDINGS / "long/long-test/Neon Scene Camera v1 ps1.mp4",
    "eye short": RECORDINGS / "supermarket-no-audio/Neon Sensor Module v1 ps1.mp4",
    "eye long": RECORDINGS / "long/long-test/Neon Sensor Module v1 ps1.mp4",
}
SCENARIOS = ["sequential", "sparse", "random", "reverse", "first", "center", "last"]
SINGLE = {"first", "center", "last"}
READERS = ["pl-video", "new", "new lazy"]
LONG_SEQUENTIAL = 10_000
LONG_REVERSE = 2_000
RANDOM_COUNT = 200
SEED = 0
SLACK = 1.1
PAIN_LIMIT = 0.5


def indices(scenario: str, length: int, long: bool) -> list[int]:
    if scenario == "sequential":
        return list(range(LONG_SEQUENTIAL if long else length))
    if scenario == "sparse":
        return list(range(0, length, 100))
    if scenario == "random":
        rng = np.random.default_rng(SEED)
        return [int(i) for i in rng.choice(length, RANDOM_COUNT, replace=False)]
    if scenario == "reverse":
        return list(
            range(length - 1, length - 1 - LONG_REVERSE, -1)
            if long
            else range(length - 1, -1, -1)
        )
    return [{"first": 0, "center": length // 2, "last": length - 1}[scenario]]


def lazy_key(
    scenario: str, picked: list[int], long: bool
) -> slice | npt.NDArray[np.int64]:
    if scenario == "sequential":
        return slice(LONG_SEQUENTIAL) if long else slice(None)
    if scenario == "sparse":
        return slice(None, None, 100)
    if scenario == "reverse":
        return slice(-1, -1 - LONG_REVERSE, -1) if long else slice(None, None, -1)
    return np.array(picked)


def run_child(reader_name: str, scenario: str, video: str) -> None:
    path = VIDEOS[video]
    long = video.endswith("long")
    with av.open(path) as container:
        length = container.streams.video[0].frames
    picked = indices(scenario, length, long)
    if reader_name == "pl-video":
        from pupil_labs.video import Reader as OldReader  # noqa: PLC0415

        start = time.perf_counter()
        old = OldReader(path)
        if scenario not in SINGLE:
            start = time.perf_counter()
        for i in picked:
            old[i].bgr  # noqa: B018
    else:
        from pupil_labs.video_reader import Reader  # noqa: PLC0415

        start = time.perf_counter()
        reader = Reader(path)
        if scenario not in SINGLE:
            start = time.perf_counter()
        if reader_name == "new lazy":
            selection = reader[lazy_key(scenario, picked, long)]
            for frame, expected in zip(selection, picked, strict=True):
                assert frame.index == expected
                frame.bgr  # noqa: B018
        else:
            for i in picked:
                reader[i].bgr  # noqa: B018
    seconds = time.perf_counter() - start
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps({"seconds": seconds, "peak_mb": peak_mb}))


@dataclass
class Cell:
    seconds: float
    peak_mb: float


def run(reader_name: str, scenario: str, video: str) -> dict[str, float]:
    out = subprocess.run(  # noqa: S603
        [sys.executable, __file__, "--child", reader_name, scenario, video],
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[str, float] = json.loads(out.stdout.strip().splitlines()[-1])
    return result


def measure(
    readers: list[str], scenario: str, video: str, runs: int
) -> dict[str, Cell]:
    """Run the readers in turn, so that a busy moment slows each of them alike."""
    results: dict[str, list[dict[str, float]]] = {r: [] for r in readers}
    for _ in range(runs):
        for reader_name in readers:
            results[reader_name].append(run(reader_name, scenario, video))
    return {
        r: Cell(
            statistics.median(x["seconds"] for x in rs),
            max(x["peak_mb"] for x in rs),
        )
        for r, rs in results.items()
    }


def warm(path: Path) -> None:
    with path.open("rb") as f:
        while f.read(1 << 24):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--scenario", nargs="*", default=SCENARIOS, choices=SCENARIOS)
    parser.add_argument(
        "--video", nargs="*", default=list(VIDEOS), choices=list(VIDEOS)
    )
    parser.add_argument("--reader", nargs="*", default=READERS, choices=READERS)
    parser.add_argument("--child", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        run_child(*args.child)
        return

    failures: list[str] = []
    print(f"{'scenario':<11}{'video':<12}" + "".join(f"{r:>28}" for r in args.reader))
    for video in args.video:
        warm(VIDEOS[video])
        for scenario in args.scenario:
            cells = measure(args.reader, scenario, video, args.runs)
            row = f"{scenario:<11}{video:<12}"
            for r in args.reader:
                cell = cells.get(r)
                row += (
                    f"{'':>28}"
                    if cell is None
                    else f"{cell.seconds:>10.3f} s {cell.peak_mb:>7.0f} MB{'':>6}"
                )
            print(row, flush=True)
            failures += check(scenario, video, cells)
    print()
    print("\n".join(failures) if failures else "All checks pass.")
    sys.exit(1 if failures else 0)


def check(scenario: str, video: str, cells: dict[str, Cell]) -> list[str]:
    failures = []
    where = f"{scenario}, {video}"
    old, new, lazy = cells.get("pl-video"), cells.get("new"), cells.get("new lazy")
    if old and new:
        if new.seconds > SLACK * old.seconds:
            failures.append(
                f"FAIL time: {where}: {new.seconds:.3f} s"
                f" > {SLACK} x {old.seconds:.3f} s"
            )
        if new.peak_mb > old.peak_mb:
            failures.append(
                f"FAIL memory: {where}: {new.peak_mb:.0f} MB > {old.peak_mb:.0f} MB"
            )
    if (
        new
        and scenario in {"center", "last"}
        and video.endswith("long")
        and new.seconds >= PAIN_LIMIT
    ):
        failures.append(
            f"FAIL pain point: {where}: {new.seconds:.3f} s >= {PAIN_LIMIT} s"
        )
    if new and lazy and lazy.seconds > new.seconds:
        failures.append(
            f"FAIL lazy: {where}: {lazy.seconds:.3f} s > {new.seconds:.3f} s"
        )
    return failures


if __name__ == "__main__":
    main()
