import time
from collections.abc import Generator, Iterable
from typing import TypeVar

import tqdm

T = TypeVar("T")


def measure_fps(
    generator: Iterable[T], total: int | None = None
) -> Generator[T, None, None]:
    n_frames = 0
    start = time.time()
    for item in tqdm.tqdm(generator, unit=" frame", total=total):
        yield item
        n_frames += 1
    end = time.time()
    taken_secs = end - start
    fps = n_frames / taken_secs
    n = int(fps / 10000.0 * 50)
    print(f"total time={round(taken_secs * 1000):5}ms {round(fps):4}fps {n * '█'}")
