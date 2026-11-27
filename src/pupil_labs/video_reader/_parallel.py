from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from queue import LifoQueue

from pupil_labs.video_reader._decoder import Decoder
from pupil_labs.video_reader.frame import Frame

type Job = tuple[int, list[int]]
"A keyframe and the frames to take from its group, in the order asked for."

WORKERS = 4
DEFAULT_THREADS = None
ALL_THREADS = 0
FEW_THREADS = 4


def frames_in_order(
    decoder: Decoder,
    open_decoder: Callable[[int | None], Decoder],
    jobs: Iterator[Job],
    group_length: int,
) -> Iterator[Frame]:
    """Yield the jobs' frames in order, decoding up to two groups' frames ahead.

    A job runs on a decoder of its own, up to four at once, while the caller takes
    the frames before it. One that takes a few frames gets a decoder with four
    threads, which loses less of its threads to a seek. The first job runs on
    `decoder` in the caller's thread, and so does one that takes most of a group
    before the previous job's when `decoder` decodes with all threads: such a
    group keeps every core busy on its own.
    """
    budget = 2 * group_length
    big_frames = decoder.threads == ALL_THREADS
    idle: dict[int | None, LifoQueue[Decoder]] = {
        DEFAULT_THREADS: LifoQueue(),
        FEW_THREADS: LifoQueue(),
    }

    def collect(
        threads: int | None, keyframe: int, targets: list[int]
    ) -> dict[int, Frame]:
        decoders = idle[threads]
        worker = open_decoder(threads) if decoders.empty() else decoders.get()
        try:
            return worker.collect(keyframe, targets)
        finally:
            decoders.put(worker)

    pending: deque[tuple[Job, Future[dict[int, Frame]] | None]] = deque()
    upcoming = next(jobs, None)
    previous: int | None = None
    in_flight = 0

    def submit_what_fits() -> None:
        nonlocal upcoming, previous, in_flight
        while upcoming is not None and (
            not pending
            or (len(pending) < WORKERS and in_flight + len(upcoming[1]) <= budget)
        ):
            keyframe, targets = upcoming
            dense = 2 * len(targets) >= group_length
            if previous is None or (dense and big_frames and keyframe < previous):
                future = None
            else:
                threads = DEFAULT_THREADS if dense else FEW_THREADS
                future = pool.submit(collect, threads, keyframe, targets)
            pending.append((upcoming, future))
            previous = keyframe
            in_flight += len(targets)
            upcoming = next(jobs, None)

    with ThreadPoolExecutor(WORKERS) as pool:
        try:
            submit_what_fits()
            while pending:
                (keyframe, targets), future = pending.popleft()
                in_flight -= len(targets)
                if future is None:
                    submit_what_fits()
                    frames = decoder.collect(keyframe, targets)
                else:
                    frames = future.result()
                    submit_what_fits()
                yield from _hand_out(frames, targets)
        finally:
            pool.shutdown(cancel_futures=True)
            for decoders in idle.values():
                while not decoders.empty():
                    decoders.get().close()


def _hand_out(frames: dict[int, Frame], targets: list[int]) -> Iterator[Frame]:
    last = {index: position for position, index in enumerate(targets)}
    for position, index in enumerate(targets):
        yield frames.pop(index) if last[index] == position else frames[index]
