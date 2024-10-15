import time

import tqdm


def measure_fps(generator, total=None):
    n_frames = 0
    start = time.time()
    for item in tqdm.tqdm(generator, unit=" frame", total=total):
        yield item
        n_frames += 1
    end = time.time()
    taken_secs = end - start
    fps = n_frames / taken_secs
    n = int(fps / 10000.0 * 50)
    print(f"total time={round(taken_secs*1000):5}ms {round(fps):4}fps {n * '█'}")
