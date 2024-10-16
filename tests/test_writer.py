import numpy as np

from pupil_labs.video_simple import Writer

from .utils import measure_fps


def test_write_ndarray(tmp_path):
    with Writer(tmp_path / "out.mp4") as video_out:
        for _ in measure_fps(range(300)):
            array = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
            video_out.write(array)
