from pathlib import Path

import numpy as np

from pupil_labs.video import Reader, Writer

from .utils import measure_fps


def test_write_ndarray(tmp_path: Path) -> None:
    with Writer(tmp_path / "out.mp4") as writer:
        for _ in measure_fps(range(300)):
            array = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
            writer.write_image(array)


def test_losslessness(tmp_path: Path) -> None:
    width = 400
    height = 300
    with Writer(tmp_path / "out.mp4", lossless=True) as writer:
        written_images = []
        for _ in measure_fps(range(10)):
            img = np.random.randint(0, 255, (3, height, width), dtype=np.uint8)
            written_images.append(img)

            # Note: the encoding is only truely lossless if yuv444p data is used.
            # When converting yuv444p to e.g. rgb24, numeric precision is lost and
            # results are slightly off.
            writer.write_image(img, pix_fmt="yuv444p")

    with Reader(tmp_path / "out.mp4") as reader:
        for img_written, frame in zip(written_images, reader):
            img_read = frame.to_ndarray(pixel_format="yuv444p")
            assert np.allclose(img_written, img_read)
