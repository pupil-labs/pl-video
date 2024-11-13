from pathlib import Path

import pytest

from pupil_labs.video.frame import PixelFormat
from pupil_labs.video.reader import Reader

from .utils import measure_fps


@pytest.mark.parametrize(
    "pixel_format",
    [
        None,
        "gray",
        "rgb24",
        "bgr24",
    ],
)
def test_decode(video_path: Path, pixel_format: PixelFormat) -> None:
    if video_path.suffix == ".mjpeg" and pixel_format is None:
        pytest.xfail("mjpeg decoding does not support pixel_format=None")

    reader = Reader(video_path)
    for frame in measure_fps(reader):
        frame.to_ndarray(pixel_format=pixel_format)


def test_decoded_frame_correctness(main_video_path: Path) -> None:
    reader = Reader(main_video_path)

    frame0 = reader[0]
    assert frame0.bgr.mean() == 186.91599114583335

    frame50 = reader[50]
    assert frame50.rgb.mean() == 163.7086623263889

    frame100 = reader[100]
    assert frame100.gray.mean() == 162.1663390625
