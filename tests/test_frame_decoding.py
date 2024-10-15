import pathlib
import av
import pytest

from pupil_labs.video_simple.reader import Reader

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
def test_decode(
    video_path: pathlib.Path,
    pixel_format,
):
    reader = Reader(video_path)
    for frame in measure_fps(reader):
        frame.to_ndarray(pixel_format=pixel_format)
