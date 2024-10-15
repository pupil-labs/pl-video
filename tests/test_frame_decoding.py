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


def test_decoded_frame_correctness(main_video_path):
    reader = Reader(main_video_path)

    frame0 = reader.by_idx[0]
    assert frame0.bgr.mean() == 186.91599114583335

    frame50 = reader.by_idx[50]
    assert frame50.rgb.mean() == 163.7086623263889

    frame100 = reader.by_idx[100]
    assert frame100.gray.mean() == 162.1663390625
