import av
import numpy as np
import pytest

from pupil_labs.video.frame import av_frame_to_ndarray_fast
from unittest.mock import patch, PropertyMock, MagicMock


@pytest.mark.parametrize(
    "width,height,line_size", [
        (384, 192, 384),
        (400, 200, 416),
    ]
)
def test_av_frame_to_ndarray_fast_contiguous(width, height, line_size):
    """
    Ensure that the output ndarray is contiguous even if the frame plane contains
    padded data.
    """
    buffer = np.zeros((height * line_size,))
    with (
        patch("numpy.frombuffer", MagicMock(return_value=buffer)) as _,
        patch("av.video.plane.VideoPlane.line_size", new_callable=PropertyMock) as mock,
    ):
        mock.return_value = line_size

        av_frame = av.VideoFrame(width, height, format="yuv420p")
        assert av_frame.planes[0].line_size == line_size

        array = av_frame_to_ndarray_fast(av_frame, "gray")
        assert array.shape == (height, width)
        assert array.flags.c_contiguous
