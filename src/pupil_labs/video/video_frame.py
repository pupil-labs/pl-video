import warnings
from dataclasses import dataclass
from typing import Any, Literal, cast

import av
import av.video.frame
import numpy as np
import numpy.typing as npt

PixelFormat = Literal["gray", "bgr24", "rgb24", "yuv420p", "yuv444p"]


@dataclass
class VideoFrame:
    av_frame: av.video.frame.VideoFrame
    index: int
    ts: int | float

    stream_type = "video"

    def __getattr__(self, key: str) -> Any:
        return getattr(self.av_frame, key)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            + ", ".join(
                f"{key}={value}"
                for key, value in [
                    ("pts", self.pts),
                    ("index", self.index),
                    ("time", f"{self.time:.5f}"),
                    ("format", f"{self.av_frame.format.name}"),
                    ("res", f"{self.av_frame.width}x{self.av_frame.height}"),
                    ("ts", f"{self.ts}"),
                ]
            )
            + ")"
        )

    @property
    def pts(self) -> int | None:
        return self.av_frame.pts

    @property
    def gray(self) -> npt.NDArray[np.uint8]:
        """
        Numpy image array in gray format
        """
        return self.to_ndarray("gray")

    @property
    def bgr(self) -> npt.NDArray[np.uint8]:
        """
        Numpy image array in BGR format
        """
        return self.to_ndarray("bgr24")

    @property
    def rgb(self) -> npt.NDArray[np.uint8]:
        """
        Numpy image array in RGB format
        """
        return self.to_ndarray("rgb24")

    def to_ndarray(self, pixel_format: PixelFormat) -> npt.NDArray[np.uint8]:
        """
        Returns an numpy array of the image for the frame
        """
        # TODO: add caching for decoded frames?
        return av_frame_to_ndarray_fast(self.av_frame, pixel_format)


def av_frame_to_ndarray_fast(
    av_frame: av.VideoFrame, pixel_format: PixelFormat | None
) -> npt.NDArray[np.uint8]:
    """
    Returns an image pixel numpy array for an av.VideoFrame in `format`
    skipping conversion by using buffers directly if possible for performance
    """
    if pixel_format == "gray":
        if av_frame.format.name == "gray":
            result = np.frombuffer(av_frame.planes[0], np.uint8).reshape(
                av_frame.height, av_frame.width
            )
        elif av_frame.format.name.startswith("yuv"):
            plane = av_frame.planes[0]
            plane_data = np.frombuffer(plane, np.uint8)
            if av_frame.height * av_frame.width == len(plane_data):
                gray = plane_data
                gray.shape = plane.height, plane.width
            else:
                gray_padded = plane_data
                gray_padded = gray_padded.reshape(-1, plane.line_size)
                gray = gray_padded[:, : plane.width]
                # gray = np.ascontiguousarray(gray)

            if av_frame.format.name == "yuv420p":
                warnings.warn(
                    "using Y plane for yuv420p gray images, range is 16-235 instead of 0-255."
                    " Use .av_frame.to_ndarray(format='gray') for full range (4x slower)",
                    stacklevel=2,
                )
                # av.to_ndarray(format='gray') returns 0-255 for gray values
                # but here reading the Y from planes for yuv420p the output
                # is limited from 16-235 instead of converted to full range 0-255
                # this is done for performance reasons
                # gray = limited_yuv420p_to_full(gray)
            result = gray

    elif pixel_format in ("bgr24", "rgb24") and av_frame.format.name == pixel_format:
        plane = av_frame.planes[0]

        # TODO(dan): find out why np.frombuffer(plane) didn't work here
        # for bgr, frombuffer is faster than array
        image = np.array(plane)

        if 3 * av_frame.height * av_frame.width != len(image):
            image = image.reshape(-1, plane.line_size)
            image = image[:, : 3 * av_frame.width]
            # image = np.ascontiguousarray(image)
            image = image.reshape(av_frame.height, av_frame.width, 3)
        else:
            image = np.frombuffer(av_frame.planes[0], np.uint8).reshape(
                av_frame.height, av_frame.width, 3
            )
        result = image
    else:
        result = av_frame.to_ndarray(format=pixel_format)

    return cast(npt.NDArray[np.uint8], result)
