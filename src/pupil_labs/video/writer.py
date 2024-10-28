from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Optional

import av
import numpy as np
import numpy.typing as npt

from pupil_labs.video.frame import PixelFormat


@dataclass
class Writer:
    """Video writer for creating videos from image arrays."""

    path: str | Path
    "Path to the video file to write to"
    lossless: bool = False
    "If True, the video will be encoded lossless H264"
    rate: int = 30
    "Frame rate of the output video"
    bit_rate: int = int(5e6)
    "Bit rate of the output video"

    def __post_init__(self) -> None:
        self.container = av.open(self.path, "w")
        self.video_stream = self.container.add_stream("h264", rate=self.rate)
        self.video_stream.bit_rate = self.bit_rate

        if self.lossless:
            self.video_stream.pix_fmt = "yuv444p"
            self.video_stream.options = {  # type: ignore
                "qp": "0",
                "preset:v": "p7",
                "tune:v": "lossless",
            }

    def write(
        self,
        image: npt.NDArray[np.uint8],
        time: Optional[float] = None,
        pix_fmt: Optional[PixelFormat] = None,
    ) -> None:
        """Write an image to the video.

        Args:
        ----
            image: The image to write. Can have 1 or 3 channels.
            time: The time of the frame in seconds.
            pix_fmt: The pixel format of the image. If None, the pixel format will be
                `gray` for 1-channel images and `bgr24` for 3-channel images.

        """
        if time is not None:
            raise NotImplementedError("Time is not yet supported")

        if self.video_stream.encoded_frame_count == 0:  # type: ignore
            if image.ndim == 2:
                height, width = image.shape
            elif image.ndim == 3:
                if image.shape[0] == 3:
                    _, height, width = image.shape
                else:
                    height, width, _ = image.shape
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")

            self.video_stream.codec_context.width = width
            self.video_stream.codec_context.height = height

        if pix_fmt is None:
            pix_fmt = "bgr24"
            if image.ndim == 2:
                pix_fmt = "gray"

        frame = av.VideoFrame.from_ndarray(image, str(pix_fmt))  # type: av.VideoFrame
        # if time is not None:
        #     frame.time = time

        packet = self.video_stream.encode(frame)
        self.container.mux(packet)

    def __enter__(self) -> "Writer":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self.container.mux(self.video_stream.encode(None))
        self.container.close()
