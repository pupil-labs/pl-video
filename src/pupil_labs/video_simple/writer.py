from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import av
import numpy as np
import numpy.typing as npt


@dataclass
class Writer:
    path: str | Path
    lossless: bool = False
    rate: int = 30

    def __post_init__(self) -> None:
        self.container = av.open(self.path, "w")
        self.video_stream = self.container.add_stream("h264", rate=self.rate)

        if self.lossless:
            self.video_stream.pix_fmt = "yuv444p"
            self.video_stream.options = {
                "qp": "0",
                "preset:v": "p7",
                "tune:v": "lossless",
            }

    def write(
        self,
        image: npt.NDArray[np.uint8],
        time: Optional[float] = None,
        pix_fmt: Optional[str] = None,
    ) -> None:
        if self.video_stream.encoded_frame_count == 0:
            self.video_stream.codec_context.width = image.shape[1]
            self.video_stream.codec_context.height = image.shape[0]

        if pix_fmt is None:
            pix_fmt = "bgr24"
            if image.ndim == 2:
                pix_fmt = "gray"

        frame = av.VideoFrame.from_ndarray(image, pix_fmt)  # type: av.VideoFrame
        # if time is not None:
        #     frame.time = time

        packet = self.video_stream.encode(frame)
        self.container.mux(packet)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.container.close()
