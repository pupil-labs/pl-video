from dataclasses import dataclass, field

import av
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    index: int
    "Position of the frame in the video."

    time: float
    "Timestamp in seconds: the one passed to the reader, or `pts` in seconds."

    pts: int
    "Presentation timestamp in the stream's time base."

    _image: av.VideoFrame = field(repr=False)

    @property
    def bgr(self) -> npt.NDArray[np.uint8]:
        return self._image.to_ndarray(format="bgr24").astype(np.uint8, copy=False)

    @property
    def rgb(self) -> npt.NDArray[np.uint8]:
        return self._image.to_ndarray(format="rgb24").astype(np.uint8, copy=False)

    @property
    def gray(self) -> npt.NDArray[np.uint8]:
        """The luma plane, as stored: 16-235 for limited-range video."""
        plane = self._image.planes[0]
        rows = np.frombuffer(plane, np.uint8).reshape(-1, plane.line_size)
        return rows[: plane.height, : plane.width].copy()
