import warnings
from enum import Enum
import av
import numpy as np
import numpy.typing as npt


class PixelFormat(Enum):
    gray = "gray"
    bgr24 = "bgr24"
    rgb24 = "rgb24"
    yuv420p = "yuv420p"
    yuv444p = "yuv444p"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class VideoFrame:
    def __init__(self, frame: av.VideoFrame, index: int):
        self.av_frame = frame
        self._index = index

    # @property
    # def dts(self) -> int:
    #     return self.av_frame.dts

    @property
    def index(self) -> int:
        return self._index

    # @index.setter
    # def index(self, index: int):
    #     self._index = index

    @property
    def pts(self) -> int | None:
        return self.av_frame.pts

    # @pts.setter
    # def pts(self, pts: int | None):
    #     self.av_frame.pts = pts

    # @property
    # def side_data(self) -> av.sidedata.sidedata.SideDataContainer:
    #     """
    #     Side data of the frame
    #     """
    #     return self.av_frame.side_data

    # @property
    # def time(self) -> float | None:
    #     """
    #     Offset in seconds of the frame in the stream
    #     """
    #     av_frame_time = self.av_frame.time
    #     if av_frame_time is not None:
    #         return av_frame_time
    #     if self.time_base and self.pts is not None:
    #         return float(self.pts * self.time_base)
    #     return None

    # @time.setter
    # def time(self, value: float | None = None) -> None:
    #     if value is None:
    #         if self.time_base is not None:
    #             raise ValueError("frame.time can not be None when time_base is set")
    #         self.pts = None
    #     else:
    #         if self.time_base is None:
    #             if self.stream and self.stream.time_base:
    #                 self.time_base = self.stream.time_base
    #             if self.time_base is None:
    #                 raise ValueError("time_base not known, can't set frame.time")
    #         self.pts = int(value / self.time_base)

    # offset_secs = time

    # @property
    # def time_base(self) -> Fraction | None:
    #     """
    #     Returns the time_base of this frame
    #     """
    #     time_base = self.av_frame.time_base
    #     # if time_base is None:
    #     #     if self.stream:
    #     #         if self.stream.time_base:
    #     #             time_base = self.stream.time_base
    #     #         elif self.stream.average_rate:
    #     #             time_base = Fraction(1, self.stream.average_rate)
    #     return time_base

    # @time_base.setter
    # def time_base(self, value) -> None:
    #     self.av_frame.time_base = value

    # def __repr__(self):

    @property
    def gray(self) -> npt.NDArray[np.uint8]:
        """
        Numpy image array in gray format
        """
        return self.to_ndarray(PixelFormat.gray)

    @property
    def bgr(self) -> npt.NDArray[np.uint8]:
        """
        Numpy image array in BGR format
        """
        return self.to_ndarray(PixelFormat.bgr24)

    @property
    def rgb(self) -> npt.NDArray[np.uint8]:
        """
        Numpy image array in RGB format
        """
        return self.to_ndarray(PixelFormat.rgb24)

    def to_ndarray(self, pixel_format: PixelFormat) -> npt.NDArray[np.uint8]:
        """
        Returns an numpy array of the image for the frame
        """
        # TODO: add caching for decoded frames?
        return av_frame_to_ndarray_fast(self.av_frame, pixel_format)


def av_frame_to_ndarray_fast(
    av_frame: av.VideoFrame, pixel_format: PixelFormat
) -> npt.NDArray[np.uint8]:
    """
    Returns an image pixel numpy array for an av.VideoFrame in `format`
    skipping conversion by using buffers directly if possible for performance
    """
    if pixel_format == PixelFormat.gray:
        if av_frame.format.name == "gray":
            return np.frombuffer(av_frame.planes[0], np.uint8).reshape(
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
                    " Use .av_frame.to_ndarray(format='gray') for full range (4x slower)"
                )
                # av.to_ndarray(format='gray') returns 0-255 for gray values
                # but here reading the Y from planes for yuv420p the output
                # is limited from 16-235 instead of converted to full range 0-255
                # this is done for performance reasons
                # gray = limited_yuv420p_to_full(gray)

            return gray
    elif pixel_format in (
        PixelFormat.bgr24,
        PixelFormat.rgb24,
    ):  # TODO(dan): is this worth it?
        if av_frame.format.name == pixel_format:
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
            return image

    if pixel_format is not None:
        pixel_format = str(pixel_format)
    return av_frame.to_ndarray(format=pixel_format)
