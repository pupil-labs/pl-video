import av


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
