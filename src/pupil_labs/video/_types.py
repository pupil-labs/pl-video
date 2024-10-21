from typing import TypeVar

import av
import av.audio.stream
import av.subtitles.subtitle
import av.video.stream
import numpy as np
import numpy.typing as npt

AVFrameTypes = av.video.frame.VideoFrame | av.audio.frame.AudioFrame | av.subtitles.subtitle.SubtitleSet

FrameType = TypeVar("FrameType")
PTSArray = npt.NDArray[np.int64]
TimesArray = npt.NDArray[np.float64]
TimestampsArray = npt.NDArray[np.float64 | np.int64] | list[int | float]
