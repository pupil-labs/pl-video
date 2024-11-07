from collections import deque
from fractions import Fraction
from functools import cached_property
from logging import Logger, getLogger
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import TracebackType
from typing import Optional, cast

import av
import av.audio
import av.stream
import av.video
import numpy as np
import numpy.typing as npt

from pupil_labs.video.frame import AudioFrame, PixelFormat, VideoFrame

av.logging.set_level(av.logging.VERBOSE)

DEFAULT_LOGGER = getLogger(__name__)


def check_pyav_video_encoder_error(encoder: str) -> str:
    """Raise error if pyav can't write with the passed in encoder

    Tries to run an encoding of a video using `encoder` since sometimes running in a
    docker container that doesn't provide gpu support but *does* have h264_nvenc
    codec available will still fail with: libav.h264_nvenc: Cannot load libcuda.so.1

    Args:
    ----
        encoder (string): eg. 'h264_nvenc'

    Returns:
    -------
        Empty string if encoding worked, error string if it failed


    """
    with NamedTemporaryFile(suffix=".mp4") as fp:
        container = av.open(fp.name, "w")
        try:
            video_stream = container.add_stream(encoder)
            video_stream.encode(None)  # type: ignore
        except Exception as e:
            return str(e)
    return ""


class Writer:
    def __init__(
        self,
        path: str | Path,
        lossless: bool = False,
        fps: int | None = None,
        bit_rate: int = 2_000_000,
        logger: Logger | None = None,
    ) -> None:
        self.path = path
        self.lossless = lossless
        self.fps = fps
        self.bit_rate = bit_rate
        self.logger = logger or DEFAULT_LOGGER
        self.container = av.open(self.path, "w")

    def write_frame(
        self,
        frame: av.audio.frame.AudioFrame
        | av.video.frame.VideoFrame
        | AudioFrame
        | VideoFrame,
        time: Optional[float] = None,
    ) -> None:
        if isinstance(frame, (AudioFrame, VideoFrame)):
            self._encode_av_frame(frame.av_frame)
        elif isinstance(frame, (av.audio.frame.AudioFrame, av.video.frame.VideoFrame)):
            self._encode_av_frame(frame)
        else:
            raise TypeError(f"invalid frame: {frame}")

    def write_image(
        self,
        image: npt.NDArray[np.uint8],
        time: Optional[float] = None,
        pix_fmt: Optional[PixelFormat] = None,
    ) -> None:
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

        frame = av.VideoFrame.from_ndarray(image, str(pix_fmt))
        if time is not None:
            frame.pts = int(time / self.video_stream.codec_context.time_base)
        self.write_frame(frame)

    @cached_property
    def video_stream(self) -> av.video.stream.VideoStream:
        # TODO(dan): what about mjpeg?

        h264_nvenc_error = check_pyav_video_encoder_error("h264_nvenc")
        if h264_nvenc_error:
            self.logger.warning(
                "could not add stream with encoder 'h264_nvenc'"
                f"using libx264 instead. Error was: {h264_nvenc_error}"
            )
            stream = self.container.add_stream("h264")
        else:
            stream = self.container.add_stream("h264_nvenc")

        stream = cast(av.video.stream.VideoStream, stream)
        stream.codec_context.time_base = Fraction(1, 90000)
        stream.codec_context.bit_rate = self.bit_rate
        stream.codec_context.pix_fmt = "yuv420p"

        # h264_nvenc encoder seems to encode at a different bitrate to requested,
        # multiplying by 10 and dividing by 8 seems to fix it (maybe it's a matter
        # issue of bits vs bytes somewhere in the encoder...)
        if stream.name == "h264_nvenc":
            stream.codec_context.bit_rate = int(stream.codec_context.bit_rate * 1.25)

        # Move atom to start so less requests when loading video in web
        stream.codec_context.options["movflags"] = "faststart"

        # bufsize at 2x bitrate seems to give better overall quality
        stream.codec_context.options["bufsize"] = f"{2 * self.bit_rate / 1000}k"

        # b frames can cause certain frames in chrome to not be seeked to correctly
        # https://bugs.chromium.org/p/chromium/issues/detail?id=66631
        stream.codec_context.options["bf"] = "0"

        if self.lossless:
            self.video_stream.codec_context.pix_fmt = "yuv444p"
            self.video_stream.codec_context.options.update({
                "qp": "0",
                "preset:v": "p7",
                "tune:v": "lossless",
            })
        return stream

    @cached_property
    def _video_frame_buffer(self) -> deque[av.video.frame.VideoFrame]:
        return deque()

    def _encode_av_audio_frame(
        self, av_frame: av.audio.frame.AudioFrame | None
    ) -> None:
        # if not hasattr(self, "first_frame"):
        #     av_frame.pts = 0
        #     self.first_frame = True
        # TODO(dan): probably need to set av_frame.rate = self.rate here
        if av_frame is not None:
            av_frame.dts = None
            self._audio_frame_buffer.append(av_frame)
            av_frame.dts = av_frame.pts = int(
                av_frame.time / self.video_stream.codec_context.time_base
            )

        packets = self.audio_stream.encode(av_frame)
        for packet in packets:
            if not self._audio_frame_buffer:
                break
            packet_frame = self._audio_frame_buffer.popleft()
            if packet.pts < 0:
                continue
            packet.dts = None
            packet.pts = packet_frame.pts
            # print(packet)
            self.container.mux([packet])

    def _encode_av_video_frame(
        self, av_frame: av.video.frame.VideoFrame | None
    ) -> None:
        if av_frame is not None:
            av_frame.dts = av_frame.pts = int(
                av_frame.time / self.video_stream.codec_context.time_base
            )
            self._video_frame_buffer.append(av_frame)

        packets = self.video_stream.encode(av_frame)
        for packet in packets:
            if not self._video_frame_buffer:
                break
            packet_frame = self._video_frame_buffer.popleft()
            packet.dts = packet.pts = packet_frame.pts
            self.container.mux([packet])

    def _encode_av_frame(
        self, av_frame: av.video.frame.VideoFrame | av.audio.frame.AudioFrame
    ) -> None:
        # print(av_frame)
        if isinstance(av_frame, av.video.frame.VideoFrame):
            return self._encode_av_video_frame(av_frame)
        elif isinstance(av_frame, av.audio.frame.AudioFrame):
            return self._encode_av_audio_frame(av_frame)
        else:
            raise TypeError(f"invalid av frame: {av_frame}")

    @cached_property
    def audio_stream(self) -> av.audio.stream.AudioStream:
        stream = self.container.add_stream("aac")
        stream = cast(av.audio.stream.AudioStream, stream)
        stream.codec_context.time_base = Fraction(1, 90000)
        # stream.codec_context.rate = 48000
        # stream.codec_context.bit_rate = 64000
        return stream

    @cached_property
    def _audio_frame_buffer(self) -> deque[av.audio.frame.AudioFrame]:
        return deque()

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
        if self.container.streams.audio:
            self._encode_av_audio_frame(None)

        if self.container.streams.video:
            self._encode_av_video_frame(None)

        self.container.close()
