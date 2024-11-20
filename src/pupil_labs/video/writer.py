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
import av.error
import av.packet
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
        encoder (string): eg. 'h264_nvenc'

    Returns:
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
        """Video writer for creating videos from image arrays.

        Args:
            path: The path to write the video to.
            lossless: If True, the video will be encoded in lossless H264.
            fps: The desired framerate of the video.
            bit_rate: The desired bit rate of the video.
            logger: Python logger to use. Decreases performance.

        """
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
        if self.fps is not None and time is not None:
            raise ValueError("Can't provide time argument when fps is set!")

        if isinstance(frame, (AudioFrame, VideoFrame)):
            av_frame = frame.av_frame
            if time is None:
                time = frame.time
        elif isinstance(frame, (av.audio.frame.AudioFrame, av.video.frame.VideoFrame)):
            av_frame = frame
            if time is None:
                time = av_frame.time
        else:
            raise TypeError(f"invalid frame: {frame}")

        if isinstance(frame, av.video.frame.VideoFrame):
            time_base = self.video_stream.codec_context.time_base
        else:
            time_base = self.audio_stream.codec_context.time_base

        if time is not None and time_base is not None:
            av_frame.pts = int(time / time_base)
        self._encode_av_frame(av_frame)

    def write_image(
        self,
        image: npt.NDArray[np.uint8],
        time: Optional[float] = None,
        pix_fmt: Optional[PixelFormat] = None,
    ) -> None:
        """Write an image to the video.

        Args:
            image: The image to write. Can have 1 or 3 channels.
            time: The time of the frame in seconds.
            pix_fmt: The pixel format of the image. If None, the pixel format will be
                `gray` for 1-channel images and `bgr24` for 3-channel images.

        """
        if pix_fmt is None:
            pix_fmt = "bgr24"
            if image.ndim == 2:
                pix_fmt = "gray"

        frame = av.VideoFrame.from_ndarray(image, str(pix_fmt))
        self.write_frame(frame, time=time)

    @cached_property
    def video_stream(self) -> av.video.stream.VideoStream:
        # TODO(dan): what about mjpeg?

        h264_nvenc_error = check_pyav_video_encoder_error("h264_nvenc")
        if h264_nvenc_error:
            self.logger.warning(
                "could not add stream with encoder 'h264_nvenc'"
                f"using libx264 instead. Error was: {h264_nvenc_error}"
            )
            encoder_name = "h264"
        else:
            encoder_name = "h264_nvenc"

        if self.fps is None:
            stream = self.container.add_stream(encoder_name)
        else:
            stream = self.container.add_stream(encoder_name, rate=self.fps)
        stream = cast(av.video.stream.VideoStream, stream)

        stream.codec_context.bit_rate = self.bit_rate
        stream.codec_context.pix_fmt = "yuv420p"
        if self.fps is None:
            stream.codec_context.time_base = Fraction(1, 90000)

        # h264_nvenc encoder seems to encode at a different bitrate to requested,
        # multiplying by 10 and dividing by 8 seems to fix it (maybe it's a matter
        # issue of bits vs bytes somewhere in the encoder...)
        if stream.name == "h264_nvenc":
            stream.codec_context.bit_rate = int(stream.codec_context.bit_rate / 1.25)

        # Move atom to start so less requests when loading video in web
        stream.codec_context.options["movflags"] = "faststart"

        # bufsize at 2x bitrate seems to give better overall quality
        stream.codec_context.options["bufsize"] = f"{2 * self.bit_rate / 1000}k"

        # b frames can cause certain frames in chrome to not be seeked to correctly
        # https://bugs.chromium.org/p/chromium/issues/detail?id=66631
        stream.codec_context.options["bf"] = "0"

        # group of pictures size
        # stream.codec_context.options["g"] = str(self.group_of-picture_size)

        if self.lossless:
            stream.codec_context.pix_fmt = "yuv444p"
            stream.codec_context.options.update({
                "qp": "0",
                "preset:v": "p7",
                "tune:v": "lossless",
            })
        return stream

    @cached_property
    def _video_frame_buffer(self) -> deque[av.video.frame.VideoFrame]:
        return deque()

    @cached_property
    def _audio_frame_buffer(self) -> deque[av.audio.frame.AudioFrame]:
        return deque()

    @cached_property
    def _av_frame_buffer(
        self,
    ) -> deque[av.audio.frame.AudioFrame | av.video.frame.VideoFrame]:
        return deque()

    def _encode_av_audio_frame(
        self, av_frame: av.audio.frame.AudioFrame | None
    ) -> None:
        self._packet_buffer.extend(self.audio_stream.encode(av_frame))
        self._mux_packets()

    def _encode_av_video_frame(
        self, av_frame: av.video.frame.VideoFrame | None
    ) -> None:
        if av_frame and self.video_stream.encoded_frame_count == 0:  # type: ignore
            self.video_stream.codec_context.width = av_frame.width
            self.video_stream.codec_context.height = av_frame.height

        self._packet_buffer.extend(self.video_stream.encode(av_frame))
        self._mux_packets()

    def _mux_packets(self, min_packets: int = 200) -> None:
        if len(self._packet_buffer) < min_packets:
            return

        for packet in self._packet_buffer:
            if packet.pts is not None:
                packet.dts = packet.pts
            # print(
            #     packet.stream.type,
            #     packet,
            #     packet.time_base,
            #     float(packet.time_base * packet.pts),
            # )

        self.container.mux(self._packet_buffer)
        self._packet_buffer.clear()

    def _encode_av_frame(
        self, av_frame: av.video.frame.VideoFrame | av.audio.frame.AudioFrame
    ) -> None:
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
    def _packet_buffer(self) -> deque[av.packet.Packet]:
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

        self._mux_packets(0)
        self.container.close()
