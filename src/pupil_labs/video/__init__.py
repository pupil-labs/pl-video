"""pupil_labs.video"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from pupil_labs.video.array_like import ArrayLike
from pupil_labs.video.frame import AudioFrame, PixelFormat, ReaderFrameType, VideoFrame
from pupil_labs.video.indexing import Indexer
from pupil_labs.video.multi_reader import MultiReader, ReaderLike
from pupil_labs.video.reader import Reader
from pupil_labs.video.writer import Writer

__all__: list[str] = [
    "__version__",
    "ArrayLike",
    "AudioFrame",
    "MultiReader",
    "Indexer",
    "PixelFormat",
    "Reader",
    "VideoFrame",
    "Writer",
    "ReaderLike",
    "ReaderFrameType",
]
