import time

import av
import numpy as np

# import pupil_labs.video as plv
from pupil_labs.video_simple.reader import Reader
import pytest
import tqdm
# from pupil_labs.video.stream.base import StreamActionCounters
# from pupil_labs.video.stream.container import StreamsContainer

from .utils import measure_fps


@pytest.fixture
def correct_pts(video_path):
    correct_pts = []
    for packet in av.open(str(video_path)).demux(video=0):
        if packet.pts is None:
            continue
        correct_pts.append(packet.pts)
    return correct_pts


@pytest.fixture
def correct_dts(video_path):
    correct_dts = []
    for packet in av.open(str(video_path)).demux(video=0):
        if packet.pts is None:
            continue
        correct_dts.append(packet.dts)
    return correct_dts


@pytest.fixture
def reader(video_path):
    return Reader(video_path)


def test_pts(reader: Reader, correct_pts):
    assert reader.pts == correct_pts


# def test_dts(reader: plv.InputContainer, correct_dts):
#     assert reader.streams.video[0].dts == correct_dts


def test_iteration(reader: Reader, correct_pts):
    frame_count = 0
    for frame, expected_pts in measure_fps(zip(reader, correct_pts)):
        assert frame.pts == expected_pts
        frame_count += 1

    assert reader.stats.seeks == 0
    assert frame_count == len(correct_pts)


def test_by_idx(reader: Reader, correct_pts):
    frame_count = 0
    for i, expected_pts in measure_fps(enumerate(correct_pts)):
        frame = reader.by_idx[i]
        assert frame.pts == expected_pts
        frame_count += 1

    assert reader.stats.seeks == 0
    assert frame_count == len(correct_pts)


def test_by_pts(reader: Reader, correct_pts):
    for expected_pts in measure_fps(correct_pts):
        frame = reader.by_pts[expected_pts]
        assert frame.pts == expected_pts

    assert reader.stats.seeks == 0


def test_seek_avoidance(reader: Reader):
    assert reader.stats.seeks == 0
    assert reader.stats.decodes == 0

    # we dont need to seek when loading the first frame
    reader.by_idx[0]
    assert reader.stats.seeks == 0
    assert reader.stats.decodes == 1

    # a second access will load the frame from buffer and not seek/decode
    reader.by_idx[0]
    assert reader.stats.seeks == 0
    assert reader.stats.decodes == 1

    # getting the second frame will also not require a seek
    reader.by_idx[1]
    assert reader.stats.seeks == 0
    assert reader.stats.decodes == 2

    # getting the 10th frame will require a seek, but the rest of the slice will not
    reader.by_idx[10:20]
    assert reader.stats.seeks == 1
    assert reader.stats.decodes == 3


# def test_external_timestamps(reader: plv.InputContainer):
#     video = reader.streams.video[0]
#     video.external_timestamps = [0.5 * i for i in range(len(video))]
#     video.seek_to_timestamp(0.3)
#     frames = iter(video.frames)
#     frame = next(frames)
#     assert frame.external_timestamp == 0.0
#     frame = next(frames)
#     assert frame.external_timestamp == 0.5
#     frame = next(frames)
#     frame = next(frames)
#     assert frame.external_timestamp == 1.5
#     video.seek_to_timestamp(0.1)
#     frame = next(frames)
#     assert frame.index == 0
#     assert frame.external_timestamp == 0.0

#     video.external_timestamps = [5 * i for i in range(len(video))]
#     video.seek_to_timestamp(3)
#     frame = next(frames)
#     assert frame.external_timestamp == 0
#     frame = next(frames)
#     assert frame.external_timestamp == 5
#     frame = next(frames)
#     assert frame.external_timestamp == 10
#     frame = next(frames)
#     assert frame.external_timestamp == 15
#     video.seek_to_timestamp(10.2)
#     frame = next(frames)
#     assert frame.index == 2
#     assert frame.external_timestamp == 10


# def test_seek_and_then_index(reader: Reader):
#     video = reader.streams.video[0]
#     frames = iter(video.frames)
#     video.seek_to_index(2)
#     frame = next(frames)
#     assert frame.index == 2


# def test_video_class(reader: plv.InputContainer):
#     video = reader.streams.video[0]
#     assert len(video.pts) > 0
#     assert len(video.dts) > 0


# def test_iteration(reader: plv.InputContainer, correct_pts):
#     video = reader.streams.video[0]
#     for i, frame in enumerate(video.frames):
#         assert frame.pts == correct_pts[i]


# def test_arbitrary_index(reader: plv.InputContainer, correct_pts):
#     video = reader.streams.video[0]
#     for i in [0, 1, 2, 10, 20]:
#         assert video.frames[i].pts == correct_pts[i]
#     for i in [-1, -10, -20]:
#         assert video.frames[i].pts == correct_pts[i]


# def test_arbitrary_slices(reader: plv.InputContainer, correct_pts):
#     video = reader.streams.video[0]
#     assert video.frames[100:101].pts == correct_pts[100:101]
#     assert video.frames[10:20].pts == correct_pts[10:20]
#     assert video.frames[20:30].pts == correct_pts[20:30]
#     assert video.frames[5:8].pts == correct_pts[5:8]


# class timeblock:
#     def __init__(self, name):
#         self.name = name

#     def __enter__(self):
#         self.start = time.time()

#     def __exit__(self, *args):
#         print(self.name, str(time.time() - self.start))


# def test_decode_speed():
#     """
#     Uses a fake video container to test that the speed of iterating a video
#     isn't slow / gets slower over time with longer recordings.
#     """

#     num_dummy_frames = 50000

#     class FakeCodecContext:
#         type = "video"

#     class FakeStream:
#         frames = num_dummy_frames
#         index = 0
#         thread_type = "AUTO"
#         codec_context = FakeCodecContext
#         stats = StreamActionCounters()
#         type = "video"
#         time_base = 1

#         def __hash__(self):
#             return 1234

#         def __eq__(self, other):
#             return True

#     dummy_stream = FakeStream()

#     class FakeStreamContainer(list):
#         def get(self, *args, **kw):
#             return [dummy_stream]

#         video = [dummy_stream]

#     class FakeFrameContainer:
#         name = "fakeframegenerator"

#         def demux(self, *args, **kwargs):
#             class Packet:
#                 pass

#             class Frame:
#                 pass

#             for i in range(num_dummy_frames):
#                 packet = Packet()
#                 packet.pts = i
#                 packet.dts = i
#                 packet.time_base = 1
#                 packet.duration = 1
#                 packet.is_keyframe = 1
#                 packet.stream = dummy_stream
#                 frame = Frame()
#                 frame.pts = i
#                 frame.time = i
#                 frame.time_base = 1
#                 packet.decode = lambda: [frame]
#                 yield packet

#         def seek(self, *args, **kwargs):
#             pass

#         streams = FakeStreamContainer([dummy_stream])

#     video = plv.InputContainer(FakeFrameContainer())

#     last_now = time.monotonic_ns()
#     processing_durations = []
#     for frame in tqdm.tqdm(video):
#         now = time.monotonic_ns()
#         processing_durations.append(now - last_now)
#         last_now = now
#     average_fps = 1e9 / np.mean(processing_durations)

#     # This is a pretty arbitrary test and may break on slower computers
#     # it checks the speed of decoding video isn't getting slower get slower if
#     # we accidentally do something like linear instead of binary search
#     # usually it's over 50k fps on a fast machine when all is good
#     assert average_fps > 50000
