from pathlib import Path
from typing import Any

import pytest

ROOT_PATH = Path(__file__).parent.parent
TEST_DATA_PATH = ROOT_PATH / "tests" / "data"


@pytest.fixture
def test_data_path() -> Path:
    return TEST_DATA_PATH


def pytest_generate_tests(metafunc: Any) -> None:
    # video_paths = metafunc.config.getoption("video_path")

    # main_video = TEST_DATA_PATH / "old/world.mp4"
    main_video = TEST_DATA_PATH / "Neon Scene Camera - audio off.mp4"

    videos_with_audio = [
        TEST_DATA_PATH / "Neon Scene Camera - audio on.mp4",
        TEST_DATA_PATH / "PI Scene Camera - audio on.mp4",
    ]

    videos_mjpeg = [
        TEST_DATA_PATH / "eye.mjpeg",
    ]

    multi_part_videos = [
        [
            TEST_DATA_PATH / "multi-part/PI world v1 ps1.mp4",
            TEST_DATA_PATH / "multi-part/PI world v1 ps2.mp4",
            TEST_DATA_PATH / "multi-part/PI world v1 ps3.mp4",
        ],
        [
            TEST_DATA_PATH / "multi-part/Neon Scene Camera v1 ps1.mp4",
            TEST_DATA_PATH / "multi-part/Neon Scene Camera v1 ps2.mp4",
            TEST_DATA_PATH / "multi-part/Neon Scene Camera v1 ps3.mp4",
        ],
    ]

    videos_other = [
        TEST_DATA_PATH / "Neon Sensor Module.mp4",
        TEST_DATA_PATH / "PI Eye Camera.mp4",
        TEST_DATA_PATH / "PI Scene Camera - audio off.mp4",
    ]

    standard_videos = [main_video, *videos_with_audio, *videos_other, *videos_mjpeg]
    if "num_frames" in metafunc.fixturenames:
        assert "video_path" in metafunc.fixturenames

        num_frames_map = {
            TEST_DATA_PATH / "old/world.mp4": 518,
            TEST_DATA_PATH / "old/world-audio.mp4": 245,
            TEST_DATA_PATH / "old/eye.mjpeg": 3868,
            TEST_DATA_PATH / "old/eye.mp4": 3868,
        }

        videos = [video for video in standard_videos if video in num_frames_map]
        metafunc.parametrize(
            "video_path, num_frames",
            [(video, num_frames_map[video]) for video in videos],
        )

    elif "video_path" in metafunc.fixturenames:
        metafunc.parametrize("video_path", standard_videos)
    elif "video_with_audio_path" in metafunc.fixturenames:
        metafunc.parametrize("video_with_audio_path", videos_with_audio)
    elif "video_mjpeg_path" in metafunc.fixturenames:
        metafunc.parametrize("video_mjpeg_path", videos_mjpeg)
    elif "main_video_path" in metafunc.fixturenames:
        metafunc.parametrize("main_video_path", [main_video])
    elif "multi_part_video_paths" in metafunc.fixturenames:
        metafunc.parametrize("multi_part_video_paths", multi_part_videos)


# indexing video stream frames is broken


# @dataclass
# class VideoTestCase:
#     video_path: Path
#     num_video_frames: int


# def pytest_addoption(parser):
#     parser.addoption(
#         "--video-path",
#         action="append",
#         default=[],
#         help="list of video_path to pass to test functions",
#     )


# def pytest_generate_tests(metafunc):
#     if "video_path" in metafunc.fixturenames:
#         video_paths = metafunc.config.getoption("video_path")
#         if not video_paths:
#             video_paths = [
#                 TEST_DATA_PATH / "world-audio.mp4",
#                 TEST_DATA_PATH / "eye.mjpeg",
#                 TEST_DATA_PATH / "eye.mp4",
#                 TEST_DATA_PATH / "world.mp4",
#             ]

#         metafunc.parametrize("video_path", video_paths)


# video_test_cases = [
#     VideoTestCase(TEST_DATA_PATH / video_path, frames)
#     for video_path, frames in [
#         ("world.mp4", 518),
#         ("world-audio.mp4", 245),
#         ("eye.mjpeg", 3868),
#         ("eye.mp4", 3868),
#     ]
# ]
