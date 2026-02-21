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
        TEST_DATA_PATH / "invalid_format_detect.mjpeg",
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
        TEST_DATA_PATH / "out_of_order_pts.mp4",
        TEST_DATA_PATH / "duplicate_pts.mp4",
        TEST_DATA_PATH / "silently_skipped_corrupt_frame.mp4",
    ]

    standard_videos = [main_video, *videos_with_audio, *videos_other, *videos_mjpeg]
    if "video_path" in metafunc.fixturenames:
        metafunc.parametrize("video_path", standard_videos)
    elif "video_with_audio_path" in metafunc.fixturenames:
        metafunc.parametrize("video_with_audio_path", videos_with_audio)
    elif "video_mjpeg_path" in metafunc.fixturenames:
        metafunc.parametrize("video_mjpeg_path", videos_mjpeg)
    elif "main_video_path" in metafunc.fixturenames:
        metafunc.parametrize("main_video_path", [main_video])
    elif "multi_part_video_paths" in metafunc.fixturenames:
        metafunc.parametrize("multi_part_video_paths", multi_part_videos)
