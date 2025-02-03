import sys
from pathlib import Path

import cv2
import numpy as np

cv2.imshow("Init", np.zeros((10, 10)))
cv2.destroyAllWindows()

import pupil_labs.video as plv


def main(video_path: Path):
    # Open video file
    with plv.Reader(video_path) as video:
        # Iterate through video frames
        for frame in video:
            # Convert video frame to BGR array
            img = frame.bgr

            cv2.imshow("Video", img)
            cv2.pollKey()

        # Iterating in reverse order is also possible
        for frame in reversed(video):
            pass

        # Index individual frames or slices
        first_frame = video[0]
        last_frame = video[-1]
        frames = video[10:20]

        # Index frames by time
        ts = video[10].time
        frame = video.by_container_timestamps[ts]
        assert frame.time == ts and frame.index == 10

        frames = video.by_container_timestamps[ts : ts + 10]
        assert frames[0].time == ts and frames[0] == 10

        # Read video properties
        print(f"Video duration: {video.duration}")
        print(f"Video resolution: {video.width}x{video.height}")
        num_frames = len(video)
        print(f"Number of frames: {num_frames}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python flexible_video_access.py path/to/recording/folder")
    main(Path(sys.argv[1]))
