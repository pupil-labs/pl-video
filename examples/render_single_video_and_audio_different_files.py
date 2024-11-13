from heapq import merge
from pathlib import Path

from tqdm import tqdm

import pupil_labs.video as plv
from pupil_labs.video.frame import VideoFrame  # import plv also works

out_path = "out.mp4"
video_path = Path("/recs/neon/long/Neon Scene Camera v1 ps1.mp4")
video2_path = Path("/recs/neon/long/Neon Scene Camera v1 ps2.mp4")
with (
    plv.Reader(source=video_path, stream="video") as reader1,
    plv.Reader(source=video2_path, stream="audio") as reader2,
    plv.Writer(out_path) as writer,
):
    progress = tqdm()
    for frame in merge(reader1, reader2, key=lambda f: f.time):
        if isinstance(frame, VideoFrame):
            progress.update()
        writer.write_frame(frame)
