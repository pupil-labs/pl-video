from heapq import merge
from pathlib import Path

from tqdm import tqdm

import pupil_labs.video as plv
from pupil_labs.video.frame import VideoFrame  # import plv also works

out_path = "out.mp4"
video_path = Path("/recs/neon/long/Neon Scene Camera v1 ps1.mp4")

with (
    plv.Reader(source=video_path, stream="video") as reader,
    plv.Writer(out_path) as writer,
):
    assert reader.audio
    progress = tqdm()
    for frame in merge(tqdm(reader), reader.audio, key=lambda frame: frame.time):
        if isinstance(frame, VideoFrame):
            progress.update()
        writer.write_frame(frame)
