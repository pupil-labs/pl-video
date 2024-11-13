from pathlib import Path

from tqdm import tqdm

import pupil_labs.video as plv  # import plv also works

out_path = "out.mp4"
video_path = Path("/recs/neon/long/Neon Scene Camera v1 ps1.mp4")
with (
    plv.Reader(source=video_path) as reader,
    plv.Writer(out_path) as writer,
):
    # on a RTX3060 neon scene video renders @ ~500fps
    for frame in tqdm(reader):
        writer.write_frame(frame)
