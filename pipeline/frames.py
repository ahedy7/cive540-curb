"""
Step 2, frames: extract 1 frame per second from every clip in pipeline/clips.csv with use = yes.

Frames go to frames/<file stem>/000001.jpg, 000002.jpg, ... Frame k is taken at k-1 seconds
into the clip, so its time is clip start + offset_s + (k - 1). Folders are named after the
original file so they stay valid if a clip's side or ID is edited later.

Clips whose folder already holds the expected number of frames are skipped.
Decoding uses the Mac's hardware decoder (VideoToolbox) when available.

Usage:
    python pipeline/frames.py                  # all clips
    python pipeline/frames.py IMG_4730.MOV     # just these files
"""
import math
import subprocess
import sys

from common import FRAMES, VIDEO, ffmpeg_exe, load_manifest

FPS = 1
JPEG_Q = 3   # ffmpeg scale 2 (best) to 31. 3 is visually the same as 2 at about 70% of the size


def extract(file, duration_s):
    out = FRAMES / file.rsplit(".", 1)[0]
    expected = math.floor(duration_s * FPS)
    if out.exists() and len(list(out.glob("*.jpg"))) >= expected - 1:
        print(f"{file}: already done ({expected} frames)")
        return
    out.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-stats",
           "-hwaccel", "videotoolbox", "-i", str(VIDEO / file),
           "-vf", f"fps={FPS}:round=down", "-q:v", str(JPEG_Q), "-start_number", "1",
           str(out / "%06d.jpg")]
    print(f"{file}: extracting about {expected} frames")
    subprocess.run(cmd, check=True)
    n = len(list(out.glob("*.jpg")))
    size = sum(p.stat().st_size for p in out.glob("*.jpg")) / 1e9
    print(f"{file}: {n} frames, {size:.2f} GB")


def main():
    m = load_manifest()
    m = m[m.use == "yes"]
    if len(sys.argv) > 1:
        m = m[m.file.isin(sys.argv[1:])]
    for r in m.itertuples():
        extract(r.file, r.duration_s)


if __name__ == "__main__":
    main()
