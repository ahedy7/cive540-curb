"""
Shared helpers for the video pipeline: paths, clip metadata, GPS tracks.

Nothing here assumes a particular phone, tracker app or file name. Clips are any .MOV or
.MP4 in video/. Tracks are any .gpx in video/ (GPX 1.0 or 1.1, any app), or a .csv of
manual anchors with columns time (ISO 8601 with time zone), lat, lon.
"""
import datetime as dt
import re
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "video"
FRAMES = ROOT / "frames"
MANIFEST = ROOT / "pipeline" / "clips.csv"
DERIVED = ROOT / "data" / "derived"
LOCAL_TZ = "America/Montreal"
MOVIE_EXT = {".mov", ".mp4", ".m4v"}


def ffmpeg_exe():
    exe = shutil.which("ffmpeg") or next((p for p in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
                                          if Path(p).exists()), None)
    if not exe:
        raise SystemExit("ffmpeg not found. Install it with: brew install ffmpeg")
    return exe


# ---------------------------------------------------------------- clip metadata
def _boxes(f, start, end):
    """Walk the boxes (atoms) of a QuickTime/MP4 file between two byte offsets."""
    pos = start
    while pos < end:
        f.seek(pos)
        head = f.read(8)
        if len(head) < 8:
            return
        size, kind = struct.unpack(">I4s", head)
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            hdr = 16
        elif size == 0:
            size = end - pos
        yield kind.decode("latin1"), pos + hdr, pos + size
        pos += size


def clip_info(path):
    """Start time (UTC), duration and phone model, read from the file header.
    Prefers Apple's creationdate (local time with offset, set when recording starts).
    Falls back to the container's creation time, which is UTC."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        moov = next((b for b in _boxes(f, 0, f.tell()) if b[0] == "moov"), None)
        if moov is None:
            raise ValueError(f"{path.name}: no moov box, not a QuickTime/MP4 file?")
        start = duration = None
        model = ""
        for kind, s, e in _boxes(f, moov[1], moov[2]):
            f.seek(s)
            if kind == "mvhd":
                version = f.read(4)[0]
                if version == 0:
                    created, _, scale, length = struct.unpack(">IIII", f.read(16))
                else:
                    created, _, scale, length = struct.unpack(">QQIQ", f.read(28))
                start = pd.Timestamp(dt.datetime(1904, 1, 1) + dt.timedelta(seconds=created), tz="UTC")
                duration = length / scale
            elif kind in ("meta", "udta"):
                text = f.read(e - s).decode("latin1")
                m = re.search(r"(20\d\d-\d\d-\d\dT\d\d:\d\d:\d\d)([-+]\d\d):?(\d\d)", text)
                if m:
                    start = pd.Timestamp(f"{m.group(1)}{m.group(2)}:{m.group(3)}").tz_convert("UTC")
                m = re.search(r"(iPhone[ \w]{0,20}|Pixel[ \w]{0,20}|SM-\w+)", text)
                if m and not model:
                    model = m.group(1).strip()
    return {"start_utc": start, "duration_s": round(duration, 2), "model": model}


# ---------------------------------------------------------------- tracks
def read_track(path):
    """GPS fixes as a DataFrame with time (UTC), lat, lon, sorted and de-duplicated."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        t = pd.read_csv(path)
        t["time"] = pd.to_datetime(t.time, utc=True)
    else:
        rows = []
        for el in ET.parse(path).getroot().iter():
            if el.tag.rsplit("}", 1)[-1] not in ("trkpt", "rtept", "wpt"):
                continue
            when = next((c.text for c in el if c.tag.rsplit("}", 1)[-1] == "time"), None)
            if when:
                rows.append((when, float(el.get("lat")), float(el.get("lon"))))
        t = pd.DataFrame(rows, columns=["time", "lat", "lon"])
        t["time"] = pd.to_datetime(t.time, utc=True, format="ISO8601")
    return t.sort_values("time").drop_duplicates("time").reset_index(drop=True)[["time", "lat", "lon"]]


def load_manifest():
    if not MANIFEST.exists():
        raise SystemExit("No clips manifest yet. Run: python pipeline/ingest.py")
    m = pd.read_csv(MANIFEST, dtype={"date": str, "gpx": str, "note": str})
    m["start_utc"] = pd.to_datetime(m.start_utc, utc=True)
    m["gpx"] = m.gpx.fillna("")
    m["offset_s"] = m.offset_s.fillna(0.0)
    return m
