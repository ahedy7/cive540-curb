"""
Step 2b, sync: find the seconds to add to each clip's clock so its frames line up with the GPS.

Clip start times can be off by a few seconds (the phone clock, or a start time typed in from
the Photos info panel). At 20 km/h each second is about 5 m, enough to put a frame on the
wrong part of a block face.

How: when the car stops, consecutive frames barely change, and the GPS speed drops to zero.
For each clip with a track, motion = mean pixel change between consecutive frames (small
greyscale copies). Speed = GPS speed at each frame's time. The offset is the shift, within
+/- MAX_SHIFT_S, where motion and speed agree best (Spearman correlation).

A result is trusted when the correlation is at least MIN_CORR and the best shift beats every
shift more than 5 s away by at least MIN_MARGIN. Trusted offsets are written to offset_s in
pipeline/clips.csv (and noted). Others are only printed, to check by hand.

Output: pipeline/sync.csv, one row per clip, and the motion curves in data/derived/motion/.

Usage:
    python pipeline/sync.py            # all clips with a track
    python pipeline/sync.py --dry-run  # don't touch clips.csv
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image

from common import DERIVED, FRAMES, MANIFEST, ROOT, VIDEO, load_manifest, read_track

MAX_SHIFT_S = 90
MIN_CORR = 0.4
MIN_MARGIN = 0.05
THUMB = (160, 90)


def motion(stem, n):
    """Mean absolute change between frame k-1 and k, for k = 1..n (frame 1 gets NaN). Cached."""
    cache = DERIVED / "motion" / f"{stem}.npy"
    if cache.exists():
        m = np.load(cache)
        if len(m) == n:
            return m
    m = np.full(n, np.nan)
    prev = None
    for k in range(1, n + 1):
        im = Image.open(FRAMES / stem / f"{k:06d}.jpg")
        im.draft("L", THUMB)   # JPEG decodes straight to a small greyscale image, much faster
        a = np.asarray(im.convert("L").resize(THUMB), dtype=np.float32)
        if prev is not None:
            m[k - 1] = np.abs(a - prev).mean()
        prev = a
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, m)
    return m


def speed_at(track, t):
    """GPS speed (m/s) at times t (seconds since 1970), from positions 1 s either side."""
    pts = gpd.GeoSeries(gpd.points_from_xy(track.lon, track.lat), crs="EPSG:4326").to_crs("EPSG:32188")
    ts = (track.time - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    x, y = pts.x.to_numpy(), pts.y.to_numpy()
    sp = np.hypot(np.interp(t + 1, ts, x) - np.interp(t - 1, ts, x),
                  np.interp(t + 1, ts, y) - np.interp(t - 1, ts, y)) / 2
    sp[(t < ts[0]) | (t > ts[-1])] = np.nan
    return sp


def best_shift(mot, track, t0):
    k = np.arange(len(mot))
    shifts = np.arange(-MAX_SHIFT_S, MAX_SHIFT_S + 1)
    corr = np.full(len(shifts), np.nan)
    mrank = pd.Series(mot)
    for i, s in enumerate(shifts):
        sp = speed_at(track, t0 + s + k)
        ok = ~np.isnan(sp) & ~np.isnan(mot)
        if ok.sum() >= 60:
            corr[i] = mrank[ok].corr(pd.Series(sp)[ok], method="spearman")
    if np.isnan(corr).all():
        return None
    i = int(np.nanargmax(corr))
    far = np.abs(shifts - shifts[i]) > 5
    margin = corr[i] - np.nanmax(np.where(far, corr, np.nan))
    return int(shifts[i]), float(corr[i]), float(margin)


def main():
    dry = "--dry-run" in sys.argv
    m = load_manifest()
    rows = []
    for c in m[(m.use == "yes") & (m.gpx != "")].itertuples():
        stem = c.file.rsplit(".", 1)[0]
        n = len(list((FRAMES / stem).glob("*.jpg")))
        if n < 60:
            print(f"{c.clip_id}: only {n} frames, skipped")
            continue
        mot = motion(stem, n)
        track = read_track(VIDEO / c.gpx)
        t0 = (c.start_utc - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds()
        res = best_shift(mot, track, t0)
        if res is None:
            print(f"{c.clip_id}: not enough frames inside the track")
            continue
        shift, corr, margin = res
        trusted = corr >= MIN_CORR and margin >= MIN_MARGIN
        rows.append({"clip_id": c.clip_id, "file": c.file, "frames": n, "shift_s": shift,
                     "corr": round(corr, 3), "margin": round(margin, 3), "trusted": trusted,
                     "old_offset_s": c.offset_s})
        print(f"{c.clip_id:14s} shift {shift:+4d} s  corr {corr:.2f}  margin {margin:.2f}  "
              f"{'trusted' if trusted else 'CHECK BY HAND'}")
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "pipeline" / "sync.csv", index=False)

    if not dry and len(out):
        man = pd.read_csv(MANIFEST, dtype=str)
        for r in out[out.trusted].itertuples():
            i = man.index[man.clip_id == r.clip_id]
            man.loc[i, "offset_s"] = str(float(r.shift_s))
            man.loc[i, "note"] = f"offset from sync.py (corr {r.corr})"
        man.to_csv(MANIFEST, index=False)
        print(f"-> offsets written to {MANIFEST.relative_to(ROOT)} for {int(out.trusted.sum())} clips")


if __name__ == "__main__":
    main()
