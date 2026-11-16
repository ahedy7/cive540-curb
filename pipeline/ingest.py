"""
Step 1, ingest: list every clip in video/, read its start time and length, and pair it with
the GPS track that overlaps it in time.

File names don't matter (IMG_4730.MOV is fine). Each clip gets an ID <date>_<side>_<part>,
e.g. 20261111_R_1, where part counts the pieces of a split recording in time order.

Output: pipeline/clips.csv, one row per clip. Edit it by hand where the guess is wrong:
    side      R or L. Defaults to R. Phones can't be told apart from the file, so set L by hand
    gpx       track file in video/ (blank = no GPS for this clip)
    offset_s  seconds to add to the clip clock so it matches the GPS clock (default 0)
    use       yes or no (no = skip, e.g. a test clip)
    note      anything
Re-running keeps your edits for clips already listed and adds new ones.

Two other sources of clips:
    pipeline/start_times.csv  clips whose export lost the recording time (the file says when it
                              was exported). Start time and model are read off the phone's Photos
                              info panel. New clips from it get side L or R from "left"/"right"
                              in the file name.
    clips.csv itself          a video deleted after its frames were extracted stays listed
Without the video, the length is the number of frames extracted (one per second), or else
the length on the Photos panel.

Usage:
    python pipeline/ingest.py
"""
import pandas as pd

from common import FRAMES, LOCAL_TZ, MANIFEST, MOVIE_EXT, ROOT, VIDEO, clip_info, read_track

STARTS = ROOT / "pipeline" / "start_times.csv"
EDITABLE = ["side", "gpx", "offset_s", "use", "note"]
SHORT_CLIP_S = 120   # clips shorter than this are probably tests. Flagged, not dropped


def n_frames(file):
    folder = FRAMES / file.rsplit(".", 1)[0]
    return len(list(folder.glob("*.jpg"))) if folder.exists() else 0


def clip_list(old):
    """file -> start (UTC), duration (s), model, side guess, from videos, start_times.csv and old rows."""
    movies = {p.name: p for p in VIDEO.iterdir() if p.suffix.lower() in MOVIE_EXT}
    starts = pd.read_csv(STARTS).set_index("file") if STARTS.exists() else pd.DataFrame()
    clips = {}
    for name in sorted(set(movies) | set(starts.index) | set(old.index)):
        info = clip_info(movies[name]) if name in movies else None
        if name in starts.index:   # the export's own time is wrong, use the phone's
            s = starts.loc[name]
            start = pd.Timestamp(s.start_local).tz_localize(LOCAL_TZ).tz_convert("UTC")
            mm, ss = map(int, s.duration.split(":"))
            duration = info["duration_s"] if info else float(n_frames(name) or mm * 60 + ss)
            side = "L" if "left" in name.lower() else "R"
            clips[name] = (start, duration, s.model, side)
        elif info:
            clips[name] = (info["start_utc"], info["duration_s"], info["model"], "R")
        else:                      # video deleted after its frames were extracted
            o = old.loc[name]
            clips[name] = (pd.Timestamp(o.start_utc), float(o.duration_s), o.model, o.side)
    return clips


def main():
    tracks = sorted(p for p in VIDEO.iterdir() if p.suffix.lower() in (".gpx", ".csv"))
    old = pd.read_csv(MANIFEST, dtype=str).set_index("file") if MANIFEST.exists() else pd.DataFrame()
    clips = clip_list(old)
    if not clips:
        raise SystemExit(f"No clips in {VIDEO} or {STARTS.name}")

    spans = {}
    for t in tracks:
        tr = read_track(t)
        if len(tr):
            spans[t.name] = (tr.time.iloc[0], tr.time.iloc[-1], len(tr))
            print(f"track {t.name}: {len(tr)} points, {tr.time.iloc[0].tz_convert(LOCAL_TZ):%Y-%m-%d %H:%M:%S} "
                  f"to {tr.time.iloc[-1].tz_convert(LOCAL_TZ):%H:%M:%S}")

    rows = []
    for file, (start, duration, model, side) in clips.items():
        end = start + pd.Timedelta(seconds=duration)
        # the track that covers most of the clip
        best, best_overlap = "", 0.0
        for name, (a, b, _) in spans.items():
            overlap = (min(end, b) - max(start, a)).total_seconds()
            if overlap > best_overlap:
                best, best_overlap = name, overlap
        rows.append({
            "file": file,
            "date": start.tz_convert(LOCAL_TZ).strftime("%Y%m%d"),
            "start_utc": start,
            "start_local": start.tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s": duration,
            "model": model,
            "gps_cover": round(best_overlap / duration, 2) if duration else 0,
            "side": side, "gpx": best, "offset_s": 0.0, "use": "yes",
            "note": "short clip, test?" if duration < SHORT_CLIP_S else "",
        })
    new = pd.DataFrame(rows)

    if len(old):  # keep hand edits
        known = new.file.isin(old.index)
        for col in EDITABLE:
            new[col] = new[col].astype(object)
            new.loc[known, col] = new.loc[known, "file"].map(old[col]).fillna("")
        new["offset_s"] = pd.to_numeric(new.offset_s, errors="coerce").fillna(0.0)

    # IDs: parts of the same date and side in time order
    new = new.sort_values("start_utc").reset_index(drop=True)
    new["part"] = new.groupby(["date", "side"]).cumcount() + 1
    new["clip_id"] = new.date + "_" + new.side + "_" + new.part.astype(str)
    cols = ["clip_id", "file", "date", "side", "part", "start_local", "start_utc", "duration_s",
            "model", "gpx", "gps_cover", "offset_s", "use", "note"]
    new[cols].to_csv(MANIFEST, index=False)

    pd.set_option("display.width", 200)
    print()
    print(new[["clip_id", "file", "start_local", "duration_s", "gpx", "gps_cover", "use", "note"]].to_string(index=False))
    print(f"\n-> {MANIFEST.relative_to(ROOT)}. Set side = L by hand for the left camera's clips, then re-run.")
    if (new.gpx == "").any():
        print("Clips with no GPS track: " + ", ".join(new.loc[new.gpx == "", "clip_id"]) +
              ". They get frames but no position until a track or anchors CSV is added.")


if __name__ == "__main__":
    main()
