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

Usage:
    python pipeline/ingest.py
"""
import pandas as pd

from common import LOCAL_TZ, MANIFEST, MOVIE_EXT, ROOT, VIDEO, clip_info, read_track

EDITABLE = ["side", "gpx", "offset_s", "use", "note"]
SHORT_CLIP_S = 120   # clips shorter than this are probably tests. Flagged, not dropped


def main():
    movies = sorted(p for p in VIDEO.iterdir() if p.suffix.lower() in MOVIE_EXT)
    tracks = sorted(p for p in VIDEO.iterdir() if p.suffix.lower() in (".gpx", ".csv"))
    if not movies:
        raise SystemExit(f"No clips in {VIDEO}")

    spans = {}
    for t in tracks:
        tr = read_track(t)
        if len(tr):
            spans[t.name] = (tr.time.iloc[0], tr.time.iloc[-1], len(tr))
            print(f"track {t.name}: {len(tr)} points, {tr.time.iloc[0].tz_convert(LOCAL_TZ):%Y-%m-%d %H:%M:%S} "
                  f"to {tr.time.iloc[-1].tz_convert(LOCAL_TZ):%H:%M:%S}")

    rows = []
    for p in movies:
        info = clip_info(p)
        start, end = info["start_utc"], info["start_utc"] + pd.Timedelta(seconds=info["duration_s"])
        # the track that covers most of the clip
        best, best_overlap = "", 0.0
        for name, (a, b, _) in spans.items():
            overlap = (min(end, b) - max(start, a)).total_seconds()
            if overlap > best_overlap:
                best, best_overlap = name, overlap
        rows.append({
            "file": p.name,
            "date": start.tz_convert(LOCAL_TZ).strftime("%Y%m%d"),
            "start_utc": start,
            "start_local": start.tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s": info["duration_s"],
            "model": info["model"],
            "gps_cover": round(best_overlap / info["duration_s"], 2) if info["duration_s"] else 0,
            "side": "R", "gpx": best, "offset_s": 0.0, "use": "yes",
            "note": "short clip, test?" if info["duration_s"] < SHORT_CLIP_S else "",
        })
    new = pd.DataFrame(rows)

    if MANIFEST.exists():  # keep hand edits
        old = pd.read_csv(MANIFEST, dtype=str).set_index("file")
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
