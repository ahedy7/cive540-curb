"""
Step 7b, review list: frames for a person to check for double parking, with validation/review.html.

Double parking is rare (a few frames in a hundred), and one photo often can't tell a truck
stopped in the lane from one waiting in traffic. The VLM finds candidates, a person confirms.
A frame goes on the list when any cached VLM answer for it (any model, with or without the
before and after frames) puts a vehicle outside the curb lane: double parked, in a bike lane
or in a bus stop. With --wide, also every frame with a delivery van, box truck or bus in any
position, to catch double parking the model called parked.

Frames are listed in time order within each clip, so one event's frames come one after another.

Output, in validation/ (gitignored, frames are not blurred):
    review/<frame_id>_[b|n|a].jpg   the frame and the frames 2 s before and after
    review.js                       the list the page reads
    review_list.csv                 the same list as a table
Open validation/review.html, answer each frame, then Export. Save the export as
validation/review_<date>.json.

Usage:
    python pipeline/review_list.py
    python pipeline/review_list.py --wide
"""
import json
import shutil
import sys

import geopandas as gpd
import pandas as pd

from common import DERIVED, ROOT
from vlm import CONTEXT_S, OUT as VLM, neighbour

OUTSIDE = {"double_parked", "in_bike_lane", "in_bus_stop"}
BIG = {"delivery_van", "box_truck", "bus"}
OUT = ROOT / "validation"
CAMERA = {"F": "forward", "L": "left", "R": "right"}


def answers():
    """frame_id -> list of (run name, VLM result) over every cached run."""
    out = {}
    for run in sorted(p for p in VLM.iterdir() if p.is_dir()):
        for f in run.glob("*.json"):
            out.setdefault(f.stem, []).append((run.name, json.loads(f.read_text())["result"]))
    return out


def claim(runs):
    """One line per run that saw something worth checking, e.g. 'haiku_ctx: box_truck double_parked'."""
    lines = []
    for name, r in runs:
        odd = [f"{v['class']} {v['position']}" for v in r["vehicles"] if v["position"] in OUTSIDE]
        big = [f"{v['class']} {v['position']}" for v in r["vehicles"] if v["class"] in BIG and v["position"] not in OUTSIDE]
        if odd or big:
            lines.append(f"{name}: " + ", ".join(odd + big))
    return " | ".join(lines)


def main():
    wide = "--wide" in sys.argv
    f = gpd.read_file(DERIVED / "frames.gpkg", ignore_geometry=True).set_index("frame_id")
    sample = gpd.read_file(DERIVED / "sample.gpkg", ignore_geometry=True).set_index("blockface_id")
    rows = []
    for fid, runs in answers().items():
        vs = [v for _, r in runs for v in r["vehicles"]]
        outside = [v for v in vs if v["position"] in OUTSIDE]
        if not (outside or (wide and any(v["class"] in BIG for v in vs))):
            continue
        face = f.at[fid, "blockface_id"] if f.at[fid, "blockface_id"] in sample.index else f.at[fid, "blockface_id_2"]
        rows.append({"frame_id": fid, "clip_id": f.at[fid, "clip_id"], "time": f.at[fid, "time_local"],
                     "camera": CAMERA[f.at[fid, "side"]], "blockface_id": face,
                     "street": sample.street.get(face, ""), "face_side": sample.side.get(face, ""),
                     "flagged_by": len({n for n, r in runs if any(v["position"] in OUTSIDE for v in r["vehicles"])}),
                     "cls": (outside or [v for v in vs if v["class"] in BIG])[0]["class"],
                     "claim": claim(runs), "path": f.at[fid, "path"]})
    s = pd.DataFrame(rows).sort_values(["clip_id", "frame_id"]).reset_index(drop=True)

    pics = OUT / "review"
    if pics.exists():
        shutil.rmtree(pics)
    pics.mkdir(parents=True)
    items = []
    for r in s.itertuples():
        now = ROOT / r.path
        item = {"id": r.frame_id, "claim": r.claim, "cls": r.cls, "street": r.street, "face_side": r.face_side,
                "camera": r.camera, "time": str(r.time)[11:19]}
        for key, p in [("before", neighbour(now, -CONTEXT_S)), ("now", now), ("after", neighbour(now, CONTEXT_S))]:
            if p:
                dst = pics / f"{r.frame_id}_{key[0]}.jpg"
                shutil.copy2(p, dst)
                item[key] = f"review/{dst.name}"
        items.append(item)
    (OUT / "review.js").write_text("window.REVIEW = " + json.dumps(items, indent=0) + ";\n")
    s.drop(columns="path").to_csv(OUT / "review_list.csv", index=False)
    print(f"{len(s)} frames to review ({int((s.flagged_by > 0).sum())} flagged outside the curb lane) "
          f"on {s.blockface_id.nunique()} faces -> validation/review.html")


if __name__ == "__main__":
    main()
