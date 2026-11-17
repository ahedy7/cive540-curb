"""
Step 6, filter: run YOLO11 (pretrained on COCO) on the frames geotag.py kept on sampled block
faces, and flag the ones worth sending to the VLM.

YOLO is only a free pre-filter. It finds vehicles, hydrants and stop signs. It can't tell a
parked car from a moving one, and COCO has no class for parking signs. Downtown, almost every
frame has a car in it, so "any vehicle" filters nothing (95% of frames). The rule instead:
    near    a vehicle box at least NEAR_H of the frame height (roughly within 20 m of the
            camera, so its position against the curb can be judged), or a hydrant. A box wider
            and taller than SELF_BOX is the car's own dashboard read as a car, and is ignored
    dup     the car is stopped and this isn't every STOP_EVERY_S-th second of the stop. A car
            waiting at a light films the same view over and over
    send    near and not dup. These go to the VLM for vehicle position and loading activity
Sign reading doesn't depend on this flag. It gets its own thin sample of frames later.

Every detection is saved, so the rule can change without re-running the model
(python pipeline/detect.py --summary). Frames already run are skipped on a re-run.

Outputs (data/derived/detect/):
    detections.csv  one row per box: frame_id, class, conf, x1 y1 x2 y2 (as a share of the
                    frame width and height)
    frames.csv      one row per frame: counts of near boxes per class, near, dup, send

Usage:
    python pipeline/detect.py            # all kept frames on sampled faces
    python pipeline/detect.py 200        # first 200, to test
    python pipeline/detect.py --summary  # only redo frames.csv from saved detections
"""
import sys

import geopandas as gpd
import pandas as pd
from ultralytics import YOLO

from common import DERIVED, ROOT

MODEL = ROOT / "models" / "yolo11m.pt"   # downloaded on first run. Medium: better than small on far, partly hidden vehicles, fine on CPU
CONF = 0.25
IMGSZ = 960
BATCH = 16
NEAR_H = 0.15          # checked by eye on 18 frames with boxes drawn, all three cameras
SELF_BOX = 0.6
STOP_EVERY_S = 10
CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 10: "hydrant", 11: "stop_sign"}
VEHICLES = {"car", "truck", "bus", "motorcycle"}
OUT = DERIVED / "detect"


def frames_to_run():
    f = gpd.read_file(DERIVED / "frames.gpkg", ignore_geometry=True)
    sample = set(gpd.read_file(DERIVED / "sample.gpkg", ignore_geometry=True).blockface_id)
    on = (f.keep & f.blockface_id.isin(sample)) | (f.keep_2 & f.blockface_id_2.isin(sample))
    f = f[on]
    return f[[(ROOT / p).exists() for p in f.path]][["frame_id", "path"]].reset_index(drop=True)


def run_yolo(limit=None):
    OUT.mkdir(parents=True, exist_ok=True)
    det_csv = OUT / "detections.csv"
    done_csv = OUT / "done.txt"   # frame IDs already run, including frames with no detection
    done = set(done_csv.read_text().split()) if done_csv.exists() else set()

    todo = frames_to_run()
    if limit:
        todo = todo.head(limit)
    todo = todo[~todo.frame_id.isin(done)]
    device = "cpu"   # MPS crashes with this torch on macOS 14 (calls a macOS 15 function)
    print(f"{len(todo)} frames to run on {device} ({len(done)} cached)")

    model = YOLO(str(MODEL))
    for i in range(0, len(todo), BATCH):
        chunk = todo.iloc[i:i + BATCH]
        res = model.predict([str(ROOT / p) for p in chunk.path], conf=CONF, imgsz=IMGSZ, device=device,
                            classes=list(CLASSES), verbose=False)
        rows = []
        for fid, r in zip(chunk.frame_id, res):
            for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                rows.append({"frame_id": fid, "class": CLASSES[int(k)], "conf": round(c, 3),
                             "x1": round(x1, 4), "y1": round(y1, 4), "x2": round(x2, 4), "y2": round(y2, 4)})
        if rows:
            pd.DataFrame(rows).to_csv(det_csv, mode="a", header=not det_csv.exists(), index=False)
        with open(done_csv, "a") as fh:   # written after the boxes, so a crash never marks a frame done without them
            fh.write("\n".join(chunk.frame_id) + "\n")
        if (i // BATCH) % 50 == 0:
            print(f"  {i + len(chunk)} / {len(todo)}")


def summary():
    """frames.csv: one row per frame run so far, with the near / dup / send flags."""
    ran = sorted(set((OUT / "done.txt").read_text().split()))
    d = pd.read_csv(OUT / "detections.csv")
    h, w = d.y2 - d.y1, d.x2 - d.x1
    near = d[(d["class"].isin(VEHICLES) & (h >= NEAR_H) & ~((h > SELF_BOX) & (w > SELF_BOX)))
             | (d["class"] == "hydrant")]
    s = near.pivot_table(index="frame_id", columns="class", values="conf", aggfunc="size", fill_value=0)
    s = s.reindex(ran, fill_value=0).rename_axis("frame_id").reset_index()
    s["near"] = s.frame_id.isin(near.frame_id)

    f = gpd.read_file(DERIVED / "frames.gpkg", ignore_geometry=True).set_index("frame_id")
    s["clip_id"] = s.frame_id.map(f.clip_id)
    s["stationary"] = s.frame_id.map(f.stationary).astype(bool)
    s["second"] = s.frame_id.str.rsplit("_", n=1).str[1].astype(int)
    s = s.sort_values(["clip_id", "second"]).reset_index(drop=True)
    stop = (~s.stationary).cumsum()   # one number per stop (a moving frame starts a new one)
    s["dup"] = s.stationary & (s.groupby(["clip_id", stop]).cumcount() % STOP_EVERY_S != 0)
    s["send"] = s.near & ~s.dup
    s.drop(columns=["second"]).to_csv(OUT / "frames.csv", index=False)
    print(f"{len(s)} frames: {int(s.near.sum())} near, {int(s.dup.sum())} repeats while stopped, "
          f"{int(s.send.sum())} to send -> {OUT.relative_to(ROOT)}/frames.csv")


def main():
    if "--summary" not in sys.argv:
        args = [a for a in sys.argv[1:] if a.isdigit()]
        run_yolo(int(args[0]) if args else None)
    summary()


if __name__ == "__main__":
    main()
