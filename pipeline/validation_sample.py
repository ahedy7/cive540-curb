"""
Step 8a, validation sample: draw the frames to hand-label and pack them with the labelling page.

Candidates: frames kept by geotag.py on a sampled block face. Stopped frames are allowed (double
parking is often in view while the car waits), the time gap below keeps out repeats. An F frame
counts for its second face too.

Draw: N_FRAMES spread evenly over the 5 sample strata, so small strata get as many checks as the
big one. A stratum without enough frames passes its leftover to the others. At most PER_FACE_CLIP frames from one face in one clip, at least MIN_GAP_S apart, so
the set isn't a run of near-identical frames. N_DOUBLE of them are flagged for a second labeller
(agreement between labellers, Cohen's kappa).

Output, in validation/ (frames/ is gitignored, never publish it: frames are not blurred):
    frames/<frame_id>.jpg   copies of the drawn frames
    frames.js               the list the page reads (a .js file so the page works from file://)
    sample.csv              the same list as a table
Open validation/label.html in a browser to label.

Usage:
    python pipeline/validation_sample.py           # 300 frames
    python pipeline/validation_sample.py 12        # a small test set
"""
import json
import shutil
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from common import DERIVED, ROOT

N_FRAMES = 300
N_DOUBLE = 50
PER_FACE_CLIP = 4
MIN_GAP_S = 5
SEED = 540
OUT = ROOT / "validation"
CAMERA = {"F": "forward", "L": "left side", "R": "right side"}


def candidates():
    f = gpd.read_file(DERIVED / "frames.gpkg", ignore_geometry=True)
    sample = gpd.read_file(DERIVED / "sample.gpkg", ignore_geometry=True).set_index("blockface_id")
    one = f[f.keep][["frame_id", "clip_id", "date", "side", "time_local", "path", "stationary", "blockface_id"]]
    two = f[f.keep_2][["frame_id", "clip_id", "date", "side", "time_local", "path", "stationary", "blockface_id_2"]]
    c = pd.concat([one, two.rename(columns={"blockface_id_2": "blockface_id"})])
    c = c[c.blockface_id.isin(sample.index)]
    c = c.drop_duplicates("frame_id")   # an F frame on two sampled faces is drawn once
    c = c[[(ROOT / p).exists() for p in c.path]]
    c["stratum"] = c.blockface_id.map(sample.stratum)
    c["street"] = c.blockface_id.map(sample.street)
    c["face_side"] = c.blockface_id.map(sample.side)
    c["second"] = c.frame_id.str.rsplit("_", n=1).str[1].astype(int)
    return c


def eligible(pool):
    """Walk a shuffled pool and keep frames that respect the per-face cap and the time gap."""
    taken, chosen = {}, []
    for r in pool.itertuples():
        key = (r.blockface_id, r.clip_id)
        near = taken.get(key, [])
        if len(near) < PER_FACE_CLIP and all(abs(r.second - t) >= MIN_GAP_S for t in near):
            taken[key] = near + [r.second]
            chosen.append(r.Index)
    return pool.loc[chosen]


def draw(c, n, rng):
    """Equal share per stratum. A stratum short of frames gives its leftover to the others."""
    pools = {s: eligible(c[c.stratum == s].sample(frac=1, random_state=rng)) for s in sorted(c.stratum.unique())}
    want, left, open_ = {s: 0 for s in pools}, n, set(pools)
    while left > 0 and open_:
        share = max(left // len(open_), 1)
        for s in sorted(open_):
            add = min(share, len(pools[s]) - want[s], left)
            want[s] += add
            left -= add
            if want[s] == len(pools[s]):
                open_.discard(s)
    for s in pools:
        print(f"  {s}: {want[s]} frames (of {len(pools[s])} eligible)")
    return pd.concat([pools[s].iloc[:want[s]] for s in pools])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_FRAMES
    rng = np.random.RandomState(SEED)
    c = candidates().reset_index(drop=True)
    print(f"{len(c)} candidate frames on {c.blockface_id.nunique()} sampled faces")
    s = draw(c, n, rng).sample(frac=1, random_state=rng).reset_index(drop=True)   # shuffled order
    s["double"] = False
    s.loc[s.sample(min(N_DOUBLE * n // N_FRAMES, len(s)), random_state=rng).index, "double"] = True

    if (OUT / "frames").exists():
        shutil.rmtree(OUT / "frames")
    (OUT / "frames").mkdir(parents=True)
    for r in s.itertuples():
        shutil.copy2(ROOT / r.path, OUT / "frames" / f"{r.frame_id}.jpg")
    s["camera"] = s.side.map(CAMERA)
    cols = ["frame_id", "date", "time_local", "camera", "blockface_id", "street", "face_side", "stratum", "double"]
    s[cols].to_csv(OUT / "sample.csv", index=False)
    items = [{"id": r.frame_id, "img": f"frames/{r.frame_id}.jpg", "camera": r.camera, "time": r.time_local,
              "street": r.street, "face_side": r.face_side, "face": r.blockface_id, "double": bool(r.double)}
             for r in s.itertuples()]
    (OUT / "frames.js").write_text("window.FRAMES = " + json.dumps(items, indent=0) + ";\n")

    print(s.groupby(["stratum", "camera"]).size().unstack(fill_value=0).to_string())
    print(f"{len(s)} frames ({int(s.double.sum())} for double labelling) -> {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
