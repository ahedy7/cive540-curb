"""
Stratified cluster sample of block faces for the video passes.

Clusters are whole blocks: both sides of a street between two intersections.

Strata:
    loading           blocks with a loading zone on either side (loading sign stretch, or an
                      official Aire de livraison within 15 m). Sampled at a higher rate.
    art_high/art_low  arterial or collector street (géobase class 5 to 7), occupied 2025
    loc_high/loc_low  premises per 100 m of curb above / at-or-below the median.
                      Local = class 0. The median is taken over the non-loading blocks.

Forced in (weight 1): every block holding one of the official loading zones that has no
matching loading sign, so the video can settle which source is right.

Inputs:  data/derived/blockfaces.gpkg, curb_supply.gpkg, premises_2025.gpkg,
         data/clipped/study_area.gpkg (official loading zones)
Outputs: data/derived/sample.gpkg layer "sample_faces", and data/derived/sample.csv
         Columns: blockface_id, block, stratum, selection (forced | random), weight,
         premises, premises_per_100m, street_class, traffic

Usage:
    python analysis/sample.py
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GPKG = ROOT / "data" / "clipped" / "study_area.gpkg"
DERIVED = ROOT / "data" / "derived"
OUT = DERIVED / "sample.gpkg"
OUT_CSV = DERIVED / "sample.csv"

SEED = 540
LOADING_NEAR_M = 15
# blocks drawn at random per stratum (each block is about 2 faces).
# High-commercial strata are oversampled. Sized so both camera loops fit in about 2 hours
# (7/7/6/6 gave 78 faces and about 164 min per morning, too long for 08:00-11:00)
BLOCKS_PER_STRATUM = {"art_high": 5, "loc_high": 5, "art_low": 4, "loc_low": 4}
LOADING_RANDOM_BLOCKS = 2  # on top of the forced loading blocks, so the stratum has a random part


def main():
    faces = gpd.read_file(DERIVED / "blockfaces.gpkg")
    faces["block"] = faces.blockface_id.str.rsplit("_", n=1).str[0]
    st = gpd.read_file(DERIVED / "curb_supply.gpkg", layer="stretches")
    prem = gpd.read_file(DERIVED / "premises_2025.gpkg")
    lz = gpd.read_file(GPKG, layer="loading_zones__Aires_de_livraison").to_crs(faces.crs)

    # loading faces, and official zones with no loading sign nearby
    load_st = st[st.use == "loading"]
    lz["near_sign_m"] = lz.geometry.apply(lambda p: load_st.distance(p).min())
    lz_faces = gpd.sjoin_nearest(lz, faces[["blockface_id", "geometry"]], max_distance=LOADING_NEAR_M)
    loading_faces = set(load_st.blockface_id) | set(lz_faces.blockface_id)
    conflict_faces = set(lz_faces[lz_faces.near_sign_m > LOADING_NEAR_M].blockface_id)

    occ = prem[prem.occupied].groupby("blockface_id").size()
    faces["premises"] = faces.blockface_id.map(occ).fillna(0).astype(int)

    blocks = faces.groupby("block").agg(curb_m=("length_m", "sum"), premises=("premises", "sum"),
                                        street_class=("street_class", "max"))
    blocks["per_100m"] = blocks.premises / blocks.curb_m * 100
    blocks["arterial"] = blocks.street_class >= 5
    loading_blocks = set(faces[faces.blockface_id.isin(loading_faces)].block)
    forced_blocks = set(faces[faces.blockface_id.isin(conflict_faces)].block)

    rest = blocks[~blocks.index.isin(loading_blocks)]
    median = rest.per_100m.median()
    blocks["stratum"] = np.where(blocks.index.isin(loading_blocks), "loading",
                                 np.where(blocks.arterial, "art_", "loc_") + np.where(blocks.per_100m > median, "high", "low"))

    rng = np.random.default_rng(SEED)
    chosen = []  # (block, stratum, selection, weight)
    for stratum, group in blocks.groupby("stratum"):
        pool = sorted(set(group.index) - forced_blocks)  # sorted, so the draw is reproducible
        n = LOADING_RANDOM_BLOCKS if stratum == "loading" else BLOCKS_PER_STRATUM[stratum]
        n = min(n, len(pool))
        draw = rng.choice(pool, size=n, replace=False) if n else []
        w = len(pool) / n if n else 0
        chosen += [(b, stratum, "random", round(w, 2)) for b in draw]
        chosen += [(b, stratum, "forced", 1.0) for b in sorted(forced_blocks & set(group.index))]

    pick = pd.DataFrame(chosen, columns=["block", "stratum", "selection", "weight"])
    sample = faces.merge(pick, on="block")
    sample["premises_per_100m"] = sample.block.map(blocks.per_100m).round(2)
    cols = ["blockface_id", "block", "stratum", "selection", "weight", "premises", "premises_per_100m",
            "street", "side", "street_class", "traffic", "length_m", "geometry"]
    sample = sample[cols].sort_values(["stratum", "block", "side"]).reset_index(drop=True)

    sample.to_file(OUT, layer="sample_faces", driver="GPKG")
    sample.drop(columns="geometry").to_csv(OUT_CSV, index=False)

    print(f"median occupied premises per 100 m (non-loading blocks): {median:.2f}")
    print(f"loading blocks: {len(loading_blocks)}, forced (official zone without a loading sign): {len(forced_blocks)}")
    frame = blocks.groupby("stratum").size().rename("blocks_in_area")
    got = sample.groupby("stratum").agg(blocks=("block", "nunique"), faces=("blockface_id", "size"),
                                        forced_faces=("selection", lambda s: (s == "forced").sum()),
                                        km=("length_m", lambda s: round(s.sum() / 1000, 2)))
    print(pd.concat([frame, got], axis=1).fillna(0).astype({"blocks": int, "faces": int, "forced_faces": int}).to_string())
    print(f"total: {sample.block.nunique()} blocks, {len(sample)} faces -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
