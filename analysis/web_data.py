"""
Data for the website (docs/): one GeoJSON of block faces with what the browser needs to rerun
the loading model, and a small JSON of model constants.

Per face, for each window's peak hour, the curb metres the loading model uses (loading,
short-term, parking = paid + free), averaged over the 10 weekday half-hours in that hour, as in
loading.py. The browser then rebuilds usable curb, pools the two faces of each street block and
runs Erlang B, so sliders (dwell, empty share, parking turned into loading) change the map live.
Python's mismatch index is kept as mi_py, so the browser result can be checked against it.

Observed double parking (dp_blocks.csv) is attached to the faces of filmed blocks.

Outputs: docs/data/blockfaces.geojson (EPSG:4326), docs/data/model.json

Usage:
    python analysis/web_data.py
"""
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from loading import (DWELLS_MIN, SPACE_M, TARGET_P, VACANT_SHARE, WINDOWS, curb_by_slot, hourly_shares)

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
OUT = ROOT / "docs" / "data"
PAID_SPACE_M = 6.0   # one paid space, as in supply.py


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shares = hourly_shares()
    peaks = {w: max(range(h0, h1), key=shares.get) for w, (h0, h1) in WINDOWS.items()}

    demand = gpd.read_file(DERIVED / "demand.gpkg", layer="demand_by_face")
    stretches = gpd.read_file(DERIVED / "curb_supply.gpkg", layer="stretches")
    curb = curb_by_slot(demand, stretches).reset_index()
    curb["hour"] = curb.minute // 60

    f = demand[["blockface_id", "street", "side", "length_m", "deliveries_low", "deliveries_high", "premises", "geometry"]].copy()
    f["block"] = f.blockface_id.str.rsplit("_", n=1).str[0]
    for w, h in peaks.items():
        c = curb[curb.hour == h].groupby("blockface_id")[["loading", "short_term", "parking"]].mean()
        k = w.replace("-", "_")
        f = f.join(c.rename(columns=lambda x: f"{x}_{k}").round(1), on="blockface_id")

    mis = pd.read_csv(DERIVED / "mismatch.csv")
    mis = mis[(mis.window == "07-11") & (mis.demand == "high") & (mis.dwell_min == 13.4)]
    f = f.merge(mis[["blockface_id", "mi"]].rename(columns={"mi": "mi_py"}), on="blockface_id", how="left")

    sample = set(gpd.read_file(DERIVED / "sample.gpkg", ignore_geometry=True).blockface_id)
    dp = pd.read_csv(DERIVED / "dp_blocks.csv", index_col=0)
    f["sampled"] = f.blockface_id.isin(sample)
    f["dp_events"] = f.block.map(dp.events).where(f.sampled)
    f["dp_km"] = f.block.map(dp.exposure_100m / 10).round(2).where(f.sampled)

    f = f.rename(columns={"blockface_id": "id", "length_m": "len", "deliveries_low": "deliv_low",
                          "deliveries_high": "deliv_high"})
    f = f.to_crs(4326)
    f.to_file(OUT / "blockfaces.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)

    model = {"space_m": SPACE_M, "paid_space_m": PAID_SPACE_M, "target_p": TARGET_P, "vacant_share": VACANT_SHARE,
             "dwells_min": list(DWELLS_MIN.values()), "windows": {w: list(v) for w, v in WINDOWS.items()},
             "peak_hour": peaks, "hour_share": {h: round(s, 5) for h, s in shares.items()}}
    (OUT / "model.json").write_text(json.dumps(model, indent=1))
    size = (OUT / "blockfaces.geojson").stat().st_size / 1e6
    print(f"{len(f)} faces -> docs/data/blockfaces.geojson ({size:.1f} MB), docs/data/model.json. Peaks: {peaks}")


if __name__ == "__main__":
    main()
