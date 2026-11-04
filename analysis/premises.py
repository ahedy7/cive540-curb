"""
Assign each 2025 commercial premise to the block face it fronts.

Input:  data/clipped/study_area.gpkg (2025 commercial survey), data/derived/blockfaces.gpkg
Output: data/derived/premises_2025.gpkg, layer "premises", EPSG:32188
        Adds blockface_id, occupied (True/False) and match (how the face was found).

Matching, in order:
    address   same street, civic number inside the face's address range, same odd/even side,
              and within 40 m of the closest face on that street
    street    same street, nearest face within 60 m (address range missing or off)
    nearest   any face within 75 m (pedestrian streets, malls with an address on a side street)
Shopping-centre units carry the mall's address, so they land on the mall's frontage face.

Usage:
    python analysis/premises.py
"""
import re
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
GPKG = ROOT / "data" / "clipped" / "study_area.gpkg"
BLOCKFACES = ROOT / "data" / "derived" / "blockfaces.gpkg"
OUT = ROOT / "data" / "derived" / "premises_2025.gpkg"
LAYER = "commercial_premises__quete_sur_l_occupation_commerciale_2025"

# survey street names that differ from the géobase
ALIASES = {"Maisonneuve": "De Maisonneuve"}
STREET_M = 60
NEAREST_M = 75
ADDR_SLACK = 10  # survey numbers can sit just outside the géobase range
ADDR_AGREE_M = 40


def civic(text):
    m = re.match(r"\s*(\d+)", str(text))
    return int(m.group(1)) if m else None


def main():
    faces = gpd.read_file(BLOCKFACES)
    cp = gpd.read_file(GPKG, layer=LAYER).to_crs(faces.crs)
    cp["occupied"] = (cp.USAGE1 != "VACANT") & (cp.VACANT_A_LOUER != "Oui")
    cp["street"] = cp.NOM_VOIE.replace(ALIASES)
    cp["civic"] = cp.CIVIQUE.map(civic)

    ids, how = [], []
    for p in cp.itertuples():
        same = faces[faces.street == p.street]
        pick, method = None, None
        if p.civic is not None and len(same):
            hit = same[(same.addr_min - ADDR_SLACK <= p.civic) & (p.civic <= same.addr_max + ADDR_SLACK)
                       & (same.addr_min % 2 == p.civic % 2)]
            # overlapping address ranges can point to a far block. Trust the address only if
            # it lands near the closest face on that street (malls sit ~90 m back, both agree)
            if len(hit) and hit.distance(p.geometry).min() <= same.distance(p.geometry).min() + ADDR_AGREE_M:
                pick, method = hit.distance(p.geometry).idxmin(), "address"
        if pick is None and len(same):
            d = same.distance(p.geometry)
            if d.min() <= STREET_M:
                pick, method = d.idxmin(), "street"
        if pick is None:
            d = faces.distance(p.geometry)
            if d.min() <= NEAREST_M:
                pick, method = d.idxmin(), "nearest"
        ids.append(faces.blockface_id[pick] if pick is not None else None)
        how.append(method or "none")
    cp["blockface_id"], cp["match"] = ids, how

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cp.drop(columns=["street"]).to_file(OUT, layer="premises", driver="GPKG")
    print(f"{len(cp)} premises, {cp.occupied.sum()} occupied -> {OUT.relative_to(ROOT)}")
    print(cp.groupby("match").occupied.agg(["size", "sum"]).rename(columns={"size": "all", "sum": "occupied"}).to_string())


if __name__ == "__main__":
    main()
