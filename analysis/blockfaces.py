"""
Build block faces (one side of a street between two intersections) for the study area.

Input:  data/clipped/study_area.gpkg (géobase centre lines + géobase double side lines)
Output: data/derived/blockfaces.gpkg, layer "blockfaces", EPSG:32188

Each block face gets a stable ID: <street>_<from>_<to>_<side>
  - from/to are the cross streets, ordered west to east (or south to north)
  - side is N, S, E or W in Montréal's street-grid sense ("north" = toward the mountain)

Usage:
    python analysis/blockfaces.py
"""
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, unary_union

ROOT = Path(__file__).resolve().parents[1]
GPKG = ROOT / "data" / "clipped" / "study_area.gpkg"
AREA = ROOT / "study_area.geojson"
OUT = ROOT / "data" / "derived" / "blockfaces.gpkg"
MTM8 = "EPSG:32188"

# Montréal's street grid is rotated. "Grid east" (along Sherbrooke) points about 34 degrees
# east of true north in MTM8. Measured from the Atwater and De Bleury corners on Sherbrooke.
GRID_EAST_BEARING = 34.0

MEDIAN_PROBES_M = (6, 12, 20)  # distances past a side line to look for the other carriageway
MEDIAN_HIT_M = 6               # other carriageway this close to a probe = median side, not a curb
BOUNDARY_STREETS = {"Sherbrooke", "René-Lévesque", "Atwater", "De Bleury"}
CORNER_STUB_M = 20  # boundary-street block ending this far outside the area = stub past a corner
MEDIAN_CROSSING_M = 30  # same street at both ends and shorter than this = median crossing
MIN_SHARE_INSIDE = 0.5  # keep blocks with at least half their length in the study area


def slug(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


def merge_lines(geoms):
    u = unary_union(list(geoms))
    return u if u.geom_type == "LineString" else linemerge(u)


def grid_angle(line):
    """Direction of a line in grid degrees: 0 = grid east, 90 = grid north."""
    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    bearing = math.degrees(math.atan2(x1 - x0, y1 - y0))  # true bearing, clockwise from north
    return (GRID_EAST_BEARING - bearing) % 360  # counter-clockwise from grid east


def load_streets():
    g = gpd.read_file(GPKG, layer="geobase__Geobase")
    no_curb = (
        g.TYP_VOIE.isin(["tunnel", "ruelle"])  # tunnels, laneways
        | g.CLASSE.isin([2, 8])                # expressway pieces, unnamed ramps
        | (g.NOM_VOIE == "Non-nommée")
    )
    pedestrian = g.CLASSE == 1                 # McTavish, Sussex: no curb, but still intersections
    g["curb"] = ~no_curb & ~pedestrian
    g["counts_as_intersection"] = ~no_curb
    return g


def node_key(pt):
    return (round(pt[0], 1), round(pt[1], 1))


def build_blocks(g):
    """Chain centre-line segments of the same street until they hit an intersection."""
    names_at = defaultdict(set)   # node -> street names that end there
    degree = defaultdict(int)
    for r in g[g.counts_as_intersection].itertuples():
        for p in (r.geometry.coords[0], r.geometry.coords[-1]):
            names_at[node_key(p)].add(r.NOM_VOIE)
            degree[node_key(p)] += 1

    def is_break(n):
        return len(names_at[n]) > 1 or degree[n] != 2

    curb = g[g.curb].reset_index(drop=True)
    parent = list(range(len(curb)))  # union-find over segments

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_node = defaultdict(list)
    for i, r in curb.iterrows():
        for p in (r.geometry.coords[0], r.geometry.coords[-1]):
            by_node[(node_key(p), r.NOM_VOIE)].append(i)
    for (n, _), segs in by_node.items():
        if not is_break(n) and len(segs) == 2:
            parent[find(segs[0])] = find(segs[1])

    curb["block"] = [find(i) for i in range(len(curb))]
    return curb, names_at, degree


def cross_street(node, own, names_at, degree, classe_of):
    others = [n for n in names_at.get(node, ()) if n != own]
    if not others:
        return "split" if degree[node] >= 3 else "end"  # carriageways splitting, or a dead end
    return max(others, key=lambda n: (classe_of.get(n, 0), n))  # prefer the bigger street


def main():
    g = load_streets()
    sides = gpd.read_file(GPKG, layer="geobase_double__Geobase_double")
    sides["ID_TRC"] = sides.ID_TRC.astype(int)
    area = gpd.read_file(AREA).to_crs(MTM8).geometry.iloc[0]
    classe_of = g.groupby("NOM_VOIE").CLASSE.max().to_dict()

    curb, names_at, degree = build_blocks(g)

    rows = []
    for _, seg in curb.groupby("block"):
        street = seg.NOM_VOIE.iloc[0]
        line = merge_lines(seg.geometry)
        if line.geom_type != "LineString":
            continue  # a block that doesn't form one line (rare). Skip rather than guess
        # share of the block's length inside the area. Boundary streets get the clip's 30 m
        # so both sides (and both carriageways) count. Stubs past the boundary drop out
        reach = area.buffer(30 if street in BOUNDARY_STREETS else 5)
        if line.intersection(reach).length / line.length < MIN_SHARE_INSIDE:
            continue
        if street in BOUNDARY_STREETS and any(
                not area.contains(Point(p)) and area.exterior.distance(Point(p)) > CORNER_STUB_M
                for p in (line.coords[0], line.coords[-1])):
            continue  # a piece of a boundary street that runs past a corner

        # orient grid west -> east, or grid south -> north
        ang = grid_angle(line)
        east_west = ang < 45 or ang > 315 or 135 < ang < 225
        if (east_west and 90 < ang < 270) or (not east_west and not (0 < ang < 180)):
            line = LineString(line.coords[::-1])
        a, b = node_key(line.coords[0]), node_key(line.coords[-1])
        frm = cross_street(a, street, names_at, degree, classe_of)
        to = cross_street(b, street, names_at, degree, classe_of)
        if frm == to and frm not in ("end", "split") and line.length < MEDIAN_CROSSING_M:
            continue  # a cross street's short piece inside a divided boulevard's median

        # traffic direction, relative to the oriented block
        dirs = set()
        for r in seg.itertuples():
            if r.SENS_CIR == 0:
                dirs.add(0)
                continue
            same = line.project(Point(r.geometry.coords[-1])) > line.project(Point(r.geometry.coords[0]))
            dirs.add(r.SENS_CIR if same else -r.SENS_CIR)
        if dirs == {1}:
            traffic = "one_way_" + ("E" if east_west else "N")
        elif dirs == {-1}:
            traffic = "one_way_" + ("W" if east_west else "S")
        else:
            traffic = "two_way"

        # side lines: which side of the oriented block each one is on
        others = curb[(curb.NOM_VOIE == street) & ~curb.ID_TRC.isin(seg.ID_TRC)].geometry.tolist()
        per_side = defaultdict(list)
        for s in sides[sides.ID_TRC.isin(seg.ID_TRC)].itertuples():
            mid = s.geometry.interpolate(0.5, normalized=True)
            d = line.project(mid)
            p0 = line.interpolate(max(d - 1, 0))
            p1 = line.interpolate(min(d + 1, line.length))
            on_centre = line.interpolate(d)
            cross = (p1.x - p0.x) * (mid.y - p0.y) - (p1.y - p0.y) * (mid.x - p0.x)
            left = cross > 0
            side = ("N" if left else "S") if east_west else ("W" if left else "E")

            # median check: push past the side line and see if the other carriageway is there
            dx, dy = mid.x - on_centre.x, mid.y - on_centre.y
            norm = math.hypot(dx, dy) or 1
            probes = [Point(mid.x + dx / norm * k, mid.y + dy / norm * k) for k in MEDIAN_PROBES_M]
            median = any(o.distance(p) < MEDIAN_HIT_M for o in others for p in probes)
            per_side[side].append((s, median))

        for side, items in per_side.items():
            if all(m for _, m in items):
                continue  # faces the median of a divided boulevard
            parts = [s.geometry for s, m in items if not m]
            geom = merge_lines(parts)
            addr = [v for s, m in items if not m for v in (s.DEBUT_ADRE, s.FIN_ADRESS) if v > 0]
            rows.append({
                "street": street, "from_street": frm, "to_street": to, "side": side,
                "length_m": round(geom.length, 1),
                "centreline_m": round(line.length, 1),
                "street_class": int(seg.CLASSE.max()),
                "traffic": traffic,
                "addr_min": int(min(addr)) if addr else None,
                "addr_max": int(max(addr)) if addr else None,
                "id_trc": ",".join(str(i) for i in sorted(seg.ID_TRC)),
                "cot_rue_id": ",".join(str(int(s.COT_RUE_ID)) for s, m in items if not m),
                "geometry": geom,
            })

    bf = gpd.GeoDataFrame(rows, crs=MTM8)
    # same street at both ends and no addresses: a connector inside a junction, nothing fronts it
    bf = bf[~((bf.from_street == bf.to_street) & bf.addr_min.isna())]
    bf["blockface_id"] = [f"{slug(r.street)}_{slug(r.from_street)}_{slug(r.to_street)}_{r.side}"
                          for r in bf.itertuples()]
    # same street, same cross streets twice (e.g. a crescent): number them along the street
    bf = bf.sort_values(["blockface_id", "addr_min"]).reset_index(drop=True)
    dup = bf.groupby("blockface_id").cumcount()
    bf.loc[dup > 0, "blockface_id"] += "_" + (dup[dup > 0] + 1).astype(str)
    bf = bf[["blockface_id"] + [c for c in bf.columns if c != "blockface_id"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bf.to_file(OUT, layer="blockfaces", driver="GPKG")
    print(f"{len(bf)} block faces, {bf.length_m.sum() / 1000:.1f} km of curb -> {OUT.relative_to(ROOT)}")
    print(bf.side.value_counts().to_string())


if __name__ == "__main__":
    main()
