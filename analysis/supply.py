"""
Curb supply: how each block face's curb is allocated, by use and weekday time window.

Inputs:
    data/derived/blockfaces.gpkg            block faces
    data/clipped/study_area.gpkg            active parking signs, paid spaces (Places), hydrants
    data/clipped/tables/paid_parking__*     paid hours per space
    analysis/rpa_rules.csv                  reviewed sign rules (from rpa_rules.py)

Output: data/derived/curb_supply.gpkg (EPSG:32188) and data/derived/supply_by_window.csv

    layer "stretches", one row per regulated piece of curb, before overlaps are resolved
        blockface_id, from_m, to_m, length_m   position along the face line, in metres
        use            no_stopping | bus_stop | no_parking | implicit_no_parking | reserved
                       | permit | loading | short_term | paid
        beneficiary    who a reserved space is for (taxi, hotel, resident_permit, ...)
        schedule       "always" or e.g. "Mo-Fr 07:00-09:30,15:30-17:30"
        season         "all_year" or "MM-DD..MM-DD"
        source         sign | places | hydrant | intersection
        ref            sign RPA code, paid space number, or hydrant ID

    table "supply_by_window" (also the CSV), one row per block face and window
        blockface_id, window (07-11 | 11-15 | 15-19), length_m,
        m_<use> for every use above plus m_free (unregulated curb)
        Metres are averaged over the weekday half-hours in the window, so a rule that
        applies 1 hour on 2 weekdays counts as a small share of its length.

Where rules overlap, the strictest wins, in the order of USE_PRIORITY.

Usage:
    python analysis/supply.py
"""
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import linemerge, nearest_points, substring

ROOT = Path(__file__).resolve().parents[1]
GPKG = ROOT / "data" / "clipped" / "study_area.gpkg"
TABLES = ROOT / "data" / "clipped" / "tables"
BLOCKFACES = ROOT / "data" / "derived" / "blockfaces.gpkg"
RULES = ROOT / "analysis" / "rpa_rules.csv"
OUT = ROOT / "data" / "derived" / "curb_supply.gpkg"
OUT_CSV = ROOT / "data" / "derived" / "supply_by_window.csv"

REFERENCE_DATE = date(2026, 11, 10)  # filming week. Seasonal rules count if in force on this date
WINDOWS = {"07-11": (7, 11), "11-15": (11, 15), "15-19": (15, 19)}
WEEKDAYS = ["Mo", "Tu", "We", "Th", "Fr"]
SLOT_MIN = 30

SNAP_M = 12             # a sign, paid space or hydrant further than this from any face is ignored
PAID_SPACE_M = 6.0      # Places has no length. Median spacing between space centres is 6.1 m
ROW_GAP_M = 9.0         # neighbours closer than this are one continuous row (90% of spacings are 5 to 9 m)
HYDRANT_CLEAR_M = 5.0   # Quebec Highway Safety Code: no parking within 5 m of a hydrant
CORNER_CLEAR_M = 5.0    # and within 5 m of an intersection

# arrow codes (FLECHE_PAN), for a viewer standing in the street facing the sign.
# Confirmed against the Agence de mobilité durable's signalec_to_curblr code
ARROW = {0: "both", 2: "left", 3: "right", 8: "both"}

# strictest first
USE_PRIORITY = ["no_stopping", "bus_stop", "no_parking", "implicit_no_parking", "reserved",
                "permit", "loading", "short_term", "paid"]


# ---------------------------------------------------------------- schedules
def parse_schedule(text):
    """'Mo-Fr 06:00-10:00,14:30-19:30 | Sa,Su 09:00-23:00' -> {day: [(start_min, end_min)]}."""
    week = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
    if text == "always":
        return {d: [(0, 1440)] for d in week}
    out = defaultdict(list)
    for part in text.split("|"):
        days_txt, times_txt = part.strip().split(" ", 1)
        if "-" in days_txt and "," not in days_txt:
            a, b = days_txt.split("-")
            days = week[week.index(a):week.index(b) + 1]
        else:
            days = days_txt.split(",")
        for slot in times_txt.split(","):
            s, e = slot.split("-")
            s = int(s[:2]) * 60 + int(s[3:]); e = int(e[:2]) * 60 + int(e[3:])
            for d in days:
                if e > s:
                    out[d].append((s, e))
                else:  # past midnight: the evening part is what matters for our daytime windows
                    out[d].append((s, 1440))
                    out[week[(week.index(d) + 1) % 7]].append((0, e))
    return dict(out)


def in_season(season, when=REFERENCE_DATE):
    if season == "all_year":
        return True
    a, b = season.split("..")
    start = date(when.year, int(a[:2]), int(a[3:]))
    end = date(when.year, int(b[:2]), int(b[3:]))
    return start <= when <= end


def slot_grid():
    """Every weekday half-hour from 07:00 to 19:00, tagged with its window."""
    slots = []
    for w, (h0, h1) in WINDOWS.items():
        for d in WEEKDAYS:
            for m in range(h0 * 60, h1 * 60, SLOT_MIN):
                slots.append((w, d, m + SLOT_MIN / 2))  # test the middle of the slot
    return slots


def active(sched, day, minute):
    return any(s <= minute < e for s, e in sched.get(day, []))


# ---------------------------------------------------------------- geometry helpers
def as_line(geom):
    """One LineString per face, so distances along it are well defined. A few faces have
    small gaps (e.g. a driveway apron in the géobase), so join their parts in order."""
    if geom.geom_type == "LineString":
        return geom
    merged = linemerge(geom)
    if merged.geom_type == "LineString":
        return merged
    return LineString([c for part in merged.geoms for c in part.coords])


def snap(points, faces, max_m=SNAP_M):
    """Nearest face for each point, with the distance along that face."""
    j = gpd.sjoin_nearest(points, faces[["blockface_id", "geometry"]], max_distance=max_m, distance_col="snap_m")
    j = j[~j.index.duplicated()].copy()
    j["d"] = [faces.geometry[i].project(p) for i, p in zip(j.index_right, j.geometry)]
    return j


def left_is_increasing(face, centre, d):
    """For a viewer in the street facing the curb at distance d, does 'left' mean increasing d?"""
    on_face = face.interpolate(d)
    on_c = nearest_points(centre, on_face)[0]
    n = np.array([on_face.x - on_c.x, on_face.y - on_c.y])  # street -> sidewalk
    n /= np.linalg.norm(n) or 1
    a, b = face.interpolate(max(d - 2, 0)), face.interpolate(min(d + 2, face.length))
    t = np.array([b.x - a.x, b.y - a.y])
    t /= np.linalg.norm(t) or 1
    return t @ np.array([-n[1], n[0]]) > 0


# ---------------------------------------------------------------- stretches from each source
def sign_stretches(faces, centres, rules):
    s = gpd.read_file(GPKG, layer="signs__Signalisation_Stationnement")
    s = s[s.DESCRIPTION_REP == "Réel"].merge(rules, left_on="CODE_RPA", right_on="code_rpa")
    s = s[s.use != "paid_zone"]  # paid metres come from Places
    j = snap(s, faces)
    print(f"signs: {len(s)} active, {len(j)} snapped to a face")

    rows = []
    for (face_i, code), g in j.groupby(["index_right", "CODE_RPA"]):
        face = faces.geometry[face_i]
        L = face.length
        poles = g.groupby("POTEAU_ID_POT").d.mean()  # one position per pole
        centre = centres[face_i]
        for r in g.itertuples():
            arrow = ARROW.get(r.FLECHE_PAN, "both")
            if arrow == "both":
                dirs = [1, -1]
            else:
                inc = left_is_increasing(face, centre, r.d)
                dirs = [1 if (arrow == "left") == inc else -1]
            d0 = poles[r.POTEAU_ID_POT]
            for k in dirs:
                # reach to the next pole with the same rule, else to the end of the face
                nxt = poles[(poles - d0) * k > 1]
                end = (nxt.min() if k > 0 else nxt.max()) if len(nxt) else (L if k > 0 else 0.0)
                rows.append((faces.blockface_id[face_i], min(d0, end), max(d0, end), r.use,
                             r.beneficiary if isinstance(r.beneficiary, str) else "", r.schedule,
                             r.season, "sign", code))
    return merge_same(rows)


def merge_same(rows):
    """Merge overlapping stretches that share face, use, schedule and source."""
    df = pd.DataFrame(rows, columns=["blockface_id", "from_m", "to_m", "use", "beneficiary",
                                     "schedule", "season", "source", "ref"])
    out = []
    for key, g in df.groupby(["blockface_id", "use", "beneficiary", "schedule", "season", "source", "ref"]):
        g = g.sort_values("from_m")
        cur_a, cur_b = None, None
        for a, b in zip(g.from_m, g.to_m):
            if cur_a is None:
                cur_a, cur_b = a, b
            elif a <= cur_b + 0.5:
                cur_b = max(cur_b, b)
            else:
                out.append((key[0], cur_a, cur_b) + key[1:])
                cur_a, cur_b = a, b
        out.append((key[0], cur_a, cur_b) + key[1:])
    return pd.DataFrame(out, columns=["blockface_id", "from_m", "to_m", "use", "beneficiary",
                                      "schedule", "season", "source", "ref"])


def paid_schedules():
    """Paid hours per space, from the type U (usage) rules in the Agence's tables."""
    reg = pd.read_csv(TABLES / "paid_parking__Réglementations.csv")
    emp = pd.read_csv(TABLES / "paid_parking__EmplacementReglementation.csv")
    rp = pd.read_csv(TABLES / "paid_parking__ReglementationPeriode.csv", encoding="latin-1")
    per = pd.read_csv(TABLES / "paid_parking__Periodes.csv", encoding="latin-1").set_index("nID")
    usage = set(reg[reg.Type == "U"].Name)
    day_cols = [("bLun", "Mo"), ("bMar", "Tu"), ("bMer", "We"), ("bJeu", "Th"), ("bVen", "Fr"),
                ("bSam", "Sa"), ("bDim", "Su")]

    def schedule_for(code):
        parts = []
        for p in rp[rp.sCode == code].noPeriode:
            if p not in per.index:
                continue
            row = per.loc[p]
            days = [d for col, d in day_cols if row[col] == 1]
            if days:
                parts.append(f"{','.join(days)} {row.dtHeureDebut[:5]}-{row.dtHeureFin[:5]}")
        return " | ".join(parts)

    sched_by_code = {c: schedule_for(c) for c in usage}
    e = emp[emp.sCodeAutocollant.isin(usage)]
    return {space: sched_by_code[code] for space, code in zip(e.sNoEmplacement, e.sCodeAutocollant)
            if sched_by_code[code]}


def paid_stretches(faces):
    pl = gpd.read_file(GPKG, layer="paid_parking__Places")
    # use the centre of each space, not the pay station it shares with its neighbours
    pl = gpd.GeoDataFrame(pl.drop(columns="geometry"), crs="EPSG:4326",
                          geometry=gpd.points_from_xy(pl.nPositionCentreLongitude, pl.nPositionCentreLatitude)).to_crs(faces.crs)
    sched = paid_schedules()
    j = snap(pl, faces)
    print(f"paid spaces: {len(pl)}, {len(j)} snapped, {sum(s in sched for s in j.sNoPlace)} with paid hours")
    rows = []
    j = j[j.sNoPlace.isin(sched)]
    for face_i, g in j.groupby("index_right"):
        L = faces.geometry[face_i].length
        g = g.sort_values("d")
        d = g.d.to_numpy()
        # a row of spaces fills the curb: neighbours up to ROW_GAP_M apart meet halfway.
        # Spaces with no close neighbour on a side get half a standard space on that side
        for k, r in enumerate(g.itertuples()):
            left = (d[k] + d[k - 1]) / 2 if k > 0 and d[k] - d[k - 1] <= ROW_GAP_M else d[k] - PAID_SPACE_M / 2
            right = (d[k] + d[k + 1]) / 2 if k < len(d) - 1 and d[k + 1] - d[k] <= ROW_GAP_M else d[k] + PAID_SPACE_M / 2
            rows.append((faces.blockface_id[face_i], max(left, 0), min(right, L), "paid", "",
                         sched[r.sNoPlace], "all_year", "places", r.sNoPlace))
    return pd.DataFrame(rows, columns=["blockface_id", "from_m", "to_m", "use", "beneficiary",
                                       "schedule", "season", "source", "ref"])


def implicit_stretches(faces):
    h = gpd.read_file(GPKG, layer="hydrants__Bornes_d_incendie")
    j = snap(h, faces)
    rows = []
    for r in j.itertuples():
        L = faces.geometry[r.index_right].length
        rows.append((faces.blockface_id[r.index_right], max(r.d - HYDRANT_CLEAR_M, 0), min(r.d + HYDRANT_CLEAR_M, L),
                     "implicit_no_parking", "", "always", "all_year", "hydrant", str(r.ID_BI)))
    print(f"hydrants: {len(h)}, {len(j)} snapped")
    for f in faces.itertuples():
        L = f.geometry.length
        for a, b in [(0, min(CORNER_CLEAR_M, L)), (max(L - CORNER_CLEAR_M, 0), L)]:
            rows.append((f.blockface_id, a, b, "implicit_no_parking", "", "always", "all_year", "intersection", ""))
    return pd.DataFrame(rows, columns=["blockface_id", "from_m", "to_m", "use", "beneficiary",
                                       "schedule", "season", "source", "ref"])


# ---------------------------------------------------------------- resolve overlaps per window
def summarise(faces, stretches):
    slots = slot_grid()
    per_window = defaultdict(int)
    for w, _, _ in slots:
        per_window[w] += 1
    rank = {u: i for i, u in enumerate(USE_PRIORITY)}
    st = stretches[stretches.season.map(in_season)].copy()
    st["sched"] = st.schedule.map(parse_schedule)

    rows = []
    for f in faces.itertuples():
        L = f.geometry.length
        mine = st[st.blockface_id == f.blockface_id]
        cuts = sorted({0.0, L, *mine.from_m.clip(0, L), *mine.to_m.clip(0, L)})
        metres = {w: defaultdict(float) for w in WINDOWS}
        for a, b in zip(cuts, cuts[1:]):
            if b - a < 0.01:
                continue
            here = mine[(mine.from_m <= a + 0.01) & (mine.to_m >= b - 0.01)]
            for w, day, minute in slots:
                uses = [u for u, sc in zip(here.use, here.sched) if active(sc, day, minute)]
                use = min(uses, key=rank.get) if uses else "free"
                metres[w][use] += (b - a) / per_window[w]
        for w in WINDOWS:
            row = {"blockface_id": f.blockface_id, "window": w, "length_m": round(L, 1)}
            for u in USE_PRIORITY + ["free"]:
                row[f"m_{u}"] = round(metres[w][u], 1)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    faces = gpd.read_file(BLOCKFACES)
    faces["geometry"] = faces.geometry.map(as_line)
    cl = gpd.read_file(GPKG, layer="geobase__Geobase").set_index("ID_TRC")
    centres = {i: cl.loc[[int(k) for k in t.split(",")]].union_all() for i, t in faces.id_trc.items()}
    rules = pd.read_csv(RULES, encoding="utf-8-sig")

    stretches = pd.concat([sign_stretches(faces, centres, rules), paid_stretches(faces), implicit_stretches(faces)],
                          ignore_index=True)
    stretches["from_m"] = stretches.from_m.round(1)
    stretches["to_m"] = stretches.to_m.round(1)
    stretches = stretches[stretches.to_m - stretches.from_m >= 0.5]
    stretches["length_m"] = (stretches.to_m - stretches.from_m).round(1)
    geom = faces.set_index("blockface_id").geometry
    stretches = gpd.GeoDataFrame(stretches, crs=faces.crs,
                                 geometry=[substring(geom[b], a, c) for b, a, c in zip(stretches.blockface_id, stretches.from_m, stretches.to_m)])

    summary = summarise(faces, stretches)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    stretches.to_file(OUT, layer="stretches", driver="GPKG")
    gpd.GeoDataFrame(summary).to_file(OUT, layer="supply_by_window", driver="GPKG")
    summary.to_csv(OUT_CSV, index=False)

    print(f"\n{len(stretches)} stretches -> {OUT.relative_to(ROOT)}")
    km = summary.groupby("window")[[c for c in summary.columns if c.startswith("m_")]].sum() / 1000
    print("\nkm of curb by use and window (all faces):")
    print(km.T.round(2).to_string())


if __name__ == "__main__":
    main()
