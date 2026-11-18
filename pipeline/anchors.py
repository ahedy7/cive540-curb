"""
Step 3a, anchors: a GPS track for the clips filmed with no GPX log, rebuilt from a route log.

The route log (pipeline/route_log.csv) was written by hand while watching the L clips: one
row per street driven, with the clock time the car entered and left it and the turn taken at
the end. Each turn is a known place (the crossing of the two streets, from the géobase) at a
known time. Between two turns the car follows the street's centre line.

Where along the street the car is at each second comes from the video itself. Frame-to-frame
pixel change (motion, from sync.py) is high when the car moves and low when it waits at a
light. It is turned into speed with a curve fitted on the L clips that do have GPS, then the
street's length is shared out over the seconds in proportion to that speed. Constant speed
would put a car waiting 40 s at a red light halfway down the block.

Open ends: the first street of a clip and the last one before a gap have only one known turn.
From there the car is walked along the street, backwards or forwards, by the predicted speed.
The direction comes from the turn (a right turn onto a street heading east means the car was
heading north before it). Seconds with no video stop the walk.

Turns logged across a short gap between clips: if the next street starts partway into its
clip, the turn is placed at that start and the previous street runs up to it.

Output: video/<date>_route_log.csv (time, lat, lon, one row per second), read by geotag.py like
a GPX. Clips with no GPX on those days are pointed at it in pipeline/clips.csv. For Friday the
file also holds the real GPX after the log ends, so a clip that runs into the GPS part uses both.
Run sync.py afterwards to find the forward camera's clock offset against the new track.

--check measures the timing method on L clips that have GPS: turns are found in the GPS track,
positions in between are rebuilt from motion (and from constant speed, for comparison) along
the true path, and compared with the GPS.

Usage:
    python pipeline/anchors.py
    python pipeline/anchors.py --check
"""
import re
import sys
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, nearest_points, substring

from common import DERIVED, FRAMES, LOCAL_TZ, MANIFEST, ROOT, VIDEO, load_manifest, read_track
from sync import motion, speed_at

CRS = "EPSG:32188"
LOG = ROOT / "pipeline" / "route_log.csv"
DAYS = {"Monday": "20261109", "Friday": "20261113"}
VIA = {("Monday", 5, 1064): ["Rue Drummond"]}   # U-turn on Sherbrooke at Drummond (log note)
TURN = {"Left Turn": "left", "Right Turn": "right"}
CONTIG_S = 2       # rows this close in time are one continuous drive
GAP_TURN_S = 60    # across a gap up to this long, a street starting mid-clip marks the turn
NEAR_CROSS_M = 30  # streets that meet without a shared node (one continues into the other)
BUFFER_M = 1200    # street search area around the study area
TURN_DEG = 60      # --check: heading change that counts as a turn


# ---------------------------------------------------------------- streets
def norm(name):
    """'Boul de Maisonneuve O' and 'De Maisonneuve' -> 'maisonneuve'."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"^(rue|av|avenue|boul|boulevard|ch|chemin)\s+", "", s)
    s = re.sub(r"\s+(o|ouest|e|est)$", "", s)
    s = re.sub(r"^(de la|du|de|des)\s+", "", s)
    s = s.replace("sainte-", "ste-").replace("saint-", "st-")
    return re.sub(r"[\s-]", "", s)


def load_streets():
    g = gpd.read_file(ROOT / "data" / "raw" / "geobase" / "Géobase.json").to_crs(CRS)
    area = gpd.read_file(ROOT / "study_area.geojson").to_crs(CRS).buffer(BUFFER_M).union_all()
    g = g[g.intersects(area) & g.TYP_VOIE.ne("tunnel")].copy()
    g["key"] = g.NOM_VOIE.map(norm)
    return {k: linemerge(list(v.geometry)) for k, v in g.groupby("key")}


def lines_of(geom):
    return list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]


def crossings(a, b):
    """Every place streets a and b meet (both carriageways of a divided street count apart)."""
    x = a.intersection(b)
    pts = [p for p in getattr(x, "geoms", [x]) if p.geom_type == "Point" and not p.is_empty]
    if not pts:
        p, q = nearest_points(a, b)
        if p.distance(q) <= NEAR_CROSS_M:
            pts = [Point((p.x + q.x) / 2, (p.y + q.y) / 2)]
    return pts


def heading(p, q):
    return np.degrees(np.arctan2(q.x - p.x, q.y - p.y)) % 360


def along(street, p, q):
    """The street's centre line from p to q, or a straight line if they are on different pieces."""
    line = min(lines_of(street), key=lambda l: l.distance(p) + l.distance(q))
    if line.distance(p) > 5 or line.distance(q) > 5:
        return LineString([p, q])
    a, b = line.project(p), line.project(q)
    if abs(a - b) < 0.5:
        return LineString([p, q])
    sub = substring(line, min(a, b), max(a, b))
    return sub if a < b else LineString(sub.coords[::-1])


def walk(street, p, deg, dist):
    """Points along the street from p, setting off in compass direction deg, at distances dist."""
    line = min(lines_of(street), key=lambda l: l.distance(p))
    s0 = line.project(p)
    ahead = line.interpolate(min(s0 + 5, line.length))
    sign = 1 if np.cos(np.radians(heading(line.interpolate(s0), ahead) - deg)) > 0 else -1
    out = []
    for d in dist:
        s = s0 + sign * d
        out.append(line.interpolate(s) if 0 <= s <= line.length else None)   # past the end: no position
    return out


# ---------------------------------------------------------------- speed from motion
def l_clips_with_gps(m):
    return m[(m.side == "L") & (m.gpx != "") & ~m.gpx.str.endswith("_route_log.csv")]


def clip_motion(c):
    stem = c.file.rsplit(".", 1)[0]
    return motion(stem, len(list((FRAMES / stem).glob("*.jpg"))))


def speed_model(m):
    """Motion -> speed (m/s) curve: median GPS speed in each motion bin, made non-decreasing."""
    mot, spd = [], []
    for c in l_clips_with_gps(m).itertuples():
        mo = clip_motion(c)
        t0 = (c.start_utc - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds() + c.offset_s
        sp = speed_at(read_track(VIDEO / c.gpx), t0 + np.arange(len(mo)) - 0.5)   # m[i]: seconds i-1 to i
        ok = ~np.isnan(mo) & ~np.isnan(sp)
        mot.append(mo[ok])
        spd.append(sp[ok])
    mot, spd = np.concatenate(mot), np.concatenate(spd)
    edges = np.unique(np.quantile(mot, np.linspace(0, 1, 26)))
    b = np.clip(np.digitize(mot, edges[1:-1]), 0, len(edges) - 2)
    x = np.array([np.median(mot[b == i]) for i in range(len(edges) - 1)])
    y = np.maximum.accumulate([np.median(spd[b == i]) for i in range(len(edges) - 1)])
    rho = pd.Series(mot).corr(pd.Series(spd), method="spearman")
    print(f"speed model: {len(mot):,} seconds from {len(l_clips_with_gps(m))} L clips with GPS, "
          f"Spearman {rho:.2f}. Stopped {y[0]:.1f} m/s, fastest bin {y[-1]:.1f} m/s")
    return lambda v: np.interp(v, x, y)


def day_speed(m, date, model):
    """Predicted speed for every UTC second of the day covered by an L clip (NaN in gaps)."""
    parts = []
    for c in m[(m.date == date) & (m.side == "L")].itertuples():
        mo = clip_motion(c)
        t0 = int((c.start_utc - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds() + c.offset_s)
        parts.append(pd.Series(model(mo), index=t0 + np.arange(len(mo))))
    return pd.concat(parts).groupby(level=0).first().sort_index()


# ---------------------------------------------------------------- route log -> segments
def segments(day, date):
    """One row per street driven: street, start and end (UTC seconds), turn at the end, vias."""
    log = pd.read_csv(LOG)
    log = log[log.day == day].reset_index(drop=True)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")

    def utc(hms):
        return int((pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:]} {hms}").tz_localize(LOCAL_TZ) - epoch).total_seconds())

    rows = []
    for r in log.itertuples():
        rows.append({"street": r.segment_name, "t0": utc(r.actual_start_time), "t1": utc(r.actual_end_time),
                     "mid_clip": r.seg_start_vid_s > 0, "turn": TURN.get(r.next_movement),
                     "via": VIA.get((day, r.video, r.seg_start_vid_s), []), "start_known": False, "end_known": False})
    segs = []   # merge a street split across two clips
    for s in rows:
        if segs and norm(segs[-1]["street"]) == norm(s["street"]) and s["t0"] - segs[-1]["t1"] <= CONTIG_S:
            segs[-1].update(t1=s["t1"], turn=s["turn"], via=segs[-1]["via"] + s["via"])
        else:
            segs.append(s)
    for a, b in zip(segs, segs[1:]):
        gap = b["t0"] - a["t1"]
        if gap <= CONTIG_S or (gap <= GAP_TURN_S and b["mid_clip"]):
            a["t1"] = b["t0"]
            a["end_known"] = b["start_known"] = True
    return segs


def place_turns(segs, streets):
    """Pick one crossing per turn (and via), so that consecutive turns are as close as possible."""
    nodes = []   # (segment index, 'end' or via number, candidate points)
    for i, s in enumerate(segs):
        a = streets[norm(s["street"])]
        for k, v in enumerate(s["via"]):
            nodes.append((i, k, crossings(a, streets[norm(v)])))
        if s["end_known"]:
            nodes.append((i, "end", crossings(a, streets[norm(segs[i + 1]["street"])])))
    for i, k, c in nodes:
        if not c:
            raise SystemExit(f"{segs[i]['street']}: no crossing found for turn {k}")
    # Viterbi over candidates with straight-line distance as the cost
    cost = [np.zeros(len(nodes[0][2]))]
    back = []
    for (_, _, prev), (_, _, cur) in zip(nodes, nodes[1:]):
        d = np.array([[p.distance(q) for p in prev] for q in cur]) + cost[-1]
        back.append(d.argmin(axis=1))
        cost.append(d.min(axis=1))
    pick = [int(cost[-1].argmin())]
    for b in reversed(back):
        pick.append(int(b[pick[-1]]))
    pick.reverse()
    for (i, k, c), j in zip(nodes, pick):
        if k == "end":
            segs[i]["end_pt"] = c[j]
            segs[i + 1]["start_pt"] = c[j]
        else:
            segs[i].setdefault("via_pts", []).append(c[j])


def share(speed, t0, t1):
    """Cumulative share of the distance covered at each second t0..t1, from predicted speed."""
    v = speed.reindex(range(t0 + 1, t1 + 1)).to_numpy()
    v = np.where(np.isnan(v), np.nanmedian(v) if np.isfinite(v).any() else 1.0, v)
    w = np.concatenate([[0], np.cumsum(v)])
    return w / w[-1] if w[-1] > 0 else np.linspace(0, 1, len(w))


def build(day, date, m, streets, model):
    segs = segments(day, date)
    place_turns(segs, streets)
    speed = day_speed(m, date, model)
    pts, report = {}, []
    # known paths first, so open ends can take their direction from a neighbour
    for s in segs:
        if s["start_known"] and s["end_known"]:
            a = streets[norm(s["street"])]
            stops = [s["start_pt"]] + s.get("via_pts", []) + [s["end_pt"]]
            s["path"] = linemerge([along(a, p, q) for p, q in zip(stops, stops[1:])])
            if s["path"].geom_type != "LineString":
                s["path"] = LineString([c for l in s["path"].geoms for c in l.coords])
    for i, s in enumerate(segs):
        a = streets[norm(s["street"])]
        if "path" in s:
            f = share(speed, s["t0"], s["t1"])
            for t, x in zip(range(s["t0"], s["t1"] + 1), f):
                pts[t] = s["path"].interpolate(x, normalized=True)
            how = f"{s['path'].length:.0f} m between turns"
        elif s["end_known"] and i + 1 < len(segs) and "path" in segs[i + 1] and s["turn"]:
            # walk back from the end turn
            nxt = segs[i + 1]["path"]
            h = heading(nxt.interpolate(0), nxt.interpolate(min(20, nxt.length)))
            h_here = (h + (90 if s["turn"] == "left" else -90)) % 360
            secs = list(range(s["t1"], s["t0"] - 1, -1))
            v = speed.reindex(range(s["t0"] + 1, s["t1"] + 1)).to_numpy()[::-1]   # v[j]: seconds t1-j-1 to t1-j
            stop = np.argmax(np.isnan(v)) if np.isnan(v).any() else len(v)
            dist = np.concatenate([[0], np.cumsum(v[:stop])])
            for t, p in zip(secs, walk(a, s["end_pt"], (h_here + 180) % 360, dist)):
                if p is not None:
                    pts[t] = p
            how = f"open start, walked back {dist[-1]:.0f} m over {len(dist) - 1} s"
        elif s["start_known"] and i > 0 and "path" in segs[i - 1] and segs[i - 1]["turn"]:
            prv = segs[i - 1]["path"]
            h = heading(prv.interpolate(max(prv.length - 20, 0)), prv.interpolate(prv.length))
            h_here = (h + (90 if segs[i - 1]["turn"] == "right" else -90)) % 360
            v = speed.reindex(range(s["t0"] + 1, s["t1"] + 1)).to_numpy()
            stop = np.argmax(np.isnan(v)) if np.isnan(v).any() else len(v)
            dist = np.concatenate([[0], np.cumsum(v[:stop])])
            for t, p in zip(range(s["t0"], s["t1"] + 1), walk(a, s["start_pt"], h_here, dist)):
                if p is not None:
                    pts[t] = p
            how = f"open end, walked on {dist[-1]:.0f} m over {len(dist) - 1} s"
        else:
            how = "no known turn next to it, skipped"
        report.append(f"  {pd.Timestamp(s['t0'], unit='s', tz='UTC').tz_convert(LOCAL_TZ):%H:%M:%S} "
                      f"{s['street']:26s} {s['t1'] - s['t0']:4d} s  {how}")
    print(f"\n{day} ({date}):")
    print("\n".join(report))
    t = sorted(pts)
    g = gpd.GeoSeries([pts[k] for k in t], crs=CRS).to_crs("EPSG:4326")
    return pd.DataFrame({"time": pd.to_datetime(t, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "lat": g.y.round(7), "lon": g.x.round(7)})


# ---------------------------------------------------------------- check on clips with GPS
def check(m, model):
    """Rebuild positions between GPS turns from motion and from constant speed, compare with GPS."""
    err_m, err_c = [], []
    for c in l_clips_with_gps(m).itertuples():
        tr = read_track(VIDEO / c.gpx)
        xy = gpd.GeoSeries(gpd.points_from_xy(tr.lon, tr.lat), crs="EPSG:4326").to_crs(CRS)
        ts = (tr.time - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
        mo = clip_motion(c)
        t0 = int((c.start_utc - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds() + c.offset_s)
        t = t0 + np.arange(len(mo))
        inside = (t >= ts[0]) & (t <= ts[-1])
        t, v = t[inside], model(mo)[inside]
        x, y = np.interp(t, ts, xy.x), np.interp(t, ts, xy.y)
        # turns: heading 10 s before vs 10 s after differs by more than TURN_DEG
        h = np.degrees(np.arctan2(np.gradient(x), np.gradient(y)))
        moving = np.hypot(np.gradient(x), np.gradient(y)) > 1
        hs = pd.Series(np.where(moving, h, np.nan)).ffill().bfill().to_numpy()
        turn = [i for i in range(10, len(t) - 10)
                if abs((hs[i + 10] - hs[i - 10] + 180) % 360 - 180) > TURN_DEG]
        cuts = [g[len(g) // 2] for g in np.split(turn, np.where(np.diff(turn) > 5)[0] + 1) if len(g)]
        for a, b in zip(cuts, cuts[1:]):
            path = LineString(np.column_stack([x[a:b + 1], y[a:b + 1]]))
            if path.length < 50:
                continue
            truth = [Point(x[i], y[i]) for i in range(a, b + 1)]
            w = np.concatenate([[0], np.cumsum(v[a + 1:b + 1])])
            for f, err in ((w / w[-1], err_m), (np.linspace(0, 1, b - a + 1), err_c)):
                err += [path.interpolate(s, normalized=True).distance(p) for s, p in zip(f, truth)]
    for name, e in (("motion", err_m), ("constant speed", err_c)):
        e = np.array(e)
        print(f"{name:15s} {len(e):,} seconds: median error {np.median(e):.1f} m, 90th pct {np.percentile(e, 90):.1f} m, "
              f"within 15 m {np.mean(e <= 15):.0%}")


def main():
    m = load_manifest()
    model = speed_model(m)
    if "--check" in sys.argv:
        check(m, model)
        return
    streets = load_streets()
    man = pd.read_csv(MANIFEST, dtype=str)
    for day, date in DAYS.items():
        track = build(day, date, m, streets, model)
        gps = sorted({g for g in m[(m.date == date)].gpx if g and not g.endswith("_route_log.csv")})
        if gps:   # the real GPX takes over where it starts
            real = pd.concat([read_track(VIDEO / g) for g in gps])
            track = track[pd.to_datetime(track.time, utc=True) < real.time.min()]
            real["time"] = real.time.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            track = pd.concat([track, real[["time", "lat", "lon"]]], ignore_index=True)
        name = f"{date}_route_log.csv"
        track.to_csv(VIDEO / name, index=False)
        # clips that start before the real GPS (or have none) use the new track
        first = pd.to_datetime(track.time, utc=True).min()
        start_gps = real.time.min() if gps else None
        for c in m[m.date == date].itertuples():
            if c.gpx == "" or (start_gps is not None and c.start_utc < pd.Timestamp(start_gps) and c.gps_cover < 0.9):
                i = man.index[man.clip_id == c.clip_id]
                man.loc[i, ["gpx", "note"]] = [name, "track from route log (anchors.py)"]
        print(f"-> video/{name}: {len(track):,} positions from {first.tz_convert(LOCAL_TZ):%H:%M:%S}")
    man.to_csv(MANIFEST, index=False)
    print(f"-> {MANIFEST.relative_to(ROOT)} updated. Next: python pipeline/sync.py, then geotag.py")


if __name__ == "__main__":
    main()
