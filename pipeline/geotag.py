"""
Steps 3 and 4, geotag and snap: give every frame a time and position, then the block face
its camera is looking at.

Works from clip times and lengths in pipeline/clips.csv, so it runs before frames are
extracted. A frame is one second of video: frame k of a clip is at start + offset_s + (k - 1).

Geotag. Position is interpolated in time between GPS fixes. Trackers skip fixes when the
phone is still (Open GPX Tracker does), so a gap is fine if the phone barely moved across it.
Frames outside the track, or in a gap where the phone moved more than GAP_MOVE_M, get no
position. Heading and speed come from the positions HEADING_S seconds either side.
While stopped, the last heading is kept.

Snap. Three camera types, set per clip in clips.csv:
    R  side window, right curb
    L  side window, left curb. Kept only on one-way streets (on two-way streets it sees
       traffic and the far curb)
    F  forward, through the windshield. Sees the curb ahead, so it is matched from a point
       LOOKAHEAD_M ahead of the car. Counts for the right curb everywhere, and for the left
       curb too on one-way streets
First the street block: the closest block face within SNAP_M that runs within MAX_ANGLE of the
travel direction. Then which of the block's two faces is right and which is left, taken from
the two curb lines themselves, so GPS error across a narrow street cannot flip the side.

Flags:
    gps_ok            frame has a trusted position
    stationary        speed under STOP_MS (frames stay, but repeat the same view)
    near_intersection within END_M of either end of the face (corners, crossing traffic)
    keep              gps_ok, snapped to a face, not near an intersection, and not an L frame
                      on a two-way street. These frames go on to the YOLO filter.
    blockface_id_2, keep_2   F frames only: the left face of a one-way street, and its keep flag

Outputs:
    data/derived/frames.gpkg  layer "frames" (points, EPSG:32188), one row per frame
    data/derived/coverage.csv sampled faces x date: frames kept and share of the face seen

Usage:
    python pipeline/geotag.py
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import linemerge

from common import DERIVED, FRAMES, LOCAL_TZ, ROOT, VIDEO, load_manifest, read_track

CRS = "EPSG:32188"
HEADING_S = 2          # heading from positions 2 s before and after
STOP_MS = 1.0          # below 1 m/s (3.6 km/h) the car is stopped
GAP_MOVE_M = 15        # a GPS gap is trusted if the phone moved less than this across it
SNAP_M = 25            # curb line within 25 m of the car (wide streets put the far curb ~20 m away)
MAX_ANGLE = 35         # face must run within 35 degrees of the travel direction
END_M = 10             # drop frames within 10 m of a face end (intersections)
LOOKAHEAD_M = 15       # F camera: the curb it shows best is about 5 to 30 m ahead
SEE_M = {"R": 15, "L": 15, "F": 30}   # curb length one frame shows, for the share-of-face-seen measure


def frame_table(m):
    rows = []
    for c in m[m.use == "yes"].itertuples():
        n = int(np.floor(c.duration_s))
        stem = c.file.rsplit(".", 1)[0]
        k = np.arange(1, n + 1)
        t = c.start_utc + pd.to_timedelta(c.offset_s + k - 1, unit="s")
        rows.append(pd.DataFrame({"frame_id": [f"{c.clip_id}_{i:06d}" for i in k], "clip_id": c.clip_id,
                                  "file": c.file, "date": c.date, "side": c.side, "time_utc": t,
                                  "path": [f"frames/{stem}/{i:06d}.jpg" for i in k], "gpx": c.gpx}))
    f = pd.concat(rows, ignore_index=True)
    f["extracted"] = [(ROOT / p).exists() for p in f.path]
    return f


def seconds(times):
    """Seconds since 1970 as floats. Safer than astype(int64), whose unit depends on the dtype."""
    return (times - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def geotag(frames):
    """Add x, y (EPSG:32188), gps_ok, speed_ms, heading_deg to the frames, track by track."""
    for col in ("x", "y", "speed_ms", "heading_deg"):
        frames[col] = np.nan
    frames["gps_ok"] = False
    for name, idx in frames[frames.gpx != ""].groupby("gpx").groups.items():
        tr = read_track(VIDEO / name)
        pts = gpd.GeoSeries(gpd.points_from_xy(tr.lon, tr.lat), crs="EPSG:4326").to_crs(CRS)
        ts = seconds(tr.time)
        tx, ty = pts.x.to_numpy(), pts.y.to_numpy()
        f = frames.loc[idx]
        t = seconds(f.time_utc)

        def at(when):
            return np.interp(when, ts, tx), np.interp(when, ts, ty)

        x, y = at(t)
        # trust: inside the track, and any gap around the frame was a stop, not a drive
        j = np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)
        gap_move = np.hypot(tx[j] - tx[j - 1], ty[j] - ty[j - 1])
        ok = (t >= ts[0]) & (t <= ts[-1]) & ((ts[j] - ts[j - 1] <= 5) | (gap_move < GAP_MOVE_M))
        x0, y0 = at(t - HEADING_S)
        x1, y1 = at(t + HEADING_S)
        speed = np.hypot(x1 - x0, y1 - y0) / (2 * HEADING_S)
        heading = np.degrees(np.arctan2(x1 - x0, y1 - y0)) % 360   # 0 = grid north of the projection
        heading = pd.Series(np.where(speed >= STOP_MS, heading, np.nan)).ffill().bfill().to_numpy()
        frames.loc[idx, ["x", "y", "speed_ms", "heading_deg"]] = np.column_stack([x, y, speed, heading])
        frames.loc[idx, "gps_ok"] = ok
    return frames


def snap(frames, faces):
    """Add, per frame, the right and left block face of the street block in view."""
    faces = faces.copy()
    faces["geometry"] = faces.geometry.map(lambda g: linemerge(g) if g.geom_type == "MultiLineString" else g)
    sindex = faces.sindex
    geoms, ids = faces.geometry.to_numpy(), faces.blockface_id.to_numpy()
    blocks = faces.blockface_id.str.rsplit("_", n=1).str[0].to_numpy()
    rows = []
    for r in frames.itertuples():
        row = {"face_right": None, "face_left": None, "dist_right": np.nan, "dist_left": np.nan,
               "along_right": np.nan, "along_left": np.nan}
        if r.gps_ok and not np.isnan(r.heading_deg):
            hx, hy = np.sin(np.radians(r.heading_deg)), np.cos(np.radians(r.heading_deg))
            ahead = LOOKAHEAD_M if r.side == "F" else 0
            at = Point(r.x + ahead * hx, r.y + ahead * hy)
            # candidates: faces near the point that run in the travel direction
            cand = {}   # index -> (distance, nearest point on the face)
            for i in sindex.query(at.buffer(SNAP_M)):
                g = geoms[i]
                d = g.distance(at)
                if d > SNAP_M:
                    continue
                parts = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
                part = min(parts, key=lambda p: p.distance(at))
                s = part.project(at)
                a, b = part.interpolate(max(s - 3, 0)), part.interpolate(min(s + 3, part.length))
                dx, dy = b.x - a.x, b.y - a.y
                norm = np.hypot(dx, dy)
                if norm and abs(dx * hx + dy * hy) / norm >= np.cos(np.radians(MAX_ANGLE)):
                    cand[i] = (d, part.interpolate(s))
            if cand:
                blk = blocks[min(cand, key=lambda i: cand[i][0])]
                pair = [i for i in cand if blocks[i] == blk]
                if len(pair) == 2:
                    i, j = pair
                    pi, pj = cand[i][1], cand[j][1]
                    right, left = (i, j) if hx * (pi.y - pj.y) - hy * (pi.x - pj.x) < 0 else (j, i)
                else:   # only one side exists: use the car's position to tell which
                    i = pair[0]
                    near = cand[i][1]
                    on_right = hx * (near.y - at.y) - hy * (near.x - at.x) < 0
                    right, left = (i, None) if on_right else (None, i)
                for name, k in (("right", right), ("left", left)):
                    if k is not None:
                        row[f"face_{name}"] = ids[k]
                        row[f"dist_{name}"] = cand[k][0]
                        row[f"along_{name}"] = geoms[k].project(at)
        rows.append(row)
    return pd.concat([frames, pd.DataFrame(rows, index=frames.index)], axis=1)


def assign(f, faces):
    """Turn right/left faces into the face(s) each frame counts for, by camera type."""
    length = faces.set_index("blockface_id").length_m
    one_way = faces.set_index("blockface_id").traffic.ne("two_way")
    use_left = f.side == "L"
    f["blockface_id"] = np.where(use_left, f.face_left, f.face_right)
    f["dist_m"] = np.where(use_left, f.dist_left, f.dist_right)
    f["along_m"] = np.where(use_left, f.along_left, f.along_right)
    f["blockface_id_2"] = np.where((f.side == "F") & f.face_left.map(one_way).fillna(False).astype(bool),
                                   f.face_left, None)
    f["along_m_2"] = np.where(f.blockface_id_2.notna(), f.along_left, np.nan)
    for sfx in ("", "_2"):
        L = f["blockface_id" + sfx].map(length)
        f["face_len_m" + sfx] = L
        along = f["along_m" + sfx]
        f["near_intersection" + sfx] = (along < END_M) | (along > L - END_M)
    two_way = f.blockface_id.map(one_way).eq(False)
    f["keep"] = f.gps_ok & f.blockface_id.notna() & ~f.near_intersection & ~(use_left & two_way)
    f["keep_2"] = f.gps_ok & f.blockface_id_2.notna() & ~f.near_intersection_2
    return f.drop(columns=["face_right", "face_left", "dist_right", "dist_left", "along_right", "along_left"])


def coverage(frames, sample):
    """Per sampled face and date: frames kept, and the share of the face's length seen."""
    cols = ["date", "side", "stationary"]
    one = frames.loc[frames.keep, cols + ["blockface_id", "along_m", "face_len_m"]]
    two = frames.loc[frames.keep_2, cols + ["blockface_id_2", "along_m_2", "face_len_m_2"]]
    two.columns = one.columns
    k = pd.concat([one, two])
    k = k[k.blockface_id.isin(sample)]
    rows = []
    for (fid, date), g in k.groupby(["blockface_id", "date"]):
        L = g.face_len_m.iat[0]
        seen = np.zeros(int(np.ceil(L)) + 1, dtype=bool)
        for s, side in zip(g.along_m, g.side):
            half = SEE_M[side] / 2
            seen[max(int(s - half), 0):int(s + half) + 1] = True
        rows.append({"blockface_id": fid, "date": date, "frames": len(g),
                     "moving_frames": int((~g.stationary).sum()),
                     "share_seen": round(min(seen.sum() / max(L - 2 * END_M, 1), 1.0), 2)})
    c = pd.DataFrame(rows, columns=["blockface_id", "date", "frames", "moving_frames", "share_seen"])
    wide = c.pivot(index="blockface_id", columns="date", values="share_seen").reindex(sorted(sample)).fillna(0)
    return c, wide


def main():
    m = load_manifest()
    faces = gpd.read_file(DERIVED / "blockfaces.gpkg")
    sample = set(gpd.read_file(DERIVED / "sample.gpkg").blockface_id)

    f = frame_table(m)
    f = geotag(f)
    f = assign(snap(f, faces), faces)
    f["stationary"] = f.speed_ms < STOP_MS
    f["in_sample"] = f.blockface_id.isin(sample) | f.blockface_id_2.isin(sample)
    f["time_local"] = f.time_utc.dt.tz_convert(LOCAL_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
    f["time_utc"] = f.time_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    g = gpd.GeoDataFrame(f, geometry=gpd.points_from_xy(f.x.fillna(0), f.y.fillna(0)), crs=CRS)
    g.loc[~g.gps_ok, "geometry"] = None
    out = DERIVED / "frames.gpkg"
    g.drop(columns=["x", "y"]).to_file(out, layer="frames", driver="GPKG")
    cov, wide = coverage(f, sample)
    cov.to_csv(DERIVED / "coverage.csv", index=False)

    # report
    pd.set_option("display.width", 200)
    print(f.groupby(["date", "side"]).agg(
        frames=("frame_id", "size"), extracted=("extracted", "sum"), gps_ok=("gps_ok", "sum"),
        on_a_face=("blockface_id", lambda s: s.notna().sum()), kept=("keep", "sum"),
        left_too=("keep_2", "sum"),
        kept_in_sample=("in_sample", lambda s: (s & f.loc[s.index, "keep"]).sum()),
        stationary=("stationary", "sum")).to_string())
    for date in sorted(f[f.gps_ok].date.unique()):
        col = wide[date] if date in wide else pd.Series(0, index=wide.index)
        print(f"\n{date}: sampled faces seen (>= 50% of length) {(col >= 0.5).sum()} of {len(sample)}, "
              f"partly (1-49%) {((col > 0) & (col < 0.5)).sum()}, not at all {(col == 0).sum()}")
    if len(wide.columns):
        best = wide.max(axis=1)
        print(f"\nall GPS days together: {(best >= 0.5).sum()} of {len(sample)} faces seen >= 50%")
        missing = best[best < 0.5].index
        print("not yet seen well:", ", ".join(missing) if len(missing) else "none")
    print(f"-> {out.relative_to(ROOT)}, {(DERIVED / 'coverage.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
