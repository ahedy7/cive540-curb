"""
Bike route through the sampled block faces, as two loops per morning:
    Loop R  camera facing right. Every sampled face you can have on your right riding legally
    Loop L  camera facing left. Left-side faces of one-way streets
Every face is filmed riding with traffic. Bidirectional bike tracks are used only to get
between blocks, never for filming.

The order is optimised: each face is a task, legal riding distances between tasks come from
shortest paths, and a local search reorders tasks to cut transit. Both loops start and end
at Peel and Sainte-Catherine.
A pilot route covers the sampled faces closest to that corner.

Inputs:  data/derived/sample.gpkg, blockfaces.gpkg, data/clipped/study_area.gpkg
Outputs: data/derived/route/route.gpx           tracks: Loop R, Loop L, Pilot
         data/derived/route/route.gpkg          the same lines, with filming vs transit parts
         data/derived/route/loop_R.png, loop_L.png, pilot.png   maps to follow on a phone

Usage:
    python analysis/route.py
"""
import math
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import geopandas as gpd
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GPKG = ROOT / "data" / "clipped" / "study_area.gpkg"
RAW_GEOBASE = ROOT / "data" / "raw" / "geobase" / "Géobase.json"
GRAPH_BUFFER_M = 300
DERIVED = ROOT / "data" / "derived"
OUT = DERIVED / "route"

START = ("Peel", "Sainte-Catherine")
PILOT_FACES = 14
FILM_KMH = 10      # slow, steady speed while filming
TRANSIT_KMH = 14
STOP_SHARE = 0.15  # extra time for lights and stops


def node(p):
    return (round(p[0], 1), round(p[1], 1))


def build_graph():
    """
    Directed street graph for a bike. Edge attrs: id_trc, length, geom (in travel direction).
    Built from the full géobase within GRAPH_BUFFER_M of the study area, so blocks you enter
    from just outside (e.g. Baile, one-way from the west) are reachable.
    """
    area = gpd.read_file(ROOT / "study_area.geojson").to_crs("EPSG:32188").buffer(GRAPH_BUFFER_M)
    g = gpd.read_file(RAW_GEOBASE, bbox=tuple(area.to_crs(4326).total_bounds)).to_crs("EPSG:32188")
    g = g[g.intersects(area.iloc[0])]
    # no expressways (class 2), ramps (8) or pedestrian streets (1). City-street underpasses
    # (tunnels of class 5, e.g. Fort and Saint-Marc) are rideable, so they stay for transit
    g = g[~g.CLASSE.isin([1, 2, 8]) & (g.NOM_VOIE != "Non-nommée")]
    bike = gpd.read_file(GPKG, layer="bike_network__Reseau_cyclable")
    two_way_bike = set(bike[bike.TYPE_VOIE2_DESC == "Piste bidirectionnelle"].ID_TRC.astype(int))
    G = nx.DiGraph()
    for r in g.itertuples():
        line = r.geometry
        fwd, back = r.SENS_CIR in (0, 1), r.SENS_CIR in (0, -1)
        contraflow_bike = r.ID_TRC in two_way_bike  # usable both ways, but not for filming
        tunnel = r.TYP_VOIE == "tunnel"
        for ok, geom, legal in [(fwd or contraflow_bike, line, fwd), (back or contraflow_bike, LineString(line.coords[::-1]), back)]:
            if not ok:
                continue
            u, v = node(geom.coords[0]), node(geom.coords[-1])
            if G.has_edge(u, v) and G[u][v]["length"] <= geom.length:
                continue
            G.add_edge(u, v, id_trc=r.ID_TRC, length=geom.length, geom=geom, legal=legal and not tunnel, street=r.NOM_VOIE)
    return G, g


def required_arcs(sample, centre):
    """
    For each sampled face, the segment directions that film it: (u, v, id_trc, loop, face).
    Face on your right riding legally -> loop R. One-way street with the face on your left -> loop L.
    """
    arcs = []
    for f in sample.itertuples():
        mid = f.geometry.interpolate(0.5, normalized=True)
        for tid in map(int, f.id_trc.split(",")):
            seg = centre.loc[tid]
            line = seg.geometry
            # which side of the digitized direction is the face on?
            d = line.project(mid)
            a, b = line.interpolate(max(d - 1, 0)), line.interpolate(min(d + 1, line.length))
            on_line = line.interpolate(d)
            cross = (b.x - a.x) * (mid.y - on_line.y) - (b.y - a.y) * (mid.x - on_line.x)
            face_right_fwd = cross < 0
            fwd_ok, back_ok = seg.SENS_CIR in (0, 1), seg.SENS_CIR in (0, -1)
            u, v = node(line.coords[0]), node(line.coords[-1])
            if face_right_fwd and fwd_ok:
                arcs.append((u, v, tid, "R", f.blockface_id))
            elif not face_right_fwd and back_ok:
                arcs.append((v, u, tid, "R", f.blockface_id))
            elif fwd_ok:
                arcs.append((u, v, tid, "L", f.blockface_id))
            else:
                arcs.append((v, u, tid, "L", f.blockface_id))
    return arcs


def make_tasks(arcs):
    """Chain a face's consecutive segment passes into one task: (start node, end node, [(u, v)])."""
    tasks = []
    for face in dict.fromkeys(a[4] for a in arcs):
        mine = [(u, v) for u, v, _, _, f in arcs if f == face]
        while mine:
            chain = [mine.pop(0)]
            grown = True
            while grown:
                grown = False
                for e in list(mine):
                    if e[0] == chain[-1][1]:
                        chain.append(e); mine.remove(e); grown = True
                    elif e[1] == chain[0][0]:
                        chain.insert(0, e); mine.remove(e); grown = True
            tasks.append((chain[0][0], chain[-1][1], chain))
    return tasks


def order_tasks(G, tasks, start):
    """
    Shortest order to ride the tasks (an asymmetric travelling-salesman problem, because of one-way
    streets). Start from the nearest-next order, then keep moving single tasks or swapping pairs
    while the total transit distance drops.
    """
    ends = [start] + [t[1] for t in tasks]
    dist = {e: nx.single_source_dijkstra_path_length(G, e, weight="length") for e in set(ends)}

    def d(a, b):
        return dist[a].get(b, 1e9)

    def cost(order):
        here, total = start, 0.0
        for i in order:
            total += d(here, tasks[i][0])
            here = tasks[i][1]
        return total + d(here, start)

    left, order, here = set(range(len(tasks))), [], start
    while left:
        i = min(left, key=lambda k: d(here, tasks[k][0]))
        order.append(i); left.remove(i); here = tasks[i][1]

    best = cost(order)
    improved = True
    while improved:
        improved = False
        for i in range(len(order)):
            for j in range(len(order)):
                if i == j:
                    continue
                cand = order[:i] + order[i + 1:]
                cand.insert(j, order[i])
                c = cost(cand)
                if c < best - 0.5:
                    order, best, improved = cand, c, True
                    break
            if improved:
                break
        if not improved:
            for i in range(len(order)):
                for j in range(i + 1, len(order)):
                    cand = order.copy(); cand[i], cand[j] = cand[j], cand[i]
                    c = cost(cand)
                    if c < best - 0.5:
                        order, best, improved = cand, c, True
                        break
                if improved:
                    break
    return order


def plan_loop(G, arcs, start):
    """Ride every required pass. Returns list of (u, v, filming) edges and passes it couldn't reach."""
    tasks = make_tasks(arcs)
    reachable = nx.descendants(G, start) | {start}
    missed = [t for t in tasks if t[0] not in reachable]
    tasks = [t for t in tasks if t[0] in reachable]
    path, here = [], start
    for i in order_tasks(G, tasks, start):
        s, e, chain = tasks[i]
        hops = nx.shortest_path(G, here, s, weight="length")
        path += [(a, b, False) for a, b in zip(hops, hops[1:])]
        path += [(u, v, True) for u, v in chain]
        here = e
    hops = nx.shortest_path(G, here, start, weight="length")
    path += [(a, b, False) for a, b in zip(hops, hops[1:])]
    return path, missed


def to_lines(G, path):
    rows = []
    for k, (a, b, filming) in enumerate(path):
        e = G[a][b]
        rows.append({"order": k, "filming": filming, "street": e["street"], "length_m": round(e["length"], 1), "geometry": e["geom"]})
    return gpd.GeoDataFrame(rows, crs="EPSG:32188")


def ride_minutes(lines):
    film = lines[lines.filming].length_m.sum() / 1000
    transit = lines[~lines.filming].length_m.sum() / 1000
    return (film / FILM_KMH + transit / TRANSIT_KMH) * 60 * (1 + STOP_SHARE), film, transit


def write_gpx(tracks, path):
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<gpx version="1.1" creator="cive540-curb" xmlns="http://www.topografix.com/GPX/1/1">',
           f'<metadata><name>CIVE 540 sample route</name><time>{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}</time></metadata>']
    for name, lines in tracks:
        ll = lines.to_crs(4326)
        out.append(f"<trk><name>{escape(name)}</name><trkseg>")
        last = None
        for geom in ll.geometry:
            for x, y in geom.coords:
                if (x, y) != last:
                    out.append(f'<trkpt lat="{y:.6f}" lon="{x:.6f}"/>')
                    last = (x, y)
        out.append("</trkseg></trk>")
    out.append("</gpx>")
    path.write_text("\n".join(out))


def draw(lines, sample_faces, streets, title, path, start_pt, highlight=None, zoom=None):
    fig, ax = plt.subplots(figsize=(9, 13))
    streets.plot(ax=ax, color="#d9d9d9", lw=0.7)
    sample_faces.plot(ax=ax, color="#fdae61", lw=4, alpha=0.6)
    if highlight is not None:
        highlight.plot(ax=ax, color="#d7191c", lw=5, alpha=0.7)
    lines[~lines.filming].plot(ax=ax, color="#2c7bb6", lw=1.2, ls="--")
    lines[lines.filming].plot(ax=ax, color="#2c7bb6", lw=2.6)
    # arrows and a running step number every ~250 m, so the order is easy to follow
    run = 0.0
    step = 1
    for geom in lines.geometry:
        if run <= 0:
            m = geom.interpolate(0.5, normalized=True)
            a = geom.interpolate(max(geom.length / 2 - 3, 0)); b = geom.interpolate(min(geom.length / 2 + 3, geom.length))
            ax.annotate("", xy=(b.x, b.y), xytext=(a.x, a.y), arrowprops=dict(arrowstyle="-|>", color="#08306b", lw=1.5, mutation_scale=16))
            ax.text(m.x, m.y, str(step), fontsize=8, weight="bold", color="white", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.2", fc="#08306b", ec="none"), clip_on=True)
            step += 1
            run = 250
        run -= geom.length
    ax.plot(start_pt.x, start_pt.y, marker="*", color="#d7191c", ms=18, mec="k")
    for name, grp in streets.groupby("NOM_VOIE"):
        if grp.length.sum() > 500:
            m = grp.union_all().interpolate(0.5, normalized=True)
            ax.text(m.x, m.y, name, fontsize=7, color="#444", alpha=0.9, clip_on=True)
    if zoom is not None:
        x0, y0, x1, y1 = zoom
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sample = gpd.read_file(DERIVED / "sample.gpkg")
    faces = gpd.read_file(DERIVED / "blockfaces.gpkg").set_index("blockface_id")
    sample["id_trc"] = sample.blockface_id.map(faces.id_trc)
    G, streets = build_graph()
    centre = streets.set_index("ID_TRC")

    # start at Peel and Sainte-Catherine
    peel = streets[streets.NOM_VOIE == START[0]].union_all()
    stc = streets[streets.NOM_VOIE == START[1]].union_all()
    corner = peel.intersection(stc).centroid
    start = min(G.nodes, key=lambda n: Point(n).distance(corner))

    arcs = required_arcs(sample, centre)
    results = {}
    for loop in ["R", "L"]:
        mine = [a for a in arcs if a[3] == loop]
        path, missed = plan_loop(G, mine, start)
        lines = to_lines(G, path)
        mins, film, transit = ride_minutes(lines)
        results[loop] = (lines, mine, missed, mins, film, transit)

    # pilot: the sampled faces nearest the start corner, both camera sides
    sample["to_start"] = sample.geometry.distance(Point(start))
    pilot_ids = set(sample.nsmallest(PILOT_FACES, "to_start").blockface_id)
    pilot_lines = []
    for loop in ["R", "L"]:
        mine = [a for a in arcs if a[3] == loop and a[4] in pilot_ids]
        if mine:
            path, _ = plan_loop(G, mine, start)
            pilot_lines.append(to_lines(G, path).assign(loop=loop))
    pilot = pd.concat(pilot_lines, ignore_index=True)
    pmins, pfilm, ptransit = ride_minutes(pilot)

    write_gpx([("Loop R (camera right)", results["R"][0]), ("Loop L (camera left)", results["L"][0]),
               ("Pilot (R then L)", pilot)], OUT / "route.gpx")
    gpkg = OUT / "route.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    for loop in ["R", "L"]:
        results[loop][0].assign(loop=loop).to_file(gpkg, layer=f"loop_{loop}", driver="GPKG")
    pilot.to_file(gpkg, layer="pilot", driver="GPKG")

    sp = Point(start)
    for loop, label in [("R", "camera RIGHT"), ("L", "camera LEFT")]:
        lines, mine, missed, mins, film, transit = results[loop]
        faces_here = sample[sample.blockface_id.isin({a[4] for a in mine})]
        draw(lines, faces_here, streets, f"Loop {loop}, {label}. {len(faces_here)} faces, {film + transit:.1f} km, about {mins:.0f} min. Start/end at the star",
             OUT / f"loop_{loop}.png", sp)
    pf = sample[sample.blockface_id.isin(pilot_ids)]
    x0, y0, x1, y1 = pf.total_bounds
    pad = 120
    draw(pilot, pf, streets, f"Pilot: {len(pf)} faces near Peel and Ste-Catherine. Loop R then L, {pfilm + ptransit:.1f} km, about {pmins:.0f} min",
         OUT / "pilot.png", sp, highlight=pf, zoom=(x0 - pad, y0 - pad, x1 + pad, y1 + pad))

    print(f"sample: {len(sample)} faces, {len(arcs)} segment passes needed")
    for loop in ["R", "L"]:
        lines, mine, missed, mins, film, transit = results[loop]
        nf = len({a[4] for a in mine})
        print(f"Loop {loop}: {nf} faces, filming {film:.1f} km + transit {transit:.1f} km, about {mins:.0f} min. Unreachable passes: {len(missed)}")
    total = results["R"][3] + results["L"][3]
    print(f"Both loops: about {total:.0f} min per morning")
    print(f"Pilot: {len(pf)} faces, {pfilm + ptransit:.1f} km, about {pmins:.0f} min")
    print(f"-> {OUT.relative_to(ROOT)}/ (route.gpx, route.gpkg, loop_R.png, loop_L.png, pilot.png)")


if __name__ == "__main__":
    main()
