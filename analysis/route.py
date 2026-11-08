"""
Car route through the sampled block faces: one loop per morning, two mounted iPhones.
    R camera  front passenger window, films the right curb. Used on every street
    L camera  rear driver-side window, films the left curb. Used only on one-way streets
So a one-way block is filmed on both sides in one pass, and a two-way block needs one pass
in each direction with the face on the right.

Car rules: one-way streets, posted turn restrictions, no bike paths, no alleys, no U-turns
(except at dead ends). Turn restrictions come from the City's road sign layer
(OBLIG. TOUT DROIT OU A DROITE = no left, etc.). Each sign is matched to the street
approach it stands on and the next intersection ahead. Signs with time limits are
treated as always in force, which is the cautious choice.

Routing runs on movements, not intersections: a node is a directed street segment and an
edge is an allowed turn onto the next one. The visiting order is optimised (asymmetric
travelling salesman, nearest-next then move and swap local search). Start and end at
Peel and Sainte-Catherine.

Inputs:  data/derived/sample.gpkg, blockfaces.gpkg, data/clipped/study_area.gpkg, raw géobase
Outputs: data/derived/route/route.gpx          one track, "Car loop"
         data/derived/route/route.gpkg         the loop, with filming vs transit parts
         data/derived/route/turn_by_turn.csv   and turn_by_turn.txt, numbered steps
         data/derived/route/car_loop.png       map with the same step numbers

Usage:
    python analysis/route.py
"""
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import geopandas as gpd
import matplotlib
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point, box
from shapely.ops import linemerge

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GPKG = ROOT / "data" / "clipped" / "study_area.gpkg"
RAW_GEOBASE = ROOT / "data" / "raw" / "geobase" / "Géobase.json"
GRAPH_BUFFER_M = 600  # wide enough that one-way streets leaving the area still connect back
DERIVED = ROOT / "data" / "derived"
OUT = DERIVED / "route"

START = ("Peel", "Sainte-Catherine")
SPEEDS_KMH = (15, 20)    # average driving speed, stops and lights included
WINDOW_MIN = 165         # 08:15 to 11:00
SIGN_TO_NODE_M = 40      # a turn sign applies to the intersection it faces, within this distance
SIGN_TO_STREET_M = 15    # and stands this close to the approach street's centre line
CORNER_MARGIN_M = 4      # and is at least this much closer to it than to the cross street
LEFT_TURN_PENALTY_M = 30  # small preference for right turns, as in most car routers
GRID_EAST_BEARING = 34   # true bearing of Montréal's grid east (same as blockfaces.py)
GRID_ROTATION = GRID_EAST_BEARING - 90  # shapely angle (counter-clockwise) that puts grid north at the top
UTURN_DEG = 125         # a turn sharper than this is a U-turn (e.g. around a median tip)
CROSSOVER_M = 40        # segments shorter than this can be median crossovers (see movement_graph)

# allowed movements for each turn sign. Everything else from that approach is banned
TURN_SIGNS = {
    "OBLIG. TOUT DROIT OU A DROITE": {"straight", "right"},
    "OBLIG. TOUT DROIT OU A GAUCHE": {"straight", "left"},
    "OBLIG. A DROITE": {"right"},
    "OBLIG. A GAUCHE": {"left"},
    "OBLIG. A DROITE OU A GAUCHE": {"left", "right"},
    "INTERDICTION VIRAGE A GAUCHE": {"straight", "right"},
    "INTERDICTION VIRAGE A DROITE": {"straight", "left"},
    "INTERDICTION VIRAGE EN U": {"straight", "left", "right"},
}


def node(p):
    return (round(p[0], 1), round(p[1], 1))


def bearing(a, b):
    """Compass bearing from point a to point b, degrees."""
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360


def end_bearing(geom, at_end, d=12):
    """Travel bearing over the last (at_end) or first d metres of a line."""
    L = geom.length
    if at_end:
        a, b = geom.interpolate(max(L - d, 0)), geom.interpolate(L)
    else:
        a, b = geom.interpolate(0), geom.interpolate(min(d, L))
    return bearing((a.x, a.y), (b.x, b.y))


def turn_kind(e_in, e_out):
    """straight / left / right / u from the bearing change between two directed segments."""
    if e_in["id_trc"] == e_out["id_trc"]:
        return "u"
    delta = (end_bearing(e_out["geom"], False) - end_bearing(e_in["geom"], True) + 540) % 360 - 180
    if abs(delta) <= 35:
        return "straight"
    if abs(delta) >= UTURN_DEG:
        return "u"
    return "right" if delta > 0 else "left"


def build_graph():
    """
    Directed street graph for a car. Edge attrs: id_trc, length, geom (in travel direction),
    street, filmable. Built from the full géobase within GRAPH_BUFFER_M of the study area.
    """
    area = gpd.read_file(ROOT / "study_area.geojson").to_crs("EPSG:32188").buffer(GRAPH_BUFFER_M)
    g = gpd.read_file(RAW_GEOBASE, bbox=tuple(area.to_crs(4326).total_bounds)).to_crs("EPSG:32188")
    g = g[g.intersects(area.iloc[0])]
    # no pedestrian streets (1), expressways (2), ramps (8), alleys or park roads.
    # City-street underpasses (class 5 tunnels, e.g. Fort) are drivable, used for transit only
    g = g[~g.CLASSE.isin([1, 2, 8]) & ~g.TYP_VOIE.isin(["ruelle", "parc"]) & (g.NOM_VOIE != "Non-nommée")]
    G = nx.DiGraph()
    for r in g.itertuples():
        for ok, geom in [(r.SENS_CIR in (0, 1), r.geometry), (r.SENS_CIR in (0, -1), LineString(r.geometry.coords[::-1]))]:
            if not ok:
                continue
            u, v = node(geom.coords[0]), node(geom.coords[-1])
            if G.has_edge(u, v) and G[u][v]["length"] <= geom.length:
                continue
            G.add_edge(u, v, id_trc=int(r.ID_TRC), length=geom.length, geom=geom, street=r.NOM_VOIE,
                       one_way=r.SENS_CIR != 0, filmable=r.TYP_VOIE != "tunnel")
    return G, g


def turn_bans(G):
    """
    Set of banned movements ((u, v), (v, w)) from the turn signs. A sign stands on the
    approach segment, on the driver's right (or either side of a one-way), and governs the
    intersection ahead.
    """
    s = gpd.read_file(GPKG, layer="road_signs__Signalisation_routiere")
    s = s[(s.DESCRIPTION_REP == "Réel") & s.DESCRIPTION_RPA.isin(TURN_SIGNS)]
    edges = list(G.edges)
    lines = gpd.GeoSeries([G[u][v]["geom"] for u, v in edges], crs="EPSG:32188")
    banned, used, log = set(), 0, []
    for sign in s.itertuples():
        p = sign.geometry
        near = lines.distance(p)
        best = None
        for k in near[near < SIGN_TO_STREET_M].sort_values().index:
            u, v = edges[k]
            e = G[u][v]
            d_end = e["geom"].length - e["geom"].project(p)
            if d_end > SIGN_TO_NODE_M:
                continue
            # corner poles sit near two streets. Only trust the sign if it is clearly
            # closer to this approach than to any other street meeting at v
            others = [near[edges.index((v, w))] for w in G.successors(v) if G[v][w]["id_trc"] != e["id_trc"]]
            others += [near[edges.index((x, v))] for x in G.predecessors(v) if G[x][v]["id_trc"] != e["id_trc"]]
            if others and min(others) - near[k] < CORNER_MARGIN_M:
                continue
            if not e["one_way"]:
                # two-way street: the sign is on the right of traffic heading into v
                on = e["geom"].interpolate(e["geom"].project(p))
                b = end_bearing(e["geom"], True)
                side = math.sin(math.radians(bearing((on.x, on.y), (p.x, p.y)) - b))
                if side < 0:
                    continue
            best = (u, v)
            break
        if best is None:
            continue
        u, v = best
        allowed = TURN_SIGNS[sign.DESCRIPTION_RPA]
        # only real intersections (3+ segments meet), and never ban every way out: in
        # both cases the sign is meant for a corner further on
        touching = {G[a][b]["id_trc"] for a, b in list(G.in_edges(v)) + list(G.out_edges(v))}
        kinds = {w: turn_kind(G[u][v], G[v][w]) for w in G.successors(v)}
        if len(touching) < 3 or not any(k in allowed for k in kinds.values()):
            continue
        used += 1
        for w, kind in kinds.items():
            if kind not in allowed and kind != "u":
                banned.add(((u, v), (v, w)))
                log.append({"sign": sign.DESCRIPTION_RPA, "pole": sign.POTEAU_ID_POT, "from": G[u][v]["street"],
                            "banned_turn": kind, "onto": G[v][w]["street"], "geometry": Point(v)})
    return banned, used, len(s), gpd.GeoDataFrame(log, crs="EPSG:32188")


def movement_graph(G, banned, keep):
    """
    Line graph: node = directed segment, edge = allowed turn. Weight = length of the next segment.

    A U-turn through a divided street's median (left onto a short crossover, then left again
    back the way you came) looks like two legal left turns. To block it, each short segment is
    split into one copy per approach, so the exit can depend on where you came from.
    Segments in `keep` (filming passes, start and end) are not split.
    """
    M = nx.DiGraph()
    for u, v in G.edges:
        M.add_node((u, v), len=G[u][v]["length"])
        out = list(G.successors(v))
        for w in out:
            nxt = ((u, v), (v, w))
            kind = turn_kind(G[u][v], G[v][w])
            # U-turns only at a true dead end: back along the same segment. A carriageway that
            # stops because the graph drops an expressway ramp is not a place to turn around
            if kind == "u" and (len(out) > 1 or G[v][w]["id_trc"] != G[u][v]["id_trc"]):
                continue
            if nxt in banned:
                continue
            M.add_edge((u, v), (v, w), weight=G[v][w]["length"] + (LEFT_TURN_PENALTY_M if kind == "left" else 0), turn=kind)

    for b in [n for n in M.nodes if M.nodes[n]["len"] < CROSSOVER_M and n not in keep]:
        preds, succs = list(M.predecessors(b)), list(M.successors(b))
        for a in preds:
            via = ("via", b, a)
            M.add_node(via, len=M.nodes[b]["len"])
            M.add_edge(a, via, **M[a][b])
            sa = segment(a)
            back_in = end_bearing(G[sa[0]][sa[1]]["geom"], True)
            for c in succs:
                sc = segment(c)
                out_b = end_bearing(G[sc[0]][sc[1]]["geom"], False)
                if abs((out_b - back_in + 540) % 360 - 180) >= UTURN_DEG:  # leaving the way you came
                    continue
                M.add_edge(via, c, **M[b][c])
        M.remove_node(b)
    return M


def segment(m):
    """The street segment a movement-graph node stands for."""
    return m[1] if m[0] == "via" else m


def required_passes(sample, centre, G):
    """
    For each sampled face, the directed segments that film it and with which camera.
    Face on the right of a legal direction -> R. Face on the left of a one-way -> L.
    Returns {(u, v): [(face, camera), ...]}.
    """
    need = {}
    for f in sample.itertuples():
        mid = f.geometry.interpolate(0.5, normalized=True)
        for tid in map(int, f.id_trc.split(",")):
            seg = centre.loc[tid]
            line = seg.geometry
            d = line.project(mid)
            a, b = line.interpolate(max(d - 1, 0)), line.interpolate(min(d + 1, line.length))
            on_line = line.interpolate(d)
            cross = (b.x - a.x) * (mid.y - on_line.y) - (b.y - a.y) * (mid.x - on_line.x)
            face_right_fwd = cross < 0
            fwd_ok, back_ok = seg.SENS_CIR in (0, 1), seg.SENS_CIR in (0, -1)
            u, v = node(line.coords[0]), node(line.coords[-1])
            if face_right_fwd and fwd_ok:
                arc, cam = (u, v), "R"
            elif not face_right_fwd and back_ok:
                arc, cam = (v, u), "R"
            elif fwd_ok:
                arc, cam = (u, v), "L"
            else:
                arc, cam = (v, u), "L"
            if not G.has_edge(*arc):
                raise SystemExit(f"{f.blockface_id}: segment {tid} is not in the car graph")
            need.setdefault(arc, []).append((f.blockface_id, cam))
    return need


def make_tasks(need, G, M):
    """
    Chain required segments that follow each other on the same street into tasks (lists of
    directed segments). Only legal movements are chained, so the two directions of a
    two-way block stay separate passes.
    """
    arcs = list(need)
    tasks = []
    while arcs:
        chain = [arcs.pop(0)]
        grown = True
        while grown:
            grown = False
            for e in list(arcs):
                same = G[e[0]][e[1]]["street"] == G[chain[0][0]][chain[0][1]]["street"]
                if same and M.has_edge(chain[-1], e) and M[chain[-1]][e]["turn"] == "straight":
                    chain.append(e); arcs.remove(e); grown = True
                elif same and M.has_edge(e, chain[0]) and M[e][chain[0]]["turn"] == "straight":
                    chain.insert(0, e); arcs.remove(e); grown = True
        tasks.append(chain)
    return tasks


def order_tasks(M, tasks, starts, ends):
    """
    Order the tasks to cut transit distance. starts = movements leaving the start corner,
    ends = movements arriving at it. Distances are shortest legal paths in the movement graph.
    """
    M = M.copy()
    for s in starts:
        M.add_edge("S", s, weight=M.nodes[s]["len"])
    for e in ends:
        M.add_edge(e, "T", weight=0)
    heads = [t[0] for t in tasks]
    dist = {}
    for src in ["S"] + [t[-1] for t in tasks]:
        dist[src] = nx.single_source_dijkstra_path_length(M, src, weight="weight")

    def d(a, b):
        return dist[a].get(b, 1e9)

    def cost(order):
        here, total = "S", 0.0
        for i in order:
            total += d(here, heads[i])
            here = tasks[i][-1]
        return total + d(here, "T")

    left, order, here = set(range(len(tasks))), [], "S"
    while left:
        i = min(left, key=lambda k: d(here, heads[k]))
        order.append(i); left.remove(i); here = tasks[i][-1]
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
    unreachable = [i for i in range(len(tasks)) if d("S", heads[i]) >= 1e9]
    return order, M, unreachable


def plan_loop(M, tasks, starts, ends):
    """The full loop as a list of directed segments, with the task each filming segment belongs to."""
    order, Ms, unreachable = order_tasks(M, tasks, starts, ends)
    if unreachable:
        raise SystemExit(f"{len(unreachable)} tasks can't be reached legally: {[tasks[i] for i in unreachable]}")
    path, here = [], "S"
    for i in order:
        hops = nx.shortest_path(Ms, here, tasks[i][0], weight="weight")
        path += [(m, False) for m in hops[1:-1] if m not in ("S", "T")]
        path += [(m, True) for m in tasks[i]]
        here = tasks[i][-1]
    hops = nx.shortest_path(Ms, here, "T", weight="weight")
    path += [(m, False) for m in hops[1:-1]]
    return path


def to_lines(G, path, need):
    rows = []
    for k, ((u, v), filming) in enumerate(path):
        e = G[u][v]
        who = need.get((u, v), []) if filming else []
        rows.append({"order": k, "filming": filming, "street": e["street"], "one_way": e["one_way"],
                     "length_m": round(e["length"], 1),
                     "faces_R": ",".join(f for f, c in who if c == "R"),
                     "faces_L": ",".join(f for f, c in who if c == "L"),
                     "u": u, "v": v, "geometry": e["geom"]})
    return gpd.GeoDataFrame(rows, crs="EPSG:32188")


def turn_by_turn(G, lines):
    """
    Group the loop into numbered steps: a new step at every change of street or of filming.
    Each step says how to get onto it (turn), how far, and which curbs are being filmed.
    """
    steps, prev = [], None
    for r in lines.itertuples():
        key = (r.street, r.filming)
        if steps and key == steps[-1]["key"]:
            s = steps[-1]
            s["length_m"] += r.length_m
            s["faces_R"] |= set(filter(None, r.faces_R.split(",")))
            s["faces_L"] |= set(filter(None, r.faces_L.split(",")))
            s["geoms"].append(r.geometry)
        else:
            turn = "start" if prev is None else (
                "continue" if prev.street == r.street else turn_kind(G[prev.u][prev.v], G[r.u][r.v]))
            steps.append({"key": key, "turn": turn, "street": r.street, "filming": r.filming,
                          "length_m": r.length_m, "faces_R": set(filter(None, r.faces_R.split(","))),
                          "faces_L": set(filter(None, r.faces_L.split(","))), "geoms": [r.geometry]})
        prev = r
    rows = []
    for n, s in enumerate(steps, 1):
        if s["turn"] == "start":
            how = f"Start on {s['street']}"
        elif s["turn"] == "continue":
            how = f"Continue on {s['street']}"
        elif s["turn"] == "straight":
            how = f"Straight onto {s['street']}"
        else:
            how = f"Turn {s['turn']} onto {s['street']}"
        if s["filming"]:
            cams = "R+L" if s["faces_L"] and s["faces_R"] else ("L only" if s["faces_L"] else "R only")
            film = f"SAMPLE, slow down ({cams})"
        else:
            film = ""
        start_pt = s["geoms"][0].interpolate(0)
        rows.append({"step": n, "instruction": how, "metres": round(s["length_m"]), "filming": film,
                     "faces_R": " ".join(sorted(s["faces_R"])), "faces_L": " ".join(sorted(s["faces_L"])),
                     "x": start_pt.x, "y": start_pt.y})
    return pd.DataFrame(rows)


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


def short_name(name):
    """Shorter street labels for the map."""
    return re.sub(r"^(Boulevard|Avenue|Rue|Chemin) (de la |de |du |des )?", "", name)


def grid_north(gdf, origin):
    """Rotate so Montréal's street grid is square on the page (grid east = 34° true points right)."""
    out = gdf.copy()
    out["geometry"] = gdf.geometry.rotate(GRID_ROTATION, origin=origin)
    return out


def draw(lines, steps, sample_faces, streets, title, path, start_pt):
    # turn everything to grid north, the way people in Montréal read the downtown map
    origin = start_pt
    lines, sample_faces, streets = (grid_north(g, origin) for g in (lines, sample_faces, streets))
    pts = grid_north(gpd.GeoDataFrame(steps, geometry=gpd.points_from_xy(steps.x, steps.y), crs="EPSG:32188"), origin)
    steps = steps.assign(x=pts.geometry.x, y=pts.geometry.y)
    x0, y0, x1, y1 = lines.total_bounds
    fig, ax = plt.subplots(figsize=(18, 18 * (y1 - y0 + 160) / (x1 - x0 + 160)))
    streets.plot(ax=ax, color="#d9d9d9", lw=0.7)
    sample_faces.plot(ax=ax, color="#fdae61", lw=5, alpha=0.6)
    lines[~lines.filming].plot(ax=ax, color="#2c7bb6", lw=1.2, ls="--")
    lines[lines.filming].plot(ax=ax, color="#2c7bb6", lw=2.8)
    for geom in lines.geometry:  # direction arrows
        if geom.length < 40:
            continue
        a = geom.interpolate(geom.length / 2 - 4); b = geom.interpolate(geom.length / 2 + 4)
        ax.annotate("", xy=(b.x, b.y), xytext=(a.x, a.y),
                    arrowprops=dict(arrowstyle="-|>", color="#08306b", lw=1.2, mutation_scale=13))
    for s in steps.itertuples():
        ax.text(s.x, s.y, str(s.step), fontsize=7, weight="bold", color="white", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.18", fc="#d7191c" if s.filming else "#08306b", ec="none"), clip_on=True, zorder=6)
    ax.plot(start_pt.x, start_pt.y, marker="*", color="#ffd700", ms=22, mec="k")
    view = box(x0 - 60, y0 - 60, x1 + 60, y1 + 60)
    for name, grp in streets.groupby("NOM_VOIE"):
        clip = grp.union_all().intersection(view)
        if clip.is_empty or clip.length < 250 or "Line" not in clip.geom_type:
            continue
        piece = linemerge(clip) if clip.geom_type == "MultiLineString" else clip
        if piece.geom_type == "MultiLineString":
            piece = max(piece.geoms, key=lambda g: g.length)
        # label in the middle of the street's visible stretch, along the street
        m = piece.interpolate(0.5, normalized=True)
        a, b = piece.interpolate(max(piece.project(m) - 10, 0)), piece.interpolate(min(piece.project(m) + 10, piece.length))
        ang = math.degrees(math.atan2(b.y - a.y, b.x - a.x))
        ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
        ax.text(m.x, m.y, short_name(name), fontsize=8, color="#333", ha="center", va="center", rotation=ang,
                rotation_mode="anchor", bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.4), clip_on=True, zorder=1.5)
    ax.set_xlim(x0 - 80, x1 + 80); ax.set_ylim(y0 - 80, y1 + 80)
    ax.annotate("grid N", xy=(0.97, 0.95), xytext=(0.97, 0.85), xycoords="axes fraction", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=2), fontsize=10, weight="bold")
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sample = gpd.read_file(DERIVED / "sample.gpkg")
    faces = gpd.read_file(DERIVED / "blockfaces.gpkg").set_index("blockface_id")
    sample["id_trc"] = sample.blockface_id.map(faces.id_trc)
    G, streets = build_graph()
    centre = streets.set_index("ID_TRC")

    peel = streets[streets.NOM_VOIE == START[0]].union_all()
    stc = streets[streets.NOM_VOIE == START[1]].union_all()
    corner = peel.intersection(stc).centroid
    start = min(G.nodes, key=lambda n: Point(n).distance(corner))
    starts = [(start, w) for w in G.successors(start)]
    ends = [(u, start) for u in G.predecessors(start)]

    need = required_passes(sample, centre, G)
    banned, signs_used, signs_all, ban_log = turn_bans(G)
    M = movement_graph(G, banned, keep=set(need) | set(starts) | set(ends))
    tasks = make_tasks(need, G, M)
    path = plan_loop(M, tasks, starts, ends)
    # every move must be a legal turn in the movement graph, and the loop must close
    moves = [m for m, _ in path]
    illegal = [(segment(a), segment(b)) for a, b in zip(moves, moves[1:]) if not M.has_edge(a, b)]
    if illegal or segment(moves[0])[0] != start or segment(moves[-1])[1] != start:
        raise SystemExit(f"route check failed: illegal moves {illegal}")
    path = [(segment(m), f) for m, f in path]
    # and no U-turns, except back along the same segment at a dead end
    segs = [m for m, _ in path]
    uturns = [G[a[0]][a[1]]["street"] for a, b in zip(segs, segs[1:])
              if turn_kind(G[a[0]][a[1]], G[b[0]][b[1]]) == "u" and G[a[0]][a[1]]["id_trc"] != G[b[0]][b[1]]["id_trc"]]
    if uturns:
        raise SystemExit(f"route check failed: U-turns on {uturns}")
    lines = to_lines(G, path, need)
    steps = turn_by_turn(G, lines)

    km = lines.length_m.sum() / 1000
    film_km = lines[lines.filming].length_m.sum() / 1000
    mins = {s: km / s * 60 for s in SPEEDS_KMH}
    covered = {f for x in lines.faces_R.tolist() + lines.faces_L.tolist() for f in x.split(",") if f}
    missing = set(sample.blockface_id) - covered

    # outputs
    write_gpx([("Car loop", lines)], OUT / "route.gpx")
    gpkg = OUT / "route.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    lines.drop(columns=["u", "v"]).to_file(gpkg, layer="car_loop", driver="GPKG")
    ban_log.to_file(gpkg, layer="turn_bans", driver="GPKG")
    steps.drop(columns=["x", "y"]).to_csv(OUT / "turn_by_turn.csv", index=False)
    with open(OUT / "turn_by_turn.txt", "w") as fh:
        fh.write(f"CIVE 540 car loop. Start and end: {START[0]} and {START[1]}. Leave 08:15.\n")
        fh.write(f"{km:.1f} km, about {mins[20]:.0f} to {mins[15]:.0f} min at 20 to 15 km/h.\n")
        fh.write("Both phones record the whole loop. SAMPLE = a sampled block, drive slowly and steadily.\n")
        fh.write("R+L = both curbs (one-way street), R only = right curb.\n\n")
        for s in steps.itertuples():
            film = f"   {s.filming}" if s.filming else ""
            fh.write(f"{s.step:3d}. {s.instruction} ({s.metres} m){film}\n")
    sp = Point(start)
    draw(lines, steps, sample, streets,
         f"Car loop: {len(sample)} faces, {km:.1f} km, about {mins[20]:.0f} to {mins[15]:.0f} min. "
         f"Start/end at the star (Peel and Ste-Catherine). Red numbers = sampled block, drive slowly",
         OUT / "car_loop.png", sp)

    print(f"turn signs: {signs_used} of {signs_all} matched to an approach, {len(banned)} movements banned")
    print(f"sample: {len(sample)} faces, {len(need)} segments to film, {len(tasks)} passes")
    print(f"faces covered: {len(covered)} of {len(sample)}. Missing: {sorted(missing)}")
    cams = pd.Series([c for v in need.values() for _, c in v]).value_counts()
    print(f"segment-face pairs by camera: {cams.to_dict()}")
    print(f"loop: {km:.1f} km ({film_km:.1f} km filming, {km - film_km:.1f} km transit), {len(steps)} steps")
    for s, m in mins.items():
        print(f"  at {s} km/h: {m:.0f} min, fits 08:15-11:00 ({WINDOW_MIN} min): {m <= WINDOW_MIN}")
    print(f"-> {OUT.relative_to(ROOT)}/ (route.gpx, route.gpkg, turn_by_turn.csv/.txt, car_loop.png)")


if __name__ == "__main__":
    main()
