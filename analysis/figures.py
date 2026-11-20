"""
Slide and report figures, as PNGs in figures/ (committed: no frames, only City data and our results).

    study_area.png    sampled blocks and the double-parking events seen from the car
    supply.png        what the curb is for on a weekday at 9:00
    demand.png        deliveries a weekday per block face (high scenario)
    shortage.png      mismatch index per face, 07-11, high demand, 13.4 min stop
    test.png          double parking per km of curb filmed: busy vs quiet, short vs not
    pipeline.png      the method in one picture

Maps are turned so the Montréal street grid runs square to the page (grid north is 56 degrees
west of true north), with a true-north arrow to show it.

Usage:
    python analysis/figures.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import chi2
from shapely import affinity

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from supply import USE_PRIORITY, active, in_season, parse_schedule   # noqa: E402

DERIVED = ROOT / "data" / "derived"
CLIPPED = ROOT / "data" / "clipped" / "study_area.gpkg"
OUT = ROOT / "figures"
ROTATE = -56         # degrees. Grid east points 34 degrees east of true north, so this puts grid north up
DPI = 200
SNAPSHOT = ("Tu", 9 * 60)   # weekday 09:00 for the curb-use map

# reference palette (dataviz skill), light mode
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#c9c8c2"
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]          # blue 100 / 250 / 400 / 550 / 700
DIV = ["#1c5cab", "#86b6ef", "#b5b4ad", "#f0a3a3", "#c23434"]          # blue arm, gray mid, red arm

plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


# ---------------------------------------------------------------- map helpers
class Turn:
    """Rotate every layer about one fixed point so they stay aligned."""
    def __init__(self, area):
        self.origin = area.geometry.iloc[0].centroid

    def __call__(self, gdf):
        g = gdf.copy()
        g["geometry"] = g.geometry.map(lambda s: affinity.rotate(s, ROTATE, origin=self.origin))
        return g


def base_map(turn, area, streets, title, subtitle):
    fig = plt.figure(figsize=(12, 5.6))
    ax = fig.add_axes([0.02, 0.13, 0.96, 0.72])
    turn(streets).plot(ax=ax, color=GRID, linewidth=0.8, zorder=1)
    turn(area).boundary.plot(ax=ax, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)
    a = turn(area).total_bounds
    ax.set_xlim(a[0] - 60, a[2] + 60)
    ax.set_ylim(a[1] - 60, a[3] + 60)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    fig.text(0.04, 0.93, title, fontsize=17, fontweight="bold", color=INK)
    fig.text(0.04, 0.885, subtitle, fontsize=11.5, color=INK2)
    north_arrow(ax)
    scale_bar(ax)
    fig.text(0.04, 0.025, "Data: Ville de Montréal and Agence de mobilité durable open data (CC BY 4.0). "
             "Analysis: CIVE 540 curb project.", fontsize=8.5, color=MUTED)
    return fig, ax


def north_arrow(ax):
    """True-north arrow in the legend row, bottom right."""
    a = np.radians(90 + ROTATE)             # true north after the turn
    tip = (0.955, 0.11)
    ax.annotate("", xy=tip, xycoords="figure fraction", xytext=(-24 * np.cos(a), -24 * np.sin(a)),
                textcoords="offset points", arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2, mutation_scale=12))
    ax.figure.text(tip[0] + 0.008, tip[1] + 0.004, "N", ha="left", va="center", fontsize=10, color=INK2)


def scale_bar(ax):
    """500 m bar in the legend row, bottom right. Measured after the map's aspect is fixed."""
    ax.apply_aspect()
    x0, x1 = ax.get_xlim()
    pos = ax.get_position()
    w = 500 / (x1 - x0) * pos.width
    right, y = 0.925, 0.085
    ax.figure.add_artist(Line2D([right - w, right], [y, y], color=INK2, lw=1.5, transform=ax.figure.transFigure))
    ax.figure.text(right - w / 2, y + 0.012, "500 m", ha="center", va="bottom", fontsize=9, color=INK2)


def legend(ax, items, title=None, loc="upper left", lw=4):
    handles = [Line2D([0], [0], color=c, lw=lw, solid_capstyle="round", label=l) for l, c in items]
    return bottom_legend(ax, handles, title)


def bottom_legend(ax, handles, title=None):
    lg = ax.figure.legend(handles=handles, title=title, loc="lower left", bbox_to_anchor=(0.035, 0.065),
                          ncol=len(handles) + 1, frameon=False, fontsize=10, title_fontsize=10,
                          handlelength=1.8, columnspacing=1.6)
    if title:   # title on the same row, to the left
        lg.get_title().set_color(INK2)
        lg._legend_box.align = "left"
    return lg


# ---------------------------------------------------------------- maps
def supply_map(turn, area, streets, faces):
    s = gpd.read_file(DERIVED / "curb_supply.gpkg", layer="stretches")
    s = s[s.season.map(in_season)]
    s = s[[active(parse_schedule(x), *SNAPSHOT) for x in s.schedule]]
    group = {"loading": "Loading or short-term", "short_term": "Loading or short-term",
             "paid": "Paid or free parking", "free": "Paid or free parking",
             "reserved": "Reserved or permit", "permit": "Reserved or permit", "bus_stop": "Reserved or permit",
             "no_stopping": "No stopping or parking", "no_parking": "No stopping or parking",
             "implicit_no_parking": "No stopping or parking"}
    colour = {"Loading or short-term": BLUE, "Paid or free parking": ORANGE,
              "Reserved or permit": AQUA, "No stopping or parking": NEUTRAL}
    fig, ax = base_map(turn, area, streets, "Most curb is parking, very little is for loading",
                       "What each metre of curb is for on a weekday at 9:00. 522 block faces, 49 km of curb.")
    turn(faces).plot(ax=ax, color=ORANGE, linewidth=1.6, zorder=3)      # no rule = free parking
    s["rank"] = s.use.map({u: i for i, u in enumerate(USE_PRIORITY)})
    for _, g in turn(s).sort_values("rank", ascending=False).groupby("rank", sort=False):   # strictest drawn last
        u = group[g.use.iat[0]]
        g.plot(ax=ax, color=colour[u], linewidth=3.2 if u.startswith("Loading") else 1.6, zorder=4 + (u.startswith("Loading")))
    legend(ax, list(colour.items()))
    return fig


def demand_map(turn, area, streets):
    d = gpd.read_file(DERIVED / "demand.gpkg", layer="demand_by_face")
    bins = [0, 1, 3, 6, 12, np.inf]
    labels = ["under 1", "1 to 3", "3 to 6", "6 to 12", "12 or more"]
    d["cls"] = pd.cut(d.deliveries_high, bins, right=False, labels=labels)
    fig, ax = base_map(turn, area, streets, "Deliveries cluster on Sainte-Catherine and the cross streets",
                       "Modelled deliveries a weekday per block face, high scenario. 1,919 in total.")
    for lab, c in zip(labels, SEQ):
        g = d[d.cls == lab]
        if len(g):
            turn(g).plot(ax=ax, color=c, linewidth=1.4 + labels.index(lab) * 0.6, zorder=3 + labels.index(lab))
    legend(ax, list(zip(labels, SEQ)), title="Deliveries a day")
    return fig


def shortage_map(turn, area, streets):
    m = gpd.read_file(DERIVED / "loading.gpkg", layer="mismatch")
    m = m[(m.window == "07-11") & (m.demand == "high") & (m.dwell_min.astype(float) == 13.4)]   # stored as text
    m = m[(m.required_m > 0) | (m.supplied_m > 0)]                  # faces with neither are left as plain street
    bins = [-1.01, -0.5, -0.25, 0.25, 0.75, 1.01]
    labels = ["More than enough", "Enough", "About right", "Short", "Very short"]
    m["cls"] = pd.cut(m.mi, bins, labels=labels)
    short = (m.mi > 0.25).sum()                                     # "clearly short", as in loading.py
    fig, ax = base_map(turn, area, streets, f"The model says {short} block faces lack loading space at the morning peak",
                       "Loading space needed vs usable curb, 7 to 11 am, high demand. Model output, not observed. "
                       "Unmarked streets: no demand and no usable curb.")
    for i, (lab, c) in enumerate(zip(labels, DIV)):
        g = m[m.cls == lab]
        if len(g):
            turn(g).plot(ax=ax, color=c, linewidth=1.6 + (i >= 3) * 1.4, zorder=4 + i)
    legend(ax, list(zip(labels, DIV)), title="Loading space (mismatch index)")
    return fig


def study_map(turn, area, streets, faces):
    sample = gpd.read_file(DERIVED / "sample.gpkg")
    e = pd.read_csv(DERIVED / "dp_events.csv")
    e = e[e.code == "double"]
    fr = gpd.read_file(DERIVED / "frames.gpkg")
    e["frame_id"] = e.frames.str.split(" .. ").str[0]
    pts = fr.set_index("frame_id").loc[e.frame_id].geometry.reset_index()
    pts = gpd.GeoDataFrame(e.reset_index(drop=True), geometry=pts.geometry.values, crs=fr.crs)
    fig, ax = base_map(turn, area, streets, f"{len(e)} double-parking events on 3 mornings of filming",
                       "62 sampled block faces in 32 blocks, filmed Nov 9, 11 and 13, 2026. Each dot is one event, checked by hand.")
    turn(faces).plot(ax=ax, color=NEUTRAL, linewidth=1.0, zorder=3)
    turn(sample).plot(ax=ax, color=BLUE, linewidth=3.0, zorder=4)
    turn(pts).plot(ax=ax, color=ORANGE, markersize=110, edgecolor=SURFACE, linewidth=2, zorder=6)
    h = [Line2D([0], [0], color=BLUE, lw=4, label="Sampled block face"),
         Line2D([0], [0], color=NEUTRAL, lw=2, label="Other block face"),
         Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor=SURFACE,
                markersize=11, label="Double-parking event")]
    bottom_legend(ax, h)
    return fig


# ---------------------------------------------------------------- chart
def poisson_ci(k, exposure):
    """Exact 95% interval for a Poisson rate (events per unit of exposure)."""
    lo = chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * k + 2) / 2
    return lo / exposure, hi / exposure


def test_chart():
    b = pd.read_csv(DERIVED / "dp_blocks.csv", index_col=0)
    b["dens"] = b.premises / b.length_m * 100
    med = b.dens.median()
    groups = [("Busy commercial\n(above median businesses\nper 100 m)", b.dens > med),
              ("Quiet\n(at or below median)", b.dens <= med),
              ("Model says short\nof loading space", b.mi > 0),
              ("Model says\nenough space", b.mi <= 0)]
    fig, ax = plt.subplots(figsize=(12, 5.6))
    xs = [0, 1, 2.6, 3.6]
    for x, (lab, mask) in zip(xs, groups):
        g = b[mask]
        k, km = int(g.events.sum()), g.exposure_100m.sum() / 10
        rate = k / km
        lo, hi = poisson_ci(k, km)
        colour = BLUE if x < 2 else ORANGE
        ax.plot([x, x], [lo, hi], color=colour, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([x], [rate], s=140, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(x + 0.09, rate, f"{rate:.2f}", va="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(x, -0.62, lab, ha="center", va="top", fontsize=10.5, color=INK)
        ax.text(x, -0.38, f"{k} events, {km:.1f} km, {len(g)} blocks", ha="center", va="top", fontsize=9, color=INK2)
    ax.set_xlim(-0.6, 4.2)
    ax.set_ylim(0, max(3.2, ax.get_ylim()[1]))
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("Double-parking events per km of curb filmed")
    ax.axvline(1.8, color=GRID, lw=1)
    ax.text(0.5, ax.get_ylim()[1] * 0.97, "Commercial activity", ha="center", fontsize=11.5, fontweight="bold", color=BLUE)
    ax.text(3.1, ax.get_ylim()[1] * 0.97, "Modelled loading shortage", ha="center", fontsize=11.5, fontweight="bold", color=ORANGE)
    fig.text(0.04, 0.93, "Double parking follows commercial activity, not the modelled shortage",
             fontsize=17, fontweight="bold", color=INK)
    fig.text(0.04, 0.885, "31 filmed blocks, 9 events. Dots are rates, lines are exact 95% Poisson intervals. "
             "Regression: business density p < 0.001, shortage p = 0.36.", fontsize=11.5, color=INK2)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.27)
    return fig


# ---------------------------------------------------------------- pipeline
def pipeline():
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 120)
    ax.set_ylim(4, 60)
    ax.axis("off")

    def box(x, y, w, title, body, colour=INK2, strong=False):
        ax.add_patch(FancyBboxPatch((x, y), w, 9.5, boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor=colour if strong else SURFACE, alpha=1,
                                    edgecolor=colour, linewidth=2 if strong else 1.2, zorder=2))
        ax.add_patch(FancyBboxPatch((x, y), w, 9.5, boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor=colour, alpha=0.10, edgecolor="none", zorder=1))
        ax.text(x + w / 2, y + 6.4, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK, zorder=3)
        ax.text(x + w / 2, y + 2.8, body, ha="center", va="center", fontsize=8.2, color=INK2, zorder=3)
        return (x, y, w)

    def arrow(a, b, rad=0.0):
        (x0, y0), (x1, y1) = a, b
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=MUTED,
                                     lw=1.2, connectionstyle=f"arc3,rad={rad}", zorder=1))

    W, G = 17, 3.5
    xs = [2 + i * (W + G) for i in range(5)]
    ya, yb = 36, 14
    ax.text(2, 47.5, "City open data", fontsize=12, fontweight="bold", color=BLUE)
    ax.text(2, 28, "Our video", fontsize=12, fontweight="bold", color=ORANGE)
    top = [("Signs + paid spaces", "3,315 signs, 2,354 spaces"), ("Curb supply", "use of each metre, by hour"),
           ("Business list", "1,287 occupied units"), ("Delivery demand", "970 to 1,919 a day"),
           ("Loading model", "Erlang B, shortage per face")]
    bot = [("2 phones, 3 mornings", "Nov 9, 11, 13, 2026"), ("28,000 frames", "1 a second, GPS matched"),
           ("AI flags (Claude Haiku)", "531 candidate frames, $3.80"), ("Human check", "partner reviewed all 531"),
           ("Double-parking events", "9 events, 31 blocks")]
    for x, (t, s) in zip(xs, top):
        box(x, ya, W, t, s, BLUE)
    for x, (t, s) in zip(xs, bot):
        box(x, yb, W, t, s, ORANGE)
    for i in range(4):
        if i != 1:                               # supply and the business list are separate inputs
            arrow((xs[i] + W, ya + 4.75), (xs[i + 1], ya + 4.75))
        arrow((xs[i] + W, yb + 4.75), (xs[i + 1], yb + 4.75))
    arrow((xs[1] + W / 2, ya + 9.5), (xs[4] + W / 2, ya + 9.5), rad=-0.12)   # supply feeds the loading model, over the top
    # the test, joining both rows
    tx = xs[4] + W + 3.5
    ax.add_patch(FancyBboxPatch((tx - 0.5, 23), 119 - tx, 22, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=INK, edgecolor=INK, zorder=2))
    ax.text(tx + (118.5 - tx) / 2, 37, "Test", ha="center", fontsize=12, fontweight="bold", color=SURFACE, zorder=3)
    ax.text(tx + (118.5 - tx) / 2, 30, "count model\nper block", ha="center", va="center", fontsize=9, color=SURFACE, zorder=3)
    arrow((xs[4] + W, ya + 4.75), (tx - 0.5, 38))
    arrow((xs[4] + W, yb + 4.75), (tx - 0.5, 28))
    fig.text(0.04, 0.93, "From free City data and two phones to a tested map of the curb", fontsize=17,
             fontweight="bold", color=INK)
    fig.text(0.04, 0.885, "Every box is a script in the repo. The video checks what the City data predicts.",
             fontsize=11.5, color=INK2)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04)
    return fig


def main():
    OUT.mkdir(exist_ok=True)
    area = gpd.read_file(ROOT / "study_area.geojson").to_crs(32188)
    streets = gpd.read_file(CLIPPED, layer="geobase__Geobase").to_crs(32188)
    faces = gpd.read_file(DERIVED / "blockfaces.gpkg").to_crs(32188)
    turn = Turn(area)
    figs = {"study_area": study_map(turn, area, streets, faces), "supply": supply_map(turn, area, streets, faces),
            "demand": demand_map(turn, area, streets), "shortage": shortage_map(turn, area, streets),
            "test": test_chart(), "pipeline": pipeline()}
    for name, fig in figs.items():
        fig.savefig(OUT / f"{name}.png", dpi=DPI)
        plt.close(fig)
        print(f"-> figures/{name}.png")


if __name__ == "__main__":
    main()
